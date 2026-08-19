// Mistral text-to-speech over the Voxtral speech HTTP API. The service requests
// PCM streaming, which arrives as server-sent events carrying base64 float32 LE
// samples at 24 kHz; each chunk is converted to 16-bit PCM and streamed
// downstream.

package mistral

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
	errs "github.com/gojargo/jargo/utils/errors"
)

const (
	ttsURL          = "https://api.mistral.ai/v1/audio/speech"
	ttsDefaultModel = "voxtral-mini-tts-2603"
	// ttsSampleRate is the native PCM rate of Mistral speech audio.
	ttsSampleRate = 24000

	// ttsEventAudioDelta is the only streamed event that carries speech. The
	// stream's other events (its closing "speech.audio.done", which reports
	// usage) carry none, and are passed over.
	ttsEventAudioDelta = "speech.audio.delta"
)

// errTTSStatus is returned when the speech API responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errTTSStatus = errors.New("mistral: tts unexpected status")

// TTSConfig configures the Mistral TTS service.
type TTSConfig struct {
	// APIKey is the Mistral API key, sent as a Bearer token. Required.
	APIKey string `validate:"required"`
	// Model is the TTS model id; empty uses a current default.
	Model string
	// Voice is the preset or custom voice id; empty lets the model choose.
	Voice string
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Mistral TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.Model == "" {
		cfg.Model = ttsDefaultModel
	}
	return tts.New("MistralTTS", &ttsSynthesizer{cfg: cfg, http: &http.Client{}})
}

type ttsSynthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports Mistral's native 24 kHz PCM rate.
func (s *ttsSynthesizer) SampleRate() int { return ttsSampleRate }

// Synthesize requests streaming speech for text and streams the converted PCM
// downstream.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	payload := map[string]any{
		"model":           s.cfg.Model,
		"input":           text,
		"response_format": "pcm",
		"stream":          true,
	}
	if s.cfg.Voice != "" {
		payload["voice_id"] = s.cfg.Voice
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ttsURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := s.http.Do(req) //nolint:gosec // request target is the service's configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errTTSStatus, resp.StatusCode, msg))
	}
	return streamEvents(resp.Body, emit)
}

// speechEvent captures both shapes a streamed speech event may take: an event
// name and audio carried either at the top level or nested under data.
type speechEvent struct {
	Event     string `json:"event"`
	AudioData string `json:"audio_data"`
	Data      struct {
		AudioData string `json:"audio_data"`
	} `json:"data"`
}

// streamEvents parses the speech SSE stream, decoding each audio chunk from
// base64 float32 PCM to 16-bit PCM and passing it to emit.
func streamEvents(r io.Reader, emit func(pcm []byte) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20)

	var eventName string
	var data strings.Builder
	dispatch := func() error {
		defer func() {
			eventName = ""
			data.Reset()
		}()
		if pcm := speechChunk(eventName, data.String()); pcm != nil {
			return emit(pcm)
		}
		return nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case line == "":
			if err := dispatch(); err != nil {
				return err
			}
		case strings.HasPrefix(line, ":"):
			// Comment/keep-alive line.
		case strings.HasPrefix(line, "event:"):
			eventName = strings.TrimSpace(line[len("event:"):])
		case strings.HasPrefix(line, "data:"):
			chunk := strings.TrimSpace(line[len("data:"):])
			if chunk == "[DONE]" {
				return nil
			}
			data.WriteString(chunk)
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return dispatch()
}

// speechChunk decodes one SSE event's audio payload to 16-bit PCM, returning nil
// when the event carries no audio: any event other than the audio delta, or an
// empty or unparseable payload.
//
// Only the delta event yields audio. Every other event is passed over, the done
// marker included, so an event that happens to carry an audio field without
// being a delta is not played.
func speechChunk(eventName, payload string) []byte {
	if payload == "" {
		return nil
	}
	var ev speechEvent
	if json.Unmarshal([]byte(payload), &ev) != nil {
		return nil
	}
	typ := eventName
	if ev.Event != "" {
		typ = ev.Event
	}
	if typ != ttsEventAudioDelta {
		return nil
	}
	audio := ev.AudioData
	if audio == "" {
		audio = ev.Data.AudioData
	}
	if audio == "" {
		return nil
	}
	raw, err := base64.StdEncoding.DecodeString(audio)
	if err != nil {
		return nil
	}
	return float32ToInt16(raw)
}

// float32ToInt16 converts float32 LE PCM samples to 16-bit LE PCM.
func float32ToInt16(data []byte) []byte {
	n := len(data) / 4
	out := make([]byte, n*2)
	for i := range n {
		f := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
		v := max(min(int32(f*32767), 32767), -32768)
		binary.LittleEndian.PutUint16(out[i*2:], uint16(int16(v)))
	}
	return out
}
