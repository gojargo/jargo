// Piper text-to-speech through a Piper HTTP server, which runs the voices out
// of process:
//
//	pip install "piper-tts[http]"
//	python -m piper.http_server -m en_US-ryan-high
//
// The server listens on port 5000 and synthesizes on POST /synthesize, so the
// base URL is http://localhost:5000/synthesize.

package piper

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// defaultHTTPSampleRate is the rate the HTTP service outputs when none is
// configured: the rate of most Piper voices ("medium" and "high").
const defaultHTTPSampleRate = 22050

// wavHeaderSize is the size of the canonical WAV header the server opens its
// response with.
const wavHeaderSize = 44

// HTTPConfig configures the Piper HTTP TTS service.
type HTTPConfig struct {
	// BaseURL is the URL of the Piper HTTP server's synthesis endpoint, which
	// requests are posted to as is (e.g. http://localhost:5000/synthesize).
	// Required. A trailing slash is removed, with a warning.
	BaseURL string `validate:"required,url"`
	// Voice is the voice the server is asked to speak in, such as
	// "en_US-ryan-high"; empty leaves the choice to the server, which uses the
	// voice it was started with.
	Voice string

	// SpeakerID is which speaker to use, for voices trained on several
	// speakers; nil leaves it to the server.
	SpeakerID *int
	// LengthScale is the phoneme length scale: below 1 is faster, above 1 is
	// slower. Nil leaves it to the server.
	LengthScale *float64
	// NoiseScale is the amount of generator noise; nil leaves it to the
	// server.
	NoiseScale *float64
	// NoiseWScale is the amount of phoneme width noise; nil leaves it to the
	// server.
	NoiseWScale *float64

	// SampleRate is the rate the service outputs; 0 uses 22050. The server
	// answers at the rate of the voice it speaks in, which its WAV header
	// gives, and the audio is resampled to this one.
	SampleRate int `validate:"gte=0"`
	// Client makes the requests; nil uses a client of its own.
	Client *http.Client
}

// Validate reports whether the configuration is usable.
func (c HTTPConfig) Validate() error { return validate.Struct(c) }

// NewHTTPTTS builds a TTS service speaking through a Piper HTTP server. Each
// utterance is posted to the server with the voice and the synthesis controls
// set, and the WAV it answers with is streamed downstream as it arrives, its
// header removed. NewTTS runs the voice in process instead.
func NewHTTPTTS(cfg HTTPConfig) *tts.Base {
	b, _ := newHTTPTTS(cfg)
	return b
}

// newHTTPTTS builds the service and returns its synthesizer alongside it.
func newHTTPTTS(cfg HTTPConfig) (*tts.Base, *httpSynthesizer) {
	if strings.HasSuffix(cfg.BaseURL, "/") {
		slog.Warn("Base URL ends with a slash, this is not allowed.", "service", "PiperHTTPTTS")
		cfg.BaseURL = strings.TrimSuffix(cfg.BaseURL, "/")
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultHTTPSampleRate
	}
	if cfg.Client == nil {
		cfg.Client = &http.Client{}
	}
	s := &httpSynthesizer{cfg: cfg, live: HTTPTTSSettings(newTTSSettings(cfg.Voice, synthesisConfig{
		speakerID:   cfg.SpeakerID,
		lengthScale: cfg.LengthScale,
		noiseScale:  cfg.NoiseScale,
		noiseWScale: cfg.NoiseWScale,
	}))}
	return tts.New("PiperHTTPTTS", s), s
}

// httpSynthesizer speaks each utterance with one request to the server.
type httpSynthesizer struct {
	cfg HTTPConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live HTTPTTSSettings
}

// SampleRate reports the PCM output rate.
func (s *httpSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder. Each request is built from it, so a change
// applies from the next utterance on.
func (s *httpSynthesizer) Settings() any { return &s.live }

// payload is the request body for text: the text and the voice, which is null
// when none is set, and each synthesis control that is set.
func (s *httpSynthesizer) payload(text string) map[string]any {
	data := map[string]any{
		"text":  text,
		"voice": s.live.Voice.AsAny(),
	}
	for _, name := range synthesisFields {
		if v, ok := settings.Get(&s.live, name); ok && v != nil {
			data[name] = v
		}
	}
	return data
}

// RunTTS requests speech for text and streams the audio as it arrives. A
// failure is yielded as an error in the context's stream.
func (s *httpSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	err := s.runTTS(ctx, text, yield)
	if err == nil || ctx.Err() != nil {
		return err
	}
	ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
	ef.Err = err
	return yield(ef)
}

// runTTS is RunTTS, returning what fails.
func (s *httpSynthesizer) runTTS(ctx context.Context, text string, yield func(f frames.Frame) error) error {
	body, err := json.Marshal(s.payload(text))
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.BaseURL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.cfg.Client.Do(req) //nolint:gosec // the target is the configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(resp.Body)
		return yield(frames.NewErrorFrame(fmt.Sprintf("Error getting audio (status: %d, error: %s)",
			resp.StatusCode, strings.ToValidUTF8(string(msg), ""))))
	}
	tts.StartUsageMetrics(ctx, text)
	return streamWAV(resp.Body, tts.ChunkSize(s.cfg.SampleRate), s.cfg.SampleRate, yield)
}

// streamWAV streams the audio body holds downstream a chunk of up to
// chunkSize bytes at a time, as it arrives. A body that opens with a WAV
// header has it removed, and its audio is labeled with the rate the header
// gives; other audio is labeled with defaultRate. Each chunk is cut to a whole
// number of samples, the odd byte carried into the next, and a last odd byte
// is padded.
func streamWAV(body io.Reader, chunkSize, defaultRate int, yield func(f frames.Frame) error) error {
	rate := 0
	stripHeader := true
	var pending []byte
	emit := func(audio []byte) error {
		r := rate
		if r == 0 {
			r = defaultRate
		}
		return yield(frames.NewTTSAudioRawFrame(audio, r, 1))
	}

	buf := make([]byte, max(chunkSize, wavHeaderSize))
	first := true
	for {
		var n int
		var readErr error
		if first {
			// The first read waits for a whole header, so one split across
			// reads is still recognized.
			n, readErr = io.ReadAtLeast(body, buf, wavHeaderSize)
			if errors.Is(readErr, io.ErrUnexpectedEOF) {
				readErr = io.EOF
			}
			first = false
		} else {
			n, readErr = body.Read(buf)
		}
		chunk := buf[:n]
		if stripHeader && n > 0 && bytes.HasPrefix(chunk, []byte("RIFF")) {
			if len(chunk) >= wavHeaderSize {
				rate = int(binary.LittleEndian.Uint32(chunk[24:28]))
			}
			chunk = chunk[min(wavHeaderSize, len(chunk)):]
			stripHeader = false
		}
		pending = append(pending, chunk...)
		if aligned := len(pending) &^ 1; aligned > 0 {
			audio := bytes.Clone(pending[:aligned])
			pending = append(pending[:0], pending[aligned:]...)
			if err := emit(audio); err != nil {
				return err
			}
		}
		if errors.Is(readErr, io.EOF) {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	if len(pending) > 0 {
		return emit(append(pending, 0))
	}
	return nil
}

// Compile-time check that the HTTP synthesizer takes settings the way the base
// applies them.
var _ tts.SettingsHolder = (*httpSynthesizer)(nil)
