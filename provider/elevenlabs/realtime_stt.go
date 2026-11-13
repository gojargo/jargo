package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// defaultRealtimeSTTHost is the ElevenLabs WebSocket host.
	defaultRealtimeSTTHost = "wss://api.elevenlabs.io"
	// realtimeSTTPath is the streaming transcription endpoint.
	realtimeSTTPath = "/v1/speech-to-text/realtime"
	// defaultRealtimeSTTModel is a current streaming transcription model.
	defaultRealtimeSTTModel = "scribe_v2_realtime"
	// realtimeSTTFallbackRate is the rate audio at a rate ElevenLabs does not
	// take is resampled to.
	realtimeSTTFallbackRate = 16000
)

// realtimeSTTRates are the rates ElevenLabs takes raw 16-bit PCM at, each
// named by its own audio format.
//
//nolint:gochecknoglobals // a fixed table
var realtimeSTTRates = []int{8000, 16000, 22050, 24000, 44100, 48000}

// realtimeSTTFormat is the audio format of raw 16-bit PCM at rate, which is one
// of realtimeSTTRates.
func realtimeSTTFormat(rate int) string { return "pcm_" + strconv.Itoa(rate) }

// ElevenLabs realtime transcription message types.
const (
	// rtEventStarted acknowledges the session.
	rtEventStarted = "session_started"
	// rtEventPartial is the transcript for the utterance in progress.
	rtEventPartial = "partial_transcript"
	// rtEventCommitted is the finalized transcript for an utterance.
	rtEventCommitted = "committed_transcript"
	// rtEventError reports a session-level failure.
	rtEventError = "error"
)

// RealtimeSTTConfig configures the ElevenLabs streaming STT service.
type RealtimeSTTConfig struct {
	// APIKey is the ElevenLabs API key, sent as the xi-api-key header. Required.
	APIKey string `validate:"required"`
	// Host overrides the WebSocket host; empty uses the hosted API.
	Host string
	// Model is the transcription model; empty uses a current streaming default.
	Model string
	// Language of the audio; the zero value lets the service detect it.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	// ElevenLabs takes 8000, 16000, 22050, 24000, 44100 or 48000 Hz; audio at
	// any other rate is resampled to 16000 Hz before it is sent.
	SampleRate int
	// Keyterms bias the transcription toward these words or phrases.
	Keyterms []string
	// SilenceThresholdSecs is how long the service waits through silence before
	// committing an utterance (0.3 to 3.0); 0 uses its default.
	SilenceThresholdSecs *float64 `validate:"omitempty,min=0.3,max=3"`
	// VADThreshold is how confident the detector must be that speech is present
	// (0 to 1); nil uses the service default.
	VADThreshold *float64 `validate:"omitempty,min=0,max=1"`
	// MinSpeechMS is the shortest run of audio treated as speech; 0 uses the
	// service default.
	MinSpeechMS int `validate:"omitempty,min=0"`
	// MinSilenceMS is the shortest run of silence treated as a pause; 0 uses the
	// service default.
	MinSilenceMS int `validate:"omitempty,min=0"`
	// FilterBackgroundAudio has the service strip background audio before it
	// transcribes; nil uses the service default.
	FilterBackgroundAudio *bool

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.ElevenLabsRealtimeTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
	// Websocket configures the connection: whether and how it reconnects, and
	// how long it waits on a peer that has stopped answering.
	Websocket wsservice.Options
}

// Validate reports whether the configuration is usable.
func (c RealtimeSTTConfig) Validate() error { return validate.Struct(c) }

// NewRealtimeSTT builds an ElevenLabs streaming STT service. Unlike NewSTT,
// which transcribes one delimited utterance per request, this holds a
// connection open and lets ElevenLabs commit each segment when its own detector
// hears a pause. That commit delimits the transcript, not the turn: the pipeline
// still decides where the turn ends.
func NewRealtimeSTT(cfg RealtimeSTTConfig) *stt.StreamService {
	if cfg.Host == "" {
		cfg.Host = defaultRealtimeSTTHost
	}
	if cfg.Model == "" {
		cfg.Model = defaultRealtimeSTTModel
	}
	return stt.NewStream("ElevenLabsRealtimeSTT", &realtimeSTTConnector{cfg: cfg}, cfg.SampleRate)
}

type realtimeSTTConnector struct {
	cfg RealtimeSTTConfig
}

// Metadata reports the transcript latency and the model. It recommends no turn
// strategies: ElevenLabs commits a segment when its own detector hears a pause,
// which delimits the transcript rather than the conversational turn, so deciding
// the turn stays with the pipeline.
func (c *realtimeSTTConnector) Metadata() stt.Metadata {
	return stt.Metadata{
		TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.ElevenLabsRealtimeTTFSP99),
		Model:   c.cfg.Model,
	}
}

// SendSampleRate is the rate ElevenLabs is sent audio at, implementing
// stt.SendRater: the service's own rate when ElevenLabs takes it, and 16 kHz
// otherwise, the input audio being resampled to it.
func (c *realtimeSTTConnector) SendSampleRate(sampleRate int) int {
	if slices.Contains(realtimeSTTRates, sampleRate) {
		return sampleRate
	}
	return realtimeSTTFallbackRate
}

// endpoint builds the session URL for audio at sampleRate, which names the
// audio format. The session is configured entirely through query parameters,
// so there is no setup message.
func (c *realtimeSTTConnector) endpoint(sampleRate int) string {
	q := url.Values{}
	q.Set("model_id", c.cfg.Model)
	q.Set("audio_format", realtimeSTTFormat(sampleRate))
	// The service commits an utterance on its own detected silence rather than
	// waiting for the client, which is what makes it a streaming service.
	q.Set("commit_strategy", "vad")
	query.SetStrOpt(q, "language_code", elevenlabsSTTLanguage(c.cfg.Language))
	query.AddAll(q, "keyterms", c.cfg.Keyterms)
	query.SetFloatOpt(q, "vad_silence_threshold_secs", c.cfg.SilenceThresholdSecs)
	query.SetFloatOpt(q, "vad_threshold", c.cfg.VADThreshold)
	if c.cfg.MinSpeechMS > 0 {
		q.Set("min_speech_duration_ms", strconv.Itoa(c.cfg.MinSpeechMS))
	}
	if c.cfg.MinSilenceMS > 0 {
		q.Set("min_silence_duration_ms", strconv.Itoa(c.cfg.MinSilenceMS))
	}
	query.SetBoolOpt(q, "filter_background_audio", c.cfg.FilterBackgroundAudio)
	return c.cfg.Host + realtimeSTTPath + "?" + q.Encode()
}

// WebsocketOptions hands the stream base the connection settings the
// configuration gives, implementing wsservice.Optioner.
func (c *realtimeSTTConnector) WebsocketOptions() wsservice.Options { return c.cfg.Websocket }

// Connect opens a transcription session for audio at sampleRate, which is the
// rate SendSampleRate gave.
func (c *realtimeSTTConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("xi-api-key", c.cfg.APIKey)

	conn, err := c.cfg.Websocket.Dial(ctx, c.endpoint(sampleRate), header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &realtimeSTTStream{
		conn:       conn,
		ctx:        ctx,
		lang:       elevenlabsSTTLanguage(c.cfg.Language),
		sampleRate: sampleRate,
	}, nil
}

type realtimeSTTStream struct {
	conn *wsutil.Conn
	ctx  context.Context
	// lang is the configured language echoed on results, or "".
	lang string
	// sampleRate is the rate of the audio the session is sent, which every
	// chunk restates.
	sampleRate int
	writeMu    sync.Mutex
}

// realtimeSTTMessage is the subset of a transcription message we read.
type realtimeSTTMessage struct {
	MessageType string `json:"message_type"`
	Text        string `json:"text"`
	Transcript  string `json:"transcript"`
	Message     string `json:"message"`
}

// transcript reads whichever field carries the text on this message.
func (m realtimeSTTMessage) transcript() string {
	if m.Text != "" {
		return m.Text
	}
	return m.Transcript
}

// Send writes a chunk of PCM as a base64 audio chunk, with the rate it is at.
// The commit flag stays false: the service decides where an utterance ends.
func (s *realtimeSTTStream) Send(audio []byte) error {
	if len(audio) == 0 {
		return nil
	}
	payload, err := json.Marshal(map[string]any{
		"message_type":  "input_audio_chunk",
		"audio_base_64": base64.StdEncoding.EncodeToString(audio),
		"commit":        false,
		"sample_rate":   s.sampleRate,
	})
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, payload)
}

// Recv reads the next transcript. A partial is the utterance so far; a committed
// transcript finalizes it and ends the turn.
func (s *realtimeSTTStream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m realtimeSTTMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		switch m.MessageType {
		case rtEventPartial:
			if text := m.transcript(); text != "" {
				return []stt.Result{{Text: text, Final: false, Language: s.lang}}, nil
			}
		case rtEventCommitted:
			if text := m.transcript(); text != "" {
				return []stt.Result{{Text: text, Final: true, EndOfTurn: true, Language: s.lang}}, nil
			}
		case rtEventError:
			return nil, fmt.Errorf("%w: %s", errSTTStatus, m.Message)
		case rtEventStarted:
			continue
		}
	}
}

// Close tears the session down.
func (s *realtimeSTTStream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
