package elevenlabs

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	// defaultRealtimeSTTHost is the ElevenLabs WebSocket host.
	defaultRealtimeSTTHost = "wss://api.elevenlabs.io"
	// realtimeSTTPath is the streaming transcription endpoint.
	realtimeSTTPath = "/v1/speech-to-text/realtime"
	// defaultRealtimeSTTModel is a current streaming transcription model.
	defaultRealtimeSTTModel = "scribe_v2_realtime"
	// realtimeSTTFormat is raw 16-bit little-endian PCM, the format the pipeline
	// carries.
	realtimeSTTFormat = "pcm_s16le_16"
	// realtimeSTTTTFSP99 is the time-to-final-segment P99 latency reported
	// downstream.
	realtimeSTTTTFSP99 = 500 * time.Millisecond
)

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
}

// Validate reports whether the configuration is usable.
func (c RealtimeSTTConfig) Validate() error { return validate.Struct(c) }

// NewRealtimeSTT builds an ElevenLabs streaming STT service. Unlike NewSTT,
// which transcribes one delimited utterance per request, this holds a
// connection open and lets ElevenLabs detect the utterance boundaries, so the
// pipeline does not need its own end-of-turn detection.
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

// Metadata tells downstream processors the service detects utterance boundaries
// itself, so the user aggregator adopts external turn strategies.
func (c *realtimeSTTConnector) Metadata() stt.Metadata {
	return stt.Metadata{
		RecommendedUserTurns: frames.UserTurnExternal,
		TTFSP99:              realtimeSTTTTFSP99,
		Model:                c.cfg.Model,
	}
}

// endpoint builds the session URL. The session is configured entirely through
// query parameters, so there is no setup message.
func (c *realtimeSTTConnector) endpoint(sampleRate int) string {
	q := url.Values{}
	q.Set("model_id", c.cfg.Model)
	q.Set("audio_format", realtimeSTTFormat)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
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
	return c.cfg.Host + realtimeSTTPath + "?" + q.Encode()
}

// Connect opens a transcription session.
func (c *realtimeSTTConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("xi-api-key", c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.endpoint(sampleRate), header, wsutil.DefaultReadLimit)
	if err != nil {
		return nil, err
	}
	return &realtimeSTTStream{
		conn: conn,
		ctx:  ctx,
		lang: elevenlabsSTTLanguage(c.cfg.Language),
	}, nil
}

type realtimeSTTStream struct {
	conn *websocket.Conn
	ctx  context.Context
	// lang is the configured language echoed on results, or "".
	lang    string
	writeMu sync.Mutex
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

// Send writes a chunk of PCM as a base64 audio chunk. The commit flag stays
// false: the service decides where an utterance ends.
func (s *realtimeSTTStream) Send(audio []byte) error {
	if len(audio) == 0 {
		return nil
	}
	payload, err := json.Marshal(map[string]any{
		"message_type": "input_audio_chunk",
		"audio_chunk":  base64.StdEncoding.EncodeToString(audio),
		"commit":       false,
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
