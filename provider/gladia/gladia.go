// Package gladia is a streaming speech-to-text service backed by Gladia's Live
// STT v2 API. A session is opened with a REST call that returns a tokenized
// WebSocket URL; audio then streams over the socket and transcripts come back as
// interim and final utterances.
package gladia

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/stt"
)

// errStatus is returned when the session REST call responds with a non-2xx
// status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("gladia: unexpected status")

const (
	liveURL = "https://api.gladia.io/v2/live"
	// defaults for the audio config sent at session init.
	defaultModel    = "solaria-1"
	defaultEncoding = "wav/pcm"
	defaultBitDepth = 16
	defaultChannels = 1
	// readLimit bounds a single WebSocket message.
	readLimit = 1 << 20
)

// LanguageConfig configures language detection and handling.
type LanguageConfig struct {
	// Languages restricts transcription to the given language codes.
	Languages []string `json:"languages,omitempty"`
	// CodeSwitching auto-detects language changes mid-stream.
	CodeSwitching *bool `json:"code_switching,omitempty"`
}

// PreProcessingConfig configures audio pre-processing.
type PreProcessingConfig struct {
	// AudioEnhancer enhances the input audio before transcription.
	AudioEnhancer *bool `json:"audio_enhancer,omitempty"`
	// SpeechThreshold sets speech-detection sensitivity (0.0-1.0).
	SpeechThreshold *float64 `json:"speech_threshold,omitempty"`
}

// MessagesConfig filters which WebSocket messages Gladia sends. Fields left nil
// are omitted.
type MessagesConfig struct {
	ReceivePartialTranscripts       *bool `json:"receive_partial_transcripts,omitempty"`
	ReceiveFinalTranscripts         *bool `json:"receive_final_transcripts,omitempty"`
	ReceiveSpeechEvents             *bool `json:"receive_speech_events,omitempty"`
	ReceivePreProcessingEvents      *bool `json:"receive_pre_processing_events,omitempty"`
	ReceiveRealtimeProcessingEvents *bool `json:"receive_realtime_processing_events,omitempty"`
	ReceivePostProcessingEvents     *bool `json:"receive_post_processing_events,omitempty"`
	ReceiveAcknowledgments          *bool `json:"receive_acknowledgments,omitempty"`
	ReceiveErrors                   *bool `json:"receive_errors,omitempty"`
}

// Config configures the Gladia STT service. Optional fields modeled as pointers,
// slices or maps are omitted from the session init when unset.
type Config struct {
	// APIKey is the Gladia API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the session-init endpoint; empty uses the hosted endpoint.
	URL string
	// Region pins the processing region ("us-west" or "eu-west"); empty omits it.
	Region string
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Encoding is the audio encoding; empty uses "wav/pcm".
	Encoding string
	// BitDepth is the audio bit depth; 0 uses 16.
	BitDepth int
	// Channels is the channel count; 0 uses 1.
	Channels int
	// Model selects the transcription model; empty uses "solaria-1".
	Model string
	// Endpointing is the silence in seconds that marks end of speech; nil omits it.
	Endpointing *float64
	// MaximumDurationWithoutEndpointing caps utterance duration in seconds without
	// silence; nil omits it.
	MaximumDurationWithoutEndpointing *int
	// LanguageConfig configures language detection; nil omits it.
	LanguageConfig *LanguageConfig
	// PreProcessing configures audio pre-processing; nil omits it.
	PreProcessing *PreProcessingConfig
	// RealtimeProcessing passes Gladia's realtime_processing block (custom
	// vocabulary, translation, NER, sentiment, etc.) through verbatim; nil omits it.
	RealtimeProcessing map[string]any
	// MessagesConfig filters received messages; nil defaults to partial+final
	// transcripts, matching jargo's needs.
	MessagesConfig *MessagesConfig
	// CustomMetadata attaches metadata to the session; nil omits it.
	CustomMetadata map[string]any
	// ExtraSettings sets arbitrary additional session-init fields not modeled
	// above; values override any field of the same name.
	ExtraSettings map[string]any
}

// Validate reports whether the configuration is usable.
func (cfg Config) Validate() error { return validate.Struct(cfg) }

// NewSTT builds a Gladia streaming STT service. It works best behind a turn
// detector: Gladia finalizes per utterance rather than per turn.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = liveURL
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.BitDepth == 0 {
		cfg.BitDepth = defaultBitDepth
	}
	if cfg.Channels == 0 {
		cfg.Channels = defaultChannels
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	return stt.NewStream("GladiaSTT", &connector{cfg: cfg, http: &http.Client{}}, cfg.SampleRate)
}

type connector struct {
	cfg  Config
	http *http.Client
}

// Connect initializes a session over REST then dials the returned WebSocket.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	wsURL, err := c.initSession(ctx, sampleRate)
	if err != nil {
		return nil, err
	}
	conn, resp, err := websocket.Dial(ctx, wsURL, nil)
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		return nil, err
	}
	conn.SetReadLimit(readLimit)
	return &stream{conn: conn, ctx: ctx}, nil
}

// settings builds the session-init body for the given sample rate.
func (cfg *Config) settings(sampleRate int) map[string]any {
	s := map[string]any{
		"encoding":    cfg.Encoding,
		"sample_rate": sampleRate,
		"bit_depth":   cfg.BitDepth,
		"channels":    cfg.Channels,
		"model":       cfg.Model,
	}
	if cfg.Endpointing != nil {
		s["endpointing"] = *cfg.Endpointing
	}
	if cfg.MaximumDurationWithoutEndpointing != nil {
		s["maximum_duration_without_endpointing"] = *cfg.MaximumDurationWithoutEndpointing
	}
	if cfg.LanguageConfig != nil {
		s["language_config"] = cfg.LanguageConfig
	}
	if cfg.PreProcessing != nil {
		s["pre_processing"] = cfg.PreProcessing
	}
	if len(cfg.RealtimeProcessing) > 0 {
		s["realtime_processing"] = cfg.RealtimeProcessing
	}
	if len(cfg.CustomMetadata) > 0 {
		s["custom_metadata"] = cfg.CustomMetadata
	}
	mc := cfg.MessagesConfig
	if mc == nil {
		on := true
		mc = &MessagesConfig{ReceivePartialTranscripts: &on, ReceiveFinalTranscripts: &on}
	}
	s["messages_config"] = mc
	maps.Copy(s, cfg.ExtraSettings)
	return s
}

func (c *connector) initSession(ctx context.Context, sampleRate int) (string, error) {
	body, err := json.Marshal(c.cfg.settings(sampleRate))
	if err != nil {
		return "", err
	}
	endpoint := c.cfg.URL
	if c.cfg.Region != "" {
		endpoint += "?" + url.Values{"region": {c.cfg.Region}}.Encode()
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("x-gladia-key", c.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode/100 != 2 {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
	}
	var out struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	return out.URL, nil
}

type stream struct {
	conn    *websocket.Conn
	ctx     context.Context
	writeMu sync.Mutex
}

// message is the subset of a Gladia transcript message we read.
type message struct {
	Type string `json:"type"`
	Data struct {
		IsFinal   bool `json:"is_final"`
		Utterance struct {
			Text     string `json:"text"`
			Language string `json:"language"`
		} `json:"utterance"`
	} `json:"data"`
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next transcript message and maps it to a result.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m message
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type != "transcript" || m.Data.Utterance.Text == "" {
			continue
		}
		return []stt.Result{{
			Text:      m.Data.Utterance.Text,
			Final:     m.Data.IsFinal,
			EndOfTurn: m.Data.IsFinal,
			Language:  m.Data.Utterance.Language,
		}}, nil
	}
}

// Close stops the session and closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"stop_recording"}`))
	s.writeMu.Unlock()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
