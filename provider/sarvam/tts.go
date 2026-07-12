package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

// errTTSProtocol is returned when Sarvam reports a TTS error message.
//
//nolint:gochecknoglobals // sentinel error
var errTTSProtocol = errors.New("sarvam: tts protocol error")

const (
	defaultTTSURL   = "wss://api.sarvam.ai/text-to-speech/ws"
	defaultTTSModel = "bulbul:v2"
	// defaultTTSLanguage is the Sarvam code used when no language is configured.
	defaultTTSLanguage = "en-IN"
	// Audio format defaults. linear16 yields raw 16-bit mono PCM.
	defaultTTSCodec   = "linear16"
	defaultTTSBitrate = "128k"
	// Buffering defaults for the WebSocket streaming protocol.
	defaultMinBufferSize  = 50
	defaultMaxChunkLength = 150
	// defaultPace is the neutral speaking rate.
	defaultPace = 1.0
	// readLimitTTS bounds a single inbound WebSocket message; audio arrives base64.
	readLimitTTS = 1 << 20
)

// ttsModelConfig captures the per-model capabilities and defaults that shape the
// synthesis request.
type ttsModelConfig struct {
	supportsPitch              bool
	supportsLoudness           bool
	supportsTemperature        bool
	defaultSampleRate          int
	defaultSpeaker             string
	paceMin, paceMax           float64
	preprocessingAlwaysEnabled bool
}

// ttsModelConfigs describes each supported bulbul model. v2 supports
// pitch/loudness; the v3 variants trade those for temperature and always
// preprocess.
//
//nolint:gochecknoglobals // static model capability table
var ttsModelConfigs = map[string]ttsModelConfig{
	"bulbul:v2": {
		supportsPitch:              true,
		supportsLoudness:           true,
		supportsTemperature:        false,
		defaultSampleRate:          22050,
		defaultSpeaker:             "anushka",
		paceMin:                    0.3,
		paceMax:                    3.0,
		preprocessingAlwaysEnabled: false,
	},
	"bulbul:v3-beta": {
		supportsPitch:              false,
		supportsLoudness:           false,
		supportsTemperature:        true,
		defaultSampleRate:          24000,
		defaultSpeaker:             "shubh",
		paceMin:                    0.5,
		paceMax:                    2.0,
		preprocessingAlwaysEnabled: true,
	},
	"bulbul:v3": {
		supportsPitch:              false,
		supportsLoudness:           false,
		supportsTemperature:        true,
		defaultSampleRate:          24000,
		defaultSpeaker:             "shubh",
		paceMin:                    0.5,
		paceMax:                    2.0,
		preprocessingAlwaysEnabled: true,
	},
}

// TTSConfig configures the Sarvam WebSocket TTS service. Optional fields modeled
// as pointers are omitted from the request when unset, and model-specific fields
// are dropped for models that do not support them.
type TTSConfig struct {
	// APIKey is the Sarvam API subscription key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the TTS model; empty uses "bulbul:v2".
	Model string `validate:"omitempty,oneof=bulbul:v2 bulbul:v3-beta bulbul:v3"`
	// Voice is the speaker id; empty uses the model's default speaker.
	Voice string
	// Language for synthesis; the zero value uses English (India). Mapped to
	// Sarvam's regional code.
	Language language.Language
	// SampleRate is the PCM rate requested and emitted downstream; 0 uses the
	// model default (22050 Hz for v2, 24000 Hz for the v3 variants).
	SampleRate int
	// EnablePreprocessing toggles text preprocessing; nil defaults to off. Always
	// forced on for the v3 variants.
	EnablePreprocessing *bool
	// Pace multiplies the speaking rate; nil uses 1.0. Clamped to the model's
	// supported range (v2: 0.3-3.0, v3: 0.5-2.0).
	Pace *float64
	// Pitch adjusts voice pitch (-0.75 to 0.75); nil omits it. Only bulbul:v2.
	Pitch *float64
	// Loudness multiplies volume (0.3 to 3.0); nil omits it. Only bulbul:v2.
	Loudness *float64
	// Temperature controls output randomness (0.01 to 1.0); nil omits it. Only
	// the v3 variants.
	Temperature *float64
	// MinBufferSize is the minimum characters buffered before audio generation;
	// nil uses 50.
	MinBufferSize *int
	// MaxChunkLength is the maximum characters processed per chunk; nil uses 150.
	MaxChunkLength *int
	// OutputAudioCodec is the audio codec; empty uses "linear16" (raw PCM).
	OutputAudioCodec string
	// OutputAudioBitrate is the audio bitrate; empty uses "128k".
	OutputAudioBitrate string
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Sarvam WebSocket TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultTTSURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultTTSModel
	}
	mc, ok := ttsModelConfigs[cfg.Model]
	if !ok {
		mc = ttsModelConfigs[defaultTTSModel]
	}

	s := &synthesizer{
		apiKey:         cfg.APIKey,
		url:            fmt.Sprintf("%s?model=%s&send_completion_event=true", cfg.URL, cfg.Model),
		model:          cfg.Model,
		langCode:       resolveTTSLanguage(cfg.Language),
		sampleRate:     firstNonZero(cfg.SampleRate, mc.defaultSampleRate),
		voice:          firstNonEmpty(cfg.Voice, mc.defaultSpeaker),
		enablePreproc:  mc.preprocessingAlwaysEnabled || (cfg.EnablePreprocessing != nil && *cfg.EnablePreprocessing),
		pace:           clampPace(cfg.Pace, mc),
		minBufferSize:  intOr(cfg.MinBufferSize, defaultMinBufferSize),
		maxChunkLength: intOr(cfg.MaxChunkLength, defaultMaxChunkLength),
		codec:          firstNonEmpty(cfg.OutputAudioCodec, defaultTTSCodec),
		bitrate:        firstNonEmpty(cfg.OutputAudioBitrate, defaultTTSBitrate),
	}
	// Drop model-specific controls the model does not support.
	if mc.supportsPitch {
		s.pitch = cfg.Pitch
	}
	if mc.supportsLoudness {
		s.loudness = cfg.Loudness
	}
	if mc.supportsTemperature {
		s.temperature = cfg.Temperature
	}
	return tts.New("SarvamTTS", s)
}

// resolveTTSLanguage maps a Language to Sarvam's regional code, defaulting to
// English (India) when unset or unsupported.
func resolveTTSLanguage(l language.Language) string {
	switch l.BaseCode() {
	case "bn":
		return "bn-IN"
	case "en":
		return "en-IN"
	case "gu":
		return "gu-IN"
	case "hi":
		return "hi-IN"
	case "kn":
		return "kn-IN"
	case "ml":
		return "ml-IN"
	case "mr":
		return "mr-IN"
	case "or":
		return "od-IN"
	case "pa":
		return "pa-IN"
	case "ta":
		return "ta-IN"
	case "te":
		return "te-IN"
	default:
		return defaultTTSLanguage
	}
}

// clampPace resolves the pace, defaulting to 1.0 and clamping to the model range.
func clampPace(pace *float64, mc ttsModelConfig) float64 {
	v := defaultPace
	if pace != nil {
		v = *pace
	}
	if v < mc.paceMin {
		return mc.paceMin
	}
	if v > mc.paceMax {
		return mc.paceMax
	}
	return v
}

func firstNonZero(v, def int) int {
	if v != 0 {
		return v
	}
	return def
}

func firstNonEmpty(v, def string) string {
	if v != "" {
		return v
	}
	return def
}

func intOr(v *int, def int) int {
	if v != nil {
		return *v
	}
	return def
}

type synthesizer struct {
	apiKey         string
	url            string
	model          string
	voice          string
	langCode       string
	sampleRate     int
	enablePreproc  bool
	minBufferSize  int
	maxChunkLength int
	codec          string
	bitrate        string
	pace           float64
	pitch          *float64
	loudness       *float64
	temperature    *float64
}

// SampleRate reports the PCM output rate.
func (s *synthesizer) SampleRate() int { return s.sampleRate }

// ttsMessage is the subset of a Sarvam TTS WebSocket message we read.
type ttsMessage struct {
	Type string `json:"type"`
	Data struct {
		Audio     string `json:"audio"`
		EventType string `json:"event_type"`
		Message   string `json:"message"`
	} `json:"data"`
}

// Synthesize opens a session, sends the config and text, flushes, and streams
// the resulting audio chunks until the completion event arrives.
func (s *synthesizer) Synthesize(ctx context.Context, text string, emit func(pcm []byte) error) error {
	header := http.Header{}
	header.Set("api-subscription-key", s.apiKey)

	conn, err := wsutil.Dial(ctx, s.url, header, readLimitTTS)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := conn.Write(ctx, websocket.MessageText, s.configMessage()); err != nil {
		return err
	}
	if err := s.writeJSON(ctx, conn, map[string]any{"type": "text", "data": map[string]any{"text": text}}); err != nil {
		return err
	}
	if err := s.writeJSON(ctx, conn, map[string]any{"type": "flush"}); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit)
}

// configMessage builds the initial config frame carrying voice and format.
func (s *synthesizer) configMessage() []byte {
	data := map[string]any{
		"target_language_code": s.langCode,
		"speaker":              s.voice,
		"speech_sample_rate":   fmt.Sprintf("%d", s.sampleRate),
		"enable_preprocessing": s.enablePreproc,
		"min_buffer_size":      s.minBufferSize,
		"max_chunk_length":     s.maxChunkLength,
		"output_audio_codec":   s.codec,
		"output_audio_bitrate": s.bitrate,
		"pace":                 s.pace,
		"model":                s.model,
	}
	if s.pitch != nil {
		data["pitch"] = *s.pitch
	}
	if s.loudness != nil {
		data["loudness"] = *s.loudness
	}
	if s.temperature != nil {
		data["temperature"] = *s.temperature
	}
	b, _ := json.Marshal(map[string]any{"type": "config", "data": data}) //nolint:errchkjson // serializable map
	return b
}

func (s *synthesizer) writeJSON(ctx context.Context, conn *websocket.Conn, msg map[string]any) error {
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, b)
}

func (s *synthesizer) receive(ctx context.Context, conn *websocket.Conn, emit func(pcm []byte) error) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var m ttsMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		switch m.Type {
		case "audio":
			pcm, err := base64.StdEncoding.DecodeString(m.Data.Audio)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "event":
			if m.Data.EventType == "final" {
				return nil
			}
		case "error":
			return fmt.Errorf("%w: %s", errTTSProtocol, m.Data.Message)
		}
	}
}
