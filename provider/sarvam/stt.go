package sarvam

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

// errSTTServer wraps an error reported by Sarvam over the STT socket.
//
//nolint:gochecknoglobals // sentinel error
var errSTTServer = errors.New("sarvam: stt server error")

const (
	transcribeURL   = "wss://api.sarvam.ai/speech-to-text/ws"
	translateURL    = "wss://api.sarvam.ai/speech-to-text-translate/ws"
	defaultSTTModel = "saaras:v3"
	// defaultInputCodec labels the input audio; matches the Sarvam encoding hint.
	defaultInputCodec = "wav"
	// readLimitSTT bounds a single inbound WebSocket message.
	readLimitSTT = 1 << 20
)

// sttModelConfig captures the per-model capabilities that gate which parameters
// are sent and which endpoint is used.
type sttModelConfig struct {
	supportsPrompt       bool
	supportsMode         bool
	supportsLanguage     bool
	supportsVADParams    bool
	defaultLanguage      string
	defaultMode          string
	useTranslateEndpoint bool
}

// sttModelConfigs describes each supported model. saarika transcribes with an
// explicit language, saaras:v2.5 auto-detects and translates, and saaras:v3
// adds mode selection and fine-grained VAD control.
//
//nolint:gochecknoglobals // static model capability table
var sttModelConfigs = map[string]sttModelConfig{
	"saarika:v2.5": {
		supportsLanguage: true,
		defaultLanguage:  "unknown",
	},
	"saaras:v2.5": {
		supportsPrompt:       true,
		useTranslateEndpoint: true,
	},
	"saaras:v3": {
		supportsMode:      true,
		supportsLanguage:  true,
		supportsVADParams: true,
		defaultLanguage:   "unknown",
		defaultMode:       "transcribe",
	},
}

// STTConfig configures the Sarvam WebSocket STT service. Optional fields modeled
// as pointers or empty strings are omitted from the connection when unset, and
// model-specific parameters are only sent for models that support them.
type STTConfig struct {
	// APIKey is the Sarvam API subscription key. Required.
	APIKey string `validate:"required"`
	// Model is the transcription model; empty uses "saaras:v3".
	Model string `validate:"omitempty,oneof=saarika:v2.5 saaras:v2.5 saaras:v3"`
	// Mode selects the saaras:v3 operation mode; empty uses the model default
	// ("transcribe"). Ignored by models without mode support.
	Mode string `validate:"omitempty,oneof=transcribe translate verbatim translit codemix"`
	// Language for transcription; the zero value uses the model default. Only
	// applies to models with language support.
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// InputAudioCodec is the input audio codec hint; empty uses "wav".
	InputAudioCodec string
	// Prompt guides transcription style/context; empty omits it. Only saaras:v2.5.
	Prompt string
	// VADSignals emits VAD signals in the response; nil omits it.
	VADSignals *bool
	// HighVADSensitivity raises VAD sensitivity; nil omits it.
	HighVADSensitivity *bool

	// The fine-grained VAD controls below apply only to saaras:v3 and are omitted
	// when nil.

	// PositiveSpeechThreshold is the probability above which a frame is speech.
	PositiveSpeechThreshold *float64
	// NegativeSpeechThreshold is the probability below which a frame is silence.
	NegativeSpeechThreshold *float64
	// MinSpeechFrames is the minimum consecutive speech frames to start a segment.
	MinSpeechFrames *int
	// FirstTurnMinSpeechFrames is the minimum speech frames for the first turn.
	FirstTurnMinSpeechFrames *int
	// NegativeFramesCount is the silence frames within the window that end a segment.
	NegativeFramesCount *int
	// NegativeFramesWindow is the sliding window size for counting negative frames.
	NegativeFramesWindow *int
	// StartSpeechVolumeThreshold is the volume (dB) below which audio is too quiet.
	StartSpeechVolumeThreshold *float64
	// InterruptMinSpeechFrames is the minimum speech frames to register a barge-in.
	InterruptMinSpeechFrames *int
	// PreSpeechPadFrames is the number of frames prepended before speech onset.
	PreSpeechPadFrames *int
	// NumInitialIgnoredFrames is the number of leading frames skipped at start.
	NumInitialIgnoredFrames *int
}

// Validate reports whether the configuration is usable.
func (c STTConfig) Validate() error { return validate.Struct(c) }

// NewSTT builds a Sarvam streaming STT service. Sarvam finalizes per utterance,
// so it works best behind a turn detector or with its own VAD signals.
func NewSTT(cfg STTConfig) *stt.StreamService {
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.InputAudioCodec == "" {
		cfg.InputAudioCodec = defaultInputCodec
	}
	mc, ok := sttModelConfigs[cfg.Model]
	if !ok {
		mc = sttModelConfigs[defaultSTTModel]
	}
	return stt.NewStream("SarvamSTT", &connector{cfg: cfg, mc: mc}, cfg.SampleRate)
}

// sarvamSTTLanguageCode maps a Language to Sarvam's regional code, or "" when
// unset or unsupported.
func sarvamSTTLanguageCode(l language.Language) string {
	switch l.BaseCode() {
	case "as":
		return "as-IN"
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
		return ""
	}
}

type connector struct {
	cfg STTConfig
	mc  sttModelConfig
}

// endpoint returns the WebSocket URL for the configured model.
func (c *connector) endpoint() string {
	if c.mc.useTranslateEndpoint {
		return translateURL
	}
	return transcribeURL
}

// languageString resolves the language code sent at connect time.
func (c *connector) languageString() string {
	if code := sarvamSTTLanguageCode(c.cfg.Language); code != "" {
		return code
	}
	return c.mc.defaultLanguage
}

// query builds the connect query string for the given sample rate.
func (c *connector) query(sampleRate int) url.Values {
	q := url.Values{}
	q.Set("model", c.cfg.Model)
	q.Set("sample_rate", strconv.Itoa(sampleRate))

	// Honor flush requests when not relying on Sarvam's own VAD signals.
	if c.cfg.VADSignals == nil || !*c.cfg.VADSignals {
		q.Set("flush_signal", "true")
	}
	if c.cfg.VADSignals != nil {
		q.Set("vad_signals", strconv.FormatBool(*c.cfg.VADSignals))
	}
	if c.cfg.HighVADSensitivity != nil {
		q.Set("high_vad_sensitivity", strconv.FormatBool(*c.cfg.HighVADSensitivity))
	}

	if c.mc.supportsVADParams {
		setFloatOpt(q, "positive_speech_threshold", c.cfg.PositiveSpeechThreshold)
		setFloatOpt(q, "negative_speech_threshold", c.cfg.NegativeSpeechThreshold)
		setIntOpt(q, "min_speech_frames", c.cfg.MinSpeechFrames)
		setIntOpt(q, "first_turn_min_speech_frames", c.cfg.FirstTurnMinSpeechFrames)
		setIntOpt(q, "negative_frames_count", c.cfg.NegativeFramesCount)
		setIntOpt(q, "negative_frames_window", c.cfg.NegativeFramesWindow)
		setFloatOpt(q, "start_speech_volume_threshold", c.cfg.StartSpeechVolumeThreshold)
		setIntOpt(q, "interrupt_min_speech_frames", c.cfg.InterruptMinSpeechFrames)
		setIntOpt(q, "pre_speech_pad_frames", c.cfg.PreSpeechPadFrames)
		setIntOpt(q, "num_initial_ignored_frames", c.cfg.NumInitialIgnoredFrames)
	}

	if lang := c.languageString(); lang != "" {
		q.Set("language_code", lang)
	}
	if mode := c.mode(); c.mc.supportsMode && mode != "" {
		q.Set("mode", mode)
	}
	if c.mc.supportsPrompt && c.cfg.Prompt != "" {
		q.Set("prompt", c.cfg.Prompt)
	}
	return q
}

// mode resolves the operation mode, defaulting to the model's mode.
func (c *connector) mode() string {
	if c.cfg.Mode != "" {
		return c.cfg.Mode
	}
	return c.mc.defaultMode
}

func setFloatOpt(q url.Values, key string, v *float64) {
	if v != nil {
		q.Set(key, strconv.FormatFloat(*v, 'g', -1, 64))
	}
}

func setIntOpt(q url.Values, key string, v *int) {
	if v != nil {
		q.Set(key, strconv.Itoa(*v))
	}
}

// Connect dials the streaming WebSocket for the configured model.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	header := http.Header{}
	header.Set("api-subscription-key", c.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, c.endpoint()+"?"+c.query(sampleRate).Encode(), header, readLimitSTT)
	if err != nil {
		return nil, err
	}
	return &stream{
		conn:       conn,
		ctx:        ctx,
		encoding:   c.encoding(),
		sampleRate: sampleRate,
	}, nil
}

// encoding derives the audio encoding hint, prefixing "audio/" when absent.
func (c *connector) encoding() string {
	if strings.HasPrefix(c.cfg.InputAudioCodec, "audio/") {
		return c.cfg.InputAudioCodec
	}
	return "audio/" + c.cfg.InputAudioCodec
}

type stream struct {
	conn       *websocket.Conn
	ctx        context.Context
	encoding   string
	sampleRate int
	writeMu    sync.Mutex
}

// sttMessage is the subset of a Sarvam STT result message we read.
type sttMessage struct {
	Type string `json:"type"`
	Data struct {
		Transcript   string `json:"transcript"`
		LanguageCode string `json:"language_code"`
		SignalType   string `json:"signal_type"`
		Message      string `json:"message"`
	} `json:"data"`
}

// Send base64-encodes the PCM chunk and writes it as a JSON audio message.
func (s *stream) Send(audio []byte) error {
	msg := map[string]any{
		"audio": map[string]any{
			"data":        base64.StdEncoding.EncodeToString(audio),
			"encoding":    s.encoding,
			"sample_rate": s.sampleRate,
		},
	}
	b, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, b)
}

// Recv reads the next message. A data message carries a finalized utterance,
// surfaced as a final, end-of-turn result; VAD signal events are skipped.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m sttMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		switch m.Type {
		case "error":
			return nil, fmt.Errorf("%w: %s", errSTTServer, m.Data.Message)
		case "data":
			text := strings.TrimSpace(m.Data.Transcript)
			if text == "" {
				continue
			}
			return []stt.Result{{
				Text:      text,
				Final:     true,
				EndOfTurn: true,
				Language:  m.Data.LanguageCode,
			}}, nil
		}
	}
}

// Close closes the socket.
func (s *stream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
