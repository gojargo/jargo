package together

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

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	ttsURL          = "wss://api.together.ai/v1/audio/speech/websocket"
	defaultTTSModel = "hexgrad/Kokoro-82M"
	defaultTTSVoice = "af_heart"
	// ttsSampleRate is Together's fixed TTS output rate; it streams 24 kHz PCM.
	ttsSampleRate = 24000
)

// errTTS wraps an error reported by the Together TTS session.
//
//nolint:gochecknoglobals // sentinel error
var errTTS = errors.New("together: tts error")

// TTSConfig configures the Together AI streaming TTS service, which streams 24 kHz
// mono PCM over an OpenAI-compatible realtime WebSocket.
type TTSConfig struct {
	// APIKey is the Together AI API key. Required.
	APIKey string `validate:"required"`
	// URL overrides the TTS WebSocket endpoint; empty uses the hosted endpoint.
	URL string
	// Model is the TTS model; empty uses a default.
	Model string
	// Voice is the voice id; empty uses a default.
	Voice string
	// Language for synthesis; the zero value uses English. Sent as the ISO 639-1
	// or lowercase locale code Together accepts.
	Language language.Language
	// MaxPartialLength caps the partial text length for streaming; nil omits it.
	MaxPartialLength *int
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// NewTTS builds a Together AI streaming TTS service.
func NewTTS(cfg TTSConfig) *tts.Base {
	return tts.New("TogetherTTS", &synthesizer{cfg: cfg.withTTSDefaults()})
}

// withTTSDefaults fills unset fields with their defaults.
func (c TTSConfig) withTTSDefaults() TTSConfig {
	if c.URL == "" {
		c.URL = ttsURL
	}
	if c.Model == "" {
		c.Model = defaultTTSModel
	}
	if c.Voice == "" {
		c.Voice = defaultTTSVoice
	}
	if c.Language == "" {
		c.Language = language.English
	}
	return c
}

// togetherLanguages are the languages this provider has been verified against,
// under the code Together takes for each.
//
//nolint:gochecknoglobals // read-only lookup table
var togetherLanguages = map[language.Language]string{
	language.English:    "en",
	language.Spanish:    "es",
	language.French:     "fr",
	language.Hindi:      "hi",
	language.Italian:    "it",
	language.Japanese:   "ja",
	language.Portuguese: "pt",
	language.Chinese:    "zh",
	language.ChineseHK:  "zh-hk",
}

// togetherLanguage maps a Language to a Together language code. Together accepts
// ISO 639-1 codes and lowercase locale codes ("zh-hk"); a regional variant
// without a verified locale code falls back to its base language code. The zero
// value maps to nothing, which sends no language.
func togetherLanguage(l language.Language) string {
	if l == "" {
		return ""
	}
	return language.Resolve(l, togetherLanguages, true)
}

type synthesizer struct {
	cfg TTSConfig
}

// SampleRate reports Together's fixed PCM output rate.
func (s *synthesizer) SampleRate() int { return ttsSampleRate }

// ttsEvent is the subset of a Together TTS event we read.
type ttsEvent struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
	Error struct {
		Message string `json:"message"`
		Code    string `json:"code"`
	} `json:"error"`
}

// endpoint builds the TTS WebSocket URL with the model, voice, language and
// partial length query parameters.
func (s *synthesizer) endpoint() string {
	params := [][2]string{{"model", s.cfg.Model}, {"voice", s.cfg.Voice}}
	if lang := togetherLanguage(s.cfg.Language); lang != "" {
		params = append(params, [2]string{"language", lang})
	}
	if s.cfg.MaxPartialLength != nil {
		params = append(params, [2]string{"max_partial_length", strconv.Itoa(*s.cfg.MaxPartialLength)})
	}
	// Kokoro blends voices with a `+`-separated name such as
	// "af_bella(2)+af_heart(1)", which has to be escaped or the server reads
	// the `+` as a space and rejects the voice.
	query := make([]string, 0, len(params))
	for _, p := range params {
		query = append(query, p[0]+"="+queryEscape(p[1]))
	}
	return s.cfg.URL + "?" + strings.Join(query, "&")
}

// queryEscape percent-encodes a query value, leaving only unreserved characters
// and `/` as they are, so a space is sent as %20 rather than `+` and a model id
// keeps its slash.
func queryEscape(v string) string {
	escaped := strings.ReplaceAll(url.QueryEscape(v), "+", "%20")
	return strings.ReplaceAll(escaped, "%2F", "/")
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, s.endpoint(), header, wsutil.DefaultReadLimit)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit)
}

// request pins the voice, then buffers the text and commits it for synthesis.
func (s *synthesizer) request(ctx context.Context, conn *wsutil.Conn, text string) error {
	msgs := []map[string]any{
		{msgType: "tts_session.updated", "session": map[string]any{"voice": s.cfg.Voice}},
		{msgType: "input_text_buffer.append", "text": text},
		{msgType: "input_text_buffer.commit"},
	}
	for _, m := range msgs {
		payload, err := json.Marshal(m)
		if err != nil {
			return err
		}
		if err := conn.Write(ctx, websocket.MessageText, payload); err != nil {
			return err
		}
	}
	return nil
}

// receive streams audio deltas until the session reports completion.
func (s *synthesizer) receive(ctx context.Context, conn *wsutil.Conn, emit func(pcm []byte) error) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var evt ttsEvent
		if json.Unmarshal(data, &evt) != nil {
			continue
		}
		switch evt.Type {
		case "conversation.item.audio_output.delta":
			if evt.Delta == "" {
				continue
			}
			pcm, err := base64.StdEncoding.DecodeString(evt.Delta)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		case "conversation.item.audio_output.done":
			return nil
		case "conversation.item.tts.failed", "error":
			return fmt.Errorf("%w: %s", errTTS, evt.Error.Message)
		}
	}
}
