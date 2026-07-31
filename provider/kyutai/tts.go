package kyutai

import (
	"context"
	"net/http"
	"net/url"
	"strings"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/vmihailenco/msgpack/v5"
)

const (
	// defaultTTSURL is the moshi-server streaming TTS endpoint on localhost.
	defaultTTSURL = "ws://127.0.0.1:8080/api/tts_streaming"
	// ttsReadLimit bounds a single inbound Audio message.
	ttsReadLimit = 1 << 20
)

// TTSConfig configures the Kyutai TTS service.
type TTSConfig struct {
	// APIKey is moshi-server's shared token (sent as the kyutai-api-key header);
	// empty uses moshi's default "public_token".
	APIKey string
	// URL overrides the moshi-server TTS WebSocket endpoint; empty uses localhost.
	URL string
	// Voice is the voice path within the kyutai/tts-voices repo. Required. Pick a
	// commercially licensed voice: the Expresso and EARS voices in that repo are
	// CC-BY-NC (non-commercial); use a CC0 or CC-BY voice in a commercial product.
	Voice string `validate:"required"`
	// SampleRate is the emitted PCM rate; 0 uses moshi's 24 kHz. The output
	// transport resamples it to its own rate.
	SampleRate int
	// Language is informational; the model itself is fixed (e.g. en_fr).
	Language language.Language
}

// Validate reports whether the configuration is usable.
func (cfg TTSConfig) Validate() error { return validate.Struct(cfg) }

// NewTTS builds a Kyutai TTS service backed by moshi-server.
func NewTTS(cfg TTSConfig) *tts.Base {
	if cfg.URL == "" {
		cfg.URL = defaultTTSURL
	}
	if cfg.APIKey == "" {
		cfg.APIKey = defaultToken
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = moshiSampleRate
	}
	return tts.New("KyutaiTTS", &synthesizer{cfg: cfg})
}

type synthesizer struct {
	cfg TTSConfig
}

// SampleRate reports the PCM rate emitted downstream.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Synthesize opens a TTS session, streams the sentence word-by-word, then emits
// the returned PCM audio chunks until moshi closes the stream.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	q := url.Values{}
	q.Set("format", "PcmMessagePack")
	q.Set("voice", s.cfg.Voice)

	header := http.Header{}
	header.Set("kyutai-api-key", s.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, s.cfg.URL+"?"+q.Encode(), header, ttsReadLimit)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.send(ctx, conn, text); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit)
}

// send streams the text one word per Text message, then an Eos to flush.
func (s *synthesizer) send(ctx context.Context, conn *websocket.Conn, text string) error {
	for word := range strings.FieldsSeq(text) {
		b, err := msgpack.Marshal(map[string]any{msgTypeKey: "Text", "text": word})
		if err != nil {
			return err
		}
		if err := conn.Write(ctx, websocket.MessageBinary, b); err != nil {
			return err
		}
	}
	b, err := msgpack.Marshal(map[string]any{msgTypeKey: "Eos"})
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageBinary, b)
}

// receive reads Audio messages, converts the float32 PCM to S16LE, and emits it.
// A normal-closure from moshi marks the end of synthesis.
func (s *synthesizer) receive(ctx context.Context, conn *websocket.Conn, emit func(pcm []byte) error) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			if websocket.CloseStatus(err) == websocket.StatusNormalClosure {
				return nil
			}
			return err
		}
		var m audioMsg
		if err := msgpack.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type == "Audio" && len(m.PCM) > 0 {
			if err := emit(float32ToInt16Bytes(m.PCM)); err != nil {
				return err
			}
		}
	}
}
