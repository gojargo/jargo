package soniox

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

const (
	defaultTTSURL   = "wss://tts-rt.soniox.com/tts-websocket"
	defaultTTSModel = "tts-rt-v1"
	// defaultVoice is one of Soniox's stock voices.
	defaultVoice = "Adrian"
	// defaultTTSSampleRate is the PCM rate requested from Soniox.
	defaultTTSSampleRate = 24000
	// ttsAudioFormat is raw 16-bit little-endian mono PCM, the format the
	// pipeline carries.
	ttsAudioFormat = "pcm_s16le"
	// keyStreamID is the field naming the stream a message belongs to.
	keyStreamID = "stream_id"
	// ttsStreamID names the single stream each synthesis opens. Soniox
	// multiplexes concurrent streams over one connection by this id; jargo
	// synthesizes one sentence per connection, so a constant is enough.
	ttsStreamID = "jargo"
)

// TTSConfig configures the Soniox streaming TTS service.
type TTSConfig struct {
	// APIKey is the Soniox API key, sent in the stream configuration. Required.
	APIKey string `validate:"required"`
	// URL overrides the synthesis WebSocket endpoint; empty uses the hosted
	// endpoint.
	URL string
	// Model is the synthesis model; empty uses a current default.
	Model string
	// Voice is a stock voice name (for example "Adrian") or the UUID of a cloned
	// voice belonging to the API key's project; empty uses a default.
	Voice string
	// Language for synthesis; the zero value leaves it unset and lets Soniox
	// decide. Soniox takes the base code, so a regional language is sent as its
	// base.
	Language language.Language
	// SampleRate is the PCM rate requested from Soniox and emitted downstream;
	// 0 uses 24 kHz. Soniox accepts 8000, 16000, 24000, 44100 and 48000.
	SampleRate int `validate:"omitempty,oneof=8000 16000 24000 44100 48000"`
	// Speed multiplies the speaking rate from 0.7 to 1.3; nil leaves it unset
	// and uses Soniox's default of 1.0.
	Speed *float64 `validate:"omitempty,min=0.7,max=1.3"`
	// WordTimestamps requests per-character timings and drives the word-aligned
	// text path: the TTS base emits a TTSTextFrame for each spoken word as its
	// audio plays, mapped back to its original written form, so the assistant
	// context records only what was actually spoken before an interruption. It
	// is on by default; set it to false to fall back to aggregated text frames.
	WordTimestamps *bool
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// wordTimestamps reports whether per-word timing is requested.
func (c TTSConfig) wordTimestamps() bool {
	return c.WordTimestamps == nil || *c.WordTimestamps
}

// withTTSDefaults fills the fields whose zero value stands for "use a default".
func (c TTSConfig) withTTSDefaults() TTSConfig {
	if c.URL == "" {
		c.URL = defaultTTSURL
	}
	if c.Model == "" {
		c.Model = defaultTTSModel
	}
	if c.Voice == "" {
		c.Voice = defaultVoice
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	return c
}

// NewTTS builds a Soniox streaming TTS service. Each sentence opens a synthesis
// stream, sends the text, and streams the audio chunks downstream as Soniox
// generates them.
func NewTTS(cfg TTSConfig) *tts.Base {
	cfg = cfg.withTTSDefaults()
	s := &ttsSynthesizer{cfg: cfg}
	if cfg.wordTimestamps() {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the base
		// takes the word-aligned path solely when timings were requested.
		return tts.New("SonioxTTS", &timedTTSSynthesizer{ttsSynthesizer: s})
	}
	return tts.New("SonioxTTS", s)
}

type ttsSynthesizer struct {
	cfg TTSConfig
}

// timedTTSSynthesizer adds word-timestamp streaming on top of ttsSynthesizer.
// It implements tts.WordTimestamps.
type timedTTSSynthesizer struct {
	*ttsSynthesizer
}

// SampleRate reports the requested PCM output rate.
func (s *ttsSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the Soniox model and voice synthesis is billed against.
func (s *ttsSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.Voice}
}

// spacelessLanguage reports whether the configured language is written without
// spaces between words. Chinese and Japanese are, so their character timings
// cannot be assembled into words by splitting on spaces.
func (s *ttsSynthesizer) spacelessLanguage() bool {
	switch s.cfg.Language.BaseCode() {
	case "zh", "ja":
		return true
	default:
		return false
	}
}

// config is the opening message. It carries the credentials, the voice
// parameters, and the stream id every later message refers to.
func (s *ttsSynthesizer) config(withTimestamps bool) map[string]any {
	cfg := map[string]any{
		"api_key":           s.cfg.APIKey,
		keyStreamID:         ttsStreamID,
		"model":             s.cfg.Model,
		"voice":             s.cfg.Voice,
		"audio_format":      ttsAudioFormat,
		"sample_rate":       s.cfg.SampleRate,
		"return_timestamps": withTimestamps,
	}
	if lang := s.cfg.Language.BaseCode(); lang != "" {
		cfg["language"] = lang
	}
	if s.cfg.Speed != nil {
		cfg["speed"] = *s.cfg.Speed
	}
	return cfg
}

// Synthesize opens a stream, sends the sentence, and streams audio chunks.
func (s *ttsSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	return s.run(ctx, text, false, emit, nil)
}

// SynthesizeTimed streams audio and reports per-word timing, implementing
// tts.WordTimestamps.
func (s *timedTTSSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, _ string,
	yield func(f frames.Frame) error,
	word func(text string, offset float64) error,
) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	return s.run(ctx, text, true, emit, word)
}

// run drives one synthesis stream end to end.
func (s *ttsSynthesizer) run(
	ctx context.Context,
	text string,
	withTimestamps bool,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	conn, err := wsutil.Dial(ctx, s.cfg.URL, nil, readLimit)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text, withTimestamps); err != nil {
		return err
	}
	return s.receive(ctx, conn, emit, word)
}

// request sends the stream configuration, the sentence, and the end marker that
// tells Soniox no more text is coming.
func (s *ttsSynthesizer) request(
	ctx context.Context,
	conn *websocket.Conn,
	text string,
	withTimestamps bool,
) error {
	messages := []map[string]any{
		s.config(withTimestamps),
		{keyStreamID: ttsStreamID, "text": text, "text_end": false},
		// An empty text with text_end closes the stream and starts the final
		// synthesis; without it Soniox waits for more text until it times out.
		{keyStreamID: ttsStreamID, "text": "", "text_end": true},
	}
	for _, msg := range messages {
		payload, err := json.Marshal(msg)
		if err != nil {
			return err
		}
		if err := conn.Write(ctx, websocket.MessageText, payload); err != nil {
			return err
		}
	}
	return nil
}

// ttsMessage is the subset of a Soniox synthesis message we read.
type ttsMessage struct {
	// Audio is a base64-encoded chunk of PCM.
	Audio string `json:"audio"`
	// Timestamps carries per-character timing; it can arrive without audio.
	Timestamps *ttsTimings `json:"timestamps"`
	// Terminated marks the end of the stream, the point at which the last word
	// is complete.
	Terminated   bool   `json:"terminated"`
	ErrorCode    any    `json:"error_code"`
	ErrorType    string `json:"error_type"`
	ErrorMessage string `json:"error_message"`
}

// ttsTimings is Soniox's per-character timing payload: one entry per character
// (spaces included) and the offset at which it starts, in seconds from the
// beginning of the synthesis.
type ttsTimings struct {
	Characters []string  `json:"characters"`
	Starts     []float64 `json:"character_start_times_seconds"`
}

// receive reads audio chunks and, when word is non-nil, timing payloads until
// Soniox reports the stream is finished.
func (s *ttsSynthesizer) receive(
	ctx context.Context,
	conn *websocket.Conn,
	emit func(pcm []byte) error,
	word func(text string, offset float64) error,
) error {
	var acc uctx.CharAccumulator
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var m ttsMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		if m.ErrorCode != nil {
			return fmt.Errorf("%w: %s", errServer, strings.TrimSpace(m.ErrorType+" "+m.ErrorMessage))
		}
		if m.Terminated {
			if word == nil {
				return nil
			}
			// A final word with no terminating space is still unreported.
			if w, ok := acc.Flush(); ok {
				return word(w.Word, w.Offset)
			}
			return nil
		}
		if m.Audio != "" {
			pcm, err := base64.StdEncoding.DecodeString(m.Audio)
			if err != nil {
				return err
			}
			if err := emit(pcm); err != nil {
				return err
			}
		}
		if word != nil && m.Timestamps != nil {
			if err := s.reportWords(&acc, m.Timestamps, word); err != nil {
				return err
			}
		}
	}
}

// reportWords turns one timing payload into words and forwards them. A language
// written without spaces is reported a character at a time, since there is no
// word boundary to split on.
func (s *ttsSynthesizer) reportWords(
	acc *uctx.CharAccumulator,
	t *ttsTimings,
	word func(text string, offset float64) error,
) error {
	var (
		words []uctx.WordTiming
		err   error
	)
	if s.spacelessLanguage() {
		words, err = uctx.CharsAsWords(t.Characters, t.Starts)
	} else {
		words, err = acc.Add(t.Characters, t.Starts)
	}
	if err != nil {
		return err
	}
	for _, w := range words {
		if err := word(w.Word, w.Offset); err != nil {
			return err
		}
	}
	return nil
}
