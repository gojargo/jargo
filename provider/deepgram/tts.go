package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/ttsstream"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/service/tts"
)

const (
	// speakPath is the synthesis route appended to the configured base URL.
	speakPath = "/v1/speak"
	// defaultTTSVoice is the Aura voice spoken when none is set.
	defaultTTSVoice = "aura-2-helena-en"
	// defaultTTSEncoding is the audio encoding jargo requests from Aura.
	defaultTTSEncoding = "linear16"
)

// TTSConfig configures the Aura TTS service over HTTP.
type TTSConfig struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// BaseURL overrides the Deepgram host, for a private or air-gapped
	// deployment. It is taken as given, scheme included; empty uses Deepgram's
	// hosted API.
	BaseURL string
	// Voice is the Aura voice model (e.g. "aura-2-helena-en"); empty uses
	// "aura-2-helena-en". Deepgram takes it as its "model" query parameter, so
	// it is also the model the synthesis is reported against.
	Voice string
	// SampleRate is the PCM rate requested from Aura and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Encoding is the audio encoding; empty uses "linear16".
	Encoding string
	// Speed multiplies the speaking rate, from 0.7 to 1.5. Nil leaves Aura at
	// its default rate. Supported by the Aura-2 English and Spanish voices;
	// Deepgram recommends staying at or above 0.9 for Spanish.
	Speed *float64
	// MipOptOut opts out of the Deepgram Model Improvement Program. See
	// https://dpgr.am/deepgram-mip for pricing impacts before setting it to
	// true. Nil leaves it unset.
	MipOptOut *bool
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// withDefaults returns a copy of the config with empty fields filled in.
func (c TTSConfig) withDefaults() TTSConfig {
	if c.Voice == "" {
		c.Voice = defaultTTSVoice
	}
	if c.SampleRate == 0 {
		c.SampleRate = defaultTTSSampleRate
	}
	if c.Encoding == "" {
		c.Encoding = defaultTTSEncoding
	}
	if c.BaseURL == "" {
		c.BaseURL = "https://" + defaultHost
	}
	return c
}

// TTSService is the Deepgram text-to-speech service NewTTS builds.
type TTSService struct {
	*tts.Base
}

// NewTTS builds a Deepgram Aura TTS service over HTTP, issuing one request per
// sentence. The voice and the speed can change while the pipeline runs (see
// TTSSettings); the next request is made with the new values.
func NewTTS(cfg TTSConfig) *TTSService {
	s := &TTSService{}
	s.Base = buildTTS(cfg, tts.WithSelf(s))
	return s
}

// buildTTS builds the service NewTTS returns, registering the
// service opts names as the processor the pipeline runs.
func buildTTS(cfg TTSConfig, opts ...tts.Option) *tts.Base {
	return tts.New("DeepgramTTS", newSynthesizer(cfg), opts...)
}

// newSynthesizer builds the HTTP synthesizer with its defaults applied.
func newSynthesizer(cfg TTSConfig) *synthesizer {
	cfg = cfg.withDefaults()
	return &synthesizer{cfg: cfg, live: newTTSSettings(cfg.Voice, cfg.Speed), http: &http.Client{}}
}

type synthesizer struct {
	cfg TTSConfig
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live *TTSSettings
	http *http.Client
}

// Metadata reports the voice synthesis is billed against, which Deepgram takes
// as the model.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Voice), VoiceID: s.live.Voice.Or(s.cfg.Voice)}
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// Settings is the configuration a caller may change while the pipeline runs,
// implementing tts.SettingsHolder. Every request is built from the settings as
// they stand, so a change applies to the next one.
func (s *synthesizer) Settings() any { return s.live }

// speakQuery builds the synthesis query string from the settings as they
// stand. container=none is required to receive headerless PCM rather than a
// WAV stream.
func (s *synthesizer) speakQuery() url.Values {
	q := url.Values{}
	q.Set("model", s.live.Voice.Or(s.cfg.Voice))
	q.Set("encoding", s.cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(s.cfg.SampleRate))
	q.Set("container", "none")
	if v, ok := s.live.Speed.Value(); ok {
		q.Set("speed", strconv.FormatFloat(v, 'f', -1, 64))
	}
	query.SetBoolOpt(q, "mip_opt_out", s.cfg.MipOptOut)
	return q
}

// RunTTS requests speech for text and streams the raw PCM downstream.
func (s *synthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := ttsstream.PCMYielder(yield, s.SampleRate())
	body, err := json.Marshal(map[string]string{"text": text})
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.cfg.BaseURL+speakPath+"?"+s.speakQuery().Encode(), bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", authToken(s.cfg.APIKey))
	req.Header.Set("Content-Type", "application/json")
	return ttsstream.Speech(s.http, req, text, emit)
}

// Compile-time checks that the synthesizer takes settings the way the base
// applies them.
var (
	_ tts.SettingsHolder = (*synthesizer)(nil)
	_ tts.Describer      = (*synthesizer)(nil)
)
