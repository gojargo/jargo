package grok

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/query"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
)

const (
	// defaultTTSWSURL is the hosted streaming synthesis WebSocket.
	defaultTTSWSURL = "wss://api.x.ai/v1/tts"
	// defaultTTSHTTPURL is the hosted batch synthesis endpoint.
	defaultTTSHTTPURL = "https://api.x.ai/v1/tts"
	// defaultVoice is a current xAI voice.
	defaultVoice = "eve"
	// defaultTTSSampleRate is the PCM rate jargo requests from xAI.
	defaultTTSSampleRate = 24000
	// ttsCodec is raw 16-bit little-endian mono PCM. xAI also offers wav, mulaw
	// and alaw, but the pipeline carries linear PCM, so the codec is fixed.
	ttsCodec = "pcm"
	// ttsLanguageAuto lets xAI detect the language.
	ttsLanguageAuto = "auto"
	// ttsReadLimit bounds a single inbound message. Audio arrives base64-encoded
	// and timing payloads can cover several seconds of speech at once.
	ttsReadLimit = 1 << 20
)

// xAI TTS message types.
const (
	// ttsClientDelta appends text to the utterance being synthesized.
	ttsClientDelta = "text.delta"
	// ttsClientDone ends the utterance and starts synthesis.
	ttsClientDone = "text.done"
	// ttsEventDelta carries a chunk of audio, timing, or both.
	ttsEventDelta = "audio.delta"
	// ttsEventDone ends the utterance's audio.
	ttsEventDone = "audio.done"
	// ttsEventError reports a server-side failure.
	ttsEventError = "error"
)

// errTTSServer wraps an error reported by the xAI TTS service.
//
//nolint:gochecknoglobals // sentinel error
var errTTSServer = errors.New("xai: tts server error")

// TTSConfig configures the xAI text-to-speech services. Both the streaming
// WebSocket service (NewTTS) and the batch HTTP one (NewHTTPTTS) read it;
// WordTimestamps applies only to the streaming service.
type TTSConfig struct {
	// APIKey is the xAI API key, sent as a Bearer token. Required.
	APIKey string `validate:"required"`
	// URL overrides the synthesis endpoint; empty uses the hosted endpoint for
	// the transport being built.
	URL string
	// Voice is the xAI voice identifier; empty uses a default.
	Voice string
	// Language for synthesis; the zero value lets xAI detect it. xAI names a
	// few languages by region, so a regional language maps to its xAI code.
	Language language.Language
	// SampleRate is the PCM rate requested from xAI and emitted downstream;
	// 0 uses 24 kHz.
	SampleRate int
	// Speed multiplies the speaking rate from 0.7 to 1.5; nil leaves it unset.
	Speed *float64 `validate:"omitempty,min=0.7,max=1.5"`
	// OptimizeStreamingLatency picks a latency optimization level (0, 1 or 2);
	// nil leaves it unset.
	OptimizeStreamingLatency *int `validate:"omitempty,min=0,max=2"`
	// TextNormalization expands numbers, dates and abbreviations before
	// synthesis; nil leaves it unset.
	TextNormalization *bool
	// WordTimestamps requests per-character timings and drives the word-aligned
	// text path: the TTS base emits a TTSTextFrame for each spoken word as its
	// audio plays, mapped back to its original written form, so the assistant
	// context records only what was actually spoken before an interruption. It
	// is on by default; set it to false to fall back to aggregated text frames.
	// The batch HTTP service ignores it.
	WordTimestamps *bool
}

// Validate reports whether the configuration is usable.
func (c TTSConfig) Validate() error { return validate.Struct(c) }

// wordTimestamps reports whether per-word timing is requested.
func (c TTSConfig) wordTimestamps() bool {
	return c.WordTimestamps == nil || *c.WordTimestamps
}

// applyTTSDefaults fills the fields whose zero value stands for "use a default".
func applyTTSDefaults(cfg TTSConfig, defaultURL string) TTSConfig {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Voice == "" {
		cfg.Voice = defaultVoice
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultTTSSampleRate
	}
	return cfg
}

// xaiLanguage maps a Language to xAI's language code. xAI names a handful of
// languages by region and takes the base code for the rest; the zero value
// leaves detection to the server.
func xaiLanguage(l language.Language) string {
	switch l.Code() {
	case "ar":
		return "ar-EG"
	case "es", "es-ES":
		return "es-ES"
	case "es-MX":
		return "es-MX"
	case "pt", "pt-PT":
		return "pt-PT"
	case "pt-BR":
		return "pt-BR"
	}
	if base := l.BaseCode(); base != "" {
		return base
	}
	return ttsLanguageAuto
}

// NewTTS builds an xAI streaming TTS service. It opens a synthesis session per
// sentence, streams the text in, and emits the audio chunks as they arrive.
func NewTTS(cfg TTSConfig) *tts.Base {
	cfg = applyTTSDefaults(cfg, defaultTTSWSURL)
	s := &ttsSynthesizer{cfg: cfg}
	if cfg.wordTimestamps() {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the base
		// takes the word-aligned path solely when timings were requested.
		return tts.New("XAITTS", &timedTTSSynthesizer{ttsSynthesizer: s})
	}
	return tts.New("XAITTS", s)
}

// NewHTTPTTS builds an xAI batch TTS service. It synthesizes each sentence in
// one request and streams the response body downstream. Prefer NewTTS: the
// streaming service starts speaking sooner and reports word timings.
func NewHTTPTTS(cfg TTSConfig) *tts.Base {
	cfg = applyTTSDefaults(cfg, defaultTTSHTTPURL)
	return tts.New("XAIHTTPTTS", &httpTTSSynthesizer{cfg: cfg, http: &http.Client{}})
}

// ttsSynthesizer streams synthesis over xAI's WebSocket.
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

// Metadata reports the xAI voice synthesis is billed against. xAI selects the
// model by voice, so there is no separate model identifier.
func (s *ttsSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.cfg.Voice}
}

// endpoint builds the session URL. xAI bakes the audio parameters into the
// handshake, so a session serves one voice, language and rate.
func (s *ttsSynthesizer) endpoint(withTimestamps bool) string {
	q := url.Values{}
	q.Set("voice", s.cfg.Voice)
	q.Set("language", xaiLanguage(s.cfg.Language))
	q.Set("codec", ttsCodec)
	q.Set("sample_rate", strconv.Itoa(s.cfg.SampleRate))
	query.SetFloatOpt(q, "speed", s.cfg.Speed)
	query.SetIntOpt(q, "optimize_streaming_latency", s.cfg.OptimizeStreamingLatency)
	query.SetBoolOpt(q, "text_normalization", s.cfg.TextNormalization)
	q.Set("with_timestamps", strconv.FormatBool(withTimestamps))
	return s.cfg.URL + "?" + q.Encode()
}

// Synthesize opens a session, sends the transcript, and streams audio chunks.
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
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	return s.run(ctx, text, true, emit, word)
}

// run drives one synthesis session end to end.
func (s *ttsSynthesizer) run(
	ctx context.Context,
	text string,
	withTimestamps bool,
	emit func(pcm []byte) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	header := http.Header{}
	header.Set("Authorization", "Bearer "+s.cfg.APIKey)

	conn, err := wsutil.Dial(ctx, s.endpoint(withTimestamps), header, ttsReadLimit)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(websocket.StatusNormalClosure, "") }()

	if err := s.request(ctx, conn, text); err != nil {
		return err
	}
	return receiveTTS(ctx, conn, emit, word)
}

// request streams the sentence in and closes the utterance so xAI starts
// synthesizing. jargo hands the base a whole sentence at a time, so one delta
// carries all of it.
func (s *ttsSynthesizer) request(ctx context.Context, conn *wsutil.Conn, text string) error {
	delta, err := json.Marshal(map[string]any{"type": ttsClientDelta, "delta": text})
	if err != nil {
		return err
	}
	if werr := conn.Write(ctx, websocket.MessageText, delta); werr != nil {
		return werr
	}
	done, err := json.Marshal(map[string]any{"type": ttsClientDone})
	if err != nil {
		return err
	}
	return conn.Write(ctx, websocket.MessageText, done)
}

// ttsMessage is the subset of an xAI TTS event we read.
type ttsMessage struct {
	Type string `json:"type"`
	// Delta is a base64-encoded chunk of PCM.
	Delta string `json:"delta"`
	// Timestamps carries per-character timing; it can arrive without audio.
	Timestamps *ttsTimings `json:"audio_timestamps"`
	Message    string      `json:"message"`
	Error      string      `json:"error"`
}

// ttsTimings is xAI's per-character timing payload: one entry per character
// (spaces and punctuation included) and the matching [start, end] second pair,
// absolute from the start of the utterance.
type ttsTimings struct {
	Chars []string    `json:"graph_chars"`
	Times [][]float64 `json:"graph_times"`
}

// receiveTTS reads audio chunks and, when word is non-nil, timing payloads until
// the utterance's audio is complete.
func receiveTTS(
	ctx context.Context,
	conn *wsutil.Conn,
	emit func(pcm []byte) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
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
		switch m.Type {
		case ttsEventDelta:
			if err := handleDelta(&acc, m, emit, word); err != nil {
				return err
			}
		case ttsEventDone:
			return flushFinalWord(&acc, word)
		case ttsEventError:
			return fmt.Errorf("%w: %s", errTTSServer, firstNonEmpty(m.Message, m.Error))
		}
	}
}

// handleDelta emits one chunk's audio and folds in the timings that came with
// it. A delta carries audio, timings, or both.
func handleDelta(
	acc *uctx.CharAccumulator,
	m ttsMessage,
	emit func(pcm []byte) error,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	if m.Delta != "" {
		pcm, err := base64.StdEncoding.DecodeString(m.Delta)
		if err != nil {
			return err
		}
		if err := emit(pcm); err != nil {
			return err
		}
	}
	if word == nil {
		return nil
	}
	return addTimings(acc, m.Timestamps, word)
}

// flushFinalWord reports the word left buffered when the utterance ends, which
// had no terminating space to close it.
func flushFinalWord(
	acc *uctx.CharAccumulator,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	if word == nil {
		return nil
	}
	if w, ok := acc.Flush(); ok {
		return word([]uctx.WordTiming{w}, tts.WordTimingOptions{})
	}
	return nil
}

// startOffsets flattens xAI's [start, end] pairs to the start of each character,
// which is what the word accumulator aligns on.
func (t *ttsTimings) startOffsets() []float64 {
	starts := make([]float64, 0, len(t.Times))
	for _, pair := range t.Times {
		if len(pair) == 0 {
			starts = append(starts, 0)
			continue
		}
		starts = append(starts, pair[0])
	}
	return starts
}

// addTimings folds one timing payload into acc, reporting every word it
// completes.
func addTimings(
	acc *uctx.CharAccumulator,
	t *ttsTimings,
	word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	if t == nil || len(t.Chars) == 0 {
		return nil
	}
	words, err := acc.Add(t.Chars, t.startOffsets())
	if err != nil {
		return err
	}
	return emitWords(words, word)
}

// emitWords forwards each assembled word to the callback.
func emitWords(words []uctx.WordTiming, word func(words []uctx.WordTiming, opts tts.WordTimingOptions) error) error {
	for _, w := range words {
		if err := word([]uctx.WordTiming{w}, tts.WordTimingOptions{}); err != nil {
			return err
		}
	}
	return nil
}

// httpTTSSynthesizer synthesizes each sentence in a single HTTP request.
type httpTTSSynthesizer struct {
	cfg  TTSConfig
	http *http.Client
}

// SampleRate reports the requested PCM output rate.
func (s *httpTTSSynthesizer) SampleRate() int { return s.cfg.SampleRate }

// Metadata reports the xAI voice synthesis is billed against.
func (s *httpTTSSynthesizer) Metadata() tts.Metadata {
	return tts.Metadata{VoiceID: s.cfg.Voice}
}

// Synthesize requests speech for text and streams the raw PCM downstream.
func (s *httpTTSSynthesizer) RunTTS(ctx context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	payload := map[string]any{
		"text":     text,
		"voice_id": s.cfg.Voice,
		"language": xaiLanguage(s.cfg.Language),
		"output_format": map[string]any{
			"codec":       ttsCodec,
			"sample_rate": s.cfg.SampleRate,
		},
	}
	if s.cfg.Speed != nil {
		payload["speed"] = *s.cfg.Speed
	}
	if s.cfg.OptimizeStreamingLatency != nil {
		payload["optimize_streaming_latency"] = *s.cfg.OptimizeStreamingLatency
	}
	if s.cfg.TextNormalization != nil {
		payload["text_normalization"] = *s.cfg.TextNormalization
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.cfg.URL, bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+s.cfg.APIKey)
	req.Header.Set("Content-Type", "application/json")
	return tts.StreamResponse(s.http, req, emit)
}

// firstNonEmpty returns the first non-empty string, or "" when both are empty.
func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
