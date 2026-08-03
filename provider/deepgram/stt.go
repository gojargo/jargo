package deepgram

import (
	"context"
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsutil"
)

const (
	listenURL       = "wss://api.deepgram.com/v1/listen"
	keepAlivePeriod = 8 * time.Second
	defaultSTTModel = "nova-3"
	defaultEncoding = "linear16"
	defaultChannels = 1
)

// Config configures the STT service. Fields left at their zero value fall back
// to jargo's defaults (which preserve its tuned turn detection); several of
// those defaults differ from Deepgram's own and are noted per field. Optional
// fields modeled as pointers or slices are omitted from the request when unset.
type Config struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// ListenURL overrides the live-transcription WebSocket endpoint; empty uses
	// Deepgram's hosted endpoint.
	ListenURL string
	// Model is the Deepgram model; empty uses "nova-3".
	Model string
	// Language is the transcription language; empty uses English (US).
	Language language.Language
	// SampleRate is the input audio sample rate; 0 uses the transport's rate.
	SampleRate int
	// Encoding is the audio encoding; empty uses "linear16".
	Encoding string
	// Channels is the channel count; 0 uses 1.
	Channels int

	// InterimResults emits partial transcripts; nil defaults to true.
	InterimResults *bool
	// SmartFormat applies Deepgram smart formatting; nil omits the param.
	SmartFormat *bool
	// Punctuate adds punctuation; nil defaults to true.
	Punctuate *bool
	// VADEvents requests Deepgram speech/utterance events; nil omits the param.
	VADEvents *bool
	// Endpointing is the silence in ms before Deepgram's speech_final; nil omits
	// the param.
	Endpointing *int
	// UtteranceEndMs is the silence in ms before an utterance-end event; nil omits
	// the param (jargo does not consume Deepgram's UtteranceEnd events).
	UtteranceEndMs *int

	// The remaining options are omitted from the request unless set, matching
	// Deepgram's defaults.

	// Numerals converts spoken numbers to numerals.
	Numerals *bool
	// ProfanityFilter filters profanity from transcripts.
	ProfanityFilter *bool
	// Diarize labels speakers in the transcript.
	Diarize *bool
	// DetectEntities enables named-entity detection.
	DetectEntities *bool
	// Dictation converts spoken commands (e.g. "comma") to punctuation.
	Dictation *bool
	// Multichannel transcribes each channel independently.
	Multichannel *bool
	// MipOptOut opts out of Deepgram's model-improvement program.
	MipOptOut *bool
	// Version pins the model version (e.g. "latest").
	Version string
	// Keywords boosts recognition of the given terms (legacy models); each entry
	// may carry an intensifier, e.g. "snuffleupagus:5".
	Keywords []string
	// Keyterm boosts recognition of the given terms (nova-3).
	Keyterm []string
	// Redact removes the given categories of sensitive data.
	Redact []string
	// Replace applies find/replace rules to the transcript.
	Replace []string
	// Search flags occurrences of the given terms.
	Search []string
	// Tag attaches billing tags to the request.
	Tag []string
	// ExtraQuery sets arbitrary additional Deepgram query parameters not modeled
	// above; values override any param of the same name set from other fields.
	ExtraQuery map[string]string
}

// Validate reports whether the configuration is usable.
func (cfg Config) Validate() error { return validate.Struct(cfg) }

// NewSTT builds a Deepgram streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.Model == "" {
		cfg.Model = defaultSTTModel
	}
	if cfg.Language == "" {
		cfg.Language = language.EnglishUS
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Channels == 0 {
		cfg.Channels = defaultChannels
	}
	if cfg.ListenURL == "" {
		cfg.ListenURL = listenURL
	}
	return stt.NewStream("DeepgramSTT", &connector{cfg: cfg, live: newSettings(cfg)}, cfg.SampleRate)
}

// deepgramLanguage maps a Language to Deepgram's code. Deepgram uses BCP-47
// codes directly, so the canonical code passes through unchanged.
func deepgramLanguage(l language.Language) string {
	return l.Code()
}

// Settings is the part of the Deepgram configuration that can change while the
// pipeline runs. It is the same set the provider itself treats as changeable;
// everything else about the connection, the endpoint, the encoding, the channel
// count, is fixed when the service is built.
//
// A change to any of it reopens the transcription session, because Deepgram
// takes these as query parameters when the session opens and has no way to be
// told about them afterwards.
type Settings struct {
	settings.STT

	// DetectEntities enables named-entity detection.
	DetectEntities settings.Opt[bool] `settings:"detect_entities"`
	// Diarize labels speakers in the transcript.
	Diarize settings.Opt[bool] `settings:"diarize"`
	// Dictation converts spoken commands (e.g. "comma") to punctuation.
	Dictation settings.Opt[bool] `settings:"dictation"`
	// Endpointing is the silence in ms before Deepgram's speech_final.
	Endpointing settings.Opt[int] `settings:"endpointing"`
	// InterimResults emits partial transcripts.
	InterimResults settings.Opt[bool] `settings:"interim_results"`
	// Keyterm boosts recognition of the given terms (nova-3).
	Keyterm settings.Opt[[]string] `settings:"keyterm"`
	// Keywords boosts recognition of the given terms (legacy models).
	Keywords settings.Opt[[]string] `settings:"keywords"`
	// Numerals converts spoken numbers to numerals.
	Numerals settings.Opt[bool] `settings:"numerals"`
	// ProfanityFilter filters profanity from transcripts.
	ProfanityFilter settings.Opt[bool] `settings:"profanity_filter"`
	// Punctuate adds punctuation.
	Punctuate settings.Opt[bool] `settings:"punctuate"`
	// Redact removes the given categories of sensitive data.
	Redact settings.Opt[[]string] `settings:"redact"`
	// Replace applies find/replace rules to the transcript.
	Replace settings.Opt[[]string] `settings:"replace"`
	// Search flags occurrences of the given terms.
	Search settings.Opt[[]string] `settings:"search"`
	// SmartFormat applies Deepgram smart formatting.
	SmartFormat settings.Opt[bool] `settings:"smart_format"`
	// UtteranceEndMs is the silence in ms before an utterance-end event.
	UtteranceEndMs settings.Opt[int] `settings:"utterance_end_ms"`
}

// newSettings is the starting state, taken from what the service was built with.
func newSettings(cfg Config) *Settings {
	s := &Settings{}
	s.Model = settings.Set(cfg.Model)
	s.Language = settings.Set(deepgramLanguage(cfg.Language))
	// The two that default to on when the caller says nothing.
	s.InterimResults = settings.Set(boolOr(cfg.InterimResults, true))
	s.Punctuate = settings.Set(boolOr(cfg.Punctuate, true))
	setOptBool(&s.SmartFormat, cfg.SmartFormat)
	setOptBool(&s.Numerals, cfg.Numerals)
	setOptBool(&s.ProfanityFilter, cfg.ProfanityFilter)
	setOptBool(&s.Diarize, cfg.Diarize)
	setOptBool(&s.DetectEntities, cfg.DetectEntities)
	setOptBool(&s.Dictation, cfg.Dictation)
	setOptInt(&s.Endpointing, cfg.Endpointing)
	setOptInt(&s.UtteranceEndMs, cfg.UtteranceEndMs)
	setOptSlice(&s.Keywords, cfg.Keywords)
	setOptSlice(&s.Keyterm, cfg.Keyterm)
	setOptSlice(&s.Redact, cfg.Redact)
	setOptSlice(&s.Replace, cfg.Replace)
	setOptSlice(&s.Search, cfg.Search)
	return s
}

func boolOr(v *bool, def bool) bool {
	if v != nil {
		return *v
	}
	return def
}

func setOptBool(o *settings.Opt[bool], v *bool) {
	if v != nil {
		*o = settings.Set(*v)
	}
}

func setOptInt(o *settings.Opt[int], v *int) {
	if v != nil {
		*o = settings.Set(*v)
	}
}

func setOptSlice(o *settings.Opt[[]string], v []string) {
	if len(v) > 0 {
		*o = settings.Set(v)
	}
}

// query builds the live-transcription query string for the given sample rate.
func (cfg *Config) query(sampleRate int, live *Settings) url.Values {
	q := url.Values{}
	q.Set("encoding", cfg.Encoding)
	q.Set("sample_rate", strconv.Itoa(sampleRate))
	q.Set("channels", strconv.Itoa(cfg.Channels))

	// Fixed when the service is built: Deepgram takes them at the session, and
	// the provider does not treat them as changeable.
	setBoolOpt(q, "vad_events", cfg.VADEvents)
	setBoolOpt(q, "multichannel", cfg.Multichannel)
	setBoolOpt(q, "mip_opt_out", cfg.MipOptOut)
	setStrOpt(q, "version", cfg.Version)
	addAll(q, "tag", cfg.Tag)

	// Changeable while the pipeline runs.
	setStrOpt(q, "model", live.Model.Or(""))
	setStrOpt(q, "language", live.Language.Or(""))
	setOptQueryBool(q, "interim_results", live.InterimResults)
	setOptQueryBool(q, "smart_format", live.SmartFormat)
	setOptQueryBool(q, "punctuate", live.Punctuate)
	setOptQueryInt(q, "endpointing", live.Endpointing)
	setOptQueryInt(q, "utterance_end_ms", live.UtteranceEndMs)
	setOptQueryBool(q, "numerals", live.Numerals)
	setOptQueryBool(q, "profanity_filter", live.ProfanityFilter)
	setOptQueryBool(q, "diarize", live.Diarize)
	setOptQueryBool(q, "detect_entities", live.DetectEntities)
	setOptQueryBool(q, "dictation", live.Dictation)
	addAll(q, "keywords", live.Keywords.Or(nil))
	addAll(q, "keyterm", live.Keyterm.Or(nil))
	addAll(q, "redact", live.Redact.Or(nil))
	addAll(q, "replace", live.Replace.Or(nil))
	addAll(q, "search", live.Search.Or(nil))

	for k, v := range cfg.ExtraQuery {
		q.Set(k, v)
	}
	return q
}

// setOptQueryBool sets key only when the setting carries a value.
func setOptQueryBool(q url.Values, key string, o settings.Opt[bool]) {
	if v, ok := o.Value(); ok {
		q.Set(key, strconv.FormatBool(v))
	}
}

// setOptQueryInt sets key only when the setting carries a value.
func setOptQueryInt(q url.Values, key string, o settings.Opt[int]) {
	if v, ok := o.Value(); ok {
		q.Set(key, strconv.Itoa(v))
	}
}

// setBoolOpt sets key only when v is non-nil.
func setBoolOpt(q url.Values, key string, v *bool) {
	if v != nil {
		q.Set(key, strconv.FormatBool(*v))
	}
}

// setStrOpt sets key only when v is non-empty.
func setStrOpt(q url.Values, key, v string) {
	if v != "" {
		q.Set(key, v)
	}
}

// addAll appends each value under key, preserving repeated query parameters.
func addAll(q url.Values, key string, vals []string) {
	for _, v := range vals {
		q.Add(key, v)
	}
}

type connector struct {
	cfg Config
	// live is what may change while the pipeline runs. The service serializes
	// reading it here against applying an update, so a session is never opened
	// from a half-written change.
	live *Settings
}

// Metadata reports the Deepgram model transcription is billed against.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{Model: c.live.Model.Or(c.cfg.Model)}
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *connector) Settings() any { return c.live }

// UpdateSettings asks for the session to be reopened whenever anything changed.
// Deepgram takes all of it as query parameters when the session opens, so a
// change reaches it only by opening another.
func (c *connector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	return true, nil
}

// ServiceLanguage names a language the way Deepgram does. It uses BCP-47 codes
// directly, so the canonical code passes through unchanged.
func (c *connector) ServiceLanguage(l language.Language) string {
	return deepgramLanguage(l)
}

// Connect dials the live transcription WebSocket for the given sample rate.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	q := c.cfg.query(sampleRate, c.live)

	header := http.Header{}
	header.Set("Authorization", authToken(c.cfg.APIKey))

	conn, err := wsutil.Dial(ctx, c.cfg.ListenURL+"?"+q.Encode(), header, 0)
	if err != nil {
		return nil, err
	}
	s := &stream{conn: conn, ctx: ctx}
	s.wg.Go(s.keepAlive)
	return s, nil
}

type stream struct {
	conn    *wsutil.Conn
	ctx     context.Context
	writeMu sync.Mutex
	wg      sync.WaitGroup
}

// dgMessage is the subset of Deepgram's live transcription result we use.
type dgMessage struct {
	Type    string `json:"type"`
	Channel struct {
		Alternatives []struct {
			Transcript string `json:"transcript"`
		} `json:"alternatives"`
	} `json:"channel"`
	IsFinal     bool `json:"is_final"`
	SpeechFinal bool `json:"speech_final"`
}

// Send writes a chunk of PCM audio as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// Recv reads the next result. A finalized result carries Deepgram's speech_final
// as the end-of-turn signal.
func (s *stream) Recv() ([]stt.Result, error) {
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m dgMessage
		if err := json.Unmarshal(data, &m); err != nil {
			continue
		}
		if m.Type != "Results" || len(m.Channel.Alternatives) == 0 {
			continue
		}
		text := m.Channel.Alternatives[0].Transcript
		if text == "" {
			continue
		}
		return []stt.Result{{Text: text, Final: m.IsFinal, EndOfTurn: m.SpeechFinal}}, nil
	}
}

// keepAlive sends a periodic KeepAlive so Deepgram does not close an idle
// connection during silence.
func (s *stream) keepAlive() {
	ticker := time.NewTicker(keepAlivePeriod)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			s.writeMu.Lock()
			_ = s.conn.Write(s.ctx, websocket.MessageText, []byte(`{"type":"KeepAlive"}`))
			s.writeMu.Unlock()
		}
	}
}

// Close asks Deepgram to flush and then closes the socket.
func (s *stream) Close() error {
	s.writeMu.Lock()
	_ = s.conn.Write(context.Background(), websocket.MessageText, []byte(`{"type":"CloseStream"}`))
	s.writeMu.Unlock()
	s.wg.Wait()
	return s.conn.Close(websocket.StatusNormalClosure, "")
}
