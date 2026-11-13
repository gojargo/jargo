package soniox

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"slices"
	"strings"
	"sync"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/service/wsservice"
	"github.com/gojargo/jargo/service/wsutil"
	errs "github.com/gojargo/jargo/utils/errors"
)

// NewSTT builds a Soniox streaming STT service.
func NewSTT(cfg Config) *stt.StreamService {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	c := &connector{cfg: cfg, live: newSTTSettings(cfg)}
	svc := stt.NewStream("SonioxSTT", c, cfg.SampleRate)
	c.svc = svc
	return svc
}

// sonioxSTTLanguages are the languages Soniox transcribes, each under the code
// it takes for it. A regional variant resolves through its base language.
//
//nolint:gochecknoglobals // read-only lookup table
var sonioxSTTLanguages = map[language.Language]string{
	language.Afrikaans:       "af",
	language.Arabic:          "ar",
	language.Azerbaijani:     "az",
	language.Belarusian:      "be",
	language.Bulgarian:       "bg",
	language.Bengali:         "bn",
	language.Bosnian:         "bs",
	language.Catalan:         "ca",
	language.Czech:           "cs",
	language.Welsh:           "cy",
	language.Danish:          "da",
	language.German:          "de",
	language.Greek:           "el",
	language.English:         "en",
	language.Spanish:         "es",
	language.Estonian:        "et",
	language.Basque:          "eu",
	language.Persian:         "fa",
	language.Finnish:         "fi",
	language.French:          "fr",
	language.Galician:        "gl",
	language.Gujarati:        "gu",
	language.Hebrew:          "he",
	language.Hindi:           "hi",
	language.Croatian:        "hr",
	language.Hungarian:       "hu",
	language.Indonesian:      "id",
	language.Italian:         "it",
	language.Japanese:        "ja",
	language.Kazakh:          "kk",
	language.Kannada:         "kn",
	language.Korean:          "ko",
	language.Lithuanian:      "lt",
	language.Latvian:         "lv",
	language.Macedonian:      "mk",
	language.Malayalam:       "ml",
	language.Marathi:         "mr",
	language.Malay:           "ms",
	language.NorwegianBokmal: "no",
	language.Dutch:           "nl",
	language.Norwegian:       "no",
	language.Punjabi:         "pa",
	language.Polish:          "pl",
	language.Portuguese:      "pt",
	language.Romanian:        "ro",
	language.Russian:         "ru",
	language.Slovak:          "sk",
	language.Slovenian:       "sl",
	language.Albanian:        "sq",
	language.Serbian:         "sr",
	language.Swedish:         "sv",
	language.Swahili:         "sw",
	language.Tamil:           "ta",
	language.Telugu:          "te",
	language.Thai:            "th",
	language.Tagalog:         "tl",
	language.Turkish:         "tr",
	language.Ukrainian:       "uk",
	language.Urdu:            "ur",
	language.Vietnamese:      "vi",
	language.Chinese:         "zh",
}

// sonioxSTTLanguage maps a language to Soniox's code for it, the base code.
// Bokmål is Norwegian to Soniox, which takes it under "no".
func sonioxSTTLanguage(l language.Language) string {
	return language.Resolve(l, sonioxSTTLanguages, true)
}

// prepareLanguageHints names each hint the way Soniox does, once: hints for two
// regions of one language name the same language twice.
func prepareLanguageHints(hints []string) []string {
	prepared := make([]string, 0, len(hints))
	for _, hint := range hints {
		code := sonioxSTTLanguage(language.Language(hint))
		if !slices.Contains(prepared, code) {
			prepared = append(prepared, code)
		}
	}
	return prepared
}

// ContextGeneralItem is one key-value pair of structured context.
type ContextGeneralItem struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

// ContextTranslationTerm maps an ambiguous or domain-specific term onto the
// translation it should take.
type ContextTranslationTerm struct {
	Source string `json:"source"`
	Target string `json:"target"`
}

// Context primes the model with what the conversation is about, for the models
// that take it.
type Context struct {
	// General is structured context as key-value pairs.
	General []ContextGeneralItem `json:"general,omitempty"`
	// Text is free-form context.
	Text string `json:"text,omitempty"`
	// Terms are words the model should expect to hear.
	Terms []string `json:"terms,omitempty"`
	// TranslationTerms fix how particular terms are translated.
	TranslationTerms []ContextTranslationTerm `json:"translation_terms,omitempty"`
}

// Settings is the part of the Soniox configuration that can change while the
// pipeline runs. Soniox is told all of it in the handshake that opens a session,
// so a change to any of it reopens the session.
type Settings struct {
	settings.STT

	// LanguageHints are the languages to expect, each named the way Soniox
	// names it when a session opens. An update giving a language and no hints
	// replaces the hints with that language, since the hints are what Soniox
	// reads.
	LanguageHints settings.Opt[[]string] `settings:"language_hints"`
	// LanguageHintsStrict confines recognition to the languages hinted.
	LanguageHintsStrict settings.Opt[bool] `settings:"language_hints_strict"`
	// Context primes the model with what the conversation is about.
	Context settings.Opt[Context] `settings:"context"`
	// EnableSpeakerDiarization labels speakers in the transcript.
	EnableSpeakerDiarization settings.Opt[bool] `settings:"enable_speaker_diarization"`
	// EnableLanguageIdentification reports the language of each token.
	EnableLanguageIdentification settings.Opt[bool] `settings:"enable_language_identification"`
	// MaxEndpointDelayMs bounds how long Soniox waits before ending a turn.
	MaxEndpointDelayMs settings.Opt[int] `settings:"max_endpoint_delay_ms"`
	// EndpointSensitivity tunes how readily Soniox calls a turn ended.
	EndpointSensitivity settings.Opt[float64] `settings:"endpoint_sensitivity"`
	// EndpointLatencyAdjustmentLevel trades endpoint latency against accuracy.
	EndpointLatencyAdjustmentLevel settings.Opt[int] `settings:"endpoint_latency_adjustment_level"`
	// ClientReferenceID labels the session in Soniox's own records.
	ClientReferenceID settings.Opt[string] `settings:"client_reference_id"`
}

// ApplyUpdate merges an update, implementing settings.Updater. Soniox reads the
// language hints when a session opens, not the language, so a language the
// update gives is moved into the hints rather than stored. Hints given in the
// same update win, and a language outside the catalog stays where it was given.
func (s *Settings) ApplyUpdate(delta any) (settings.Changed, error) {
	hint, move := languageAsHint(delta)
	kept := s.Language
	changed, err := settings.ApplyFields(s, delta)
	if err != nil || !move {
		return changed, err
	}
	s.Language = kept
	delete(changed, "language")
	hints := []string{hint}
	if old, ok := s.LanguageHints.Value(); !ok || !slices.Equal(old, hints) {
		changed["language_hints"] = s.LanguageHints.AsAny()
		s.LanguageHints = settings.Set(hints)
	}
	return changed, nil
}

// languageAsHint is the language an update gives, when it is one to move into
// the hints: a language the catalog names, in an update giving no hints of its
// own.
func languageAsHint(delta any) (string, bool) {
	given, err := settings.Given(delta)
	if err != nil {
		return "", false
	}
	if _, ok := given["language_hints"]; ok {
		return "", false
	}
	code, _ := given["language"].(string)
	if code == "" || !slices.Contains(language.All(), language.Language(code)) {
		return "", false
	}
	return code, true
}

// newSTTSettings is the starting state, taken from what the service was built
// with.
func newSTTSettings(cfg Config) *Settings {
	s := &Settings{}
	s.Model = settings.Set(cfg.Model)
	// The language the service is built with becomes the hint, as a language
	// update does, unless hints are given alongside it, which win.
	switch {
	case len(cfg.LanguageHints) > 0:
		hints := make([]string, 0, len(cfg.LanguageHints))
		for _, l := range cfg.LanguageHints {
			hints = append(hints, l.Code())
		}
		s.LanguageHints = settings.Set(hints)
		if cfg.Language != "" {
			s.Language = settings.Set(sonioxSTTLanguage(cfg.Language))
		}
	case slices.Contains(language.All(), cfg.Language):
		s.LanguageHints = settings.Set([]string{cfg.Language.Code()})
	case cfg.Language != "":
		s.Language = settings.Set(cfg.Language.Code())
	}
	setOpt(&s.LanguageHintsStrict, cfg.LanguageHintsStrict)
	setOpt(&s.EnableSpeakerDiarization, cfg.EnableSpeakerDiarization)
	setOpt(&s.EnableLanguageIdentification, cfg.EnableLanguageIdentification)
	setOpt(&s.MaxEndpointDelayMs, cfg.MaxEndpointDelayMs)
	setOpt(&s.EndpointSensitivity, cfg.EndpointSensitivity)
	setOpt(&s.EndpointLatencyAdjustmentLevel, cfg.EndpointLatencyAdjustmentLevel)
	if cfg.ClientReferenceID != "" {
		s.ClientReferenceID = settings.Set(cfg.ClientReferenceID)
	}
	if cfg.Context != nil {
		s.Context = settings.Set(*cfg.Context)
	}
	return s
}

// setOpt gives o a value when the configuration carried one.
func setOpt[T any](o *settings.Opt[T], v *T) {
	if v != nil {
		*o = settings.Set(*v)
	}
}

type connector struct {
	cfg Config
	// live is what may change while the pipeline runs: the model and the
	// language, which is the set Soniox treats as changeable and that jargo's
	// configuration carries.
	live *Settings
	// svc is the service this connector drives, for reporting an error Soniox
	// sends in a session. It is nil for a connector built outside NewSTT.
	svc *stt.StreamService
}

// Metadata reports the transcript latency the turn strategies size their
// wait by.
func (c *connector) Metadata() stt.Metadata {
	return stt.Metadata{TTFSP99: stt.TTFSP99Or(c.cfg.TTFSP99, stt.SonioxTTFSP99)}
}

// Settings is the configuration a caller may change while the pipeline runs.
func (c *connector) Settings() any { return c.live }

// Keepalive asks for an idle session to be kept open, implementing
// stt.Keepaliver: Soniox closes a session that goes quiet, so one idle for a
// second is sent Soniox's own keepalive message, checked for every five.
func (c *connector) Keepalive() stt.KeepaliveOptions {
	return stt.KeepaliveOptions{Timeout: keepaliveTimeout, Interval: keepaliveInterval}
}

// UpdateSettings asks for the session to be reopened whenever anything changed.
// Soniox is told all of this in the handshake that opens the session, so a
// change reaches it only by opening another.
func (c *connector) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	return true, nil
}

// ServiceLanguage names a language the way Soniox does, by its base code.
func (c *connector) ServiceLanguage(l language.Language) string {
	return sonioxSTTLanguage(l)
}

// WebsocketOptions hands the stream base the connection settings the
// configuration gives, implementing wsservice.Optioner.
func (c *connector) WebsocketOptions() wsservice.Options { return c.cfg.Websocket }

// Connect dials the WebSocket, authenticated by the API key in its handshake
// header, and sends the config message.
func (c *connector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	conn, err := c.cfg.Websocket.Dial(ctx, c.cfg.URL, authHeader(c.cfg.APIKey), readLimit)
	if err != nil {
		return nil, err
	}

	if err := conn.Write(ctx, websocket.MessageText, c.config(sampleRate)); err != nil {
		_ = conn.Close(websocket.StatusInternalError, "config failed")
		return nil, err
	}
	return &stream{conn: conn, ctx: ctx, report: c.reportError}, nil
}

// reportError puts an error Soniox sent in a session on the pipeline. Its code
// is an HTTP status, which gives the error its category. The session is left
// to Soniox to end.
func (c *connector) reportError(ctx context.Context, code json.RawMessage, message string) {
	msg := "Soniox STT error: " + message
	if len(code) > 0 {
		msg = fmt.Sprintf("Soniox STT error %s: %s", code, message)
	}
	if c.svc == nil {
		slog.Error(msg)
		return
	}
	var opts []processor.ErrorOption
	if category := classifyErrorCode(code); category != errs.Unset {
		opts = append(opts, processor.WithErrorCategory(category))
	}
	c.svc.PushError(ctx, msg, nil, false, opts...)
}

func (c *connector) config(sampleRate int) []byte {
	cfg := map[string]any{
		"model":        c.live.Model.Or(c.cfg.Model),
		"audio_format": audioFormat,
		"sample_rate":  sampleRate,
		"num_channels": 1,
		// Always on. A VAD stop finalizes sooner, and the endpoint still ends
		// the transcript of speech the VAD missed; it never ends the turn.
		"enable_endpoint_detection": true,
	}
	// Everything Soniox is told in the handshake and that may change while the
	// pipeline runs. A setting with no value is left out, so Soniox applies its
	// own default rather than being sent a zero that means something else.
	if hints, ok := c.live.LanguageHints.Value(); ok {
		cfg["language_hints"] = prepareLanguageHints(hints)
	}
	putOpt(cfg, "language_hints_strict", c.live.LanguageHintsStrict)
	putOpt(cfg, "context", c.live.Context)
	putOpt(cfg, "enable_speaker_diarization", c.live.EnableSpeakerDiarization)
	putOpt(cfg, "enable_language_identification", c.live.EnableLanguageIdentification)
	putOpt(cfg, "max_endpoint_delay_ms", c.live.MaxEndpointDelayMs)
	putOpt(cfg, "endpoint_sensitivity", c.live.EndpointSensitivity)
	putOpt(cfg, "endpoint_latency_adjustment_level", c.live.EndpointLatencyAdjustmentLevel)
	putOpt(cfg, "client_reference_id", c.live.ClientReferenceID)

	b, _ := json.Marshal(cfg) //nolint:errchkjson // map of known-serializable values
	return b
}

// putOpt writes a setting into the handshake only when it carries a value.
func putOpt[T any](cfg map[string]any, key string, o settings.Opt[T]) {
	if v, ok := o.Value(); ok {
		cfg[key] = v
	}
}

type stream struct {
	conn    *wsutil.Conn
	ctx     context.Context
	writeMu sync.Mutex
	// report puts an error Soniox sent on the pipeline.
	report func(ctx context.Context, code json.RawMessage, message string)

	// final is the final tokens of the utterance not yet ended.
	final []token
	// problem is an error Soniox sent behind results, reported once those
	// results have gone out.
	problem *message
	// finished is set once Soniox has ended the session behind results, and
	// ends the read that follows them.
	finished bool
}

// Send writes a chunk of PCM as a binary frame.
func (s *stream) Send(audio []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, audio)
}

// SendKeepalive sends Soniox's own keepalive message in place of the silence,
// implementing stt.KeepaliveSender.
func (s *stream) SendKeepalive([]byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, []byte(keepaliveMessage))
}

// Finalize asks Soniox for the final tokens of the speech the VAD heard end,
// implementing stt.Finalizer, so the transcript arrives sooner than the
// endpoint would bring it. Soniox answers with the "<fin>" marker.
func (s *stream) Finalize() error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageText, []byte(finalizeMessage))
}

// Recv reads the next batch of tokens. Final tokens accumulate until a final
// "<end>" or "<fin>" marker, which flushes them as the utterance with EndOfTurn
// set; the tokens after the marker belong to the next one. Once the message is
// read, what is held, final and provisional together, surfaces as an interim
// result, since the final tokens only go out formatted as a whole at the
// marker. An error Soniox sends flushes the utterance and is reported, and the
// session carries on; the finished message flushes it and ends the session.
func (s *stream) Recv() ([]stt.Result, error) {
	if m := s.problem; m != nil {
		s.problem = nil
		s.reportProblem(m)
	}
	if s.finished {
		return nil, io.EOF
	}
	for {
		_, data, err := s.conn.Read(s.ctx)
		if err != nil {
			return nil, err
		}
		var m message
		if err := json.Unmarshal(data, &m); err != nil {
			slog.Warn("soniox stt: received a message that is not JSON", "message", string(data), "err", err)
			continue
		}
		results := s.handle(&m)
		if m.isError() {
			results = s.flush(results)
			if len(results) > 0 {
				s.problem = &m
			} else {
				s.reportProblem(&m)
			}
		}
		if m.Finished {
			results = s.flush(results)
			if len(results) == 0 {
				return nil, io.EOF
			}
			s.finished = true
		}
		if len(results) > 0 {
			return results, nil
		}
	}
}

// handle reads the tokens of one message into results.
func (s *stream) handle(m *message) []stt.Result {
	var results []stt.Result
	var provisional []token
	for _, t := range m.Tokens {
		switch {
		case !t.IsFinal:
			provisional = append(provisional, t)
		case t.isEnd():
			results = s.flush(results)
		default:
			s.final = append(s.final, t)
		}
	}
	if len(s.final) > 0 || len(provisional) > 0 {
		// The final tokens go out as interim too, so that the transcript is
		// pushed formatted as a whole once the marker arrives.
		text := tokensText(s.final) + tokensText(provisional)
		results = append(results, stt.Result{Text: text, Final: false})
	}
	return results
}

// flush ends the utterance held, adding it to results. With nothing held it
// adds nothing, so a marker behind the one that already ended the utterance
// ends nothing more. The transcript is labeled with the language most of its
// tokens were identified as. Every transcript Soniox gives ends at a marker, so
// every one is final.
func (s *stream) flush(results []stt.Result) []stt.Result {
	if len(s.final) == 0 {
		return results
	}
	tokens := s.final
	s.final = nil
	return append(results, stt.Result{
		Text:      tokensText(tokens),
		Final:     true,
		EndOfTurn: true,
		Language:  languageFromTokens(tokens),
	})
}

// tokensText is the text of tokens, joined as Soniox spaced them.
func tokensText(tokens []token) string {
	var b strings.Builder
	for _, t := range tokens {
		b.WriteString(t.Text)
	}
	return b.String()
}

// languageFromTokens is the language most of the tokens were identified as,
// the first of them on a tie, or none when no token names a language the
// catalog knows.
func languageFromTokens(tokens []token) string {
	counts := map[string]int{}
	var order []string
	for _, t := range tokens {
		if t.Language == "" || !slices.Contains(language.All(), language.Language(t.Language)) {
			continue
		}
		if counts[t.Language] == 0 {
			order = append(order, t.Language)
		}
		counts[t.Language]++
	}
	best := ""
	for _, l := range order {
		if counts[l] > counts[best] {
			best = l
		}
	}
	return best
}

// reportProblem reports an error Soniox sent.
func (s *stream) reportProblem(m *message) {
	if s.report != nil {
		s.report(s.ctx, m.ErrorCode, m.ErrorMessage)
	}
}

// Stop tells Soniox the audio is complete as the pipeline ends, implementing
// stt.Stopper, with the empty frame that ends a session. It is sent while the
// socket is still open, before the service tears the session down. The final
// tokens still to come are not waited for: they would hold the shutdown open
// for a transcript that starts no further turn. The send is best effort, and
// the teardown goes on whether or not it succeeds.
func (s *stream) Stop() error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(s.ctx, websocket.MessageBinary, []byte{})
}

// Close closes the socket. A session canceled, or closed to reopen it with new
// settings, is closed without the end-of-audio frame, which only ending the
// pipeline sends.
func (s *stream) Close() error {
	return s.conn.Close(websocket.StatusNormalClosure, "")
}

// message is the subset of a Soniox result message we use.
type message struct {
	Tokens       []token         `json:"tokens"`
	Finished     bool            `json:"finished"`
	ErrorCode    json.RawMessage `json:"error_code"`
	ErrorMessage string          `json:"error_message"`
}

// isError reports whether the message carries an error, by its code or by its
// message.
func (m *message) isError() bool {
	code := string(m.ErrorCode)
	return (code != "" && code != "null" && code != "0") || m.ErrorMessage != ""
}

type token struct {
	Text    string `json:"text"`
	IsFinal bool   `json:"is_final"`
	// Language is the language Soniox identified the token as, when language
	// identification is on.
	Language string `json:"language"`
}

// isEnd reports whether the token is one of the markers that end an utterance.
func (t token) isEnd() bool { return t.Text == endToken || t.Text == finToken }
