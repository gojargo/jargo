package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsutil"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/gojargo/jargo/utils/text"
)

// GenerationConfig guides Cartesia generation; it applies to the sonic-3 series
// of models. Fields left at their zero value are omitted.
type GenerationConfig struct {
	// Volume multiplies the generated speech volume (0.5 to 2.0).
	Volume *float64 `json:"volume,omitempty"`
	// Speed multiplies the speaking rate (0.6 to 1.5).
	Speed *float64 `json:"speed,omitempty"`
	// Emotion guides the emotional tone (e.g. "neutral", "excited", "sad").
	Emotion string `json:"emotion,omitempty"`
}

// NewTTS builds a Cartesia TTS service.
func NewTTS(cfg Config) *tts.Base {
	cfg = ttsDefaults(cfg)

	s := &synthesizer{cfg: cfg}
	var b *tts.Base
	if cfg.WordTimestamps {
		// Only the timestamp-aware type implements tts.WordTimestamps, so the base
		// takes the word-aligned path solely when the caller opts in.
		b = tts.New("CartesiaTTS", &timedSynthesizer{synthesizer: s})
	} else {
		b = tts.New("CartesiaTTS", s)
	}
	// Always hold off on a sentence boundary inside a spell tag. The tag is full
	// of periods that end no sentence, and splitting it would hand Cartesia half
	// a tag. The preferred way to use Cartesia's markup is to have a text
	// transform insert it for the synthesizer alone; this keeps markup that
	// reaches the service by any other route intact.
	if tok := b.TextTokenizer(); tok != nil {
		b.SetTextAggregator(text.NewSkipTagsAggregator(cfg.TextAggregation, tok,
			[]text.StartEndTags{{Start: "<spell>", End: "</spell>"}}))
	}
	return b
}

// ttsDefaults fills in what the caller left unset.
func ttsDefaults(cfg Config) Config {
	if cfg.URL == "" {
		cfg.URL = defaultURL
	}
	if cfg.Version == "" {
		cfg.Version = defaultVersion
	}
	if cfg.Model == "" {
		cfg.Model = defaultModel
	}
	if cfg.SampleRate == 0 {
		cfg.SampleRate = defaultSampleRate
	}
	if cfg.Encoding == "" {
		cfg.Encoding = defaultEncoding
	}
	if cfg.Container == "" {
		cfg.Container = defaultContainer
	}
	if cfg.TextAggregation == "" {
		cfg.TextAggregation = frames.AggregationSentence
	}
	// Cartesia warns against the middle ground of client-side sentence
	// aggregation plus the server's default 3000ms buffer. With no value chosen,
	// send 0 when sentences are aggregated here and leave it unset when tokens
	// are streamed, so the server's managed buffering applies to the one shape it
	// suits.
	if cfg.MaxBufferDelayMs == nil && cfg.TextAggregation != frames.AggregationToken {
		zero := 0
		cfg.MaxBufferDelayMs = &zero
	}
	return cfg
}

// synthesizer speaks over one WebSocket held open for the session, with each
// turn given a context of its own on it.
//
// A context is what keeps the turns apart on a shared connection. Every message
// carries the id of the context it belongs to, so audio the server had already
// generated for a turn the user cut off arrives after the interruption and is
// dropped rather than spoken over the next one. It is also what lets the
// sentences of one turn stream continuously: each is sent on the same context
// with continue set, and the turn is flushed once its last sentence has gone.
type synthesizer struct {
	cfg  Config
	host tts.AudioContextHost

	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	// mu guards the connection. It is never held across a call into the host:
	// appending blocks while the audio plays out, and the next sentence has to
	// be able to go out while that happens.
	mu       sync.Mutex
	conn     *wsutil.Conn
	connStop context.CancelFunc
	// reading records that the loop owning incoming messages is running, so it
	// is started once and survives the reconnects it makes itself.
	reading bool
}

// timedSynthesizer adds word-timestamp streaming on top of synthesizer. It
// implements tts.WordTimestamps.
type timedSynthesizer struct {
	*synthesizer
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// Metadata reports the Cartesia model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.cfg.Model, VoiceID: s.cfg.VoiceID}
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// spacelessLanguage reports whether the configured language is written without
// spaces between words. Chinese and Japanese are, and this provider reports
// their characters separately but grouped into one message, which is what makes
// the message rather than the character the unit worth reporting.
func (s *synthesizer) spacelessLanguage() bool {
	switch s.cfg.Language.BaseCode() {
	case "zh", "ja":
		return true
	default:
		return false
	}
}

// Start dials the connection when the pipeline starts, so the handshake is paid
// while the transport is still negotiating rather than in front of the first
// thing the bot says. It implements tts.Starter.
func (s *synthesizer) Start(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.connect(ctx); err != nil {
		// Best effort: the first sentence dials again and reports the failure.
		slog.Debug("cartesia tts connect on start failed", "err", err)
	}
}

// Close releases the connection, implementing tts.Closer.
func (s *synthesizer) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	conn := s.conn
	if conn == nil {
		return nil
	}
	if s.connStop != nil {
		s.connStop()
		s.connStop = nil
	}
	s.conn = nil
	return conn.Close(websocket.StatusNormalClosure, "")
}

// connect dials the connection, and starts the loop that owns the messages
// coming back on it the first time. Callers hold mu.
func (s *synthesizer) connect(ctx context.Context) error {
	if s.conn != nil {
		return nil
	}
	header := http.Header{}
	header.Set("X-API-Key", s.cfg.APIKey)
	header.Set("Cartesia-Version", s.cfg.Version)
	conn, err := wsutil.Dial(ctx, s.cfg.URL, header, readLimit)
	if err != nil {
		return err
	}
	s.conn = conn
	if s.reading {
		// The loop is already running and dialed this one for itself.
		return nil
	}
	// The reader outlives the call that dialed: it runs until the connection is
	// closed for good, not until this sentence is done.
	runCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	s.connStop = stop
	s.reading = true
	go s.readLoop(runCtx)
	return nil
}

// disconnect closes the connection so the next write dials a fresh one. Callers
// hold mu.
func (s *synthesizer) disconnect() {
	if s.conn == nil {
		return
	}
	_ = s.conn.Close(websocket.StatusInternalError, "")
	s.conn = nil
}

// write sends one JSON message, serialized against the other writers.
func (s *synthesizer) write(ctx context.Context, conn *wsutil.Conn, msg map[string]any) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return conn.Write(ctx, websocket.MessageText, payload)
}

// readLoop owns the connection's incoming messages for as long as the service
// lasts.
//
// Cartesia closes a connection that has carried nothing for five minutes, and it
// offers no keepalive, so a read ending is usually that rather than a failure.
// The loop dials again for it, which keeps the socket warm: the next turn pays
// no handshake in front of the first thing the bot says.
func (s *synthesizer) readLoop(ctx context.Context) {
	defer func() {
		s.mu.Lock()
		s.reading = false
		s.mu.Unlock()
	}()
	for {
		s.mu.Lock()
		conn := s.conn
		s.mu.Unlock()
		if conn == nil || ctx.Err() != nil {
			return
		}

		s.readMessages(ctx, conn)
		if ctx.Err() != nil {
			return
		}

		slog.Debug("cartesia tts connection was disconnected (timeout?), reconnecting")
		s.mu.Lock()
		if s.conn == conn {
			s.conn = nil
		}
		err := s.connect(ctx)
		s.mu.Unlock()
		if err != nil {
			slog.Warn("reconnecting to cartesia tts failed", "err", err)
			return
		}
	}
}

// readMessages reads until the connection ends, dispatching each message.
func (s *synthesizer) readMessages(ctx context.Context, conn *wsutil.Conn) {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return
		}
		var m wsMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		s.dispatch(&m)
	}
}

// dispatch delivers one message to the context it names.
//
// A message for a context that is no longer open is skipped: an interruption
// abandons a turn mid-flight, and audio the server had already generated for it
// arrives afterwards. Attributing by context id keeps that audio out of the next
// turn, which is what the ids are for.
func (s *synthesizer) dispatch(m *wsMessage) {
	host := s.host
	if host == nil || m.ContextID == "" || !host.AudioContextAvailable(m.ContextID) {
		return
	}

	switch m.Type {
	case msgDone:
		s.finishContext(m.ContextID, host)
	case msgTimestamps:
		if !s.cfg.WordTimestamps {
			return
		}
		batch, opts := normalizeWordTimings(m.WordTimestamps, s.spacelessLanguage())
		if len(batch) > 0 {
			host.AddWordTimestamps(m.ContextID, batch, opts)
		}
	case msgChunk:
		pcm, err := base64.StdEncoding.DecodeString(m.Data)
		if err != nil {
			slog.Warn("cartesia tts sent audio that could not be decoded", "err", err)
			return
		}
		f := frames.NewTTSAudioRawFrame(pcm, s.SampleRate(), 1)
		f.ContextID = m.ContextID
		host.AppendToAudioContext(m.ContextID, f)
	case msgError:
		slog.Error("cartesia tts reported an error",
			"context", m.ContextID, "message", m.Message)
		s.finishContext(m.ContextID, host)
	case msgFlushDone:
		// A boundary marker within a context, which Cartesia emits per submission
		// when its own buffering is disabled. Each turn already has a context of
		// its own and every chunk carries its id, so there is nothing to do.
	default:
		slog.Warn("cartesia tts sent a message of an unknown type", "type", m.Type)
	}
}

// finishContext closes a context out: the stop frame rides the queue behind the
// audio, so it is pushed only once the last of that audio has been.
func (s *synthesizer) finishContext(contextID string, host tts.AudioContextHost) {
	stopped := frames.NewTTSStoppedFrame()
	stopped.ContextID = contextID
	host.AppendToAudioContext(contextID, stopped)
	host.RemoveAudioContext(contextID)
}

// RunTTS sends one sentence on the turn's context. It yields nothing: the call
// returns once the text is sent, so the next sentence goes out while this one is
// still being generated, and the audio arrives on the reader, which appends it
// to the context it belongs to.
func (s *synthesizer) RunTTS(
	ctx context.Context, text, contextID string, _ func(f frames.Frame) error,
) error {
	s.mu.Lock()
	if err := s.connect(ctx); err != nil {
		s.mu.Unlock()
		return err
	}
	conn := s.conn
	s.mu.Unlock()

	// continue says more of this turn may follow, which is what lets the
	// sentences of one turn stream as a single utterance. The flush at the end of
	// the turn is what closes it.
	err := s.write(ctx, conn, s.request(text, contextID, true))
	if err == nil {
		return nil
	}

	// The server's view of the connection is unknown from here, so it is dropped
	// and dialed again. The context is closed out rather than left waiting on
	// audio that is not coming.
	if host := s.host; host != nil && host.AudioContextAvailable(contextID) {
		s.finishContext(contextID, host)
	}
	s.mu.Lock()
	if s.conn == conn {
		s.disconnect()
		if connectErr := s.connect(ctx); connectErr != nil {
			slog.Debug("cartesia tts reconnect after a failed send failed", "err", connectErr)
		}
	}
	s.mu.Unlock()
	return err
}

// FlushAudio closes the turn's context, telling Cartesia the text is complete so
// it generates what it is still holding. It implements tts.AudioFlusher.
func (s *synthesizer) FlushAudio(ctx context.Context, contextID string) {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil || contextID == "" {
		return
	}
	if err := s.write(ctx, conn, s.request("", contextID, false)); err != nil {
		slog.Warn("flushing the cartesia tts context failed", "context", contextID, "err", err)
	}
}

// OnAudioContextInterrupted stops Cartesia generating into a context nobody is
// listening to, implementing tts.AudioContextInterrupter.
func (s *synthesizer) OnAudioContextInterrupted(_ context.Context, contextID string) {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil || contextID == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), cancelWriteTimeout)
	defer cancel()
	msg := map[string]any{"context_id": contextID, "cancel": true}
	if err := s.write(ctx, conn, msg); err != nil {
		slog.Warn("canceling the cartesia tts context failed", "context", contextID, "err", err)
	}
}

// request is one synthesis message: the text to add to a context, and whether
// more of the turn may follow.
func (s *synthesizer) request(text, contextID string, more bool) map[string]any {
	msg := map[string]any{
		fieldTranscript: text,
		"continue":      more,
		"context_id":    contextID,
		"model_id":      s.cfg.Model,
		"voice":         map[string]any{"mode": "id", "id": s.cfg.VoiceID},
		"output_format": map[string]any{
			"container":   s.cfg.Container,
			"encoding":    s.cfg.Encoding,
			"sample_rate": s.cfg.SampleRate,
		},
	}
	if s.cfg.WordTimestamps {
		msg["add_timestamps"] = true
		// The timings are matched against the text as it was written, so the
		// normalized spelling of a number or an abbreviation would not match.
		msg["use_normalized_timestamps"] = false
	}
	if s.cfg.MaxBufferDelayMs != nil {
		msg["max_buffer_delay_ms"] = *s.cfg.MaxBufferDelayMs
	}
	if lang := cartesiaLanguage(s.cfg.Language); lang != "" {
		msg["language"] = lang
	}
	if s.cfg.GenerationConfig != nil {
		msg["generation_config"] = s.cfg.GenerationConfig
	}
	if s.cfg.PronunciationDictID != "" {
		msg["pronunciation_dict_id"] = s.cfg.PronunciationDictID
	}
	return msg
}

// RunTTSTimed sends the sentence like RunTTS, implementing tts.WordTimestamps.
// The timings arrive on the reader with the audio, so neither callback is used
// here; implementing the interface is what puts the base on the word-aligned
// path.
func (s *timedSynthesizer) RunTTSTimed(
	ctx context.Context,
	text, contextID string,
	_ func(f frames.Frame) error,
	_ func(words []uctx.WordTiming, opts tts.WordTimingOptions) error,
) error {
	return s.RunTTS(ctx, text, contextID, nil)
}

// wsMessage is the subset of a Cartesia WebSocket message we read.
type wsMessage struct {
	Type           string         `json:"type"`
	ContextID      string         `json:"context_id"`
	Data           string         `json:"data"`
	Message        string         `json:"message"`
	WordTimestamps *wsWordTimings `json:"word_timestamps"`
}

// wsWordTimings is the payload of a Cartesia "timestamps" message: parallel
// arrays of spoken words and their start times, in seconds from the start of the
// synthesis.
type wsWordTimings struct {
	Words []string  `json:"words"`
	Start []float64 `json:"start"`
}

// Compile-time checks that the synthesizer answers to everything the base asks
// of a provider whose audio arrives on a receive loop of its own.
var (
	_ tts.ContextSynthesizer      = (*synthesizer)(nil)
	_ tts.AudioFlusher            = (*synthesizer)(nil)
	_ tts.AudioContextInterrupter = (*synthesizer)(nil)
	_ tts.Starter                 = (*synthesizer)(nil)
	_ tts.Closer                  = (*synthesizer)(nil)
	_ tts.Describer               = (*synthesizer)(nil)
	_ tts.WordTimestamps          = (*timedSynthesizer)(nil)
	_ tts.ContextSynthesizer      = (*timedSynthesizer)(nil)
)

// cartesiaLanguages are the languages this provider has been verified against,
// each under the code Cartesia takes for it.
//
//nolint:gochecknoglobals // lookup table, read-only
var cartesiaLanguages = map[language.Language]string{
	language.Arabic:     "ar",
	language.Bulgarian:  "bg",
	language.Bengali:    "bn",
	language.Czech:      "cs",
	language.Danish:     "da",
	language.German:     "de",
	language.English:    "en",
	language.Greek:      "el",
	language.Spanish:    "es",
	language.Finnish:    "fi",
	language.French:     "fr",
	language.Gujarati:   "gu",
	language.Hebrew:     "he",
	language.Hindi:      "hi",
	language.Croatian:   "hr",
	language.Hungarian:  "hu",
	language.Indonesian: "id",
	language.Italian:    "it",
	language.Japanese:   "ja",
	language.Georgian:   "ka",
	language.Kannada:    "kn",
	language.Korean:     "ko",
	language.Malayalam:  "ml",
	language.Marathi:    "mr",
	language.Malay:      "ms",
	language.Dutch:      "nl",
	language.Norwegian:  "no",
	language.Odia:       "or",
	language.Punjabi:    "pa",
	language.Polish:     "pl",
	language.Portuguese: "pt",
	language.Romanian:   "ro",
	language.Russian:    "ru",
	language.Slovak:     "sk",
	language.Swedish:    "sv",
	language.Tamil:      "ta",
	language.Telugu:     "te",
	language.Thai:       "th",
	language.Tagalog:    "tl",
	language.Turkish:    "tr",
	language.Ukrainian:  "uk",
	language.Urdu:       "ur",
	language.Vietnamese: "vi",
	language.Chinese:    "zh",
}

// cartesiaLanguage maps a Language to Cartesia's language code. Cartesia takes
// the base code, so a language this provider was not verified against is still
// sent, under its own base code and with a warning, rather than dropped. The
// zero value is the one that sends nothing, which leaves Cartesia on its own
// default.
func cartesiaLanguage(l language.Language) string {
	if l == "" {
		return ""
	}
	return language.Resolve(l, cartesiaLanguages, true)
}

// cartesiaTagPattern matches the markup this provider accepts in the text it is
// given and reports back in its timings.
//
//nolint:gochecknoglobals // compiled once, read-only
var cartesiaTagPattern = regexp.MustCompile(`(?i)</?(?:spell|emotion|break|volume|speed)\b[^>]*>`)

// stripCartesiaTags removes that markup from one reported token, collapsing the
// whitespace it leaves behind. A token that was nothing but markup comes back
// empty and is dropped by the caller.
//
// A tag standing between two alphanumeric characters becomes a space, since it
// is the only thing separating two words ("to<spell>1234"). Anywhere else it
// comes out entirely: a space there would join nothing, and it would split a
// word from its own punctuation ("<spell>1234</spell>." has to stay "1234.",
// not "1234 .", or the token no longer matches the text that was sent for
// synthesis).
func stripCartesiaTags(text string) string {
	var b strings.Builder
	prev := 0
	for _, m := range cartesiaTagPattern.FindAllStringIndex(text, -1) {
		b.WriteString(text[prev:m[0]])
		if alnumBefore(text, m[0]) && alnumAt(text, m[1]) {
			b.WriteByte(' ')
		}
		prev = m[1]
	}
	b.WriteString(text[prev:])
	return strings.TrimSpace(strings.Join(strings.Fields(b.String()), " "))
}

// alnumBefore reports whether the character ending at i is a letter or a digit.
func alnumBefore(text string, i int) bool {
	if i <= 0 {
		return false
	}
	r, _ := utf8.DecodeLastRuneInString(text[:i])
	return isAlnum(r)
}

// alnumAt reports whether the character starting at i is a letter or a digit.
func alnumAt(text string, i int) bool {
	if i >= len(text) {
		return false
	}
	r, _ := utf8.DecodeRuneInString(text[i:])
	return isAlnum(r)
}

func isAlnum(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) }

// normalizeWordTimings turns one batch of reported timings into the tokens the
// base tracks words with.
//
// The markup this provider was given comes back in the tokens it reports, and
// has to come off before anything downstream tries to match them against the
// text. A language written without spaces between words has its characters
// reported separately but grouped into one message, so the message is joined
// into a single token at the offset of its first character: that is the unit a
// reader of the language recognizes, where the characters alone are not.
func normalizeWordTimings(
	wt *wsWordTimings, spaceless bool,
) ([]uctx.WordTiming, tts.WordTimingOptions) {
	opts := tts.WordTimingOptions{IncludesInterFrameSpaces: spaceless}
	if wt == nil {
		return nil, opts
	}

	if spaceless {
		var combined strings.Builder
		for _, w := range wt.Words {
			combined.WriteString(stripCartesiaTags(w))
		}
		if combined.Len() == 0 || len(wt.Start) == 0 {
			return nil, opts
		}
		return []uctx.WordTiming{{Word: combined.String(), Offset: wt.Start[0]}}, opts
	}

	batch := make([]uctx.WordTiming, 0, len(wt.Words))
	for i, w := range wt.Words {
		cleaned := stripCartesiaTags(w)
		if cleaned == "" {
			continue
		}
		var start float64
		if i < len(wt.Start) {
			start = wt.Start[i]
		}
		batch = append(batch, uctx.WordTiming{Word: cleaned, Offset: start})
	}
	return batch, opts
}
