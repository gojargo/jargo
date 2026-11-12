package cartesia

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/coder/websocket"
	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttscartesia "github.com/gojargo/jargo/adapter/tts/cartesia"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/service/wsservice"
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

// serviceReporter is what the synthesizer needs of the service it runs in: a
// way to report errors, and whether it is still usable.
type serviceReporter interface {
	PushError(ctx context.Context, msg string, err error, fatal bool, opts ...processor.ErrorOption)
	ReportConnectionError(ctx context.Context, ef *frames.ErrorFrame, forceTreatAsPermanent bool)
	Usable() bool
}

// detachedService stands in for the service until NewTTS attaches the real
// one: it logs what would be reported, and stays usable.
type detachedService struct{}

func (detachedService) PushError(_ context.Context, msg string, err error, _ bool, _ ...processor.ErrorOption) {
	slog.Error(msg, "err", err)
}

func (detachedService) ReportConnectionError(_ context.Context, ef *frames.ErrorFrame, _ bool) {
	slog.Error(ef.Error)
}

func (detachedService) Usable() bool { return true }

// errNotConnected is returned by a sentence that found no connection to send
// on, the dial having failed and been reported.
//
//nolint:gochecknoglobals // sentinel error
var errNotConnected = errors.New("cartesia tts: websocket not connected")

// NewTTS builds a Cartesia TTS service.
func NewTTS(cfg Config) *tts.Base {
	cfg = ttsDefaults(cfg)

	s := newSynthesizer(cfg)
	// Always word-aligned: Cartesia reports when each word is spoken, so the
	// base emits each word's text as its audio plays and the conversation keeps
	// only what was heard before an interruption.
	b := tts.New("CartesiaTTS", &timedSynthesizer{synthesizer: s})
	s.svc = b
	// Always hold off on a sentence boundary inside a spell tag. The tag is full
	// of periods that end no sentence, and splitting it would hand Cartesia half
	// a tag. The preferred way to use Cartesia's markup is to have a text
	// transform insert it for the synthesizer alone; this keeps markup that
	// reaches the service by any other route intact.
	b.SetTextAggregator(text.NewSkipTagsAggregator(cfg.TextAggregation, b.TextAggregationLanguage(),
		[]text.StartEndTags{{Start: "<spell>", End: "</spell>"}}))
	s.expressionSettings = b.ExpressionSettings
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
		cfg.TextAggregation = frames.TextTypeSentence
	}
	// Cartesia warns against the middle ground of client-side sentence
	// aggregation plus the server's default 3000ms buffer. With no value chosen,
	// send 0 when sentences are aggregated here and leave it unset when tokens
	// are streamed, so the server's managed buffering applies to the one shape it
	// suits.
	if cfg.MaxBufferDelayMs == nil && cfg.TextAggregation != frames.TextTypeToken {
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
	cfg Config
	// live is what a caller may change while the pipeline runs. It is seeded
	// from cfg, and is what every request is built from.
	live *TTSSettings
	host tts.AudioContextHost
	// expressionSettings is what the service's expression adapter asks to send
	// with the request being synthesized, in an expressive conversation.
	expressionSettings func() map[string]any

	// writeMu serializes writes; the connection permits one writer at a time.
	writeMu sync.Mutex

	// svc is the service this synthesizes for, which reports errors and says
	// whether it is still usable.
	svc serviceReporter
	// ws runs the receive loop and restores a lost connection with backoff.
	ws *wsservice.Base

	// mu guards the connection. It is never held across a call into the host:
	// appending blocks while the audio plays out, and the next sentence has to
	// be able to go out while that happens.
	mu   sync.Mutex
	conn *wsutil.Conn
	// closed records that the read on conn ended: the connection is gone.
	closed bool
	// receiveStop ends the receive loop, and receiveDone is closed once it has
	// ended. Both are nil while no loop runs.
	receiveStop context.CancelFunc
	receiveDone chan struct{}
}

// newSynthesizer builds the synthesizer with its settings seeded, which is the
// only way it should be built: the settings store is not optional.
func newSynthesizer(cfg Config) *synthesizer {
	s := &synthesizer{cfg: cfg, live: newTTSSettings(cfg), svc: detachedService{}}
	s.ws = wsservice.New(s, wsservice.Config{})
	return s
}

// timedSynthesizer adds word-timestamp streaming on top of synthesizer. It
// implements tts.WordTimestamps.
type timedSynthesizer struct {
	*synthesizer
}

// SetAudioContextHost records the host this provider appends its audio to,
// implementing tts.ContextSynthesizer.
func (s *synthesizer) SetAudioContextHost(h tts.AudioContextHost) { s.host = h }

// ExpressionAdapter implements tts.ExpressionAdapterProvider: Cartesia's
// emotions, laughter and pauses.
func (s *synthesizer) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	return ttscartesia.CartesiaExpressionAdapter{}
}

// generationConfig is a request's generation config: the settings', with what
// its expression asks for, or nil when it holds nothing.
func generationConfig(config *GenerationConfig, expression map[string]any) map[string]any {
	values := map[string]any{}
	if config != nil {
		if config.Volume != nil {
			values["volume"] = *config.Volume
		}
		if config.Speed != nil {
			values["speed"] = *config.Speed
		}
		if config.Emotion != "" {
			values["emotion"] = config.Emotion
		}
	}
	maps.Copy(values, expression)
	if len(values) == 0 {
		return nil
	}
	return values
}

// expression is what the expression adapter asks to send with the request.
func expression(settings func() map[string]any) map[string]any {
	if settings == nil {
		return nil
	}
	return settings()
}

// Metadata reports the Cartesia model and voice synthesis is billed against.
func (s *synthesizer) Metadata() tts.Metadata {
	return tts.Metadata{Model: s.live.Model.Or(s.cfg.Model), VoiceID: s.live.Voice.Or(s.cfg.VoiceID)}
}

// SampleRate reports the requested PCM output rate.
func (s *synthesizer) SampleRate() int { return s.cfg.SampleRate }

// spacelessLanguage reports whether the configured language is written without
// spaces between words. Chinese and Japanese are, and this provider reports
// their characters one at a time, with each punctuation mark on its own.
func (s *synthesizer) spacelessLanguage() bool {
	code := s.live.Language.Or(cartesiaLanguage(s.cfg.Language))
	base, _, _ := strings.Cut(code, "-")
	switch strings.ToLower(base) {
	case "zh", "ja":
		return true
	default:
		return false
	}
}

// Start connects when the pipeline starts, so the handshake is paid while the
// transport is still negotiating rather than in front of the first thing the
// bot says. It implements tts.Starter.
func (s *synthesizer) Start(ctx context.Context) {
	s.connect(ctx)
}

// Close disconnects, implementing tts.Closer.
func (s *synthesizer) Close() error {
	s.disconnect(context.Background())
	return nil
}

// connect readies a fresh connection, opens it, and starts the receive loop on
// it. A connection that cannot be opened is reported, and leaves no loop
// running: the next sentence tries again.
func (s *synthesizer) connect(ctx context.Context) {
	s.ws.Connect()
	_ = s.ConnectWebsocket(ctx)
	s.mu.Lock()
	if s.conn == nil || s.receiveStop != nil {
		s.mu.Unlock()
		return
	}
	// The loop outlives the call that connected: it runs until the service
	// disconnects, not until this sentence is done.
	runCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	s.receiveStop, s.receiveDone = stop, done
	s.mu.Unlock()
	go func() {
		defer close(done)
		s.ws.ReceiveTaskHandler(runCtx, s.reportError)
		s.mu.Lock()
		if s.receiveDone == done {
			s.receiveStop, s.receiveDone = nil, nil
		}
		s.mu.Unlock()
	}()
}

// disconnect marks the disconnect deliberate, stops the receive loop, and
// closes the connection.
func (s *synthesizer) disconnect(ctx context.Context) {
	s.ws.Disconnect()
	s.mu.Lock()
	stop, done := s.receiveStop, s.receiveDone
	s.receiveStop, s.receiveDone = nil, nil
	s.mu.Unlock()
	if stop != nil {
		stop()
		select {
		case <-done:
		case <-time.After(receiveStopTimeout):
			slog.Debug("cartesia tts receive loop did not stop in time")
		}
	}
	s.closeWebsocket(ctx)
}

// ConnectWebsocket opens the connection, implementing wsservice.Handler. A
// failure is reported here and leaves no connection behind, which the receive
// loop reads as a failed attempt.
func (s *synthesizer) ConnectWebsocket(ctx context.Context) error {
	if s.Connected() {
		return nil
	}
	header := http.Header{}
	header.Set("X-API-Key", s.cfg.APIKey)
	header.Set("Cartesia-Version", s.cfg.Version)
	// Applied last, so a deployment can route or authorize the handshake its own
	// way, overriding what the service sets.
	for k, v := range s.cfg.Headers {
		header.Set(k, v)
	}
	conn, err := wsutil.Dial(ctx, s.cfg.URL, header, readLimit)
	if err != nil {
		ef := frames.NewErrorFrame(fmt.Sprintf("Unknown error occurred: %v", err))
		ef.Err = err
		s.svc.ReportConnectionError(ctx, ef, false)
		return nil
	}
	s.mu.Lock()
	s.conn, s.closed = conn, false
	s.mu.Unlock()
	return nil
}

// DisconnectWebsocket readies a redial, implementing wsservice.Handler. The
// contexts open on the connection being replaced never get the rest of their
// audio, and the new one has never seen them, so they are let go first.
func (s *synthesizer) DisconnectWebsocket(ctx context.Context) error {
	if s.host != nil {
		s.host.ReplaceConnection()
	}
	s.closeWebsocket(ctx)
	return nil
}

// closeWebsocket closes the connection, if there is one still open.
func (s *synthesizer) closeWebsocket(ctx context.Context) {
	s.mu.Lock()
	conn, open := s.conn, s.conn != nil && !s.closed
	s.conn, s.closed = nil, false
	s.mu.Unlock()
	if !open {
		return
	}
	if err := conn.Close(websocket.StatusNormalClosure, ""); err != nil {
		s.svc.PushError(ctx, fmt.Sprintf("Unknown error occurred: %v", err), err, false)
	}
}

// Connected reports whether there is a live connection, implementing
// wsservice.Handler.
func (s *synthesizer) Connected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.conn != nil && !s.closed
}

// Usable reports whether the service can still do its job, implementing
// wsservice.Handler.
func (s *synthesizer) Usable() bool { return s.svc.Usable() }

// reportError reports a connection error from the receive loop.
func (s *synthesizer) reportError(ctx context.Context, ef *frames.ErrorFrame, forceTreatAsPermanent bool) {
	s.svc.ReportConnectionError(ctx, ef, forceTreatAsPermanent)
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

// ReceiveMessages reads and dispatches messages until the connection fails,
// implementing wsservice.Handler.
//
// Cartesia closes a connection that has carried nothing for five minutes, and it
// offers no keepalive, so a connection the server closes gracefully is usually
// that rather than a failure, and is dialed again at once. That keeps the socket
// warm: the next turn pays no handshake in front of the first thing the bot
// says. A read that fails goes back to the receive loop, which restores the
// connection with backoff.
func (s *synthesizer) ReceiveMessages(ctx context.Context) error {
	for {
		s.mu.Lock()
		conn := s.conn
		s.mu.Unlock()
		if conn == nil {
			return wsservice.ErrNotConnected
		}
		err := s.readMessages(ctx, conn)
		s.mu.Lock()
		if s.conn == conn {
			s.closed = true
		}
		s.mu.Unlock()
		if ctx.Err() != nil || !gracefulClose(err) {
			return err
		}
		slog.Debug("cartesia tts connection was disconnected (timeout?), reconnecting")
		if s.host != nil {
			s.host.ReplaceConnection()
		}
		_ = s.ConnectWebsocket(ctx)
	}
}

// gracefulClose reports whether err is the server closing the connection the
// way it does after an idle timeout.
func gracefulClose(err error) bool {
	switch websocket.CloseStatus(err) {
	case websocket.StatusNormalClosure, websocket.StatusGoingAway:
		return true
	default:
		return false
	}
}

// readMessages reads until the connection ends, dispatching each message, and
// returns what ended it.
func (s *synthesizer) readMessages(ctx context.Context, conn *wsutil.Conn) error {
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return err
		}
		var m wsMessage
		if json.Unmarshal(data, &m) != nil {
			continue
		}
		m.raw = data
		s.dispatch(ctx, &m)
	}
}

// dispatch delivers one message to the context it names.
//
// A message for a context that is no longer open is skipped: an interruption
// abandons a turn mid-flight, and audio the server had already generated for it
// arrives afterwards. Attributing by context id keeps that audio out of the next
// turn, which is what the ids are for.
func (s *synthesizer) dispatch(ctx context.Context, m *wsMessage) {
	host := s.host
	if host == nil || m.ContextID == "" || !host.AudioContextAvailable(m.ContextID) {
		return
	}

	switch m.Type {
	case msgDone:
		s.finishContext(m.ContextID, host)
	case msgTimestamps:
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
		// The turn's speech stops now rather than behind audio that is not
		// coming, and the failure is reported through the service with the
		// server's message as it was sent. There are no metrics to stop: they are
		// taken per RunTTS call, which has returned by the time the server
		// answers.
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = m.ContextID
		_ = host.PushFrame(ctx, stopped, processor.Downstream)
		host.PushError(ctx, "Error: "+string(m.raw), nil, false)
		host.ResetActiveAudioContext()
	case msgFlushDone:
		// A boundary marker within a context, which Cartesia emits per submission
		// when its own buffering is disabled. Each turn already has a context of
		// its own and every chunk carries its id, so there is nothing to do.
	default:
		// Reported rather than only logged: a message type the service does not
		// know is a change on the server's side the application should hear of.
		host.PushError(ctx, "Error, unknown message type: "+string(m.raw), nil, false)
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
	if !s.Connected() {
		s.connect(ctx)
	}
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return errNotConnected
	}

	// continue says more of this turn may follow, which is what lets the
	// sentences of one turn stream as a single utterance. The flush at the end of
	// the turn is what closes it.
	err := s.write(ctx, conn, s.request(text, contextID, true))
	if err == nil {
		tts.StartUsageMetrics(ctx, text)
		return nil
	}

	// The server's view of the connection is unknown from here, so it is dropped
	// and dialed again. The context is closed out rather than left waiting on
	// audio that is not coming.
	if host := s.host; host != nil && host.AudioContextAvailable(contextID) {
		s.finishContext(contextID, host)
	}
	s.disconnect(ctx)
	s.connect(ctx)
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
		"model_id":      s.live.Model.Or(s.cfg.Model),
		"voice":         map[string]any{"mode": "id", "id": s.live.Voice.Or(s.cfg.VoiceID)},
		"output_format": map[string]any{
			"container":   s.cfg.Container,
			"encoding":    s.cfg.Encoding,
			"sample_rate": s.cfg.SampleRate,
		},
	}
	msg["add_timestamps"] = true
	// The timings are matched against the text as it was written, so the
	// normalized spelling of a number or an abbreviation would not match.
	msg["use_normalized_timestamps"] = false
	if s.cfg.MaxBufferDelayMs != nil {
		msg["max_buffer_delay_ms"] = *s.cfg.MaxBufferDelayMs
	}
	if lang := s.live.Language.Or(cartesiaLanguage(s.cfg.Language)); lang != "" {
		msg["language"] = lang
	}
	if gen := generationConfig(
		s.live.GenerationConfig.Or(s.cfg.GenerationConfig), expression(s.expressionSettings)); gen != nil {
		msg["generation_config"] = gen
	}
	if dict := s.live.PronunciationDictID.Or(s.cfg.PronunciationDictID); dict != "" {
		msg["pronunciation_dict_id"] = dict
	}
	if accent := s.live.Accent.Or(s.cfg.Accent); accent != "" {
		msg["accent"] = accent
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
	// raw is the message as it arrived, which an error is reported with.
	raw []byte
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

// cartesiaLanguages are the languages Cartesia supports, each under the code
// it takes for it. A regional locale Cartesia supports keeps its region:
// Mexican Spanish goes out as "es-MX", not as Spanish. Norwegian Bokmal is
// Cartesia's "no", and Filipino its "tl".
//
//nolint:gochecknoglobals // lookup table, read-only
var cartesiaLanguages = map[language.Language]string{
	language.Arabic:            "ar",
	language.Bulgarian:         "bg",
	language.BulgarianBG:       "bg-BG",
	language.Bengali:           "bn",
	language.BengaliIN:         "bn-IN",
	language.Czech:             "cs",
	language.CzechCZ:           "cs-CZ",
	language.Danish:            "da",
	language.DanishDK:          "da-DK",
	language.German:            "de",
	language.GermanAT:          "de-AT",
	language.GermanDE:          "de-DE",
	language.Greek:             "el",
	language.GreekGR:           "el-GR",
	language.English:           "en",
	language.EnglishAU:         "en-AU",
	language.EnglishCA:         "en-CA",
	language.EnglishGB:         "en-GB",
	language.EnglishIE:         "en-IE",
	language.EnglishIN:         "en-IN",
	language.EnglishNZ:         "en-NZ",
	language.EnglishSG:         "en-SG",
	language.EnglishUS:         "en-US",
	language.EnglishZA:         "en-ZA",
	language.Spanish:           "es",
	language.SpanishAR:         "es-AR",
	language.SpanishCL:         "es-CL",
	language.SpanishCO:         "es-CO",
	language.SpanishDO:         "es-DO",
	language.SpanishES:         "es-ES",
	language.SpanishMX:         "es-MX",
	language.SpanishPA:         "es-PA",
	language.SpanishPE:         "es-PE",
	language.SpanishPR:         "es-PR",
	language.SpanishUS:         "es-US",
	language.Finnish:           "fi",
	language.FinnishFI:         "fi-FI",
	language.French:            "fr",
	language.FrenchBE:          "fr-BE",
	language.FrenchCA:          "fr-CA",
	language.FrenchFR:          "fr-FR",
	language.Gujarati:          "gu",
	language.GujaratiIN:        "gu-IN",
	language.Hebrew:            "he",
	language.HebrewIL:          "he-IL",
	language.Hindi:             "hi",
	language.HindiIN:           "hi-IN",
	language.Croatian:          "hr",
	language.CroatianHR:        "hr-HR",
	language.Hungarian:         "hu",
	language.HungarianHU:       "hu-HU",
	language.Indonesian:        "id",
	language.IndonesianID:      "id-ID",
	language.Italian:           "it",
	language.ItalianIT:         "it-IT",
	language.Japanese:          "ja",
	language.JapaneseJP:        "ja-JP",
	language.Georgian:          "ka",
	language.GeorgianGE:        "ka-GE",
	language.Kannada:           "kn",
	language.KannadaIN:         "kn-IN",
	language.Korean:            "ko",
	language.KoreanKR:          "ko-KR",
	language.Malayalam:         "ml",
	language.MalayalamIN:       "ml-IN",
	language.Marathi:           "mr",
	language.MarathiIN:         "mr-IN",
	language.Malay:             "ms",
	language.MalayMY:           "ms-MY",
	language.Dutch:             "nl",
	language.DutchBE:           "nl-BE",
	language.DutchNL:           "nl-NL",
	language.NorwegianBokmal:   "no",
	language.Norwegian:         "no",
	language.NorwegianBokmalNO: "no-NO",
	language.Odia:              "or",
	language.OdiaIN:            "or-IN",
	language.Punjabi:           "pa",
	language.PunjabiIN:         "pa-IN",
	language.Polish:            "pl",
	language.PolishPL:          "pl-PL",
	language.Portuguese:        "pt",
	language.PortugueseBR:      "pt-BR",
	language.PortuguesePT:      "pt-PT",
	language.Romanian:          "ro",
	language.RomanianRO:        "ro-RO",
	language.Russian:           "ru",
	language.RussianRU:         "ru-RU",
	language.Slovak:            "sk",
	language.SlovakSK:          "sk-SK",
	language.Swedish:           "sv",
	language.SwedishSE:         "sv-SE",
	language.Tamil:             "ta",
	language.TamilIN:           "ta-IN",
	language.Telugu:            "te",
	language.TeluguIN:          "te-IN",
	language.Thai:              "th",
	language.ThaiTH:            "th-TH",
	language.Filipino:          "tl",
	language.Tagalog:           "tl",
	language.FilipinoPH:        "tl-PH",
	language.Turkish:           "tr",
	language.TurkishTR:         "tr-TR",
	language.Ukrainian:         "uk",
	language.UkrainianUA:       "uk-UA",
	language.Urdu:              "ur",
	language.UrduIN:            "ur-IN",
	language.Vietnamese:        "vi",
	language.VietnameseVN:      "vi-VN",
	language.Chinese:           "zh",
	language.ChineseCN:         "zh-CN",
}

// cartesiaLanguage maps a Language to Cartesia's language code. A locale
// Cartesia does not list is still sent, under its base code and with a
// warning, rather than dropped. The zero value is the one that sends nothing,
// which leaves Cartesia on its own default.
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

// normalizeWordTimings turns one timestamps message into the tokens to report:
// each entry keeps its own start, with Cartesia's markup taken off and an
// entry left empty by that dropped, so one entry in is at most one token out,
// whatever the language.
//
// Cartesia reports Chinese and Japanese one character at a time, with each
// punctuation mark as a token of its own. For those the base is asked to merge
// a mark into the character before it ("す", "。" read "す。"), and told the
// tokens carry their own spacing.
func normalizeWordTimings(
	wt *wsWordTimings, spaceless bool,
) ([]uctx.WordTiming, tts.WordTimingOptions) {
	opts := tts.WordTimingOptions{IncludesInterFrameSpaces: spaceless, PreMergeTokens: spaceless}
	if wt == nil {
		return nil, opts
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
