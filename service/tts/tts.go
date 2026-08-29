// Package tts is the shared base for text-to-speech services. The base
// aggregates incoming text into sentences and sends each one on the turn's audio
// context, a queue that holds a turn's audio in the order it was generated and
// is drained downstream by a goroutine of its own. Bracketing a turn with
// TTSStarted/TTSStopped, sentence aggregation, and the HTTP response streaming
// helper live here; the context machinery is in audiocontext.go.
//
// A provider implements Synthesize and has its audio appended for it, or
// implements ContextSynthesizer and appends the audio itself as its receive loop
// reads it. Either way the audio reaches the pipeline through a context, so one
// sentence never waits on the one before it.
package tts

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/onset"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/telemetry/metrics"
	uctx "github.com/gojargo/jargo/utils/context"
	errs "github.com/gojargo/jargo/utils/errors"
	ttstext "github.com/gojargo/jargo/utils/text"
	"github.com/google/uuid"
)

// errStatus is returned when a provider responds with a non-200 status.
//
//nolint:gochecknoglobals // sentinel error
var errStatus = errors.New("tts: unexpected status")

// readChunk is the size of each audio read from an HTTP stream.
const readChunk = 4096

// ttfaMaxBufferSeconds bounds how much leading audio is buffered while scanning
// for a speech onset before giving up on the time-to-first-audio measurement.
const ttfaMaxBufferSeconds = 3

// Synthesizer turns text into speech. SampleRate reports the PCM rate of the
// audio it produces.
//
// RunTTS synthesizes one unit of text and hands each frame it produces to yield,
// which routes them to the synthesis context they belong to. Most providers
// yield audio and nothing else, and can build those frames with PCMYielder;
// yielding is the contract so a provider that has more to report, such as an
// error mid-stream, can say so in the same ordered stream as its audio.
//
// contextID names the synthesis the text belongs to. A provider that keeps a
// server-side stream open across a turn needs it; one that answers each call on
// its own can ignore it.
type Synthesizer interface {
	SampleRate() int
	RunTTS(ctx context.Context, text, contextID string, yield func(f frames.Frame) error) error
}

// PCMYielder adapts a frame yield to a callback taking raw 16-bit mono PCM, for
// a provider whose output is audio and nothing else.
func PCMYielder(yield func(f frames.Frame) error, rate int) func(pcm []byte) error {
	return func(pcm []byte) error {
		if len(pcm) == 0 {
			return nil
		}
		return yield(frames.NewTTSAudioRawFrame(pcm, rate, 1))
	}
}

// WordTimestamps is an optional interface a Synthesizer may implement to report
// per-word timing aligned to the audio it produces. When the provider driving a
// Base implements it, the Base tracks word completion and emits playback-aligned
// TTSTextFrames that map each spoken word back to its original written form — so
// the assistant context records only what was actually spoken before an
// interruption, in its written form. A Synthesizer that does not implement this
// interface is unaffected: the Base pushes text and synthesizes audio exactly as
// before, and no TTSTextFrames are produced.
type WordTimestamps interface {
	Synthesizer
	// RunTTSTimed synthesizes like RunTTS and additionally reports word timing
	// via word: each call carries a batch of tokens in spoken order, with the
	// offset of each in seconds from the beginning of this synthesis. A batch
	// rather than one token at a time, because normalizing a token stream can
	// need the token before it, which the base cannot see one call at a time.
	RunTTSTimed(
		ctx context.Context,
		text, contextID string,
		yield func(f frames.Frame) error,
		word func(words []uctx.WordTiming, opts WordTimingOptions) error,
	) error
}

// WordTimingOptions says how a batch of word timings should be treated, for the
// shapes a provider's token stream can take.
type WordTimingOptions struct {
	// IncludesInterFrameSpaces marks each token as carrying whatever spacing
	// separates it from the ones around it, so a consumer joining them adds none
	// of its own. Set it for a language written without spaces between words,
	// Chinese and Japanese, where a provider reports tokens that already read as
	// continuous text and inserting spaces between them is wrong. Off by default:
	// a stream of words in a spaced language needs the separator supplied.
	//
	// Leave it off when PreMergeTokens is set. Merging produces clean word
	// strings, which is the spaced shape.
	IncludesInterFrameSpaces bool
	// PreMergeTokens merges punctuation- and space-only tokens into the word
	// before them, for a provider that reports those as tokens of their own
	// rather than attached to the word they belong to. Off by default: a
	// provider whose stream is already word-level needs no merging, and merging
	// it anyway would join tokens that were meant to stand apart.
	PreMergeTokens bool
}

// Metadata describes a TTS service to the observability layer.
type Metadata struct {
	// Model is the provider's model identifier, e.g. "eleven_flash_v2_5". It
	// labels the metrics and is what a cost-tracking backend prices the
	// synthesis against, so it should be the identifier the provider bills
	// under.
	Model string
	// VoiceID is the provider's voice identifier, or "" when the service has no
	// notion of a voice separate from the model.
	VoiceID string
}

// Describer is an optional interface a Synthesizer implements to describe the
// model it synthesizes with. A Synthesizer that does not implement it still
// reports character usage on its spans; only the model label is missing, and
// with it the ability to price the synthesis.
type Describer interface {
	Metadata() Metadata
}

// SettingsHolder is an optional interface a Synthesizer implements when part of
// what it was built with can change while the pipeline runs: the voice it speaks
// in, the language, the model. The value returned is the provider's own store, a
// pointer to a settings value, which an update is merged into.
type SettingsHolder interface {
	Settings() any
}

// SettingsUpdater is an optional interface a Synthesizer implements to act on a
// settings change, with what changed and what each field held before. A provider
// whose new settings only take effect on a fresh connection reconnects here,
// which it can do because it owns its connection. A Synthesizer that holds
// settings without implementing this still has them updated; it picks them up
// the next time it synthesizes.
type SettingsUpdater interface {
	SettingsHolder
	UpdateSettings(ctx context.Context, changed settings.Changed) error
}

// LanguageNamer is an optional interface a Synthesizer implements to name a
// language the way its provider does. Providers disagree on the codes, so a
// caller naming a language neutrally has it converted before it is stored,
// leaving the store holding the code the provider itself uses. Without that the
// stored value and the next update would be in different vocabularies, and a
// change would be reported for two spellings of the same language.
type LanguageNamer interface {
	ServiceLanguage(l language.Language) string
}

// Closer is an optional interface a Synthesizer implements when it holds a
// resource open across syntheses, such as a connection it reuses rather than
// redialing per sentence. The Base closes it when the pipeline tears down. A
// Synthesizer that does not implement it is unaffected.
type Closer interface {
	Close() error
}

// Starter is an optional interface a Synthesizer implements when it has setup to
// do before the first synthesis, such as dialing the connection it will reuse.
// The Base calls it when the pipeline starts — the counterpart to Closer at
// teardown — so the cost lands while the transport is still negotiating rather
// than in front of the first thing the bot says. It runs off the frame goroutine.
// A Synthesizer that does not implement it is unaffected.
type Starter interface {
	Start(ctx context.Context)
}

// Base is the shared TTS processor. It aggregates text into sentences,
// synthesizes each one, and routes the audio through an audio context so it
// reaches the pipeline in the order it was generated. See audiocontext.go.
type Base struct {
	*service.Base
	syn     Synthesizer
	meta    Metadata
	filters []ttstext.Filter

	// aggregator groups streamed text into the units the provider is given.
	aggregator ttstext.Aggregator
	// sequencer keeps the frames of a synthesis in the order the text was
	// spoken, so the conversation records it that way.
	sequencer *uctx.AggregatedFrameSequencer
	// aggregatorErr is the failure to build the default aggregator, reported
	// when the pipeline starts rather than swallowed.
	aggregatorErr error

	// Audio contexts. serial orders playback across contexts; audioContexts
	// holds each open context's queue. See audiocontext.go.
	audioCtxMu    sync.Mutex
	audioContexts map[string]*audioContext
	// ttsContexts records what each context carries beyond its audio. It is read
	// on the drain goroutine, long after the call that set it, which is why the
	// answers are kept here rather than passed along with each frame.
	ttsContexts map[string]ttsContext
	// pendingResponseEnd holds the frame ending a model response, keyed by the
	// context its speech was sent on, until that speech has been heard. Several
	// contexts can be in flight at once, so each is held under its own key and
	// released by its own context finishing.
	pendingResponseEnd map[string]*frames.LLMFullResponseEndFrame
	serial             *serialQueue
	ctxCancel          context.CancelFunc
	// ctxDone is closed when the drain goroutine exits, so a graceful end can
	// wait for the queue it asked to shut down.
	ctxDone chan struct{}
	ctxWG   sync.WaitGroup

	// yieldsSync records whether the provider answered the last call with its
	// audio, which is what says who closes the context out.
	yieldsSync bool

	// wordLastPTS is when the last word pushed downstream was spoken, so a
	// context starting while another is still playing times its words after it.
	wordLastPTS time.Duration

	// turnContext is the context every sentence of the current turn is sent on,
	// empty between turns. Reusing it is what keeps the provider from opening a
	// cold context per sentence.
	turnContext string

	// llmResponseStarted reports whether a model response is driving the speech.
	// It is what tells an utterance the service says on its own from one the
	// model composed: the first has no response around it to close the assistant
	// turn, so the service closes it itself.
	llmResponseStarted bool

	// Pausing frame handling while a turn's audio is generated. See pause.go.
	pauseStateMu sync.Mutex
	// pauseOpts is how the service was configured; pausing is off by default.
	pauseOpts PauseOptions

	// outMu guards how what leaves the service is shaped: the silence padding
	// each utterance and the transport stream the audio is addressed to.
	outMu       sync.Mutex
	silence     SilenceOptions
	destination string
	// processingText records that text for this turn reached the provider, so
	// there is audio worth waiting for.
	processingText bool
	// botSpeaking records that the audio is confirmed playing, which is one of
	// the two things that say a pause has something to wait for.
	botSpeaking bool

	// Contexts that complete with no audio at all. See silentcontext.go.
	silentMu sync.Mutex
	// zeroAudioLimit is how many such contexts in a row write the service off;
	// 0 reports them without ever doing so.
	zeroAudioLimit int
	// zeroAudioRun is how many have completed in a row.
	zeroAudioRun int

	// textAgg times how long grouping text into sentences takes. See
	// textaggregation.go.
	textAgg textAggregationMetrics

	// metaMu guards meta, which the model labeling the metrics is read from
	// and which a settings update can change while a turn is being measured.
	metaMu sync.Mutex
}

// model is the identifier the synthesis is measured and priced against.
func (b *Base) model() string {
	b.metaMu.Lock()
	defer b.metaMu.Unlock()
	return b.meta.Model
}

// New builds a TTS Base named name driven by syn. The concrete service passes
// itself as syn and embeds the returned Base.
func New(name string, syn Synthesizer) *Base {
	b := &Base{syn: syn, zeroAudioLimit: defaultZeroAudioContextLimit}
	if tok, err := ttstext.NewPunktEnglish(); err != nil {
		b.aggregatorErr = err
	} else {
		b.aggregator = ttstext.NewSimpleAggregator(frames.AggregationSentence, tok)
		b.sequencer = uctx.NewAggregatedFrameSequencer(name, false, tok)
	}
	if d, ok := syn.(Describer); ok {
		b.meta = d.Metadata()
	}
	b.Base = service.New(name, b)
	if cs, ok := syn.(ContextSynthesizer); ok {
		cs.SetAudioContextHost(b)
	}
	return b
}

// Cleanup releases the Synthesizer's resources, when it holds any, and tears
// down the processor.
func (b *Base) Cleanup(ctx context.Context) error {
	b.stopAudioContexts()
	if c, ok := b.syn.(Closer); ok {
		_ = c.Close()
	}
	return b.Base.Cleanup(ctx)
}

// SetTextAggregator sets how streamed text is grouped into the units handed to
// the provider, replacing the default (English sentences). Pass an aggregator
// built over the language the bot speaks, or one that aggregates by token to
// stream text through as it arrives. Call this before the pipeline starts.
func (b *Base) SetTextAggregator(a ttstext.Aggregator) {
	b.aggregator = a
	b.aggregatorErr = nil
}

// SetTextFilters sets the text-normalization filters applied to each sentence
// just before synthesis — for example a text.VoiceFormatter that strips
// Markdown and spells out numbers, currency and dates. Filters run in order.
// Call this before the pipeline starts; the filter set is not safe to change
// while it is running.
func (b *Base) SetTextFilters(filters ...ttstext.Filter) {
	b.filters = filters
}

// Setup opens the synthesizer's connection, for a synthesizer that dials one.
//
// Dialing here rather than on the StartFrame is what keeps it off the frame
// path: a pipeline sets its processors up at once, so several services connect
// together instead of one after another as the frame reaches them, and a service
// that cannot connect is reported and left unusable before the pipeline starts,
// in time for a switcher to move off it.
func (b *Base) Setup(ctx context.Context, s processor.Setup) error {
	if err := b.Base.Setup(ctx, s); err != nil {
		return err
	}
	if st, ok := b.syn.(Starter); ok {
		// The connection outlives this call, so it is not tied to a context that
		// ends with it.
		st.Start(context.WithoutCancel(ctx))
	}
	return nil
}

// ProcessFrame aggregates text into sentences and synthesizes them.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	// Text the LLM service stamped to skip synthesis passes straight through: it
	// belongs in the conversation, but it is not to be spoken. The response
	// brackets are stamped with it too, so the turn they describe is not opened
	// and closed here for speech that never happens.
	if skipsTTS(f) {
		return b.PushFrame(ctx, f, dir)
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		return b.handleText(ctx, &fr.TextFrame, f, dir)
	case *frames.TextFrame:
		return b.handleText(ctx, fr, f, dir)
	case *frames.TTSSpeakFrame:
		return b.handleSpeak(ctx, fr, dir)
	case *frames.LLMFullResponseStartFrame:
		b.setLLMResponding(true)
		// A new turn gets one context id, shared by all of its sentences.
		b.turnContext = b.createContextID()
		if c, ok := b.syn.(TurnContextCreator); ok {
			c.OnTurnContextCreated(ctx, b.turnContext)
		}
		// Through the serialization queue, so this frame is emitted only after any
		// context already queued has drained rather than racing ahead of it.
		return b.queueSerial(ctx, f, dir)
	case *frames.LLMFullResponseEndFrame, *frames.EndFrame:
		return b.handleTurnEnd(ctx, f, dir)
	case *frames.InterruptionFrame:
		// An interruption stops every clock this service is running, this one
		// included: what it was measuring will never finish.
		b.stopTextAggregationMetrics(ctx)
		b.handleInterruption(ctx)
		return b.PushFrame(ctx, f, dir)
	case *frames.CancelFrame:
		b.handleInterruption(ctx)
		b.stopAudioContexts()
		return b.PushFrame(ctx, f, dir)
	case *frames.StartFrame:
		return b.handleStart(ctx, f, dir)
	case *frames.TTSUpdateSettingsFrame:
		return b.handleSettingsUpdate(ctx, fr, dir)
	case *frames.BotStartedSpeakingFrame, *frames.BotStoppedSpeakingFrame:
		return b.handleBotSpeaking(ctx, f, dir)
	default:
		return b.queueSerial(ctx, f, dir)
	}
}

// handleSettingsUpdate applies an update addressed to this service, and leaves
// one meant for another service untouched for that one.
func (b *Base) handleSettingsUpdate(
	ctx context.Context, f *frames.TTSUpdateSettingsFrame, dir processor.Direction,
) error {
	if !f.TargetsService(b) {
		return b.PushFrame(ctx, f, dir)
	}
	b.updateSettings(ctx, f)
	return nil
}

// skipsTTS reports whether the frame was stamped to bypass synthesis. Only the
// frames describing the model's output carry the stamp, and an unstamped frame
// is not one to skip.
func skipsTTS(f frames.Frame) bool {
	var skip *bool
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		skip = fr.SkipTTS
	case *frames.TextFrame:
		skip = fr.SkipTTS
	case *frames.LLMFullResponseStartFrame:
		skip = fr.SkipTTS
	case *frames.LLMFullResponseEndFrame:
		skip = fr.SkipTTS
	}
	return skip != nil && *skip
}

// updateSettings merges an update into the provider's own settings and lets it
// act on what changed. A provider whose new settings only take effect on a fresh
// connection reconnects for itself: unlike a transcription session, which the
// base owns, a synthesizer owns its own connection.
func (b *Base) updateSettings(ctx context.Context, f *frames.TTSUpdateSettingsFrame) {
	holder, ok := b.syn.(SettingsHolder)
	if !ok {
		slog.Warn("settings update for a service whose provider has none", "service", b.Name())
		return
	}
	store := holder.Settings()

	delta, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, store)
	if err != nil {
		b.PushError(ctx, "tts: settings update", err, false)
		return
	}
	if !ok {
		return
	}

	// Naming the language the provider's way before applying is what keeps the
	// comparison honest: the store holds the provider's code, so a neutral name
	// meaning the same language must be converted first or it reads as a change
	// when nothing changed.
	b.nameLanguage(delta)

	changed, err := settings.Apply(store, delta)
	if err != nil {
		b.PushError(ctx, "tts: settings update", err, false)
		return
	}
	if len(changed) == 0 {
		return
	}
	slog.Info("updated settings", "service", b.Name(), "fields", changed.String())
	b.SettingsUpdated(ctx)

	if changed.Has("model") {
		// The model labels the synthesis this service reports and is what the
		// characters are priced against, so a model that changed mid-call has to
		// relabel what follows.
		name, _ := settings.Get(store, "model")
		model, _ := name.(string)
		b.metaMu.Lock()
		b.meta.Model = model
		b.metaMu.Unlock()
	}

	if updater, ok := b.syn.(SettingsUpdater); ok {
		if err := updater.UpdateSettings(ctx, changed); err != nil {
			b.PushError(ctx, "tts: settings update", err, false)
		}
	}
}

// nameLanguage rewrites a language the delta gives into the code the provider
// uses, when the provider says how. A code the provider does not recognize is
// left as it came, since it may be one the service accepts directly.
func (b *Base) nameLanguage(delta any) {
	namer, ok := b.syn.(LanguageNamer)
	if !ok {
		return
	}
	value, ok := settings.Get(delta, "language")
	if !ok {
		return
	}
	code, ok := value.(string)
	if !ok || code == "" {
		return
	}
	named := namer.ServiceLanguage(language.Language(code))
	if named == "" || named == code {
		return
	}
	if err := settings.SetNamed(delta, "language", named); err != nil {
		slog.Warn("naming the language the provider's way failed", "service", b.Name(), "err", err)
	}
}

// handleTurnEnd closes a turn out: an LLMFullResponseEndFrame ends the model's
// response, an EndFrame ends the pipeline. Both flush whatever text did not land
// on a sentence boundary and close the context it was sent on.
func (b *Base) handleTurnEnd(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	_, isEnd := f.(*frames.EndFrame)
	if isEnd {
		// The pipeline is ending. The serialization queue is shut down and
		// waited on first, so the audio a provider is still delivering, and
		// anything queued behind it, reaches the output before the frame that
		// stops the pipeline rather than being cut off by it.
		b.drainAudioContexts()
	}
	if err := b.flush(ctx); err != nil {
		return err
	}
	// Pause before the flag is cleared: a turn that sent no text, a function
	// call and nothing else, has no audio to wait for.
	b.maybePauseFrameProcessing()
	b.setProcessingText(false)
	// Taken before the turn is closed out, which is what forgets the id.
	turnContext := b.turnContext
	b.onTurnContextCompleted(ctx)
	if isEnd {
		return b.PushFrame(ctx, f, dir)
	}
	// A provider that times its words has its response end held back until the
	// audio for it has been heard, and stamped with the moment of the last word,
	// so it cannot overtake the words it is ending. One that does not time them
	// pushes text a whole unit at a time on the serialization queue, and the end
	// goes out behind that.
	if end, ok := f.(*frames.LLMFullResponseEndFrame); ok && b.wordPath() {
		if turnContext != "" {
			b.holdResponseEnd(turnContext, end)
		}
		return nil
	}
	return b.queueSerial(ctx, f, dir)
}

// handleBotSpeaking tracks whether the bot's audio is playing, which is what
// releases a service paused waiting for it.
func (b *Base) handleBotSpeaking(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if _, started := f.(*frames.BotStartedSpeakingFrame); started {
		b.setBotSpeaking(true)
		return b.queueSerial(ctx, f, dir)
	}
	b.setBotSpeaking(false)
	b.maybeResumeFrameProcessing()
	return b.queueSerial(ctx, f, dir)
}

// queueSerial routes a downstream frame through the serialization queue so it is
// emitted in the order it arrived relative to the contexts already queued. A
// system frame, or anything going upstream, overtakes the queue by design.
func (b *Base) queueSerial(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if _, isSystem := f.(frames.SystemFrame); isSystem || dir != processor.Downstream {
		return b.PushFrame(ctx, f, dir)
	}
	b.audioCtxMu.Lock()
	serial := b.serial
	b.audioCtxMu.Unlock()
	if serial == nil {
		return b.PushFrame(ctx, f, dir)
	}
	serial.push(serialItem{frame: f})
	return nil
}

// createContextID returns the turn's context id, so every sentence of a turn is
// synthesized on one context, or a fresh one outside a turn.
func (b *Base) createContextID() string {
	if b.turnContext != "" {
		// Keep the open context from timing out while the next sentence is still
		// being generated.
		b.refreshAudioContext(b.turnContext)
		return b.turnContext
	}
	return uuid.NewString()
}

// onTurnContextCompleted closes the turn out once its text has all been sent.
// The audio for it may still be arriving: for a provider that streams on its own
// receive loop the context is removed when the provider marks it final, and for
// one that returns its audio inline the base closes it here.
func (b *Base) onTurnContextCompleted(ctx context.Context) {
	id := b.turnContext
	b.turnContext = ""
	if id == "" {
		return
	}
	if ctx.Err() != nil {
		// Interrupted while the turn's last sentence was still being synthesized.
		// The interruption tears the context down, and the words it was holding
		// back are meant to stay unspoken.
		//
		// The check is here because a goroutine cannot be stopped from outside:
		// canceling only cancels the context, so the call that was in flight
		// runs to its end and has to notice for itself. Closing the turn out here
		// would speak what the interruption was meant to cut off.
		return
	}
	if b.yieldingSync() && b.AudioContextAvailable(id) {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = id
		b.AppendToAudioContext(id, stopped)
		b.RemoveAudioContext(id)
	}
	// Anything the provider is still holding is flushed before the context is
	// closed, so a provider that waits to be told generates what it has.
	if f, ok := b.syn.(AudioFlusher); ok && b.AudioContextAvailable(id) {
		f.FlushAudio(ctx, id)
	}
	if c, ok := b.syn.(TurnContextCompleter); ok && b.AudioContextAvailable(id) {
		c.OnTurnContextCompleted(ctx, id)
	}
}

// handleInterruption drops everything queued for playback. The audio is no
// longer wanted, and the provider is told so it stops generating into contexts
// nobody is listening to.
func (b *Base) handleInterruption(ctx context.Context) {
	b.setProcessingText(false)
	b.setBotSpeaking(false)
	b.setLLMResponding(false)
	if b.aggregator != nil {
		b.aggregator.Reset()
	}
	for _, f := range b.filters {
		if i, ok := f.(ttstext.InterruptibleFilter); ok {
			i.HandleInterruption()
		}
	}
	if b.sequencer != nil {
		b.sequencer.Clear()
	}
	b.turnContext = ""
	b.audioCtxMu.Lock()
	b.wordLastPTS = 0
	b.audioCtxMu.Unlock()
	b.stopAudioContexts()
	b.audioCtxMu.Lock()
	if b.serial != nil {
		b.serial.reset()
	}
	b.audioCtxMu.Unlock()
	if c, ok := b.syn.(AudioContextInterrupter); ok {
		for _, id := range b.openAudioContexts() {
			c.OnAudioContextInterrupted(ctx, id)
		}
	}
	b.startAudioContexts(ctx)
	// The frame goroutine may be paused here: an interruption arriving while an
	// uninterruptible frame was being handled flushes the queue but leaves the
	// goroutine in place, and no BotStoppedSpeakingFrame is coming for audio that
	// was never played. Resume, or the service stays paused for good.
	b.maybeResumeFrameProcessing()
}

// handleSpeak speaks fixed text immediately, bypassing sentence aggregation.
//
// The frame is consumed rather than forwarded: it is a request to speak, and
// what the pipeline behind this service is told is the speech it produced. The
// conversation is built from what was actually spoken, never from the text on
// its way to the synthesizer, so this utterance reaches it as TTSTextFrames, per
// word where the provider times them and as one whole unit where it does not.
// The caller's answer about whether it belongs there is carried to those frames,
// so an utterance the service says rather than the assistant (a phrase covering
// a tool call, a stall while something is fetched) stays out of the conversation
// when the caller asked for that.
func (b *Base) handleSpeak(ctx context.Context, fr *frames.TTSSpeakFrame, _ processor.Direction) error {
	appendToContext := fr.AppendToContext
	// With no model response driving this, nothing else will close the assistant
	// turn the speech opens, so the service closes it once the words are out.
	pushAssistantAggregation := appendToContext && !b.llmResponding()
	// A fixed utterance is independent of any LLM turn, so it gets a context of
	// its own that is opened and closed around it. Whether a turn was mid-flight
	// is saved and put back, so speaking this does not look like the end of one.
	saved := b.turnContext
	savedProcessing := b.isProcessingText()
	b.turnContext = b.createContextID()
	if c, ok := b.syn.(TurnContextCreator); ok {
		c.OnTurnContextCreated(ctx, b.turnContext)
	}
	// A fixed utterance was never aggregated, so it stands as one sentence and
	// the text is its own written form.
	spoken := ttstext.Aggregation{Text: fr.Text, Type: frames.AggregationSentence}
	if err := b.pushTTSFrames(ctx, spoken, appendToContext, pushAssistantAggregation); err != nil {
		return err
	}
	b.onTurnContextCompleted(ctx)
	// Text went to the provider, so pause for the audio it will produce.
	b.maybePauseFrameProcessing()
	b.turnContext = saved
	b.setProcessingText(savedProcessing)
	return nil
}

// handleStart forwards the StartFrame, announces the service, and gives the
// provider its chance to set up before the first sentence.
func (b *Base) handleStart(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.PushFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := b.aggregatorErr; err != nil {
		// Fatal: with no way to group text into units there is nothing to speak.
		b.PushError(ctx, "tts text aggregator unavailable", err, true)
		return err
	}
	b.startAudioContexts(ctx)
	return nil
}

// handleText folds a text frame's text into the aggregator. The frame itself is
// not forwarded: what reaches the pipeline is the text the synthesizer actually
// spoke, as TTSTextFrames timed to the audio, so the conversation records what
// was said rather than what was generated. A consumer that wants the text as the
// model produced it watches for it where the model pushes it.
func (b *Base) handleText(ctx context.Context, tf *frames.TextFrame, _ frames.Frame, _ processor.Direction) error {
	// The model has started answering, which is when the wait for the first
	// sentence starts. A transcription is a distinct type and does not land here.
	b.startTextAggregationMetrics()
	return b.aggregate(ctx, tf.Text)
}

// aggregate folds text into the aggregator and synthesizes every unit it
// completes.
func (b *Base) aggregate(ctx context.Context, text string) error {
	if b.aggregator == nil {
		return nil
	}
	for _, agg := range b.aggregator.Aggregate(text) {
		if agg.Type != frames.AggregationToken {
			// The first sentence is complete, which is what the aggregation clock
			// was waiting for. Later ones find it already stopped.
			b.stopTextAggregationMetrics(ctx)
		}
		if err := b.pushTTSFrames(ctx, agg, true, false); err != nil {
			return err
		}
	}
	return nil
}

// flush synthesizes text left buffered when a response ends without a boundary.
func (b *Base) flush(ctx context.Context) error {
	if b.aggregator == nil {
		return nil
	}
	rest, ok := b.aggregator.Flush()
	// The response is over. Stop the clock whether or not anything was left, for
	// a response that never completed a sentence at all.
	b.stopTextAggregationMetrics(ctx)
	if !ok || strings.TrimSpace(rest.Text) == "" {
		return nil
	}
	return b.pushTTSFrames(ctx, rest, true, false)
}

// setLLMResponding records whether a model response is driving the speech.
func (b *Base) setLLMResponding(v bool) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	b.llmResponseStarted = v
}

// llmResponding reports whether a model response is driving the speech.
func (b *Base) llmResponding() bool {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.llmResponseStarted
}

// setYieldsSync records whether the provider yielded its audio.
func (b *Base) setYieldsSync(v bool) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	b.yieldsSync = v
}

// yieldingSync reports whether the provider answers with its audio.
func (b *Base) yieldingSync() bool {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.yieldsSync
}

// wordPath reports whether the driving provider reports word timings, in which
// case the base emits playback-aligned TTSTextFrames for interruption-accurate
// context.
func (b *Base) wordPath() bool {
	_, ok := b.syn.(WordTimestamps)
	return ok
}

// PushFrame pushes a frame on, closing the assistant turn behind an utterance
// the service said on its own.
//
// It is done here rather than where the stop frame is built because there is
// more than one place that builds one, the base's own and a provider's, and
// every one of them passes through here.
func (b *Base) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if stopped, ok := f.(*frames.TTSStoppedFrame); ok {
		if stopped.ContextID != "" {
			if err := b.maybeCloseAssistantTurn(ctx, stopped.ContextID); err != nil {
				return err
			}
		}
		// Ahead of the stop frame, so the padding is part of the utterance
		// rather than something arriving after it has been called finished.
		if err := b.pushTrailingSilence(ctx); err != nil {
			return err
		}
	}
	b.stampDestination(f)
	return b.Base.PushFrame(ctx, f, dir)
}

// maybeCloseAssistantTurn tells the aggregator to commit, for a context that
// asked for it when it was opened. The answer is taken rather than read, so the
// turn is closed once however many stop frames the context produces.
func (b *Base) maybeCloseAssistantTurn(ctx context.Context, contextID string) error {
	c, known := b.takeTTSContext(contextID)
	if !known || !c.pushAssistantAggregation {
		return nil
	}
	push := frames.NewLLMAssistantPushAggregationFrame()
	// The words of this utterance travel the transport's clock queue, which
	// holds each until its moment. A frame with no timing of its own takes the
	// other queue and would overtake them, closing the turn before the last
	// words had been said, so it is timed just past the last of them.
	if pts := b.lastWordPTS(); pts > 0 {
		push.Base().SetPTS(int64(pts) + 1)
	}
	return b.Base.PushFrame(ctx, push, processor.Downstream)
}

// pushTTSFrames synthesizes one piece of text on the turn's audio context,
// opening the context if this is the turn's first sentence. original is the
// pre-filter text; the configured filters produce the text sent to the provider.
// appendToContext says whether what is spoken here belongs in the conversation;
// it is recorded on the context and stamped onto every frame emitted from it.
// pushAssistantAggregation says whether the assistant turn has to be closed
// once it has been said.
//
// The audio does not go downstream from here. It is appended to the context and
// pushed by the context's drain loop, so a provider whose audio arrives later on
// its own receive loop and one that returns it inline reach the pipeline the
// same way, in the order the audio was generated.
func (b *Base) pushTTSFrames(
	ctx context.Context, agg ttstext.Aggregation, appendToContext, pushAssistantAggregation bool,
) error {
	original := agg.Text
	filtered := original
	for _, f := range b.filters {
		// A stateful filter is told the interruption is over just before it is
		// asked for text again, which is the point it can resume from.
		if i, ok := f.(ttstext.InterruptibleFilter); ok {
			i.ResetInterruption()
		}
		filtered = f.Filter(filtered)
	}
	if strings.TrimSpace(filtered) == "" {
		return nil
	}
	// Text is on its way to the provider, so there is audio to wait for at the
	// end of the turn. Set after the filters: a filter that strips the text to
	// nothing would otherwise leave the flag latched and, with pausing enabled,
	// pause the service waiting for audio that never comes.
	b.setProcessingText(true)
	contextID := b.createContextID()

	// Announce what is about to be spoken, before the audio describing it opens.
	// A consumer that wants the text ahead of hearing it reads this: an RTVI
	// client starts its segment from it, and the progress frames that follow
	// refer back to it. It carries the text as written, before the filters, and
	// does not itself go into the conversation, which is built from what was
	// actually spoken.
	aggregated := frames.NewAggregatedTextFrame(original, agg.Type)
	// The written form this unit was cut from. It differs from the text only
	// when the unit came from a matched pattern, where the text is the content
	// and the delimiters around it are what the model actually wrote.
	aggregated.RawText = agg.Original()
	aggregated.ContextID = contextID
	aggregated.WillBeSpoken = true
	aggregated.AppendToContext = false

	// Recorded before the context opens, so the drain loop has the answers in
	// hand by the time the first frame of the context reaches it.
	b.setTTSContext(contextID, ttsContext{
		appendToContext:          appendToContext,
		pushAssistantAggregation: pushAssistantAggregation,
	})

	if !b.AudioContextAvailable(contextID) {
		// Serialized so it lands immediately before the start of the context it
		// describes, rather than racing ahead of audio still draining from the
		// turn before.
		_ = b.queueSerial(ctx, aggregated, processor.Downstream)
		b.CreateAudioContext(contextID)
		started := frames.NewTTSStartedFrame()
		started.ContextID = contextID
		b.AppendToAudioContext(contextID, started)
	} else {
		b.AppendToAudioContext(contextID, aggregated)
	}
	c := b.audioContextFor(contextID)
	if c == nil {
		return nil
	}
	c.addText(filtered)
	b.pushSequencerFrames(ctx, b.sequencer.RegisterSpoken(
		aggregated, contextID, filtered, appendToContext, b.wordPath(), false))
	return b.runTTS(ctx, c, contextID, original, filtered, appendToContext)
}

// runTTS asks the provider to speak one unit of text and routes what it yields
// to the synthesis context. A provider whose audio arrives later on its own
// receive loop yields nothing here and appends it itself.
func (b *Base) runTTS(
	ctx context.Context, c *audioContext, contextID, original, filtered string, appendToContext bool,
) error {
	// Every frame the provider yields is routed to its context, so audio and
	// anything else it reports stay in the order it produced them.
	yielded := false
	yield := func(f frames.Frame) error {
		if _, audio := f.(*frames.TTSAudioRawFrame); audio {
			yielded = true
		}
		b.AppendToAudioContext(contextID, f)
		return nil
	}
	var err error
	usable := b.Usable()
	if !usable {
		// A service that can no longer work cannot synthesize anything, and one
		// that connects on demand would attempt a handshake per request. The
		// bookkeeping around this still runs, so the turn completes with no audio
		// rather than stalling. The text that goes unspoken is named: silence
		// from the bot is otherwise hard to trace back to the service that caused
		// it.
		slog.WarnContext(ctx, "service is no longer usable, not speaking",
			"service", b.Name(), "text", filtered)
	} else if wt, ok := b.syn.(WordTimestamps); ok {
		word := func(words []uctx.WordTiming, opts WordTimingOptions) error {
			b.AddWordTimestamps(contextID, words, opts)
			return nil
		}
		err = wt.RunTTSTimed(ctx, filtered, contextID, yield, word)
	} else {
		err = b.syn.RunTTS(ctx, filtered, contextID, yield)
	}
	if err != nil && ctx.Err() == nil {
		c.span.RecordError(err)
		b.PushError(ctx, "tts synthesis failed", err, false)
	}
	// Whether the provider answered here decides who closes the context: one that
	// yielded its audio is finished, one that did not is still delivering. A
	// service that can no longer work was never asked, so nothing is coming and
	// the base closes the context itself rather than leaving the turn open on
	// audio that will never arrive.
	b.setYieldsSync(yielded || !usable)
	if !b.wordPath() {
		// With no word timings there is nothing to place the text against, so the
		// whole unit goes in as one, carrying the text as written rather than what
		// the filters made of it.
		//
		// This is the only thing that puts the turn into the conversation: the
		// text the model produced is consumed here and never forwarded, and
		// without word timings no per-word frames are produced either. It cannot
		// depend on the provider having answered inline. One that delivers its
		// audio later on its own receive loop yields nothing here, and gating on
		// that left every one of its turns out of the context entirely.
		text := frames.NewTTSTextFrame(original)
		text.ContextID = contextID
		text.AppendToContext = appendToContext
		b.AppendToAudioContext(contextID, text)
		b.pushSequencerFrames(ctx, b.sequencer.CompleteSpokenSlot())
	}
	return nil
}

// emitTiming records the synthesis's time-to-first-byte, time-to-first-audible
// sample, processing time and character count to OpenTelemetry (always) and,
// when in-band metrics are enabled, downstream as a MetricsFrame for the RTVI
// client.
func (b *Base) emitTiming(ctx context.Context, chars int, m *ttfaMeter, processing time.Duration) {
	model := b.model()
	metrics.RecordProcessing(ctx, "tts", b.Name(), model, processing.Seconds())
	metrics.RecordTTSCharacters(ctx, b.Name(), model, int64(chars))
	if m.hadTTFB {
		metrics.RecordTTFB(ctx, "tts", b.Name(), model, m.ttfb.Seconds())
	}
	if m.hadTTFA {
		metrics.RecordTTFA(ctx, "tts", b.Name(), model, m.ttfa.Seconds())
	}
	if !b.MetricsEnabled() {
		return
	}
	base := frames.BaseMetricsData{Processor: b.Name(), Model: b.model()}
	data := []frames.MetricsData{
		frames.ProcessingMetricsData{BaseMetricsData: base, Value: processing},
		frames.TTSUsageMetricsData{BaseMetricsData: base, Value: chars},
	}
	if m.hadTTFB {
		data = append(data, frames.TTFBMetricsData{BaseMetricsData: base, Value: m.ttfb})
	}
	if m.hadTTFA {
		data = append(data, frames.TTFAMetricsData{
			BaseMetricsData: base,
			TTFA:            m.ttfa,
			TTFB:            m.ttfb,
			LeadingSilence:  m.leadingSilence,
		})
	}
	_ = b.PushFrame(ctx, frames.NewMetricsFrame(data...), processor.Downstream)
}

// ServiceMetadataFrame implements service.MetadataDescriber, describing this
// synthesizer to the rest of the pipeline.
func (b *Base) ServiceMetadataFrame() frames.ServiceMetadata {
	return frames.NewServiceMetadataFrame(b.Name())
}

// ttfaMeter measures a synthesis's time-to-first-byte and time-to-first-audible
// sample from the emitted PCM. TTFA extends TTFB: once the first byte arrives it
// scans the leading audio for a sustained speech onset and reports TTFB plus the
// leading-silence duration.
type ttfaMeter struct {
	rate int
	// report says whether this synthesis is measured at all. It is false once
	// the pipeline has had the only measurement it asked for, and nothing is
	// timed or scanned from then on.
	report bool

	hadTTFB bool
	ttfb    time.Duration

	active         bool
	buf            []byte
	hadTTFA        bool
	ttfa           time.Duration
	leadingSilence time.Duration
}

// observe folds one non-empty PCM chunk into the measurement, relative to the
// synthesis start time.
func (m *ttfaMeter) observe(pcm []byte, start time.Time) {
	if !m.report {
		return
	}
	if !m.hadTTFB {
		m.hadTTFB = true
		m.ttfb = time.Since(start)
		m.active = true
	}
	if !m.active {
		return
	}
	m.buf = append(m.buf, pcm...)
	if idx := onset.Detect(m.buf, m.rate, 1); idx >= 0 {
		m.leadingSilence = time.Duration(float64(idx) / float64(m.rate) * float64(time.Second))
		m.ttfa = m.ttfb + m.leadingSilence
		m.hadTTFA = true
		m.active = false
		m.buf = nil
		return
	}
	if len(m.buf) >= ttfaMaxBufferSeconds*m.rate*2 {
		// No onset within a bounded window of audio; stop scanning.
		m.active = false
		m.buf = nil
	}
}

// StreamResponse issues req and streams the raw-PCM response body to emit in
// chunks. It is the shared body-reading loop for HTTP TTS providers. Pair it
// with PCMYielder to turn the chunks into frames.
func StreamResponse(client *http.Client, req *http.Request, emit func(pcm []byte) error) error {
	resp, err := client.Do(req) //nolint:gosec // request target is the service's configured endpoint
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return errs.NewHTTPStatusError(resp.StatusCode, fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg))
	}
	buf := make([]byte, readChunk)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			chunk := make([]byte, n)
			copy(chunk, buf[:n])
			if perr := emit(chunk); perr != nil {
				return perr
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// CanGenerateMetrics reports that this service times synthesis and reports
// the result, so the pipeline counts it when it collects the processors that
// report metrics.
func (b *Base) CanGenerateMetrics() bool { return true }
