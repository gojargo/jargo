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
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/gojargo/jargo/audio/onset"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/metrics"
	uctx "github.com/gojargo/jargo/utils/context"
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
	*processor.Base
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
	serial        *serialQueue
	ctxCancel     context.CancelFunc
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

	// Pausing frame handling while a turn's audio is generated. See pause.go.
	pauseStateMu sync.Mutex
	// pauseOpts is how the service was configured; pausing is off by default.
	pauseOpts PauseOptions
	// processingText records that text for this turn reached the provider, so
	// there is audio worth waiting for.
	processingText bool
	// botSpeaking records that the audio is confirmed playing, which is what
	// says the watchdog is not needed.
	botSpeaking bool
	// watchdogCancel stops the armed watchdog, nil when none is armed.
	watchdogCancel context.CancelFunc

	// textAgg times how long grouping text into sentences takes. See
	// textaggregation.go.
	textAgg textAggregationMetrics
}

// New builds a TTS Base named name driven by syn. The concrete service passes
// itself as syn and embeds the returned Base.
func New(name string, syn Synthesizer) *Base {
	b := &Base{syn: syn}
	if tok, err := ttstext.NewPunktEnglish(); err != nil {
		b.aggregatorErr = err
	} else {
		b.aggregator = ttstext.NewSimpleAggregator(frames.AggregationSentence, tok)
		b.sequencer = uctx.NewAggregatedFrameSequencer(name, false, tok)
	}
	if d, ok := syn.(Describer); ok {
		b.meta = d.Metadata()
	}
	b.Base = processor.New(name, b)
	if cs, ok := syn.(ContextSynthesizer); ok {
		cs.SetAudioContextHost(b)
	}
	return b
}

// Cleanup releases the Synthesizer's resources, when it holds any, and tears
// down the processor.
func (b *Base) Cleanup(ctx context.Context) error {
	b.cancelPauseWatchdog()
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

// ProcessFrame aggregates text into sentences and synthesizes them.
func (b *Base) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.LLMTextFrame:
		return b.handleText(ctx, &fr.TextFrame, f, dir)
	case *frames.TextFrame:
		return b.handleText(ctx, fr, f, dir)
	case *frames.TTSSpeakFrame:
		return b.handleSpeak(ctx, fr, dir)
	case *frames.LLMFullResponseStartFrame:
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
	case *frames.BotStartedSpeakingFrame, *frames.BotStoppedSpeakingFrame:
		return b.handleBotSpeaking(ctx, f, dir)
	default:
		return b.queueSerial(ctx, f, dir)
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
	b.maybePauseFrameProcessing(ctx)
	b.setProcessingText(false)
	b.onTurnContextCompleted(ctx)
	if isEnd {
		return b.PushFrame(ctx, f, dir)
	}
	return b.queueSerial(ctx, f, dir)
}

// handleBotSpeaking tracks whether the bot's audio is playing, which is what
// releases a service paused waiting for it.
func (b *Base) handleBotSpeaking(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if _, started := f.(*frames.BotStartedSpeakingFrame); started {
		// The audio for this turn is confirmed playing, so the watchdog is not
		// needed: the stopped frame resumes once playback finishes.
		b.setBotSpeaking(true)
		b.cancelPauseWatchdog()
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
		b.AppendToAudioContext(id, frames.NewTTSStoppedFrame())
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
func (b *Base) handleSpeak(ctx context.Context, fr *frames.TTSSpeakFrame, dir processor.Direction) error {
	// The conversation is built from what was actually spoken, never from the
	// text on its way to the synthesizer. Word timings or not, this utterance
	// reaches the context as TTSTextFrames: per word where the provider times
	// them, as one whole unit where it does not. Letting the aggregator record
	// the fixed text as well would enter it twice.
	fr.AppendToContext = false
	if err := b.PushFrame(ctx, fr, dir); err != nil {
		return err
	}
	// A fixed utterance is independent of any LLM turn, so it gets a context of
	// its own that is opened and closed around it. Whether a turn was mid-flight
	// is saved and put back, so speaking this does not look like the end of one.
	saved := b.turnContext
	savedProcessing := b.isProcessingText()
	b.turnContext = b.createContextID()
	if c, ok := b.syn.(TurnContextCreator); ok {
		c.OnTurnContextCreated(ctx, b.turnContext)
	}
	if err := b.pushTTSFrames(ctx, fr.Text); err != nil {
		return err
	}
	b.onTurnContextCompleted(ctx)
	// Text went to the provider, so pause for the audio it will produce.
	b.maybePauseFrameProcessing(ctx)
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
	b.broadcastMetadata(ctx)
	if st, ok := b.syn.(Starter); ok {
		// Detached: the setup outlives this frame, and a turn canceled by an
		// interruption must not abandon a connection half-dialed.
		go st.Start(context.WithoutCancel(ctx))
	}
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
		if err := b.pushTTSFrames(ctx, agg.Text); err != nil {
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
	return b.pushTTSFrames(ctx, rest.Text)
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

// pushTTSFrames synthesizes one piece of text on the turn's audio context,
// opening the context if this is the turn's first sentence. original is the
// pre-filter text; the configured filters produce the text sent to the provider.
//
// The audio does not go downstream from here. It is appended to the context and
// pushed by the context's drain loop, so a provider whose audio arrives later on
// its own receive loop and one that returns it inline reach the pipeline the
// same way, in the order the audio was generated.
func (b *Base) pushTTSFrames(ctx context.Context, original string) error {
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
	aggregated := frames.NewAggregatedTextFrame(original, frames.AggregationSentence)
	aggregated.ContextID = contextID
	aggregated.WillBeSpoken = true
	aggregated.AppendToContext = false

	if !b.AudioContextAvailable(contextID) {
		// Serialized so it lands immediately before the start of the context it
		// describes, rather than racing ahead of audio still draining from the
		// turn before.
		_ = b.queueSerial(ctx, aggregated, processor.Downstream)
		b.CreateAudioContext(contextID)
		b.AppendToAudioContext(contextID, frames.NewTTSStartedFrame())
	} else {
		b.AppendToAudioContext(contextID, aggregated)
	}
	c := b.audioContextFor(contextID)
	if c == nil {
		return nil
	}
	// Providers bill per character, so count runes: len would charge an accented
	// character twice.
	c.addChars(utf8.RuneCountInString(filtered))
	b.pushSequencerFrames(ctx, b.sequencer.RegisterSpoken(
		aggregated, contextID, filtered, true, b.wordPath(), false))
	return b.runTTS(ctx, c, contextID, original, filtered)
}

// runTTS asks the provider to speak one unit of text and routes what it yields
// to the synthesis context. A provider whose audio arrives later on its own
// receive loop yields nothing here and appends it itself.
func (b *Base) runTTS(ctx context.Context, c *audioContext, contextID, original, filtered string) error {
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
	if wt, ok := b.syn.(WordTimestamps); ok {
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
	// yielded its audio is finished, one that did not is still delivering.
	b.setYieldsSync(yielded)
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
	metrics.RecordProcessing(ctx, "tts", b.Name(), b.meta.Model, processing.Seconds())
	metrics.RecordTTSCharacters(ctx, b.Name(), b.meta.Model, int64(chars))
	if m.hadTTFB {
		metrics.RecordTTFB(ctx, "tts", b.Name(), b.meta.Model, m.ttfb.Seconds())
	}
	if m.hadTTFA {
		metrics.RecordTTFA(ctx, "tts", b.Name(), b.meta.Model, m.ttfa.Seconds())
	}
	if !b.MetricsEnabled() {
		return
	}
	base := frames.BaseMetricsData{Processor: b.Name(), Model: b.meta.Model}
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

// broadcastMetadata pushes the TTS service's metadata frame downstream at
// pipeline start so downstream processors can discover the service.
func (b *Base) broadcastMetadata(ctx context.Context) {
	_ = b.PushFrame(ctx, frames.NewServiceMetadataFrame(b.Name()), processor.Downstream)
}

// ttfaMeter measures a synthesis's time-to-first-byte and time-to-first-audible
// sample from the emitted PCM. TTFA extends TTFB: once the first byte arrives it
// scans the leading audio for a sustained speech onset and reports TTFB plus the
// leading-silence duration.
type ttfaMeter struct {
	rate int

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
		return fmt.Errorf("%w %d: %s", errStatus, resp.StatusCode, msg)
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
