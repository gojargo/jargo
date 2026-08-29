package tts

import (
	"context"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/telemetry/tracing"
	uctx "github.com/gojargo/jargo/utils/context"
	"go.opentelemetry.io/otel/trace"
)

// stopFrameTimeout is how long an audio context waits for more audio before it
// is taken as finished and its stop frame is pushed.
const stopFrameTimeout = 3 * time.Second

// AudioContextHost is the part of the Base a ContextSynthesizer pushes its audio
// into. A provider whose audio arrives on its own receive loop appends frames to
// the context they belong to and removes the context when the provider has
// marked it final.
type AudioContextHost interface {
	// AppendToAudioContext adds a frame to an existing context's queue.
	AppendToAudioContext(contextID string, f frames.Frame)
	// AppendWordToAudioContext adds one spoken token to a context's queue, with
	// the offset in seconds where it starts in the context's audio, so its text
	// frame is pushed only once the audio that speaks it has gone out.
	AppendWordToAudioContext(contextID, word string, offset float64)
	// AddWordTimestamps adds a batch of spoken tokens to a context's queue,
	// normalizing them as opts asks first. It is the entry point for a provider
	// that reports timings on its own receive loop rather than through the
	// callback, so both arrive normalized the same way.
	AddWordTimestamps(contextID string, words []uctx.WordTiming, opts WordTimingOptions)
	// RemoveAudioContext marks a context for deletion once the audio already
	// queued for it has been pushed.
	RemoveAudioContext(contextID string)
	// AudioContextAvailable reports whether the context is still open.
	AudioContextAvailable(contextID string) bool
}

// ContextSynthesizer is an optional interface a Synthesizer implements when its
// audio does not come back from RunTTS but arrives later on a receive loop of
// its own. Such a provider yields nothing: RunTTS returns once the text is sent,
// so the next sentence goes out while the current one is still being generated,
// and the audio is appended to the context it belongs to as it arrives.
//
// A provider that yields its audio needs none of this. Either way the audio
// reaches the pipeline through an audio context.
type ContextSynthesizer interface {
	Synthesizer
	// SetAudioContextHost hands the provider the host it appends audio to. The
	// Base calls it once, when the service is built.
	SetAudioContextHost(h AudioContextHost)
}

// TurnContextCreator is an optional interface a Synthesizer implements to open
// its server-side context as soon as the turn starts, before any text flows.
type TurnContextCreator interface {
	OnTurnContextCreated(ctx context.Context, contextID string)
}

// TurnContextCompleter is an optional interface a Synthesizer implements to
// close its server-side context once the turn's text has all been sent. The
// audio for it may still be arriving.
type TurnContextCompleter interface {
	OnTurnContextCompleted(ctx context.Context, contextID string)
}

// AudioContextInterrupter is an optional interface a Synthesizer implements when
// it has cleanup to do for a context an interruption cut short, such as telling
// the provider to stop generating into it.
type AudioContextInterrupter interface {
	OnAudioContextInterrupted(ctx context.Context, contextID string)
}

// AudioFlusher is an optional interface a Synthesizer implements when it holds
// text server-side until told to generate it. The Base flushes the turn's
// context once its text has all been sent.
type AudioFlusher interface {
	FlushAudio(ctx context.Context, contextID string)
}

// AudioContextCompleter is an optional interface a Synthesizer implements when
// it has state to release once a context has finished playing out.
type AudioContextCompleter interface {
	OnAudioContextCompleted(ctx context.Context, contextID string)
}

// ctxItem is one entry on an audio context's queue: a frame to push, a spoken
// token to turn into a text frame, a keepalive that only resets the idle
// timeout, or the sentinel that ends the context.
type ctxItem struct {
	frame frames.Frame
	word  string
	// offset is where the word starts, in seconds from the beginning of the
	// context's audio.
	offset float64
	// includesInterFrame reports whether the token carries whatever spacing
	// separates it from the ones around it, so a consumer joining them adds none
	// of its own.
	includesInterFrame bool
	isWord             bool
	keepalive          bool
	end                bool
}

// audioContext is one context's queue of audio, in the order the provider
// generated it, along with the sentence trackers its words are attributed to and
// the measurement of what the synthesis cost.
//
// The timing is measured here rather than at the synthesis call because that is
// where both kinds of provider meet: one returns its audio inline, the other
// delivers it later on its own receive loop, and only the queue sees both.
type audioContext struct {
	mu     sync.Mutex
	items  []ctxItem
	notify chan struct{}

	start time.Time
	rate  int
	chars int
	// texts are the sentences spoken on this context, in the order they were
	// sent. One context covers a whole utterance, which sentence aggregation
	// splits into several synthesis calls, so the span reports the utterance
	// rather than whichever sentence happened to be last.
	texts []string
	meter ttfaMeter
	span  trace.Span
	// spanCtx carries span, so the usage recorded when the context finishes
	// lands on the span that covered it.
	spanCtx context.Context //nolint:containedctx // the span outlives the call that opened it

	// initialWordTimestamp is the clock reading the context's words are timed
	// from, fixed on its first chunk of audio. Negative until then.
	initialWordTimestamp time.Duration
	// pendingWords holds words that arrived before any audio did, so there was
	// no baseline yet to time them against.
	pendingWords []ctxItem
}

func newAudioContext(rate int, report bool, spanCtx context.Context, span trace.Span) *audioContext {
	return &audioContext{
		notify:               make(chan struct{}, 1),
		start:                time.Now(),
		rate:                 rate,
		meter:                ttfaMeter{rate: rate, report: report},
		span:                 span,
		spanCtx:              spanCtx,
		initialWordTimestamp: -1,
	}
}

// addText records a sentence sent on this context, and the characters it cost.
// Providers bill per character, and the count is in runes because that is the
// unit they bill in: an accented character is one character, not the two bytes
// it occupies.
func (c *audioContext) addText(text string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.texts = append(c.texts, text)
	c.chars += utf8.RuneCountInString(text)
}

// observe folds one chunk of this context's audio into the measurement.
func (c *audioContext) observe(pcm []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.meter.observe(pcm, c.start)
}

// startWordTimestamps fixes the point the context's words are timed from, on the
// first chunk of its audio, and returns the words that were waiting for it. now
// is the pipeline clock; lastPTS carries the timeline across contexts whose
// audio overlaps, so a later context's words cannot be timed before an earlier
// context's last word.
func (c *audioContext) startWordTimestamps(now, lastPTS time.Duration) []ctxItem {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.initialWordTimestamp >= 0 {
		return nil
	}
	// A context whose audio starts while an earlier one is still playing times
	// its words after that one's last, so the timeline only moves forward.
	c.initialWordTimestamp = max(now, lastPTS)
	waiting := c.pendingWords
	c.pendingWords = nil
	return waiting
}

// wordPTS reports when a word at offset seconds into the context is spoken. It
// reports ok=false while no audio has arrived, since there is nothing to time
// it against yet.
func (c *audioContext) wordPTS(offset float64) (time.Duration, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.initialWordTimestamp < 0 {
		return 0, false
	}
	return c.initialWordTimestamp + time.Duration(offset*float64(time.Second)), true
}

// holdWord keeps a word that arrived before any audio, to be timed once the
// baseline is fixed.
func (c *audioContext) holdWord(it ctxItem) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.pendingWords = append(c.pendingWords, it)
}

// measurement reports what the context cost, once it has finished draining.
func (c *audioContext) measurement() (chars int, meter ttfaMeter, elapsed time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.chars, c.meter, time.Since(c.start)
}

// spoken is everything sent on this context, as one utterance.
func (c *audioContext) spoken() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return strings.Join(c.texts, " ")
}

// push appends an item and wakes the drain loop. It never blocks.
func (c *audioContext) push(it ctxItem) {
	c.mu.Lock()
	c.items = append(c.items, it)
	c.mu.Unlock()
	select {
	case c.notify <- struct{}{}:
	default:
	}
}

// get returns the next item, waiting up to timeout for one. It reports timedOut
// when nothing arrived in time and ok=false when ctx was canceled.
func (c *audioContext) get(ctx context.Context, timeout time.Duration) (it ctxItem, ok, timedOut bool) {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for {
		c.mu.Lock()
		if len(c.items) > 0 {
			next := c.items[0]
			c.items = c.items[1:]
			c.mu.Unlock()
			return next, true, false
		}
		c.mu.Unlock()
		select {
		case <-c.notify:
		case <-timer.C:
			return ctxItem{}, false, true
		case <-ctx.Done():
			return ctxItem{}, false, false
		}
	}
}

// serialItem is what the serialization queue carries: an audio context to drain,
// a frame to emit at exactly this position in the output stream, or the sentinel
// that shuts the queue down once everything queued ahead of it has been
// processed.
type serialItem struct {
	contextID string
	frame     frames.Frame
	end       bool
}

// serialQueue is an unbounded FIFO of serialization items. Unbounded because its
// consumer blocks pushing audio downstream at playout speed while its producer
// is the frame goroutine, which must not be held up behind playback.
type serialQueue struct {
	mu     sync.Mutex
	items  []serialItem
	notify chan struct{}
}

func newSerialQueue() *serialQueue {
	return &serialQueue{notify: make(chan struct{}, 1)}
}

func (q *serialQueue) push(it serialItem) {
	q.mu.Lock()
	q.items = append(q.items, it)
	q.mu.Unlock()
	select {
	case q.notify <- struct{}{}:
	default:
	}
}

// reset drops everything queued, for an interruption.
func (q *serialQueue) reset() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = nil
}

func (q *serialQueue) get(ctx context.Context) (serialItem, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			it := q.items[0]
			q.items = q.items[1:]
			q.mu.Unlock()
			return it, true
		}
		q.mu.Unlock()
		select {
		case <-q.notify:
		case <-ctx.Done():
			return serialItem{}, false
		}
	}
}

// startAudioContexts brings up the drain goroutine and the context registry.
func (b *Base) startAudioContexts(ctx context.Context) {
	b.stopAudioContexts()
	done := make(chan struct{})
	b.audioCtxMu.Lock()
	b.audioContexts = map[string]*audioContext{}
	b.ttsContexts = map[string]ttsContext{}
	// Held response ends go with the contexts they were waiting on. An
	// interruption comes through here, and what it cut off is not to be reported
	// as having finished.
	b.pendingResponseEnd = map[string]*frames.LLMFullResponseEndFrame{}
	b.serial = newSerialQueue()
	b.ctxDone = done
	b.audioCtxMu.Unlock()
	loopCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	b.ctxCancel = cancel
	b.ctxWG.Add(1)
	go b.audioContextLoop(loopCtx, done)
}

// stopAudioContexts tears the drain goroutine down and waits for it. What was
// queued is dropped, which is what an interruption or a cancel wants; a graceful
// end goes through drainAudioContexts instead.
func (b *Base) stopAudioContexts() {
	cancel := b.ctxCancel
	b.ctxCancel = nil
	if cancel == nil {
		return
	}
	cancel()
	b.ctxWG.Wait()
	b.abandonOpenAudioContexts()
}

// abandonOpenAudioContexts closes the spans of the contexts still open once the
// drain loop has stopped. The context that was playing closed its own span on
// the way out; these are the ones queued behind it, which an interruption drops
// without ever speaking. Their spans would otherwise stay open and never be
// exported, so the utterance the user cut off would vanish from the trace
// instead of being recorded as cut off.
func (b *Base) abandonOpenAudioContexts() {
	b.audioCtxMu.Lock()
	open := b.audioContexts
	b.audioContexts = map[string]*audioContext{}
	b.audioCtxMu.Unlock()
	for _, c := range open {
		b.abandonAudioContext(c)
	}
}

// traceSettings renders the provider's settings for a synthesis span. A provider
// that keeps no settings of its own contributes none.
func (b *Base) traceSettings() map[string]any {
	holder, ok := b.syn.(SettingsHolder)
	if !ok {
		return nil
	}
	given, err := settings.Given(holder.Settings())
	if err != nil {
		return nil
	}
	return given
}

// drainAudioContexts shuts the serialization queue down gracefully and returns
// once it has drained. The sentinel is queued behind everything already there,
// so reaching it means every context queued ahead has played out in full and
// every frame behind them has been pushed, including audio a provider was still
// delivering on its own receive loop.
//
// It is what an EndFrame waits on: the frame that stops the pipeline must not
// overtake the speech it was queued behind. An interruption arriving meanwhile
// tears the loop down, and the wait ends with it.
func (b *Base) drainAudioContexts() {
	b.audioCtxMu.Lock()
	serial, done := b.serial, b.ctxDone
	b.audioCtxMu.Unlock()
	if serial == nil || done == nil {
		return
	}
	serial.push(serialItem{end: true})
	<-done
}

// CreateAudioContext opens a context and queues it for playback behind whatever
// is already queued, so contexts are played in the order they were created.
func (b *Base) CreateAudioContext(contextID string) {
	// The context is played out on the serialization queue, long after the frame
	// that opened it was processed, so the span is parented explicitly to the
	// turn being spoken rather than through a context that is already gone.
	spanCtx, span := b.StartSpan(context.Background(), "tts")
	tracing.SetTTSAttributes(span, tracing.TTSAttributes{
		Service:  b.TypeName(),
		Model:    b.meta.Model,
		VoiceID:  b.meta.VoiceID,
		Settings: b.traceSettings(),
	})
	b.audioCtxMu.Lock()
	if b.audioContexts == nil {
		b.audioContexts = map[string]*audioContext{}
	}
	b.audioContexts[contextID] = newAudioContext(b.syn.SampleRate(), b.BeginTTFB(), spanCtx, span)
	serial := b.serial
	b.audioCtxMu.Unlock()
	if serial != nil {
		serial.push(serialItem{contextID: contextID})
	}
}

// AppendToAudioContext adds a frame to an open context.
func (b *Base) AppendToAudioContext(contextID string, f frames.Frame) {
	if c := b.audioContextFor(contextID); c != nil {
		c.push(ctxItem{frame: f})
	}
}

// AddWordTimestamps queues a batch of spoken tokens for a context, normalizing
// them as opts asks before they are queued.
//
// Normalizing belongs here rather than in each provider: one that skips it
// reports tokens nothing downstream expects, and there is no way to tell from
// the outside which of them did.
func (b *Base) AddWordTimestamps(contextID string, words []uctx.WordTiming, opts WordTimingOptions) {
	if opts.PreMergeTokens {
		words = uctx.MergePunctTokens(words)
	}
	c := b.audioContextFor(contextID)
	if c == nil {
		return
	}
	for _, w := range words {
		c.push(ctxItem{
			word:               w.Word,
			offset:             w.Offset,
			includesInterFrame: opts.IncludesInterFrameSpaces,
			isWord:             true,
		})
	}
}

// AppendWordToAudioContext adds one spoken token to an open context, so its text
// frame is pushed in step with the audio around it.
func (b *Base) AppendWordToAudioContext(contextID, word string, offset float64) {
	if c := b.audioContextFor(contextID); c != nil {
		c.push(ctxItem{word: word, offset: offset, isWord: true})
	}
}

// RemoveAudioContext marks a context for deletion. The drain loop reaches the
// sentinel once it has pushed everything queued ahead of it, which is what keeps
// the last of the audio from being cut off.
func (b *Base) RemoveAudioContext(contextID string) {
	if c := b.audioContextFor(contextID); c != nil {
		c.push(ctxItem{end: true})
	}
}

// AudioContextAvailable reports whether the context is still open.
func (b *Base) AudioContextAvailable(contextID string) bool {
	return b.audioContextFor(contextID) != nil
}

// refreshAudioContext resets a context's idle timeout without emitting
// anything, for a turn whose next sentence is still being generated.
func (b *Base) refreshAudioContext(contextID string) {
	if c := b.audioContextFor(contextID); c != nil {
		c.push(ctxItem{keepalive: true})
	}
}

// ttsContext is what a synthesis context carries beyond its audio.
type ttsContext struct {
	// appendToContext reports whether what is spoken on this context belongs in
	// the conversation.
	appendToContext bool
	// pushAssistantAggregation reports whether the assistant aggregator has to be
	// told to commit once this context has finished speaking. It is set for an
	// utterance the service says on its own, which has no model response around
	// it to close the assistant turn.
	pushAssistantAggregation bool
}

// setTTSContext records what a context carries.
func (b *Base) setTTSContext(contextID string, c ttsContext) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	if b.ttsContexts == nil {
		b.ttsContexts = map[string]ttsContext{}
	}
	b.ttsContexts[contextID] = c
}

// appendsToContext reports what setTTSContext recorded for a context. A context
// nothing was recorded for appends, which is the default everywhere else.
func (b *Base) appendsToContext(contextID string) bool {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	c, ok := b.ttsContexts[contextID]
	return !ok || c.appendToContext
}

// holdResponseEnd keeps the frame ending a model response until the speech sent
// on contextID has been heard.
func (b *Base) holdResponseEnd(contextID string, f *frames.LLMFullResponseEndFrame) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	if b.pendingResponseEnd == nil {
		b.pendingResponseEnd = map[string]*frames.LLMFullResponseEndFrame{}
	}
	b.pendingResponseEnd[contextID] = f
}

// takeResponseEnd returns the frame held for a context and forgets it.
func (b *Base) takeResponseEnd(contextID string) (*frames.LLMFullResponseEndFrame, bool) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	f, ok := b.pendingResponseEnd[contextID]
	delete(b.pendingResponseEnd, contextID)
	return f, ok
}

// takeTTSContext returns what a context carries and forgets it, so the answer is
// acted on once.
func (b *Base) takeTTSContext(contextID string) (ttsContext, bool) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	c, ok := b.ttsContexts[contextID]
	delete(b.ttsContexts, contextID)
	return c, ok
}

// deleteTTSContext forgets a context that has finished.
func (b *Base) deleteTTSContext(contextID string) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	delete(b.ttsContexts, contextID)
}

// audioContextFor looks a context up, or nil when it is not open.
func (b *Base) audioContextFor(contextID string) *audioContext {
	if contextID == "" {
		return nil
	}
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.audioContexts[contextID]
}

// hasOpenAudioContexts reports whether any context is still open, which is what
// says audio may yet arrive for the turn.
func (b *Base) hasOpenAudioContexts() bool {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return len(b.audioContexts) > 0
}

// openAudioContexts lists the contexts still open, for an interruption.
func (b *Base) openAudioContexts() []string {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	out := make([]string, 0, len(b.audioContexts))
	for id := range b.audioContexts {
		out = append(out, id)
	}
	return out
}

// deleteAudioContext forgets a context that has been fully drained.
func (b *Base) deleteAudioContext(contextID string) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	delete(b.audioContexts, contextID)
}

// audioContextLoop drains the serialization queue, preserving downstream frame
// order: a context is played out in full before whatever was queued behind it.
// It runs until the queue is torn down or the shutdown sentinel is reached,
// closing done on its way out so a graceful end knows the queue has drained.
func (b *Base) audioContextLoop(ctx context.Context, done chan struct{}) {
	defer b.ctxWG.Done()
	defer close(done)
	b.audioCtxMu.Lock()
	serial := b.serial
	b.audioCtxMu.Unlock()
	if serial == nil {
		return
	}
	for {
		it, ok := serial.get(ctx)
		if !ok {
			return
		}
		if it.end {
			// Everything queued ahead of the sentinel has been processed.
			return
		}
		if it.frame != nil {
			_ = b.PushFrame(ctx, it.frame, processor.Downstream)
			continue
		}
		if it.contextID == "" {
			continue
		}
		b.handleAudioContext(ctx, it.contextID)
		b.deleteAudioContext(it.contextID)
		b.deleteTTSContext(it.contextID)
		if c, ok := b.syn.(AudioContextCompleter); ok {
			c.OnAudioContextCompleted(ctx, it.contextID)
		}
	}
}

// handleAudioContext pushes one context's frames downstream until the context is
// marked finished, or until it has been idle long enough to be taken as
// finished on its own.
func (b *Base) handleAudioContext(ctx context.Context, contextID string) {
	c := b.audioContextFor(contextID)
	if c == nil {
		return
	}
	shouldPushStop := false
	// Whether any audio arrived for this context at all, which is what says the
	// provider actually spoke rather than accepting the request in silence.
	receivedAudio := false
	for {
		it, ok, timedOut := c.get(ctx, stopFrameTimeout)
		if timedOut {
			break
		}
		if !ok {
			return
		}
		if it.keepalive {
			continue
		}
		if it.end {
			break
		}
		if it.isWord {
			b.pushWord(ctx, c, contextID, it)
			continue
		}
		switch fr := it.frame.(type) {
		case *frames.TTSStartedFrame:
			shouldPushStop = true
			// Stamped here rather than where the frame is built: this is the one
			// point every started frame passes through, the base's own and the ones
			// a provider emits for itself.
			fr.AppendToContext = b.appendsToContext(contextID)
		case *frames.TTSAudioRawFrame:
			// The first chunk of audio is what the context's words are timed
			// from: it is the point the audio starts being heard.
			receivedAudio = true
			c.observe(fr.Audio)
			waiting := c.startWordTimestamps(b.now(), b.lastWordPTS())
			_ = b.PushFrame(ctx, it.frame, processor.Downstream)
			for _, w := range waiting {
				b.pushWord(ctx, c, contextID, w)
			}
			continue
		case *frames.TTSStoppedFrame:
			b.applyForceComplete(ctx, contextID)
			shouldPushStop = false
		}
		_ = b.PushFrame(ctx, it.frame, processor.Downstream)
	}
	b.applyForceComplete(ctx, contextID)
	if shouldPushStop {
		stopped := frames.NewTTSStoppedFrame()
		stopped.ContextID = contextID
		_ = b.PushFrame(ctx, stopped, processor.Downstream)
	}
	b.releaseResponseEnd(ctx, contextID)
	b.finishAudioContext(ctx, c)
	b.recordContextAudioOutcome(ctx, contextID, receivedAudio)
}

// releaseResponseEnd reports the end of the model's response now that the audio
// for this context has been heard, stamped with the moment its last word was
// spoken so it lands behind that word rather than ahead of it.
//
// The frame held at the end of the turn is the one pushed, id and all, so a
// consumer recognizing a frame it has already seen is not told twice that the
// same response ended. A response that was under way with nothing held for this
// context is reported with a fresh frame rather than not at all.
func (b *Base) releaseResponseEnd(ctx context.Context, contextID string) {
	if !b.wordPath() {
		// Nothing to release: the end went out on the serialization queue, behind
		// the text this provider pushes a whole unit at a time.
		return
	}
	f, ok := b.takeResponseEnd(contextID)
	if !ok {
		if !b.llmResponding() {
			return
		}
		f = frames.NewLLMFullResponseEndFrame()
	}
	b.setLLMResponding(false)
	f.SetPTS(int64(b.lastWordPTS()))
	_ = b.PushFrame(ctx, f, processor.Downstream)
}

// finishAudioContext records what the context cost and closes its span.
func (b *Base) finishAudioContext(ctx context.Context, c *audioContext) {
	chars, meter, elapsed := c.measurement()
	attrs := tracing.TTSAttributes{
		Service:        b.TypeName(),
		Model:          b.meta.Model,
		VoiceID:        b.meta.VoiceID,
		Text:           c.spoken(),
		CharacterCount: &chars,
		Settings:       b.traceSettings(),
	}
	if meter.hadTTFB {
		ttfb := meter.ttfb.Seconds()
		attrs.TTFB = &ttfb
	}
	if meter.hadTTFA {
		// Time to the first audible sample has no key in the GenAI conventions,
		// which measure the response rather than when it starts being heard.
		attrs.Extra = map[string]any{"metrics.ttfa": meter.ttfa.Seconds()}
	}
	tracing.SetTTSAttributes(c.span, attrs)
	tracing.SetTTSUsage(c.spanCtx, b.meta.Model, chars)
	b.emitTiming(ctx, chars, &meter, elapsed)
	c.span.End()
}

// abandonAudioContext closes the span of a context an interruption dropped
// before it was ever played. The contexts queued behind the one being spoken are
// discarded whole when the user cuts in, and without this their spans would stay
// open and never be exported.
func (b *Base) abandonAudioContext(c *audioContext) {
	chars, _, _ := c.measurement()
	tracing.SetTTSAttributes(c.span, tracing.TTSAttributes{
		Service:        b.TypeName(),
		Model:          b.meta.Model,
		VoiceID:        b.meta.VoiceID,
		Text:           c.spoken(),
		CharacterCount: &chars,
		Settings:       b.traceSettings(),
		Extra:          map[string]any{"tts.interrupted": true},
	})
	c.span.End()
}

// pushWord stamps one spoken token with the moment it is heard and hands it to
// the sequencer, which attributes it to the sentence it belongs to and returns
// the frames that follow from it. The transport holds each frame until its
// moment, so an interruption before then leaves the word unspoken. A token
// arriving before any audio has nothing to be timed against yet and waits for
// the first chunk.
func (b *Base) pushWord(ctx context.Context, c *audioContext, contextID string, it ctxItem) {
	pts, timed := c.wordPTS(it.offset)
	if !timed {
		c.holdWord(it)
		return
	}
	b.pushSequencerFrames(ctx,
		b.sequencer.ProcessWord(it.word, int64(pts), contextID, it.includesInterFrame))
}

// pushSequencerFrames emits what the sequencer returned, recording when the last
// word was spoken so a later context times its own words after it.
func (b *Base) pushSequencerFrames(ctx context.Context, out []frames.Frame) {
	for _, f := range out {
		if _, ok := f.(*frames.TTSTextFrame); ok {
			if pts, has := f.Base().PTS(); has {
				b.setLastWordPTS(time.Duration(pts))
			}
		}
		_ = b.PushFrame(ctx, f, processor.Downstream)
	}
}

// now reads the pipeline clock, or zero before the pipeline has one.
func (b *Base) now() time.Duration {
	clk := b.Clock()
	if clk == nil {
		return 0
	}
	return clk.Time()
}

// lastWordPTS is when the last word pushed was spoken, which keeps the timeline
// moving forward across contexts whose audio overlaps.
func (b *Base) lastWordPTS() time.Duration {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.wordLastPTS
}

// setLastWordPTS records when the last word pushed was spoken.
func (b *Base) setLastWordPTS(pts time.Duration) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	if pts > b.wordLastPTS {
		b.wordLastPTS = pts
	}
}

// applyForceComplete closes out a context whose audio has ended, emitting the
// text of any sentence the provider never finished reporting words for and
// releasing the frames that were waiting behind it.
func (b *Base) applyForceComplete(ctx context.Context, contextID string) {
	b.pushSequencerFrames(ctx, b.sequencer.ForceComplete(contextID, int64(b.lastWordPTS())))
}
