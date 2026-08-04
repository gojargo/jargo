package tts

import (
	"context"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/tracing"
	uctx "github.com/gojargo/jargo/utils/context"
	"go.opentelemetry.io/otel/attribute"
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
	offset    float64
	isWord    bool
	keepalive bool
	end       bool
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

func newAudioContext(rate int, spanCtx context.Context, span trace.Span) *audioContext {
	return &audioContext{
		notify:               make(chan struct{}, 1),
		start:                time.Now(),
		rate:                 rate,
		meter:                ttfaMeter{rate: rate},
		span:                 span,
		spanCtx:              spanCtx,
		initialWordTimestamp: -1,
	}
}

// addChars records the characters of a sentence sent on this context. Providers
// bill per character.
func (c *audioContext) addChars(n int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.chars += n
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
	spanCtx, span := tracing.Tracer().Start(b.Tracing().Parent(context.Background()), "tts")
	span.SetAttributes(
		attribute.String("tts.service", b.Name()),
		attribute.Int("tts.sample_rate", b.syn.SampleRate()),
		attribute.String("gen_ai.output.type", "speech"),
	)
	if b.meta.VoiceID != "" {
		span.SetAttributes(attribute.String("gen_ai.request.voice", b.meta.VoiceID))
	}
	b.audioCtxMu.Lock()
	if b.audioContexts == nil {
		b.audioContexts = map[string]*audioContext{}
	}
	b.audioContexts[contextID] = newAudioContext(b.syn.SampleRate(), spanCtx, span)
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
	for _, w := range words {
		b.AppendWordToAudioContext(contextID, w.Word, w.Offset)
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

// audioContextFor looks a context up, or nil when it is not open.
func (b *Base) audioContextFor(contextID string) *audioContext {
	if contextID == "" {
		return nil
	}
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.audioContexts[contextID]
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
		case *frames.TTSAudioRawFrame:
			// The first chunk of audio is what the context's words are timed
			// from: it is the point the audio starts being heard.
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
		_ = b.PushFrame(ctx, frames.NewTTSStoppedFrame(), processor.Downstream)
	}
	b.finishAudioContext(ctx, c)
}

// finishAudioContext records what the context cost and closes its span.
func (b *Base) finishAudioContext(ctx context.Context, c *audioContext) {
	chars, meter, elapsed := c.measurement()
	c.span.SetAttributes(attribute.Int("tts.chars", chars))
	if meter.hadTTFB {
		c.span.SetAttributes(attribute.Int64("tts.ttfb_ms", meter.ttfb.Milliseconds()))
	}
	if meter.hadTTFA {
		c.span.SetAttributes(attribute.Int64("tts.ttfa_ms", meter.ttfa.Milliseconds()))
	}
	tracing.SetTTSUsage(c.spanCtx, b.meta.Model, chars)
	b.emitTiming(ctx, chars, &meter, elapsed)
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
	b.pushSequencerFrames(ctx, b.sequencer.ProcessWord(it.word, int64(pts), contextID, false))
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
