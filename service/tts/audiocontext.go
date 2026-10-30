package tts

import (
	"cmp"
	"context"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/gojargo/jargo/expressive/spoken"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/telemetry/tracing"
	uctx "github.com/gojargo/jargo/utils/context"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
)

// defaultStopFrameTimeout is how long an audio context waits for more audio
// before it is taken as finished and its stop frame is pushed.
const defaultStopFrameTimeout = 3 * time.Second

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
	// RotateTurnContext closes the turn's context out and opens a fresh one for
	// the sentences still to come. See Base.RotateTurnContext.
	RotateTurnContext(ctx context.Context)
	// ReplaceConnection closes every open context and moves a turn in progress
	// to a new context ID, for a provider replacing its connection. See
	// Base.ReplaceConnection.
	ReplaceConnection()
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

	// seq is the context's place in creation order.
	seq uint64

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
	// uninterruptible marks a context an interruption does not drop while it
	// waits to play.
	uninterruptible bool
	frame           frames.Frame
	end             bool
	// reached is closed when the drain goroutine reaches the sentinel, or when
	// the queue is torn down before it could.
	reached chan struct{}
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

// reset drops everything queued, for an interruption, except what is
// uninterruptible: frames such as a FunctionCallResultFrame, which must not be
// lost mid-flight, and the contexts of uninterruptible speech. The shutdown
// sentinel is kept too: it is what a graceful end, itself uninterruptible, is
// waiting on.
func (q *serialQueue) reset() {
	q.mu.Lock()
	defer q.mu.Unlock()
	kept := q.items[:0:0]
	for _, it := range q.items {
		switch {
		case it.end,
			it.frame != nil && !frames.Interruptible(it.frame),
			it.contextID != "" && it.uninterruptible:
			kept = append(kept, it)
		}
	}
	q.items = kept
}

// releaseEnds drops the shutdown sentinels still queued and releases whoever is
// waiting on them, for a queue torn down before they were reached.
func (q *serialQueue) releaseEnds() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = slices.DeleteFunc(q.items, func(it serialItem) bool {
		if it.end && it.reached != nil {
			close(it.reached)
		}
		return it.end
	})
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
	b.audioCtxMu.Lock()
	b.audioContexts = map[string]*audioContext{}
	b.ttsContexts = map[string]ttsContext{}
	b.audioRemainders = map[string]audioRemainder{}
	b.serial = newSerialQueue()
	b.playingContext = ""
	b.audioTornDown = false
	b.audioCtxMu.Unlock()
	b.createAudioContextTask(ctx)
}

// createAudioContextTask starts the drain goroutine if it is not running. The
// serialization queue is kept: an interruption restarts the goroutine without
// losing the uninterruptible contexts and frames still queued on it.
func (b *Base) createAudioContextTask(ctx context.Context) {
	if b.ctxCancel != nil {
		return
	}
	loopCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	b.ctxCancel = cancel
	b.ctxWG.Add(1)
	go b.audioContextLoop(loopCtx)
}

// stopAudioContexts tears the drain goroutine down and waits for it. What was
// queued is dropped, which is what a cancel wants; a graceful end goes through
// drainAudioContexts instead, and is released here if it was still waiting.
func (b *Base) stopAudioContexts() {
	if b.stopAudioContextLoop() {
		b.abandonOpenAudioContexts()
	}
	b.audioCtxMu.Lock()
	b.audioTornDown = true
	if b.serial != nil {
		b.serial.releaseEnds()
	}
	b.audioCtxMu.Unlock()
}

// stopAudioContextLoop cancels the drain goroutine and waits for it, leaving the
// contexts it had not finished registered. It reports whether a goroutine was
// running.
func (b *Base) stopAudioContextLoop() bool {
	cancel := b.ctxCancel
	b.ctxCancel = nil
	if cancel == nil {
		return false
	}
	cancel()
	b.ctxWG.Wait()
	return true
}

// interruptAudioContexts drops the interruptible contexts for an interruption,
// while the synthesis and playback of uninterruptible speech carry on.
//
// The drain goroutine is stopped only when what it is playing can be cut off;
// otherwise it goes on playing it. Either way the queue behind it loses
// everything interruptible. The provider is told about each context dropped
// while it is still registered, so a hook that looks the context up still finds
// it, and only then is the context removed. The one being played is among them
// when it was cut off: the drain loop leaves it in place.
func (b *Base) interruptAudioContexts(ctx context.Context) {
	preserved := b.uninterruptibleContexts()
	interruptPlaying := !slices.Contains(preserved, b.playingContextID())
	if interruptPlaying {
		b.stopAudioContextLoop()
		// The next words are timed after the uninterruptible ones still to play.
		b.audioCtxMu.Lock()
		b.wordLastPTS = b.uninterruptibleLastPTS
		b.audioCtxMu.Unlock()
	}

	b.audioCtxMu.Lock()
	if b.serial != nil {
		b.serial.reset()
	}
	b.audioCtxMu.Unlock()
	b.sequencer.Clear(preserved...)

	interrupter, _ := b.syn.(AudioContextInterrupter)
	for _, id := range b.openAudioContexts() {
		if slices.Contains(preserved, id) {
			continue
		}
		if interrupter != nil {
			interrupter.OnAudioContextInterrupted(ctx, id)
		}
		b.removeInterruptedAudioContext(id)
	}

	if interruptPlaying {
		b.setPlayingContextID("")
	}
	if !slices.Contains(preserved, b.turnContextID()) {
		b.setTurnContextID("")
	}
}

// uninterruptibleContexts lists the synthesis contexts an interruption leaves
// in place.
func (b *Base) uninterruptibleContexts() []string {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	var out []string
	for id, c := range b.ttsContexts {
		if !c.interruptible {
			out = append(out, id)
		}
	}
	return out
}

// playingContextID is the context the drain goroutine is playing, or "" when it
// is between contexts.
func (b *Base) playingContextID() string {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.playingContext
}

// setPlayingContextID records the context the drain goroutine is playing.
func (b *Base) setPlayingContextID(id string) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	b.playingContext = id
}

// removeInterruptedAudioContext forgets a context an interruption dropped and
// closes its span as cut off.
func (b *Base) removeInterruptedAudioContext(contextID string) {
	b.audioCtxMu.Lock()
	c := b.audioContexts[contextID]
	delete(b.audioContexts, contextID)
	delete(b.audioRemainders, contextID)
	delete(b.ttsContexts, contextID)
	b.audioCtxMu.Unlock()
	if c != nil {
		b.abandonAudioContext(c)
	}
}

// abandonOpenAudioContexts closes the spans of the contexts still open once the
// drain loop has stopped: the one that was playing and the ones queued behind
// it, which are dropped without being spoken in full. Their spans would
// otherwise stay open and never be exported, so the utterance cut off would
// vanish from the trace instead of being recorded as cut off.
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
// drops what is interruptible and may restart the loop; the sentinel survives
// it, so the wait goes on until the loop running then has reached it too, behind
// the uninterruptible speech still to play.
func (b *Base) drainAudioContexts() {
	b.audioCtxMu.Lock()
	if b.serial == nil || b.audioTornDown {
		b.audioCtxMu.Unlock()
		return
	}
	reached := make(chan struct{})
	b.serial.push(serialItem{end: true, reached: reached})
	b.audioCtxMu.Unlock()
	<-reached
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
	c := newAudioContext(b.syn.SampleRate(), b.BeginTTFB(), spanCtx, span)
	if old, ok := b.audioContexts[contextID]; ok {
		// A context opened again under the same id keeps its place.
		c.seq = old.seq
	} else {
		b.audioCtxSeq++
		c.seq = b.audioCtxSeq
	}
	b.audioContexts[contextID] = c
	serial := b.serial
	// Uninterruptible speech keeps its place in the queue through an
	// interruption. A context nothing was recorded for is interruptible.
	tc, known := b.ttsContexts[contextID]
	uninterruptible := known && !tc.interruptible
	b.audioCtxMu.Unlock()
	if serial != nil {
		serial.push(serialItem{contextID: contextID, uninterruptible: uninterruptible})
	}
}

// AppendToAudioContext adds a frame to an open context.
//
// A provider can take longer than the idle timeout to send any audio, and the
// drain loop then takes its context as finished. Audio that arrives afterwards
// for the turn still being spoken opens the context again, queued behind what
// is already there, rather than being dropped.
func (b *Base) AppendToAudioContext(contextID string, f frames.Frame) {
	if contextID != "" && f != nil && !b.AudioContextAvailable(contextID) && contextID == b.turnContextID() {
		slog.Debug("recreating audio context", "service", b.Name(), "context", contextID)
		b.CreateAudioContext(contextID)
	}
	c := b.audioContextFor(contextID)
	if c == nil {
		return
	}
	// A frame on a context belongs to it. Providers often leave the id unset,
	// and it is what marks the frame as part of uninterruptible speech when it is
	// pushed.
	stampContextID(f, contextID)
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		if !b.alignAudioFrame(contextID, fr) {
			return
		}
	case *frames.TTSStoppedFrame:
		// The context's audio ends here, so a held-back partial sample must be
		// queued before the stop frame that follows it.
		b.flushAudioRemainder(contextID, c)
	}
	c.push(ctxItem{frame: f})
}

// audioRemainder is the bytes of a sample split across audio chunks, with the
// format they were delivered in.
type audioRemainder struct {
	audio       []byte
	sampleRate  int
	numChannels int
}

// alignAudioFrame trims an audio frame to whole samples, carrying a split
// sample forward.
//
// A provider may cut its PCM stream at any byte, so a chunk can end mid-sample.
// The partial sample is held back and put in front of the context's next frame,
// which keeps every queued frame sample-aligned for whatever reads audio frame
// by frame downstream (metrics, resamplers, filters). A frame that is already
// aligned passes through untouched.
//
// It reports false when the frame holds less than one whole sample and is to be
// dropped.
func (b *Base) alignAudioFrame(contextID string, f *frames.TTSAudioRawFrame) bool {
	blockSize := 2 * max(f.NumChannels, 1)
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	remainder, held := b.audioRemainders[contextID]
	delete(b.audioRemainders, contextID)
	if !held && len(f.Audio)%blockSize == 0 {
		return true
	}
	audio := append(append([]byte{}, remainder.audio...), f.Audio...)
	aligned := len(audio) - len(audio)%blockSize
	if aligned < len(audio) {
		if b.audioRemainders == nil {
			b.audioRemainders = map[string]audioRemainder{}
		}
		b.audioRemainders[contextID] = audioRemainder{
			audio:       audio[aligned:],
			sampleRate:  f.SampleRate,
			numChannels: f.NumChannels,
		}
	}
	if aligned == 0 {
		return false
	}
	f.Audio = audio[:aligned]
	return true
}

// flushAudioRemainder queues a context's held-back partial sample, zero-padded
// to a whole one.
func (b *Base) flushAudioRemainder(contextID string, c *audioContext) {
	b.audioCtxMu.Lock()
	remainder, held := b.audioRemainders[contextID]
	delete(b.audioRemainders, contextID)
	b.audioCtxMu.Unlock()
	if !held {
		return
	}
	audio := make([]byte, 2*max(remainder.numChannels, 1))
	copy(audio, remainder.audio)
	f := frames.NewTTSAudioRawFrame(audio, remainder.sampleRate, remainder.numChannels)
	f.ContextID = contextID
	c.push(ctxItem{frame: f})
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
		// The context's audio ends here, so a held-back partial sample must be
		// queued before the end marker.
		b.flushAudioRemainder(contextID, c)
		c.push(ctxItem{end: true})
	}
}

// AudioContextAvailable reports whether the context is still open.
func (b *Base) AudioContextAvailable(contextID string) bool {
	return b.audioContextFor(contextID) != nil
}

// ReplaceConnection closes every open audio context, and moves a turn in
// progress to a new context ID, for a provider replacing its connection.
//
// A provider keeps its context state per connection and does not resume it on a
// new one, so audio still due on the old connection never arrives. Every open
// context is closed (the audio already received still plays), and the rest of a
// turn in progress opens a fresh context with the provider, rather than reusing
// an ID the new connection has never seen, which a provider may drop without an
// error or speak without the settings it was opened with. A provider holding
// one connection across requests calls it before it dials a replacement.
func (b *Base) ReplaceConnection() {
	for _, id := range b.openAudioContexts() {
		b.RemoveAudioContext(id)
	}
	b.audioCtxMu.Lock()
	if b.turnContext != "" {
		b.turnContext = uuid.NewString()
	}
	b.audioCtxMu.Unlock()
}

// RotateTurnContext closes the turn's context out and opens a fresh one for the
// sentences still to come.
//
// It is for a provider whose settings are fixed for the length of a context: a
// voice or a model changed mid-turn takes effect only on a new one, so the turn
// is split in two and the rest of it is spoken in the new voice.
//
// The sentence the old context was still holding is finalized first, so its
// already-heard prefix is still reported. Without that a change landing
// mid-sentence would abandon it before it was promoted, and that prefix would
// go missing from the transcript.
// It runs on the frame goroutine, which is where the turn's context is owned and
// where a settings update is handled.
func (b *Base) RotateTurnContext(ctx context.Context) {
	current := b.turnContextID()
	if current == "" {
		// Nothing is being spoken, so the next turn opens its own context with
		// the new settings already in force.
		return
	}

	b.pushSequencerFrames(ctx, b.sequencer.Finalize(current), current)
	if b.AudioContextAvailable(current) {
		if f, ok := b.syn.(AudioFlusher); ok {
			f.FlushAudio(ctx, current)
		}
	}
	b.setTurnContextID(uuid.NewString())
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
	// interruptible reports whether an interruption may discard this synthesis.
	// Uninterruptible speech goes on being synthesized and played through one,
	// and every frame it produces is marked uninterruptible so the transport
	// keeps it too.
	interruptible bool
	// spoken is the delivery and sounds of the words the provider reports
	// speaking in this context, created with its first expressive unit. Part of
	// expressive conversation (alpha).
	spoken *spoken.SpokenExpression
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

// ttsContextOrDefault returns what a context carries, recording def for it
// first when nothing was. A TTSSpeakFrame records its context before its text
// gets here; the sentences of a model response take the default.
func (b *Base) ttsContextOrDefault(contextID string, def ttsContext) ttsContext {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	if b.ttsContexts == nil {
		b.ttsContexts = map[string]ttsContext{}
	}
	if c, ok := b.ttsContexts[contextID]; ok {
		return c
	}
	b.ttsContexts[contextID] = def
	return def
}

// applyContextInterruptible makes a frame of an uninterruptible context
// uninterruptible too, and notes when the last uninterruptible word is spoken.
func (b *Base) applyContextInterruptible(f frames.Frame) {
	if id := frameContextID(f); id != "" {
		b.audioCtxMu.Lock()
		c, ok := b.ttsContexts[id]
		b.audioCtxMu.Unlock()
		if ok && !c.interruptible {
			f.Base().SetInterruptible(false)
		}
	}
	if _, word := f.(*frames.TTSTextFrame); word && !frames.Interruptible(f) {
		if pts, has := f.Base().PTS(); has && pts != 0 {
			b.audioCtxMu.Lock()
			b.uninterruptibleLastPTS = time.Duration(pts)
			b.audioCtxMu.Unlock()
		}
	}
}

// frameContextID is the synthesis context a frame names, or "" for a frame
// that names none.
func frameContextID(f frames.Frame) string {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		return fr.ContextID
	case *frames.TTSStartedFrame:
		return fr.ContextID
	case *frames.TTSStoppedFrame:
		return fr.ContextID
	case *frames.TTSTextFrame:
		return fr.ContextID
	case *frames.AggregatedTextFrame:
		return fr.ContextID
	case *frames.AggregatedTextProgressFrame:
		return fr.ContextID
	}
	return ""
}

// stampContextID names the context on a frame that belongs to it, when the
// frame does not name one already.
func stampContextID(f frames.Frame, contextID string) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		if fr.ContextID == "" {
			fr.ContextID = contextID
		}
	case *frames.TTSStartedFrame:
		if fr.ContextID == "" {
			fr.ContextID = contextID
		}
	case *frames.TTSStoppedFrame:
		if fr.ContextID == "" {
			fr.ContextID = contextID
		}
	case *frames.TTSTextFrame:
		if fr.ContextID == "" {
			fr.ContextID = contextID
		}
	case *frames.AggregatedTextFrame:
		if fr.ContextID == "" {
			fr.ContextID = contextID
		}
	}
}

// appendsToContext reports what setTTSContext recorded for a context. A context
// nothing was recorded for appends, which is the default everywhere else.
func (b *Base) appendsToContext(contextID string) bool {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	c, ok := b.ttsContexts[contextID]
	return !ok || c.appendToContext
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

// openAudioContexts lists the contexts still open in the order they were
// created, for an interruption.
func (b *Base) openAudioContexts() []string {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	out := make([]string, 0, len(b.audioContexts))
	for id := range b.audioContexts {
		out = append(out, id)
	}
	slices.SortFunc(out, func(x, y string) int {
		return cmp.Compare(b.audioContexts[x].seq, b.audioContexts[y].seq)
	})
	return out
}

// deleteAudioContext forgets a context that has been fully drained.
func (b *Base) deleteAudioContext(contextID string) {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	delete(b.audioContexts, contextID)
	// If audio resumes after a timeout, the context is recreated without the
	// partial sample held here.
	delete(b.audioRemainders, contextID)
}

// audioContextLoop drains the serialization queue, preserving downstream frame
// order: a context is played out in full before whatever was queued behind it.
// It runs until the queue is torn down or the shutdown sentinel is reached,
// which it reports so a graceful end knows the queue has drained.
func (b *Base) audioContextLoop(ctx context.Context) {
	defer b.ctxWG.Done()
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
			if it.reached != nil {
				close(it.reached)
			}
			return
		}
		if it.frame != nil {
			b.stampLLMResponseFrame(it.frame)
			_ = b.PushFrame(ctx, it.frame, processor.Downstream)
			continue
		}
		// A context an interruption removed before it played is skipped.
		if it.contextID == "" || !b.AudioContextAvailable(it.contextID) {
			continue
		}
		b.setPlayingContextID(it.contextID)
		if !b.handleAudioContext(ctx, it.contextID) {
			// Cut off mid-play: the context stays registered so whoever stopped
			// the loop can report it interrupted rather than completed.
			return
		}
		b.deleteAudioContext(it.contextID)
		b.deleteTTSContext(it.contextID)
		if c, ok := b.syn.(AudioContextCompleter); ok {
			c.OnAudioContextCompleted(ctx, it.contextID)
		}
		b.setPlayingContextID("")
		b.releaseFramesIfAudioContextsEnded()
	}
}

// handleAudioContext pushes one context's frames downstream until the context is
// marked finished, or until it has been idle long enough to be taken as
// finished on its own. It reports false when the loop was canceled before the
// context ended.
func (b *Base) handleAudioContext(ctx context.Context, contextID string) bool {
	c := b.audioContextFor(contextID)
	if c == nil {
		return true
	}
	shouldPushStop := false
	// Whether any audio arrived for this context at all, which is what says the
	// provider actually spoke rather than accepting the request in silence.
	receivedAudio := false
	for {
		it, ok, timedOut := c.get(ctx, b.stopFrameTimeout)
		if timedOut {
			break
		}
		if !ok {
			return false
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
	b.finishAudioContext(ctx, c)
	b.recordContextAudioOutcome(ctx, contextID, receivedAudio)
	return true
}

// stampLLMResponseFrame sends the end of a model response after its words. The
// start of a response records where its words begin; the end is stamped with
// the moment its last word is spoken, so it lands behind that word rather than
// ahead of it. A response that timed no words leaves the end unstamped, right
// behind its start.
func (b *Base) stampLLMResponseFrame(f frames.Frame) {
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		b.audioCtxMu.Lock()
		b.replyStartPTS = b.wordLastPTS
		b.audioCtxMu.Unlock()
	case *frames.LLMFullResponseEndFrame:
		if !b.wordPath() {
			return
		}
		b.audioCtxMu.Lock()
		last, start := b.wordLastPTS, b.replyStartPTS
		b.audioCtxMu.Unlock()
		if last != start {
			fr.SetPTS(int64(last))
		}
	}
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
	b.pushWordFrames(ctx,
		b.sequencer.ProcessWord(it.word, int64(pts), contextID, it.includesInterFrame))
}

// pushSequencerFrames emits what the sequencer returned for text registered on a
// context, in audio order.
//
// These frames (skipped text the spoken text before it has released and,
// streaming tokens, a promoted sentence and the words replayed against it)
// follow text queued in a context whose audio and text the drain loop emits.
// They go through that context's queue, so one goroutine emits everything in
// order; pushed from here they would overtake it. With the context gone, they
// are pushed directly.
func (b *Base) pushSequencerFrames(ctx context.Context, out []frames.Frame, contextID string) {
	for _, f := range out {
		if contextID != "" && b.AudioContextAvailable(contextID) {
			b.AppendToAudioContext(contextID, f)
			continue
		}
		_ = b.PushFrame(ctx, f, processor.Downstream)
	}
}

// pushWordFrames emits the word frames the sequencer built, recording when the
// last word was spoken so a later context times its own words after it.
func (b *Base) pushWordFrames(ctx context.Context, out []frames.Frame) {
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
	b.pushWordFrames(ctx, b.sequencer.ForceComplete(contextID, int64(b.lastWordPTS())))
}
