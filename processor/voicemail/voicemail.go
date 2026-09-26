// Package voicemail detects whether an outbound call reached a person or went to
// voicemail.
//
// A bot that places a call needs to know whether a person answered or the call
// went to voicemail. The Detector listens to what the other side says and asks
// a classifier; until it has an answer, its TTSGate holds the bot's speech back
// so a voicemail greeting is never talked over.
//
// Any classifier will do. A classifier/llm classifier needs a service that runs
// a one-shot inference, so a realtime LLM cannot back the detector.
package voicemail

import (
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/classifier"
	llmclassifier "github.com/gojargo/jargo/classifier/llm"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/notify"
)

// The events a Detector raises. Each handler receives the detector as its
// source, which it can push frames through.
const (
	// EventConversationDetected is raised when a person answered.
	EventConversationDetected = "on_conversation_detected"
	// EventVoicemailDetected is raised when the call went to voicemail and the
	// greeting has been quiet for VoicemailResponseDelay.
	EventVoicemailDetected = "on_voicemail_detected"
)

const (
	defaultVoicemailResponseDelay = 2 * time.Second
	defaultDecisionTimeout        = time.Second

	verdictConversation = "conversation"
	verdictVoicemail    = "voicemail"
	// questionName is the name the question is asked, and answered, under.
	questionName = "voicemail"
)

// errNoClassifier reports a detector given nothing to decide with.
var errNoClassifier = errors.New("voicemail: a detector needs a classifier")

// Question is the question put to the classifier, with the transcript so far as
// the state.
//
//nolint:gochecknoglobals // the question the detector asks, fixed
var Question = classifier.ChoiceQuestion{
	Instructions: "A bot has placed an outbound phone call. This is what was heard after the " +
		"call connected. Decide whether a person answered or the call went to " +
		"voicemail.",
	Options: []classifier.Option{
		{
			Name: verdictConversation,
			Description: "a person answered: a greeting such as 'hello?', 'hi', 'yeah?' or " +
				"'John speaking'; a question to the caller such as 'who is this?' or " +
				"'can I help you?'; spontaneous speech that expects a reply",
		},
		{
			Name: verdictVoicemail,
			Description: "an automated greeting or carrier message: 'you've reached', 'leave a " +
				"message', 'I'm not available right now', 'call me back', 'mailbox is " +
				"full', 'not in service', 'all circuits are busy', 'our office is " +
				"currently closed'",
		},
	},
}

// Config configures a Detector.
type Config struct {
	// Classifier decides between a person and a voicemail. It is asked a choice
	// question with the transcript so far. Required, unless LLM is given.
	Classifier classifier.Classifier
	// VoicemailResponseDelay is how long the greeting must be quiet after a
	// voicemail verdict before EventVoicemailDetected is raised, so the message
	// is left after the greeting ends and the recording starts; zero uses 2s.
	VoicemailResponseDelay time.Duration `validate:"min=0"`
	// DecisionTimeout is how long the caller must be quiet after they stop
	// speaking before the latest answer decides; zero uses 1s. A greeting
	// resumes after its pauses while a person stays quiet, so a verdict on a
	// fragment such as "hi, this is Sam" is not acted on until the caller has
	// really stopped.
	DecisionTimeout time.Duration `validate:"min=0"`

	// LLM is a service to build the classifier from, when Classifier is nil.
	//
	// Deprecated: build a classifier/llm classifier and pass it as Classifier.
	LLM llm.Inferencer
	// CustomSystemPrompt is put ahead of the classifier's own instructions for
	// the LLM.
	//
	// Deprecated: build a classifier/llm classifier with the instructions you
	// want and pass it as Classifier.
	CustomSystemPrompt string
}

// Detector decides whether a person answered an outbound call or it went to
// voicemail.
//
// Placed after the STT service, it passes every frame through and collects what
// the other side says. After each transcription it asks its classifier, in the
// background, whether the call reached a person or a voicemail, and keeps asking
// as more is said. Once the caller has been quiet for DecisionTimeout, the
// latest answer decides and the detector acts:
//
//   - conversation: the bot's held-back speech is released and the call goes on
//     as normal, and EventConversationDetected is raised.
//   - voicemail: the held-back speech is dropped, the pipeline is interrupted,
//     no further input reaches the conversation, and EventVoicemailDetected is
//     raised once the greeting has finished, so the handler can leave a message.
//
// The pipeline places the detector after the STT service and its gate after
// the TTS service:
//
//	detector, err := voicemail.New(voicemail.Config{Classifier: c})
//	events.OnSignal(detector.Events(), voicemail.EventVoicemailDetected, func(ctx context.Context) {
//		_ = detector.PushFrame(ctx, frames.NewTTSSpeakFrame("Please call me back."), processor.Downstream)
//	})
//	pipe := pipeline.New(in, stt, detector.Detector(), userAgg, llm, tts, detector.Gate(), out, assistantAgg)
type Detector struct {
	*processor.Base
	cfg        Config
	classifier classifier.Classifier
	gate       *TTSGate

	conversation *notify.EventNotifier
	voicemail    *notify.EventNotifier

	// segments is what was heard, queued for the classifying goroutine, which
	// owns the transcript and the answer.
	segments *segmentQueue

	// ctx is what the detector's own goroutines run under.
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup

	mu         sync.Mutex
	transcript []string
	lastResult *classifier.ChoiceResult
	decision   string
	// Two silence timers, each restarted by speech. The decision timer acts on
	// the latest answer once the caller has stopped; after a voicemail verdict,
	// the message timer raises the event once the greeting has.
	userSpeaking  bool
	decisionTimer context.CancelFunc
	messageTimer  context.CancelFunc
	messageLeft   bool
}

// New builds a voicemail detector.
func New(cfg Config) (*Detector, error) {
	if cfg.VoicemailResponseDelay == 0 {
		cfg.VoicemailResponseDelay = defaultVoicemailResponseDelay
	}
	if cfg.DecisionTimeout == 0 {
		cfg.DecisionTimeout = defaultDecisionTimeout
	}
	c := cfg.Classifier
	if c == nil {
		if cfg.LLM == nil {
			return nil, errNoClassifier
		}
		slog.Warn("voicemail: Config.LLM is deprecated, pass a classifier instead")
		// The classifier's own instructions, which ask for a JSON object, come
		// last and win.
		var instructions string
		if cfg.CustomSystemPrompt != "" {
			instructions = cfg.CustomSystemPrompt + "\n\n" + llmclassifier.DefaultInstructions
		}
		built, err := llmclassifier.New(llmclassifier.Config{LLM: cfg.LLM, Instructions: instructions})
		if err != nil {
			return nil, err
		}
		c = built
	}
	d := &Detector{
		cfg:          cfg,
		classifier:   c,
		conversation: notify.NewEventNotifier(),
		voicemail:    notify.NewEventNotifier(),
		segments:     newSegmentQueue(),
	}
	d.Base = processor.New("VoicemailDetector", d)
	d.gate = NewTTSGate(d.conversation, d.voicemail)
	d.Events().Register(EventConversationDetected, false)
	d.Events().Register(EventVoicemailDetected, false)
	events.On(c.Events(), classifier.EventMetrics, d.onClassifierMetrics)
	return d, nil
}

// Detector is the processor to place after the STT service: the detector itself.
func (d *Detector) Detector() *Detector { return d }

// Gate is the processor to place after the TTS service, holding speech until
// the decision is made.
func (d *Detector) Gate() *TTSGate { return d.gate }

// Classifier is what decides between a person and a voicemail.
func (d *Detector) Classifier() classifier.Classifier { return d.classifier }

// Setup sets up the processor and its classifier, and starts classifying.
func (d *Detector) Setup(ctx context.Context, s processor.Setup) error {
	if err := d.Base.Setup(ctx, s); err != nil {
		return err
	}
	if err := d.classifier.Setup(ctx); err != nil {
		return err
	}
	d.ctx, d.cancel = context.WithCancel(ctx)
	d.wg.Go(func() { d.classifySegments(d.ctx) })
	return nil
}

// Cleanup stops the classifying and the timers, and cleans up the classifier.
func (d *Detector) Cleanup(ctx context.Context) error {
	err := d.Base.Cleanup(ctx)
	d.mu.Lock()
	d.cancelDecisionTimerLocked()
	d.cancelMessageTimerLocked()
	d.mu.Unlock()
	if d.cancel != nil {
		d.cancel()
	}
	d.wg.Wait()
	return errors.Join(err, d.classifier.Cleanup(ctx))
}

// ProcessFrame collects transcriptions, keeps the silence timers, and holds
// everything but the frames that end or control the pipeline back after a
// voicemail verdict.
func (d *Detector) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := d.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	d.mu.Lock()
	switch fr := f.(type) {
	case *frames.TranscriptionFrame:
		if text := strings.TrimSpace(fr.Text); text != "" {
			d.segments.put(text)
			// A transcription often lands after the caller has stopped; the
			// silence timer restarts from it.
			if !d.userSpeaking && d.decision == "" {
				d.restartDecisionTimerLocked()
			}
		}
	case *frames.UserStartedSpeakingFrame:
		d.userSpeaking = true
		d.cancelDecisionTimerLocked()
		d.cancelMessageTimerLocked()
	case *frames.UserStoppedSpeakingFrame:
		d.userSpeaking = false
		switch {
		case d.decision == "":
			d.restartDecisionTimerLocked()
		case d.decision == verdictVoicemail && !d.messageLeft:
			d.restartMessageTimerLocked()
		}
	}
	voicemail := d.decision == verdictVoicemail
	d.mu.Unlock()

	// After a voicemail verdict nothing more should reach the conversation, only
	// the frames that end or control the pipeline.
	if voicemail && !passesClosedGate(f) {
		return nil
	}
	return d.PushFrame(ctx, f, dir)
}

// passesClosedGate reports whether a frame still flows after a voicemail
// verdict: the lifecycle frames that end or control the pipeline.
func passesClosedGate(f frames.Frame) bool {
	switch f.(type) {
	case frames.SystemFrame, *frames.EndFrame, *frames.StopFrame, frames.WorkerFrame:
		return true
	default:
		return false
	}
}

// classifySegments classifies the transcript every time it grows. Segments
// that arrive during a classification are taken together, so a burst of
// transcriptions costs one call on the full transcript.
func (d *Detector) classifySegments(ctx context.Context) {
	for {
		segments, ok := d.segments.take(ctx)
		if !ok {
			return
		}
		d.mu.Lock()
		d.transcript = append(d.transcript, segments...)
		transcript := strings.Join(d.transcript, " ")
		decided := d.decision != ""
		d.mu.Unlock()
		// Nothing to ask once the verdict is in.
		if !decided {
			d.classify(ctx, transcript)
		}
		// Lets the silence timer know the transcript is fully classified.
		d.segments.done(len(segments))
	}
}

// classify asks the classifier about the transcript and keeps its answer.
func (d *Detector) classify(ctx context.Context, transcript string) {
	// Any failure is one missed answer; the silence timer must never be left
	// waiting on a classification that will not come.
	results, err := d.classifier.Choice(ctx, transcript,
		[]classifier.Named[classifier.ChoiceQuestion]{{Name: questionName, Question: Question}})
	if err != nil {
		slog.WarnContext(ctx, "voicemail: classification failed", "detector", d.Name(), "error", err)
		return
	}
	result := results[questionName]
	slog.DebugContext(ctx, "voicemail: classified", "detector", d.Name(),
		"choice", result.Choice, "confidence", result.Confidence, "transcript", transcript)
	// No verdict acts from here: "hi, this is Sam" is what a person says and how
	// a greeting starts, and only the silence that follows tells them apart, so
	// the silence timer decides.
	d.mu.Lock()
	d.lastResult = &result
	d.mu.Unlock()
}

// restartDecisionTimerLocked starts the decision timer over. The caller holds mu.
func (d *Detector) restartDecisionTimerLocked() {
	d.cancelDecisionTimerLocked()
	if d.ctx == nil {
		return
	}
	ctx, cancel := context.WithCancel(d.ctx)
	d.decisionTimer = cancel
	d.wg.Go(func() { d.decideAfterSilence(ctx) })
}

// cancelDecisionTimerLocked stops the decision timer, if it is running. The
// caller holds mu.
func (d *Detector) cancelDecisionTimerLocked() {
	if d.decisionTimer != nil {
		d.decisionTimer()
		d.decisionTimer = nil
	}
}

// decideAfterSilence acts on the latest answer once the caller has been quiet
// long enough. A greeting goes on after its pauses; a person stops and waits. So
// this is how every verdict comes to act.
func (d *Detector) decideAfterSilence(ctx context.Context) {
	if !sleep(ctx, d.cfg.DecisionTimeout) {
		return
	}
	// The answer for everything heard so far, once a call in flight is done.
	// This cannot hang: the classifier answers or fails within a timeout of its
	// own, and every segment is marked done either way.
	if !d.segments.join(ctx) {
		return
	}
	d.mu.Lock()
	if d.decision != "" || ctx.Err() != nil {
		d.mu.Unlock()
		return
	}
	verdict := verdictConversation
	if d.lastResult != nil {
		slog.DebugContext(ctx, "voicemail: deciding after silence", "detector", d.Name(),
			"choice", d.lastResult.Choice, "confidence", d.lastResult.Confidence, "after", d.cfg.DecisionTimeout)
		if d.lastResult.Choice == verdictVoicemail {
			verdict = verdictVoicemail
		}
	} else {
		slog.WarnContext(ctx, "voicemail: no answer from the classifier after the silence, assuming a conversation",
			"detector", d.Name(), "after", d.cfg.DecisionTimeout)
	}
	d.decision = verdict
	d.mu.Unlock()
	d.decide(verdict)
}

// decide acts on the verdict: releases the held speech, or drops it.
func (d *Detector) decide(verdict string) {
	ctx := d.ctx
	if verdict == verdictVoicemail {
		slog.DebugContext(ctx, "voicemail: VOICEMAIL detected", "detector", d.Name())
		d.voicemail.Notify()
		if err := d.BroadcastInterruption(ctx); err != nil {
			slog.WarnContext(ctx, "voicemail: interrupting failed", "detector", d.Name(), "error", err)
		}
		d.mu.Lock()
		d.restartMessageTimerLocked()
		d.mu.Unlock()
		return
	}
	slog.DebugContext(ctx, "voicemail: CONVERSATION detected", "detector", d.Name())
	d.conversation.Notify()
	d.Events().Call(ctx, EventConversationDetected, d)
}

// restartMessageTimerLocked starts the message timer over. The caller holds mu.
func (d *Detector) restartMessageTimerLocked() {
	d.cancelMessageTimerLocked()
	if d.ctx == nil {
		return
	}
	ctx, cancel := context.WithCancel(d.ctx)
	d.messageTimer = cancel
	d.wg.Go(func() { d.leaveMessageAfterQuiet(ctx) })
}

// cancelMessageTimerLocked stops the message timer, if it is running. The
// caller holds mu.
func (d *Detector) cancelMessageTimerLocked() {
	if d.messageTimer != nil {
		d.messageTimer()
		d.messageTimer = nil
	}
}

// leaveMessageAfterQuiet raises EventVoicemailDetected once the greeting has
// been quiet for the delay. It fires once: a beep or a prompt heard while the
// message is being left does not start it over.
func (d *Detector) leaveMessageAfterQuiet(ctx context.Context) {
	if !sleep(ctx, d.cfg.VoicemailResponseDelay) {
		return
	}
	d.mu.Lock()
	if d.messageLeft || ctx.Err() != nil {
		d.mu.Unlock()
		return
	}
	d.messageLeft = true
	d.mu.Unlock()
	d.Events().Call(ctx, EventVoicemailDetected, d)
}

// onClassifierMetrics pushes the classifier's metrics into the pipeline.
func (d *Detector) onClassifierMetrics(ctx context.Context, data []frames.MetricsData) {
	if err := d.PushFrame(ctx, frames.NewMetricsFrame(data...), processor.Downstream); err != nil {
		slog.DebugContext(ctx, "voicemail: pushing classifier metrics failed", "error", err)
	}
}

// sleep waits for d, reporting false when ctx ends first.
func sleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// segmentQueue is what was heard, waiting to be classified. join waits until
// every segment put has been marked done.
type segmentQueue struct {
	mu         sync.Mutex
	items      []string
	unfinished int
	// changed is closed and replaced whenever the queue changes, waking the
	// goroutines waiting on it.
	changed chan struct{}
}

func newSegmentQueue() *segmentQueue { return &segmentQueue{changed: make(chan struct{})} }

// put queues a segment.
func (q *segmentQueue) put(s string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.items = append(q.items, s)
	q.unfinished++
	q.signalLocked()
}

// take waits for a segment and takes it together with any that arrived
// meanwhile, reporting false once ctx ends.
func (q *segmentQueue) take(ctx context.Context) ([]string, bool) {
	for {
		q.mu.Lock()
		if len(q.items) > 0 {
			items := q.items
			q.items = nil
			q.mu.Unlock()
			return items, true
		}
		changed := q.changed
		q.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return nil, false
		}
	}
}

// done marks n segments as classified.
func (q *segmentQueue) done(n int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.unfinished -= n
	q.signalLocked()
}

// join waits until every segment put has been marked done, reporting false once
// ctx ends.
func (q *segmentQueue) join(ctx context.Context) bool {
	for {
		q.mu.Lock()
		if q.unfinished <= 0 {
			q.mu.Unlock()
			return true
		}
		changed := q.changed
		q.mu.Unlock()
		select {
		case <-changed:
		case <-ctx.Done():
			return false
		}
	}
}

func (q *segmentQueue) signalLocked() {
	close(q.changed)
	q.changed = make(chan struct{})
}

// TTSGate holds the bot's speech until the voicemail decision is made.
//
// Placed right after the TTS service. TTS frames are buffered while the decision
// is pending; every other frame passes through. A conversation verdict releases
// the buffered frames in order, a voicemail verdict discards them, since they
// were meant for a person.
type TTSGate struct {
	*processor.Base
	conversation notify.Notifier
	voicemail    notify.Notifier

	cancel context.CancelFunc
	wg     sync.WaitGroup

	// mu is held while the buffer is released, so a frame arriving meanwhile
	// waits and goes out after the ones held ahead of it.
	mu     sync.Mutex
	buffer []heldFrame
	gating bool
}

// heldFrame is a frame the gate holds, with the direction it was going.
type heldFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

// NewTTSGate builds a gate released by conversation and emptied by voicemail.
func NewTTSGate(conversation, voicemail notify.Notifier) *TTSGate {
	g := &TTSGate{conversation: conversation, voicemail: voicemail, gating: true}
	g.Base = processor.New("TTSGate", g)
	return g
}

// Setup sets up the processor and starts waiting for the verdict.
func (g *TTSGate) Setup(ctx context.Context, s processor.Setup) error {
	if err := g.Base.Setup(ctx, s); err != nil {
		return err
	}
	ctx, g.cancel = context.WithCancel(ctx)
	g.wg.Go(func() { g.waitForConversation(ctx) })
	g.wg.Go(func() { g.waitForVoicemail(ctx) })
	return nil
}

// Cleanup stops waiting for the verdict.
func (g *TTSGate) Cleanup(ctx context.Context) error {
	err := g.Base.Cleanup(ctx)
	if g.cancel != nil {
		g.cancel()
	}
	g.wg.Wait()
	return err
}

// ProcessFrame buffers TTS frames while the decision is pending, and passes the
// rest.
func (g *TTSGate) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := g.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	g.mu.Lock()
	if g.gating && isTTS(f) {
		g.buffer = append(g.buffer, heldFrame{frame: f, dir: dir})
		g.mu.Unlock()
		return nil
	}
	g.mu.Unlock()
	return g.PushFrame(ctx, f, dir)
}

// isTTS reports whether a frame is the bot's speech.
func isTTS(f frames.Frame) bool {
	switch f.(type) {
	case *frames.TTSStartedFrame, *frames.TTSStoppedFrame, *frames.TTSTextFrame, *frames.TTSAudioRawFrame:
		return true
	default:
		return false
	}
}

func (g *TTSGate) waitForConversation(ctx context.Context) {
	if !g.conversation.Wait(ctx) {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.gating = false
	for _, h := range g.buffer {
		if err := g.PushFrame(ctx, h.frame, h.dir); err != nil {
			slog.DebugContext(ctx, "voicemail: releasing held speech failed", "error", err)
		}
	}
	g.buffer = nil
}

func (g *TTSGate) waitForVoicemail(ctx context.Context) {
	if !g.voicemail.Wait(ctx) {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.gating = false
	g.buffer = nil
}

// Compile-time interface checks.
var (
	_ processor.Processor = (*Detector)(nil)
	_ processor.Processor = (*TTSGate)(nil)
)
