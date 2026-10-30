package turns

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strings"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

const (
	defaultClassifierShortTimeout          = 5 * time.Second
	defaultClassifierLongTimeout           = 10 * time.Second
	defaultClassifierClassificationTimeout = time.Second
)

// turnCompletionQuestionName is the name the turn-completion question is asked
// under.
const turnCompletionQuestionName = "turn"

// The turn-completion verdicts, as the LLM's markers report them and as the
// classifier's options name them.
const (
	kindComplete = "complete"
	kindShort    = "short"
	kindLong     = "long"
)

// TurnCompletionQuestion is the question ClassifierStop puts to its classifier.
// The state is the user's turn so far and, when the strategy has the
// conversation, the assistant's last message.
//
//nolint:gochecknoglobals // a fixed question, as the classifier is given it
var TurnCompletionQuestion = classifier.ChoiceQuestion{
	Instructions: "The user is talking to a voice assistant. Their words come from speech " +
		"recognition, without punctuation, and may have been cut off. Decide whether " +
		"the user's turn is complete. Complete means conversationally complete, not " +
		"long: one word can be a complete answer, a question is complete, a " +
		"correction is complete.",
	Options: []classifier.Option{
		{Name: kindComplete, Description: "the user has taken their turn and the assistant should answer"},
		{Name: kindShort, Description: "the user stopped mid-sentence and will continue in a few seconds: " +
			"the last words leave a phrase open, such as ending on a conjunction, a preposition, " +
			"an article, or an unfinished list or number"},
		{Name: kindLong, Description: "the user needs time to think or asked the assistant to wait, or has " +
			"only acknowledged the question without answering it"},
	},
}

// ClassifierStopConfig configures a ClassifierStop strategy.
type ClassifierStopConfig struct {
	// Classifier decides whether the turn is complete. Required.
	Classifier classifier.Classifier `validate:"required"`
	// Context is the conversation, so the classifier sees the assistant's last
	// message along with the user's turn: whether "that's interesting" is an
	// answer or an acknowledgement depends on what was asked. Nil leaves it
	// out.
	Context *frames.LLMContext
	// ShortTimeout is how long to wait after a "short" verdict before
	// reporting that the user is done; 0 uses 5s.
	ShortTimeout time.Duration `validate:"min=0"`
	// LongTimeout is how long to wait after a "long" verdict before reporting
	// that the user is done; 0 uses 10s.
	LongTimeout time.Duration `validate:"min=0"`
	// ClassificationTimeout is how long to wait for the classifier before
	// reporting without it; 0 uses 1s.
	ClassificationTimeout time.Duration `validate:"min=0"`
}

// ClassifierStop decides the user is done when a classifier says their turn is
// complete.
//
// Each time the user pauses with new words transcribed, the strategy asks its
// classifier whether the turn so far is conversationally complete:
//
//   - "complete": it reports that the user is done.
//   - "short": the user was cut off mid-sentence. It reports after ShortTimeout
//     unless they continue.
//   - "long": the user asked for time or only acknowledged the question. It
//     reports after LongTimeout unless they continue.
//
// It asks once the pause's words are in: when the transcription service
// finalizes their transcript, or when its p99 latency has passed since the user
// stopped. Asking sooner could judge the turn without its last words.
//
// If the classifier fails or takes longer than ClassificationTimeout, it
// reports right away, so a turn never waits on a classifier that cannot answer.
// A pause with no words transcribed is not classified, and the controller's
// stop timeout ends such a turn.
//
// Next to a turn analyzer such as TurnAnalyzerStop, in a StrategyModeAll list,
// the default, the turn ends once both have decided: the analyzer hears that
// more is coming after words that read complete, but holds a complete turn
// until it decides too. As the only stop strategy, it ends a turn as soon as
// the classifier finds it complete. In a StrategyModeFirst list the classifier
// can only end a turn sooner.
//
// ClassifierUserTurnStrategies adds it after the default stop strategies.
type ClassifierStop struct {
	StopStrategyBase
	classifier            classifier.Classifier
	convo                 *frames.LLMContext
	shortTimeout          time.Duration
	longTimeout           time.Duration
	classificationTimeout time.Duration

	// ctx bounds the classifications, from Setup to Cleanup.
	ctx       context.Context //nolint:containedctx // the strategy's lifetime, held for its goroutines
	cancelCtx context.CancelFunc
	metricsID events.HandlerID
	hasMetric bool

	turnOpen     bool
	userSpeaking bool
	text         string
	// classifiedText is the turn text last given to the classifier, so a pause
	// without new words does not ask again. hasClassified records that there
	// is one.
	classifiedText string
	hasClassified  bool
	sttTimeout     time.Duration
	txFinalized    bool
	// classification counts the classifications started, so a verdict that
	// lands after a newer one started, or after it was canceled, is dropped.
	classification   uint64
	cancelClassify   context.CancelFunc
	cancelWait       func()
	cancelSTTWait    func()
	sttWaitScheduled bool
}

// NewClassifierStop builds a stop strategy that asks a classifier whether the
// turn is complete. It fails on a configuration without a classifier.
func NewClassifierStop(cfg ClassifierStopConfig) (*ClassifierStop, error) {
	if err := validate.Struct(cfg); err != nil {
		return nil, fmt.Errorf("turns: classifier stop: %w", err)
	}
	s := &ClassifierStop{
		classifier:            cfg.Classifier,
		convo:                 cfg.Context,
		shortTimeout:          orDuration(cfg.ShortTimeout, defaultClassifierShortTimeout),
		longTimeout:           orDuration(cfg.LongTimeout, defaultClassifierLongTimeout),
		classificationTimeout: orDuration(cfg.ClassificationTimeout, defaultClassifierClassificationTimeout),
	}
	s.EnableUserSpeakingFrames = true
	return s, nil
}

// orDuration is d, or def when d is zero.
func orDuration(d, def time.Duration) time.Duration {
	if d == 0 {
		return def
	}
	return d
}

// Classifier is the classifier that decides whether a turn is complete.
func (s *ClassifierStop) Classifier() classifier.Classifier { return s.classifier }

// DecisionScheduled reports whether a "short" or "long" wait is running, which
// decides when it ends.
func (s *ClassifierStop) DecisionScheduled() bool { return s.cancelWait != nil }

// Setup sets the classifier up and passes on the metrics of each of its calls.
func (s *ClassifierStop) Setup(processor.Setup) error {
	s.env.locked(func() {
		s.ctx, s.cancelCtx = context.WithCancel(context.Background())
		s.metricsID = events.On(s.classifier.Events(), classifier.EventMetrics,
			func(_ context.Context, data []frames.MetricsData) {
				s.env.locked(func() { s.Push(frames.NewMetricsFrame(data...), processor.Downstream) })
			})
		s.hasMetric = true
	})
	if err := s.classifier.Setup(s.ctx); err != nil {
		return fmt.Errorf("turns: setting up the classifier: %w", err)
	}
	return nil
}

// Cleanup cancels what is pending and cleans the classifier up. The controller
// calls it with its lock held.
func (s *ClassifierStop) Cleanup() {
	s.cancelPending()
	s.cancelSTTTimer()
	if s.hasMetric {
		s.classifier.Events().Remove(classifier.EventMetrics, s.metricsID)
		s.hasMetric = false
	}
	if s.cancelCtx != nil {
		s.cancelCtx()
	}
	if err := s.classifier.Cleanup(context.Background()); err != nil {
		slog.Warn("turns: cleaning up the classifier failed", "err", err)
	}
}

// TurnStarted starts a fresh turn with no text.
func (s *ClassifierStop) TurnStarted() {
	s.resetTurn()
	s.turnOpen = true
}

// TurnStopped forgets the turn and any pending decision.
func (s *ClassifierStop) TurnStopped() {
	s.resetTurn()
	s.turnOpen = false
}

func (s *ClassifierStop) resetTurn() {
	s.cancelPending()
	s.cancelSTTTimer()
	s.text = ""
	s.classifiedText, s.hasClassified = "", false
	s.txFinalized = false
}

// Process collects the turn's text and classifies it once the user pauses and
// the words are in.
func (s *ClassifierStop) Process(f frames.Frame) ProcessFrameResult {
	switch fr := f.(type) {
	case *frames.STTMetadataFrame:
		s.sttTimeout = fr.TTFSP99Latency
	case *frames.VADUserStartedSpeakingFrame, *frames.UserStartedSpeakingFrame,
		*frames.ProposedUserStartedSpeakingFrame:
		s.handleSpeechStarted()
	case *frames.VADUserStoppedSpeakingFrame:
		s.handleVADSpeechStopped(fr)
	case *frames.UserStoppedSpeakingFrame, *frames.ProposedUserStoppedSpeakingFrame:
		// These stops come after the transcript of the user's words. The VAD's
		// comes before it.
		s.handleSpeechStopped(0)
	case *frames.InterimTranscriptionFrame:
		// More of the user's words are on their way.
		s.txFinalized = false
	case *frames.TranscriptionFrame:
		s.handleTranscription(fr)
	}
	return Continue
}

// handleSpeechStarted readies a new classification. The controller clears the
// stop decisions on speech, so the turn is classified again at the next pause,
// even without new words. A verdict still on its way is about words the user
// has gone past.
func (s *ClassifierStop) handleSpeechStarted() {
	s.userSpeaking = true
	s.classifiedText, s.hasClassified = "", false
	s.txFinalized = false
	s.cancelPending()
	s.cancelSTTTimer()
}

// handleVADSpeechStopped waits the STT's p99 latency, which counts from when the
// user stopped speaking: the VAD reports that StopSecs later.
func (s *ClassifierStop) handleVADSpeechStopped(fr *frames.VADUserStoppedSpeakingFrame) {
	stopped := fr.Timestamp
	if stopped.IsZero() {
		stopped = time.Now()
	}
	deadline := stopped.Add(-time.Duration(fr.StopSecs * float64(time.Second))).Add(s.sttTimeout)
	s.handleSpeechStopped(time.Until(deadline))
}

func (s *ClassifierStop) handleSpeechStopped(sttWait time.Duration) {
	s.userSpeaking = false
	s.cancelSTTTimer()
	if sttWait > 0 {
		s.sttWaitScheduled = true
		s.cancelSTTWait = s.after(sttWait, func() {
			s.cancelSTTWait = nil
			s.sttWaitScheduled = false
			s.maybeClassify()
		})
	}
	s.maybeClassify()
}

func (s *ClassifierStop) handleTranscription(fr *frames.TranscriptionFrame) {
	if fr.Finalized {
		s.txFinalized = true
	}
	if said := strings.TrimSpace(fr.Text); said != "" {
		s.text = strings.TrimSpace(s.text + " " + said)
	}
	s.maybeClassify()
}

func (s *ClassifierStop) maybeClassify() {
	if !s.turnOpen || s.userSpeaking || s.text == "" {
		return
	}
	// Until the STT finalizes the pause's words or its p99 latency passes, more
	// of them may be on their way.
	if s.sttWaitScheduled && !s.txFinalized {
		return
	}
	if s.hasClassified && s.text == s.classifiedText {
		return
	}
	s.cancelPending()
	s.classifiedText, s.hasClassified = s.text, true
	s.startClassification(s.text)
}

// startClassification asks the classifier about text, off the lock, and acts on
// the verdict under it.
func (s *ClassifierStop) startClassification(text string) {
	s.classification++
	id := s.classification
	parent := s.ctx
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithTimeout(parent, s.classificationTimeout)
	s.cancelClassify = cancel
	state := s.state(text)
	go func() {
		defer cancel()
		results, err := s.classifier.Choice(ctx, state, []classifier.Named[classifier.ChoiceQuestion]{
			{Name: turnCompletionQuestionName, Question: TurnCompletionQuestion},
		})
		if ctx.Err() != nil && !errors.Is(ctx.Err(), context.DeadlineExceeded) {
			// Canceled: the user spoke again, or the turn is over.
			return
		}
		s.env.locked(func() {
			if id != s.classification || s.cancelClassify == nil {
				return
			}
			s.cancelClassify = nil
			s.handleVerdict(text, results, err)
		})
	}()
}

// state is what the classifier is asked about: the user's turn and, when the
// strategy has the conversation, the assistant's last message.
func (s *ClassifierStop) state(text string) map[string]string {
	if assistant, ok := s.lastAssistantMessage(); ok {
		return map[string]string{"assistant": assistant, "user": text}
	}
	return map[string]string{"user": text}
}

func (s *ClassifierStop) handleVerdict(text string, results map[string]classifier.ChoiceResult, err error) {
	result, ok := results[turnCompletionQuestionName]
	if err != nil || !ok {
		if err == nil {
			err = fmt.Errorf("%w: no answer to the turn-completion question", classifier.ErrClassifier)
		}
		slog.Warn("turns: no classification, leaving the turn to the others", "err", err)
		s.ReportStopped()
		return
	}
	slog.Debug("turns: turn classified", "choice", result.Choice, "confidence", result.Confidence, "text", text)
	switch result.Choice {
	case kindShort:
		s.startWait(s.shortTimeout)
	case kindLong:
		s.startWait(s.longTimeout)
	default:
		s.ReportStopped()
	}
}

func (s *ClassifierStop) startWait(d time.Duration) {
	s.cancelWait = s.after(d, func() {
		slog.Debug("turns: the user did not continue", "after", d)
		s.cancelWait = nil
		s.ReportStopped()
	})
}

// cancelPending drops a classification on its way and a running wait.
func (s *ClassifierStop) cancelPending() {
	if s.cancelClassify != nil {
		s.cancelClassify()
		s.cancelClassify = nil
	}
	if s.cancelWait != nil {
		s.cancelWait()
		s.cancelWait = nil
	}
}

func (s *ClassifierStop) cancelSTTTimer() {
	if s.cancelSTTWait != nil {
		s.cancelSTTWait()
		s.cancelSTTWait = nil
	}
	s.sttWaitScheduled = false
}

// lastAssistantMessage is the text of the conversation's last assistant
// message, when there is one with text.
func (s *ClassifierStop) lastAssistantMessage() (string, bool) {
	if s.convo == nil {
		return "", false
	}
	for _, m := range slices.Backward(s.convo.Messages()) {
		if m.Role != frames.RoleAssistant {
			continue
		}
		return m.Text, m.Text != ""
	}
	return "", false
}

// ClassifierUserTurnStrategies builds stop strategies that let a classifier
// decide whether the user's turn is complete.
//
// It is an alternative to FilterIncompleteUserTurnStrategies for an LLM that
// does not follow the turn-completion marker instructions well. A classifier
// decides from the user's words, and the bot's last line, whether the user
// finished their thought, and the LLM only answers complete turns.
//
// ClassifierStop, which asks at every pause with new words, is added after the
// stop strategies given, or after the default ones (Smart Turn) when none are
// given, and the turn ends once all of them have decided. Smart Turn hears from
// how the user sounds that more is coming after words that read complete, such
// as "I want to go to Paris", which the classifier cannot tell from the words. A
// turn the classifier found complete then waits for Smart Turn too, up to its
// silence fallback when Smart Turn finds it incomplete. For the classifier
// alone, build the stop list yourself with NewClassifierStop.
//
// With UserTurnStrategiesParams{StopMergeMode: MergeModeExtend}, the classifier
// strategy and any stop strategies given join the ones a service recommends
// instead. It fails on a configuration without a classifier.
func ClassifierUserTurnStrategies(stop []StopStrategy, cfg ClassifierStopConfig) (UserTurnStrategies, error) {
	strategy, err := NewClassifierStop(cfg)
	if err != nil {
		return UserTurnStrategies{}, err
	}
	given := append(append([]StopStrategy(nil), stop...), strategy)
	if len(stop) == 0 {
		stop = DefaultStopStrategies()
	}
	return UserTurnStrategies{
		Stop: append(append([]StopStrategy(nil), stop...), strategy),
		// Extending a service's stop strategies adds the classifier strategy
		// and the stop strategies given, never the default ones.
		stopExtension:    given,
		hasStopExtension: true,
	}, nil
}

var _ StopStrategy = (*ClassifierStop)(nil)
