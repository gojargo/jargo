package llmwithbackend

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/goid"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
	"github.com/google/uuid"
)

// Predictive delegation (PredictiveDelegation, passed as
// Config.PredictiveDelegation) is an optimization that can speed up responses
// that need backend involvement. Normally, a text frontend spends a whole
// inference deciding to call delegate before the backend can start. Predictive
// delegation uses a fast yes/no classifier to determine whether it needs to wait
// for that inference or can go ahead and trigger an early backend delegation.
//
// How it works: when a user's turn ends, the frontend runs in parallel (in a
// race) with the classifier (see package classifier) that decides whether the
// turn needs the backend.
//
//   - If the classifier says "yes" before the frontend has produced output, the
//     frontend's inference is canceled, and a delegate call to the backend is
//     made right away, in its place.
//   - Otherwise, the frontend handles the turn as usual, having waited for
//     nothing (it may still choose to delegate).
//
// predictiveDelegationRace tracks whether a classifier "yes" arrives before the
// frontend's first output, and so whether the turn is delegated early or handled
// by the frontend as usual. A classifier "no" plays no part. The frontend is
// normally idle when a turn ends (the user's speech interrupts it), so its
// inference starts at once; a "yes" that comes while it is still busy is held
// until it starts. The race is attached to the frontend as its output gate
// (llm.OutputGate), which the frontend consults from its ProcessFrame, PushFrame
// and the running of its function calls.

// raceState is where the predictive delegation race for a user's turn stands.
// Its zero value is before the first user turn starts a race.
type raceState int

const (
	// racePending is when the frontend has been asked to reply to the user's
	// turn but has not started its inference yet: normally for only a moment,
	// longer if it is still busy with an earlier response. A classifier "yes"
	// now is held until the inference starts.
	racePending raceState = iota + 1
	// raceRacing is when the frontend's inference is running and has not
	// produced any output yet. A classifier "yes" now, or one held from
	// racePending, delegates the turn early.
	raceRacing
	// raceFrontendWon is when the frontend's inference produced output before
	// any classifier "yes", or the turn was interrupted. The frontend handles
	// the turn as usual.
	raceFrontendWon
	// raceClassifierWon is when a classifier "yes" came first. The turn is
	// delegated early, and the frontend's inference is canceled.
	raceClassifierWon
)

// earlyDelegationFrame asks the frontend to delegate a user's turn now, in place
// of its own inference.
//
// PredictiveDelegation sends it when its classifier says "yes". If the
// frontend's inference for the turn has not produced anything yet, the frontend
// makes the given calls and cancels the inference; otherwise the frame does
// nothing. A system frame, so the frontend handles it at once, while the
// inference runs on another goroutine.
type earlyDelegationFrame struct {
	frames.BaseSystemFrame
	// turn is the context frame of the user's turn.
	turn *frames.LLMContextFrame
	// functionCalls are the calls made in place of the inference (the delegate
	// call).
	functionCalls []frames.ToolCall
}

func newEarlyDelegationFrame(turn *frames.LLMContextFrame, calls []frames.ToolCall) *earlyDelegationFrame {
	return &earlyDelegationFrame{
		BaseSystemFrame: frames.NewBaseSystemFrame("EarlyDelegationFrame"),
		turn:            turn,
		functionCalls:   calls,
	}
}

// predictiveDelegationRace tracks whether a classifier "yes" arrives before the
// frontend's first output for a turn.
//
// Early delegation makes the frame's calls, then cancels the frontend's
// inference with CancelFrameInProgress, which keeps the frontend's queue,
// including the acknowledgement run the call starts. Downstream sees an ordinary
// response holding the call; any other output the inference pushes before it is
// canceled is dropped.
//
// The frontend's inference is recognized by the goroutine its frames and calls
// come from, the frontend's process goroutine; everything else the frontend
// pushes is left alone. Every check is made under one lock, so the frontend's
// first output and early delegation cannot interleave.
type predictiveDelegationRace struct {
	mu   sync.Mutex
	turn *frames.LLMContextFrame
	// frontendInference is the frontend's process goroutine, which runs its
	// inference for the turn and goes on to process the frames after it.
	frontendInference uint64
	// state is zero until the first user turn starts a race; after that, the
	// latest race's state, which stays put once decided until the
	// next turn.
	state    raceState
	frontend *llm.Base
	// heldEarlyDelegation is a "yes" that arrived while racePending, held until
	// the inference starts its response.
	heldEarlyDelegation *earlyDelegationFrame
}

// start starts a race for a user's turn, just before the turn is passed to the
// frontend.
func (r *predictiveDelegationRace) start(f *frames.LLMContextFrame) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.turn = f
	r.frontendInference = 0
	r.state = racePending
	r.heldEarlyDelegation = nil
}

// ProcessFrame implements llm.OutputGate: it follows the race through the frames
// the frontend processes. Racing starts when the frontend's inference for the
// turn starts, an interruption leaves the turn to the frontend, and an early
// delegation frame is carried out while racing.
func (r *predictiveDelegationRace) ProcessFrame(ctx context.Context, service *llm.Base, f frames.Frame) error {
	switch fr := f.(type) {
	case *frames.LLMContextFrame:
		r.mu.Lock()
		if fr == r.turn && r.state == racePending {
			r.frontend = service
			r.frontendInference = goid.Current()
			r.state = raceRacing
		}
		r.mu.Unlock()
	case *frames.InterruptionFrame:
		// The interruption cancels the inference: there is nothing left for
		// early delegation to replace.
		r.mu.Lock()
		if r.state == racePending || r.state == raceRacing {
			r.state = raceFrontendWon
		}
		r.mu.Unlock()
	case *earlyDelegationFrame:
		return r.delegateEarly(ctx, service, fr)
	}
	return nil
}

// AllowPush implements llm.OutputGate: whether the frontend may push a frame,
// noting its inference's first output. It reports false only for what the
// inference pushes after the turn was delegated early and before its
// cancellation lands, other than the end of its response.
func (r *predictiveDelegationRace) AllowPush(f frames.Frame) bool {
	r.mu.Lock()
	if !r.calledFromFrontendInference() {
		r.mu.Unlock()
		return true
	}
	held := r.noteFrontendInferencePush(f)
	allow := true
	if r.state == raceClassifierWon {
		_, isSystem := f.(frames.SystemFrame)
		_, isEnd := f.(*frames.LLMFullResponseEndFrame)
		allow = isSystem || isEnd
	}
	frontend := r.frontend
	r.mu.Unlock()
	if held != nil && frontend != nil {
		// Carried out on the frontend's input goroutine, like any early
		// delegation frame.
		if err := frontend.QueueFrame(context.Background(), held, processor.Downstream); err != nil {
			slog.Debug("predictive delegation: could not queue the held early delegation", "err", err)
		}
	}
	return allow
}

// AllowFunctionCalls implements llm.OutputGate: whether the frontend may run
// function calls, noting its inference's first call. It reports false only for
// calls the inference reaches after the turn was delegated early and before its
// cancellation lands.
func (r *predictiveDelegationRace) AllowFunctionCalls() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.calledFromFrontendInference() {
		return true
	}
	r.noteFrontendInferenceFunctionCalls()
	return r.state != raceClassifierWon
}

// calledFromFrontendInference reports whether the caller is the frontend's
// process goroutine, running its inference for the turn. The caller holds mu.
func (r *predictiveDelegationRace) calledFromFrontendInference() bool {
	return r.frontendInference != 0 && goid.Current() == r.frontendInference
}

// noteFrontendInferencePush notes a frame the frontend's inference pushes while
// racing. The start of its response releases an early delegation held from
// racePending, which it returns for the caller to queue. Any other output but a
// system frame is the inference's first, and leaves the turn to the frontend.
// The caller holds mu.
func (r *predictiveDelegationRace) noteFrontendInferencePush(f frames.Frame) *earlyDelegationFrame {
	if r.state != raceRacing {
		return nil
	}
	switch f.(type) {
	case *frames.LLMFullResponseStartFrame:
		held := r.heldEarlyDelegation
		r.heldEarlyDelegation = nil
		return held
	case frames.SystemFrame:
	default:
		r.state = raceFrontendWon
	}
	return nil
}

// noteFrontendInferenceFunctionCalls notes the frontend's inference reaching
// function calls. While racing, its first call leaves the turn to the frontend.
// The caller holds mu.
func (r *predictiveDelegationRace) noteFrontendInferenceFunctionCalls() {
	if r.state == raceRacing {
		r.state = raceFrontendWon
	}
}

// delegateEarly makes the frame's calls and cancels the inference, if the turn
// is racing. A frame that arrives while the turn is pending is held until the
// inference starts.
func (r *predictiveDelegationRace) delegateEarly(
	ctx context.Context, frontend *llm.Base, f *earlyDelegationFrame,
) error {
	r.mu.Lock()
	turn := f.turn
	if turn == nil || turn != r.turn {
		r.mu.Unlock()
		return nil
	}
	switch r.state {
	case racePending:
		slog.DebugContext(ctx, "predictive delegation: turn pending; holding the early delegation")
		r.heldEarlyDelegation = f
		r.mu.Unlock()
		return nil
	case raceRacing:
	default:
		slog.DebugContext(ctx, "predictive delegation: not delegating early (the race is decided)")
		r.mu.Unlock()
		return nil
	}
	slog.DebugContext(ctx, "predictive delegation: delegating early, canceling the frontend's inference")
	r.state = raceClassifierWon
	r.mu.Unlock()
	if err := frontend.RunFunctionCalls(ctx, turn.Context, f.functionCalls); err != nil {
		return err
	}
	frontend.CancelFrameInProgress(turn)
	return nil
}

var _ llm.OutputGate = (*predictiveDelegationRace)(nil)

// delegationPredictionThreshold is how likely a classifier "yes" must be for a
// turn to be delegated early.
const delegationPredictionThreshold = 0.6

// delegationPredictionHistoryLength is how many of the conversation's latest
// messages the classifier sees.
const delegationPredictionHistoryLength = 12

// Who said each message of the conversation the classifier sees.
const (
	speakerUser      = "user"
	speakerAssistant = "assistant"
	speakerBackend   = "backend"
)

// needsBackendQuestion is the name the classifier's one question is asked
// under.
const needsBackendQuestion = "needs_backend"

// predictiveFrontend is what predictive delegation needs of the frontend: a
// text LLM service built on the LLM service base.
type predictiveFrontend interface {
	SetOutputGate(g llm.OutputGate)
	SystemInstruction() string
	FilterIncompleteUserTurns() bool
}

// ErrPredictiveDelegationNeedsTextFrontend is reported when predictive
// delegation is given a speech-to-speech frontend.
var ErrPredictiveDelegationNeedsTextFrontend = errors.New("predictive delegation needs a text frontend: " +
	"a speech-to-speech frontend replies to the user's audio before the turn's transcript reaches its " +
	"context, so there is no turn to classify in time")

// ErrPredictiveDelegationNeedsNoArguments is reported when predictive
// delegation is given a request strategy whose delegate call needs arguments.
var ErrPredictiveDelegationNeedsNoArguments = errors.New("predictive delegation makes its delegate call " +
	"with no arguments")

// PredictiveDelegationConfig configures a PredictiveDelegation.
type PredictiveDelegationConfig struct {
	// Classifier is the yes/no classifier to ask. It answers within a bound of
	// its own; an answer that comes after the frontend's first output is
	// ignored.
	Classifier classifier.Classifier `validate:"required"`
}

// PredictiveDelegation is an optimization that can speed up responses that need
// backend involvement.
//
// Alpha. It uses a fast yes/no classifier to determine, after each user request,
// whether it needs to wait for the frontend's inference or can go ahead and
// trigger an early backend delegation. See the note on predictive delegation in
// this file for how it works. Pass one as Config.PredictiveDelegation, which
// puts it just in front of the frontend; it needs a text frontend and a request
// strategy whose delegate call needs no arguments, such as the default
// TranscriptBackendRequestStrategy.
//
// It races every turn that ends with a new user message, spoken or typed, and so
// also a user-role message the app appends and runs the frontend on. The
// classifier is asked whether that message needs the backend, given the
// frontend's system instruction (where the app says what the backend is for, and
// the pairing instruction when to delegate), the delegate tool's description,
// and the latest messages of the conversation, the backend's included. A turn
// the frontend sees as speculative (eager end of turn), or with incomplete-turn
// filtering on, is left to the frontend.
//
//	jevClassifier, _ := jev.New(jev.Config{APIKey: key})
//	pd, _ := llmwithbackend.NewPredictiveDelegation(
//		llmwithbackend.PredictiveDelegationConfig{Classifier: jevClassifier})
//	llm, _ := llmwithbackend.New(llmwithbackend.Config{
//		Frontend: openaiLLM, Backend: backend, PredictiveDelegation: pd,
//	})
type PredictiveDelegation struct {
	*processor.Base

	classifier classifier.Classifier
	race       predictiveDelegationRace
	frontend   predictiveFrontend
	question   classifier.YesNoQuestion

	mu sync.Mutex
	// lastRaced is the user message last raced, so a context frame ending with
	// it is not raced again.
	lastRaced    racedMessage
	classifyStop context.CancelFunc
	classifyDone chan struct{}
	warned       map[string]bool
	// reporting says the classifier's metrics are already reported here.
	reporting bool
}

// racedMessage says which message of which conversation was raced. Messages
// are values, so a message is known by where it sits.
type racedMessage struct {
	context *frames.LLMContext
	index   int
	text    string
}

// NewPredictiveDelegation builds predictive delegation.
func NewPredictiveDelegation(cfg PredictiveDelegationConfig) (*PredictiveDelegation, error) {
	if err := validate.Struct(cfg); err != nil {
		return nil, fmt.Errorf("predictive delegation: %w", err)
	}
	p := &PredictiveDelegation{classifier: cfg.Classifier, warned: map[string]bool{}}
	p.Base = processor.New("PredictiveDelegation", p)
	return p, nil
}

// bind attaches the race to the frontend and builds the classifier's question.
func (p *PredictiveDelegation) bind(frontend predictiveFrontend, connector *BackendConnector) {
	p.frontend = frontend
	frontend.SetOutputGate(&p.race)
	p.question = classifier.YesNoQuestion{
		Instructions: "Does the user's latest message need to be handed to the assistant's backend " +
			"now? The handoff works like this: " + connector.RequestStrategy().ToolDescription(),
		No: "The assistant's instructions say not to hand this message to the backend.",
	}
}

// Setup sets up the classifier, reporting its metrics in the pipeline.
func (p *PredictiveDelegation) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	if err := p.classifier.Setup(ctx); err != nil {
		return err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.reporting {
		events.On(p.classifier.Events(), classifier.EventMetrics,
			func(ctx context.Context, data []frames.MetricsData) {
				_ = p.PushFrame(ctx, frames.NewMetricsFrame(data...), processor.Downstream)
			})
		p.reporting = true
	}
	return nil
}

// Cleanup stops any classification in progress and cleans up the classifier.
func (p *PredictiveDelegation) Cleanup(ctx context.Context) error {
	p.cancelClassification()
	err := p.classifier.Cleanup(ctx)
	return errors.Join(err, p.Base.Cleanup(ctx))
}

// ProcessFrame races each new user turn, and passes every frame on.
func (p *PredictiveDelegation) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch f.(type) {
	case *frames.InterruptionFrame, *frames.UserStartedSpeakingFrame:
		p.cancelClassification()
	}
	if cf, ok := f.(*frames.LLMContextFrame); ok && dir == processor.Downstream && p.shouldStartRace(ctx, cf) {
		p.cancelClassification()
		p.race.start(cf)
		state := p.classifierState(cf)
		if err := p.PushFrame(ctx, cf, dir); err != nil {
			return err
		}
		p.startClassification(ctx, cf, state)
		return nil
	}
	return p.PushFrame(ctx, f, dir)
}

// shouldStartRace reports whether a context frame should start a race: when it
// asks for a reply to a user message that has not been raced yet, unless the
// turn is speculative (eager end of turn) or the frontend filters incomplete
// user turns.
func (p *PredictiveDelegation) shouldStartRace(ctx context.Context, f *frames.LLMContextFrame) bool {
	if f.Speculation {
		p.warnOnce(ctx, "speculation",
			"predictive delegation: speculative turns (eager end of turn) are left to the frontend")
		return false
	}
	if p.frontend.FilterIncompleteUserTurns() {
		p.warnOnce(ctx, "filtering",
			"predictive delegation: turns are left to the frontend while it filters incomplete user turns")
		return false
	}
	messages := f.Context.Messages()
	if len(messages) == 0 {
		return false
	}
	last := messages[len(messages)-1]
	if last.IsLLMSpecific() || last.Role != frames.RoleUser || len(last.ToolResults) > 0 {
		return false
	}
	raced := racedMessage{context: f.Context, index: len(messages) - 1, text: last.Text}
	p.mu.Lock()
	defer p.mu.Unlock()
	if raced == p.lastRaced {
		return false
	}
	p.lastRaced = raced
	return true
}

// classifierState is what the classifier is told about the turn.
type classifierState struct {
	// Assistant is the instruction the frontend's model reads, the pairing
	// instruction included, so the classifier follows the same rules for
	// delegating.
	Assistant    string              `json:"assistant"`
	Conversation []classifierMessage `json:"conversation"`
}

// classifierMessage is one message of the conversation as the classifier sees
// it: who spoke, and what they said or did.
type classifierMessage struct {
	Speaker string
	Text    string
	// Action is what the speaker did, for a message that is not something said.
	Action string
}

// MarshalJSON writes the message with the speaker first, and either its text or
// its action.
func (m classifierMessage) MarshalJSON() ([]byte, error) {
	if m.Action != "" {
		return json.Marshal(struct {
			Speaker string `json:"speaker"`
			Action  string `json:"action"`
		}{m.Speaker, m.Action})
	}
	return json.Marshal(struct {
		Speaker string `json:"speaker"`
		Text    string `json:"text"`
	}{m.Speaker, m.Text})
}

func (p *PredictiveDelegation) classifierState(f *frames.LLMContextFrame) classifierState {
	return classifierState{
		Assistant:    p.frontend.SystemInstruction(),
		Conversation: renderForClassifier(f.Context.Messages()),
	}
}

// startClassification asks the classifier about a turn on a goroutine of its
// own.
func (p *PredictiveDelegation) startClassification(
	ctx context.Context, f *frames.LLMContextFrame, state classifierState,
) {
	classifyCtx, stop := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	p.mu.Lock()
	p.classifyStop, p.classifyDone = stop, done
	p.mu.Unlock()
	go func() {
		defer close(done)
		p.classify(classifyCtx, f, state)
	}()
}

// classify asks the classifier about a turn; on a "yes", it asks the frontend to
// delegate it early.
func (p *PredictiveDelegation) classify(ctx context.Context, f *frames.LLMContextFrame, state classifierState) {
	answers, err := p.classifier.YesNo(ctx, state, []classifier.Named[classifier.YesNoQuestion]{
		{Name: needsBackendQuestion, Question: p.question},
	})
	if ctx.Err() != nil {
		return
	}
	if err != nil {
		slog.WarnContext(ctx, "predictive delegation: the classifier failed", "err", err)
		return
	}
	probability := answers[needsBackendQuestion].Probability
	slog.DebugContext(ctx, fmt.Sprintf("predictive delegation: classifier says %.2f", probability))
	if probability < delegationPredictionThreshold {
		return
	}
	call := frames.ToolCall{
		ID:   "early_delegation_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:16],
		Name: DelegateToolName,
		Args: json.RawMessage(`{}`),
	}
	if err := p.PushFrame(ctx, newEarlyDelegationFrame(f, []frames.ToolCall{call}),
		processor.Downstream); err != nil {
		slog.DebugContext(ctx, "predictive delegation: could not ask for an early delegation", "err", err)
	}
}

// cancelClassification stops the classification in progress, if any, and waits
// for it to end.
func (p *PredictiveDelegation) cancelClassification() {
	p.mu.Lock()
	stop, done := p.classifyStop, p.classifyDone
	p.classifyStop, p.classifyDone = nil, nil
	p.mu.Unlock()
	if stop != nil {
		stop()
		<-done
	}
}

func (p *PredictiveDelegation) warnOnce(ctx context.Context, key, message string) {
	p.mu.Lock()
	warned := p.warned[key]
	p.warned[key] = true
	p.mu.Unlock()
	if !warned {
		slog.WarnContext(ctx, message)
	}
}

// renderForClassifier is the conversation's latest messages, as speaker and
// text, for the classifier.
func renderForClassifier(messages []frames.Message) []classifierMessage {
	var rendered []classifierMessage
	for _, m := range messages {
		if m.IsLLMSpecific() {
			continue
		}
		switch {
		case m.Role == frames.RoleUser && len(m.ToolResults) == 0:
			rendered = append(rendered, classifierMessage{Speaker: speakerUser, Text: m.Text})
		case m.Role == frames.RoleAssistant:
			if m.Text != "" {
				rendered = append(rendered, classifierMessage{Speaker: speakerAssistant, Text: m.Text})
			}
			for _, call := range m.ToolCalls {
				if call.Name == DelegateToolName {
					rendered = append(rendered, classifierMessage{
						Speaker: speakerAssistant, Action: "handed the request to the backend",
					})
				}
			}
		case isBackendMessage(m):
			rendered = append(rendered, classifierMessage{
				Speaker: speakerBackend, Text: strings.TrimSuffix(m.Text, "\n\n"+RelayNote),
			})
		}
	}
	if len(rendered) > delegationPredictionHistoryLength {
		rendered = rendered[len(rendered)-delegationPredictionHistoryLength:]
	}
	return rendered
}
