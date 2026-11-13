package eval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/classifier"
	llmclassifier "github.com/gojargo/jargo/classifier/llm"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/ordered"
	"github.com/gojargo/jargo/service/llm"
)

// fenced matches the markdown code fences a judge model wraps its JSON in when
// it ignores the instruction not to.
//
//nolint:gochecknoglobals // compiled once
var fenced = regexp.MustCompile("(?m)^```(?:json)?[ \t]*|[ \t]*```$")

// The verdicts a judge can return.
const (
	// VerdictYes means the reply satisfies the criterion.
	VerdictYes = "yes"
	// VerdictNo means the reply gives a substantive answer that fails it.
	VerdictNo = "no"
	// VerdictContinue means the bot is still working toward its answer, and the
	// criterion should be judged again once more text arrives.
	VerdictContinue = "continue"
)

// JudgeVerdict is the outcome of a single judge call.
type JudgeVerdict struct {
	// Verdict is VerdictYes, VerdictNo or VerdictContinue.
	Verdict string
	// Reason is the explainer's one-sentence reason for the verdict, or the
	// classifier's probabilities when nothing explained it.
	Reason string
	// RawResponse is the judge's raw answer, for diagnostics.
	RawResponse string
	// Confidence is how sure the judge is of the verdict, from 0 to 1; nil when
	// it did not say.
	Confidence *float64
	// Probabilities is the probability of each outcome the classifier was
	// offered; empty when the question failed.
	Probabilities map[string]float64
}

// Passed reports whether the verdict is a definite yes.
func (v JudgeVerdict) Passed() bool { return v.Verdict == VerdictYes }

// Judge decides whether the bot's most recent reply satisfies a
// natural-language criterion, which is what a scenario's `judge:` assertion
// asks. It is fed the conversation as the scenario plays: the harness records
// each user turn and each segment of the bot's reply, and Evaluate judges the
// most recent reply in that context. That is what lets a terse or ambiguous
// reply be resolved, one that would not make sense on its own.
//
// A judge is per-scenario, because the conversation it holds is. The harness
// treats it as optional, so a scenario with no `judge:` assertion needs none. A
// judge that also has a Close(context.Context) error method is closed when the
// run ends, however it ends.
type Judge interface {
	// AddUserMessage records a user turn in the conversation.
	AddUserMessage(text string)
	// AddAssistantMessage appends a segment of the bot's current reply.
	AddAssistantMessage(text string)
	// Evaluate judges the conversation so far against criterion. A judge that
	// cannot answer reports VerdictNo with the reason, rather than failing: an
	// unavailable judge is a failed assertion, not a broken run.
	Evaluate(ctx context.Context, criterion string) JudgeVerdict
	// EvaluateCall judges a tool call the bot made against criterion, by the
	// call's name and arguments, with the conversation so far as context. It is
	// how a scenario checks what an argument subset cannot match verbatim.
	//
	// A call is not a partial reply, so there is nothing to wait for and the
	// verdict is yes or no; a judge answering VerdictContinue is read as a no.
	EvaluateCall(ctx context.Context, name string, args map[string]any, criterion string) JudgeVerdict
}

// judgeCloser is a judge holding something open, released when the run ends.
type judgeCloser interface {
	Close(ctx context.Context) error
}

// closeJudge releases what a judge holds open, if it holds anything.
func closeJudge(ctx context.Context, judge Judge) {
	if c, ok := judge.(judgeCloser); ok {
		if err := c.Close(ctx); err != nil {
			slog.WarnContext(ctx, "eval: closing the judge failed", "error", err)
		}
	}
}

// defaultJudgeMaxTokens caps the explainer's answer for one reason: enough for
// a short sentence in JSON, and no room for the model to start explaining
// itself at length.
const defaultJudgeMaxTokens = 200

// defaultExplainBelow is the confidence under which a yes is explained too.
const defaultExplainBelow = 0.75

// explainerTimeout is how long the explainer has to answer a question before it
// is asked again.
const explainerTimeout = 60 * time.Second

// explainSystemInstruction is what the explainer is told for every question.
// The classifier has decided: the explainer is told the verdict and says what
// in the conversation supports it, so a reason always explains the verdict
// that counts.
const explainSystemInstruction = "You explain the verdicts a judge gave on a conversation between a user and a " +
	"bot under test. The 'user' messages are the user; the 'assistant' messages are the bot's " +
	"replies. The judge's verdicts are final. For each verdict you are asked about, say in " +
	"one short sentence what the bot said or did that makes it right, naming it; never " +
	"just restate the verdict, and never argue for a different verdict. When a criterion " +
	"does not apply to a reply, say what the reply does instead and why that is outside " +
	"what the criterion covers. " +
	"When the bot spoke its replies, its text is an automatic speech-to-text " +
	"transcription, so it may contain homophones, misspellings, split or merged words, " +
	"and missing punctuation; read it by its intended spoken meaning. " +
	"Respond ONLY with the JSON object the question describes, on a single line, with no " +
	"other text or markdown."

// explainReplyAsk is the ask for a reply's verdict, after the conversation so
// far: the outcome, then the criterion.
const explainReplyAsk = "The judge's verdict: the bot's most recent reply %s this criterion.\n\n" +
	"Criterion: %s\n\n" +
	`Answer with {"reason": "<one short sentence>"}.`

// explainCallAsk is the ask for a function call's verdict, the conversation so
// far being context: the call's name, its arguments as JSON, the outcome, then
// the criterion.
const explainCallAsk = "The bot called the function `%s` with arguments `%s`. The judge's verdict: " +
	"this call %s this criterion.\n\n" +
	"Criterion: %s\n\n" +
	`Answer with {"reason": "<one short sentence>"}.`

// judgeFailed is the reason a verdict carries when its question failed.
const judgeFailed = "judge call failed"

// The questions the judge asks, each followed by the criterion and the notes
// that go with every question; see instructions.
const (
	replyQuestion = "Does `latest_bot_reply`, the bot's most recent reply, following `conversation`, " +
		"satisfy this criterion?"
	callQuestion = "Does the bot's function `call`, judged by its name and arguments, satisfy this " +
		"criterion?"
	callNote          = "`conversation` is context only."
	transcriptionNote = "The bot's text may be an automatic speech-to-text transcription: judge its " +
		"intended spoken meaning, never its spelling ('for' may mean 'four', 'to' may mean 'two')."
)

// replyOutcomes are the answers to a reply's question, in order.
//
//nolint:gochecknoglobals // the options the judge offers, fixed
var replyOutcomes = []classifier.Option{
	{Name: VerdictYes, Description: "The bot has given its answer, and the answer satisfies the criterion."},
	{Name: VerdictNo, Description: "The bot has given its answer, and the answer does not satisfy the " +
		"criterion. A reply that waits for the user (asking them to take their time or to go on) instead " +
		"of giving what the criterion asks for is a no."},
	{Name: VerdictContinue, Description: "The bot is still working toward its answer: it only greets, " +
		"says it is checking or looking something up and will report back, or the reply is an obviously " +
		"incomplete fragment."},
}

// errNoClassifier reports a judge given nothing to decide with.
var errNoClassifier = errors.New("eval: a judge needs a classifier to decide the verdicts")

// EvalJudgeConfig configures an EvalJudge.
type EvalJudgeConfig struct {
	// Classifier decides the verdicts. The judge sets it up before its first
	// question and cleans it up when it is closed. Required.
	Classifier classifier.Classifier
	// Explainer is the LLM asked for the reason behind a no or an unsure
	// verdict; nil reports the probabilities alone.
	Explainer llm.Inferencer
	// ExplainBelow is the confidence under which a yes is explained too; nil
	// uses 0.75. Zero limits the explainer to the no verdicts, and a value above
	// 1 sends it every verdict.
	ExplainBelow *float64
	// AllowContinue is whether a reply may be judged continue: the bot is still
	// working toward its answer, so the harness waits for more of it. Nil allows
	// it. A suite whose judged replies are all final answers turns it off, and a
	// reply is then yes or no.
	AllowContinue *bool
	// MaxTokens caps the explainer's answer for one reason; zero uses 200,
	// enough for a short sentence.
	MaxTokens int
}

// EvalJudge judges a conversation with a classifier, and asks an LLM for the
// reasons.
//
// Every question goes to the classifier, which answers with an option and a
// probability for each of the options it was offered. A reply is judged yes, no
// or continue, with the conversation before it and the reply itself sent apart,
// so the classifier knows which part to judge. A tool call is judged yes or no
// by its name and arguments. Each verdict carries the classifier's confidence,
// and its reason starts out as the probabilities it gave (P(yes)=0.97). Each
// question's verdict is cached by its criterion and state.
//
// A classifier gives no reasons, so the judge asks an LLM of its own, the
// explainer, for them: for every no, and for every yes less sure than
// ExplainBelow, but never for a continue. The explainer is told the verdict and
// asked only what in the conversation supports it, so it never changes a
// verdict; the reason it gives takes the place of the probabilities, which the
// verdict still carries.
//
// A question that fails is asked once more; a second failure gives a no with
// the reason "judge call failed".
type EvalJudge struct {
	classifier    classifier.Classifier
	explainer     *explainer
	explainBelow  float64
	allowContinue bool
	outcomes      []classifier.Option

	setupOnce sync.Once

	mu sync.Mutex
	// transcript is the conversation the judge evaluates against, grown by the
	// harness over the scenario.
	transcript []transcriptEntry
	cache      map[string]JudgeVerdict
}

// transcriptEntry is one entry of the conversation the judge keeps: a user turn,
// a segment of the bot's reply, or a tool call the bot made.
type transcriptEntry struct {
	role    string
	content string
}

const (
	roleUser      = "user"
	roleAssistant = "assistant"
	roleTool      = "tool"
	roleBot       = "bot"
)

// NewEvalJudge builds a judge.
func NewEvalJudge(cfg EvalJudgeConfig) (*EvalJudge, error) {
	if cfg.Classifier == nil {
		return nil, errNoClassifier
	}
	j := &EvalJudge{
		classifier:    cfg.Classifier,
		explainBelow:  defaultExplainBelow,
		allowContinue: cfg.AllowContinue == nil || *cfg.AllowContinue,
		cache:         map[string]JudgeVerdict{},
	}
	if cfg.ExplainBelow != nil {
		j.explainBelow = *cfg.ExplainBelow
	}
	j.outcomes = replyOutcomes
	if !j.allowContinue {
		j.outcomes = replyOutcomes[:2]
	}
	if cfg.Explainer != nil {
		maxTokens := cfg.MaxTokens
		if maxTokens == 0 {
			maxTokens = defaultJudgeMaxTokens
		}
		j.explainer = newExplainer(cfg.Explainer, maxTokens)
	}
	return j, nil
}

// NewLLMJudge builds a judge that classifies with inf and explains with it too,
// e.g. eval.NewLLMJudge(chat.NewLLM(chat.LLMConfig{APIKey: key})).
//
// Deprecated: build a classifier/llm classifier over the service and pass it to
// NewEvalJudge, with the service as the Explainer for the reasons.
func NewLLMJudge(inf llm.Inferencer) *EvalJudge {
	c, err := llmclassifier.New(llmclassifier.Config{LLM: inf})
	if err != nil {
		// A nil service is the only way to fail, and that is a caller's mistake
		// made where it is made.
		panic(err) //nolint:forbidigo // a mistake in the program, refused where it is made
	}
	j, _ := NewEvalJudge(EvalJudgeConfig{Classifier: c, Explainer: inf})
	return j
}

// Classifier is the classifier that decides the verdicts.
func (j *EvalJudge) Classifier() classifier.Classifier { return j.classifier }

// AddUserMessage records a user turn, so a later reply is judged in context (a
// terse "that's four" answering "what is two plus two?", say).
func (j *EvalJudge) AddUserMessage(text string) { j.add(roleUser, text) }

// AddAssistantMessage adds a segment of the bot's current reply to the
// conversation the judge sees. Consecutive segments are one reply.
func (j *EvalJudge) AddAssistantMessage(text string) { j.add(roleAssistant, text) }

// AddToolCall records a tool call the bot made, on one line, e.g.
// book({"time": "6pm"}). A reply and a call are judged on the spoken
// conversation only, so it is kept but not shown to them.
func (j *EvalJudge) AddToolCall(text string) { j.add(roleTool, text) }

func (j *EvalJudge) add(role, text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	j.mu.Lock()
	defer j.mu.Unlock()
	j.transcript = append(j.transcript, transcriptEntry{role: role, content: text})
}

// spoken is the conversation without the tool calls, and the whole transcript.
func (j *EvalJudge) spoken() (spoken, all []transcriptEntry) {
	j.mu.Lock()
	defer j.mu.Unlock()
	all = append([]transcriptEntry(nil), j.transcript...)
	for _, e := range all {
		if e.role != roleTool {
			spoken = append(spoken, e)
		}
	}
	return spoken, all
}

// Evaluate judges whether the bot's latest reply satisfies criterion, in the
// conversation so far. It gives a yes, a no, or, when continue is allowed, a
// continue when the bot is still working toward its answer, cached by criterion
// and conversation. A final verdict that needs a reason is explained, and a
// failed question is a no with the failure as its reason.
func (j *EvalJudge) Evaluate(ctx context.Context, criterion string) JudgeVerdict {
	spoken, all := j.spoken()
	entries := numberedTurns(spoken)
	latest := ""
	if n := len(entries); n > 0 && entries[n-1].role == roleBot {
		latest = entries[n-1].content
		entries = entries[:n-1]
	}
	state := ordered.Object{
		{Key: "conversation", Value: conversation(entries)},
		{Key: "latest_bot_reply", Value: latest},
	}
	key := cacheKey("reply", criterion, state)
	if v, ok := j.cached(key); ok {
		return v
	}

	question := classifier.ChoiceQuestion{
		Instructions: instructions(replyQuestion, "Criterion", criterion, ""),
		Options:      j.outcomes,
	}
	var verdict JudgeVerdict
	answers, err := askTwice(ctx, j, func() (map[string]classifier.ChoiceResult, error) {
		return j.classifier.Choice(ctx, state, []classifier.Named[classifier.ChoiceQuestion]{
			{Name: "verdict", Question: question},
		})
	})
	if err != nil {
		verdict = failed(VerdictNo)
	} else {
		verdict = replyVerdict(answers["verdict"], question)
		if verdict.Verdict != VerdictContinue && j.explainer != nil && j.needsReason(verdict) {
			verdict = withReason(verdict, j.explainer.explain(ctx, all, criterion, verdict))
		}
	}
	j.store(key, verdict)
	return verdict
}

// EvaluateCall judges whether a tool call the bot made satisfies criterion, in
// the conversation so far. It gives a yes or a no, cached by call, criterion and
// conversation, and explained when it needs a reason.
func (j *EvalJudge) EvaluateCall(
	ctx context.Context, name string, args map[string]any, criterion string,
) JudgeVerdict {
	spoken, all := j.spoken()
	state := ordered.Object{
		{Key: "conversation", Value: conversation(numberedTurns(spoken))},
		{Key: "call", Value: ordered.Object{{Key: "name", Value: name}, {Key: "arguments", Value: argsOrEmpty(args)}}},
	}
	key := cacheKey("call", criterion, state)
	if v, ok := j.cached(key); ok {
		return v
	}

	question := classifier.YesNoQuestion{Instructions: instructions(callQuestion, "Criterion", criterion, callNote)}
	var verdict JudgeVerdict
	answers, err := askTwice(ctx, j, func() (map[string]classifier.YesNoResult, error) {
		return j.classifier.YesNo(ctx, state, []classifier.Named[classifier.YesNoQuestion]{
			{Name: "answer", Question: question},
		})
	})
	if err != nil {
		verdict = failed(VerdictNo)
	} else {
		verdict = yesNoVerdict(answers["answer"])
		if j.explainer != nil && j.needsReason(verdict) {
			verdict = withReason(verdict, j.explainer.explainCall(ctx, all, name, args, criterion, verdict))
		}
	}
	j.store(key, verdict)
	return verdict
}

// Close releases what the judge holds open; the session calls it when the run
// ends.
func (j *EvalJudge) Close(ctx context.Context) error {
	// A question still setting the classifier up finishes first.
	j.setupOnce.Do(func() {})
	return j.classifier.Cleanup(ctx)
}

func (j *EvalJudge) cached(key string) (JudgeVerdict, bool) {
	j.mu.Lock()
	defer j.mu.Unlock()
	v, ok := j.cache[key]
	return v, ok
}

func (j *EvalJudge) store(key string, v JudgeVerdict) {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.cache[key] = v
}

// setup wires the classifier up before the first question. A classifier that
// cannot be set up is only a warning: the question goes out regardless.
func (j *EvalJudge) setup(ctx context.Context) {
	j.setupOnce.Do(func() {
		if err := j.classifier.Setup(ctx); err != nil {
			slog.WarnContext(ctx, "eval: the judge could not set its classifier up", "error", err)
		}
	})
}

// needsReason reports whether a verdict is worth explaining: a no, or an unsure
// one.
func (j *EvalJudge) needsReason(v JudgeVerdict) bool {
	unsure := v.Confidence != nil && *v.Confidence < j.explainBelow
	return v.Verdict == VerdictNo || unsure
}

// askTwice is the classifier's answer, asked once more if it failed.
func askTwice[R any](ctx context.Context, j *EvalJudge, ask func() (R, error)) (R, error) {
	j.setup(ctx)
	var (
		answers R
		err     error
	)
	for attempt := 1; attempt <= 2; attempt++ {
		answers, err = ask()
		if err == nil {
			slog.DebugContext(ctx, "eval: judge answered", "answers", answers)
			return answers, nil
		}
		if attempt == 1 {
			slog.WarnContext(ctx, "eval: judge question failed, asking again", "error", err)
		} else {
			slog.ErrorContext(ctx, "eval: judge question failed", "error", err)
		}
	}
	return answers, err
}

// numbered is one entry of the conversation, with each bot reply joined into
// one numbered turn.
type numbered struct {
	role    string
	turn    int
	content string
}

// numberedTurns joins each bot reply into one numbered turn. A bot turn is a run
// of reply segments with nothing else between them.
func numberedTurns(transcript []transcriptEntry) []numbered {
	var (
		entries []numbered
		turn    int
	)
	for _, e := range transcript {
		if e.role == roleAssistant {
			if n := len(entries); n > 0 && entries[n-1].role == roleBot {
				entries[n-1].content += " " + e.content
				continue
			}
			turn++
			entries = append(entries, numbered{role: roleBot, turn: turn, content: e.content})
			continue
		}
		entries = append(entries, numbered{role: e.role, content: e.content})
	}
	return entries
}

// conversation is the conversation as the state carries it: a speaker, the
// text, and a bot turn's number.
func conversation(entries []numbered) []ordered.Object {
	out := make([]ordered.Object, 0, len(entries))
	for _, e := range entries {
		obj := ordered.Object{{Key: "speaker", Value: e.role}}
		if e.role == roleBot {
			obj.Set("turn", e.turn)
		}
		obj.Set("text", e.content)
		out = append(out, obj)
	}
	return out
}

// instructions writes a question's instructions: the question, the labeled
// criterion, and the notes. The criterion gets its final punctuation so the note
// after it reads as a new sentence, and every question carries the
// transcription note.
func instructions(question, label, criterion, note string) string {
	criterion = strings.TrimSpace(criterion)
	if !strings.HasSuffix(criterion, ".") && !strings.HasSuffix(criterion, "!") &&
		!strings.HasSuffix(criterion, "?") {
		criterion += "."
	}
	parts := []string{question, label + ": " + criterion}
	if note != "" {
		parts = append(parts, note)
	}
	return strings.Join(append(parts, transcriptionNote), " ")
}

// replyVerdict is a reply's verdict: the choice of yes, no or continue.
func replyVerdict(answer classifier.ChoiceResult, question classifier.ChoiceQuestion) JudgeVerdict {
	probabilities := ordered.Object{}
	for _, o := range question.OptionNames() {
		if p, ok := answer.Probabilities[o]; ok {
			probabilities.Set(o, p)
		}
	}
	raw, _ := ordered.Marshal(ordered.Object{
		{Key: "choice", Value: answer.Choice},
		{Key: "probabilities", Value: probabilities},
		{Key: "confidence", Value: answer.Confidence},
	})
	confidence := answer.Confidence
	return JudgeVerdict{
		Verdict:       answer.Choice,
		Reason:        describeProbabilities(probabilities),
		RawResponse:   string(raw),
		Confidence:    &confidence,
		Probabilities: maps.Clone(answer.Probabilities),
	}
}

// yesNoVerdict is a yes/no verdict from the probability of yes.
func yesNoVerdict(answer classifier.YesNoResult) JudgeVerdict {
	p := answer.Probability
	verdict := VerdictNo
	if answer.IsYes() {
		verdict = VerdictYes
	}
	raw, _ := ordered.Marshal(ordered.Object{{Key: "probability", Value: p}})
	confidence := max(p, 1-p)
	return JudgeVerdict{
		Verdict:       verdict,
		Reason:        fmt.Sprintf("P(yes)=%.2f", p),
		RawResponse:   string(raw),
		Confidence:    &confidence,
		Probabilities: map[string]float64{VerdictYes: p, VerdictNo: 1 - p},
	}
}

// describeProbabilities writes the probability of each outcome, as a verdict's
// reason.
func describeProbabilities(probabilities ordered.Object) string {
	parts := make([]string, 0, len(probabilities))
	for _, f := range probabilities {
		parts = append(parts, fmt.Sprintf("P(%s)=%.2f", f.Key, f.Value))
	}
	return strings.Join(parts, ", ")
}

// failed is the verdict a failed question gives.
func failed(verdict string) JudgeVerdict {
	return JudgeVerdict{Verdict: verdict, Reason: judgeFailed}
}

// withReason is the verdict with the explainer's reason in place of the
// probabilities, when it gave one.
func withReason(verdict JudgeVerdict, reason string) JudgeVerdict {
	if reason != "" {
		verdict.Reason = reason
	}
	return verdict
}

// argsOrEmpty is the call's arguments, or an empty object for a call that
// carried none, so the judge is always shown the same shape.
func argsOrEmpty(args map[string]any) map[string]any {
	if args == nil {
		return map[string]any{}
	}
	return args
}

// cacheKey hashes what a question was about: its kind, criterion and state.
func cacheKey(parts ...any) string {
	encoded, err := ordered.Marshal(parts)
	if err != nil {
		encoded = fmt.Append(nil, parts...)
	}
	sum := sha256.Sum256(encoded)
	return hex.EncodeToString(sum[:])
}

// explainer asks an LLM why the classifier gave its verdicts, telling it each
// verdict.
//
// Each question names the reason it wants in a response schema, so a service
// that can enforce one answers in exactly that shape. A question whose call
// fails, or whose answer gives no reason, is asked once more. Its answers are
// cached by question and conversation, so explaining the same verdict twice
// costs one call. A reason is empty when it was still not given.
type explainer struct {
	inf       llm.Inferencer
	maxTokens int

	mu    sync.Mutex
	cache map[string]string
}

func newExplainer(inf llm.Inferencer, maxTokens int) *explainer {
	return &explainer{inf: inf, maxTokens: maxTokens, cache: map[string]string{}}
}

// explain is why the bot's latest reply got verdict on criterion: a
// one-sentence reason, or "" when none was given.
func (e *explainer) explain(
	ctx context.Context, transcript []transcriptEntry, criterion string, verdict JudgeVerdict,
) string {
	return e.reason(ctx, transcript, fmt.Sprintf(explainReplyAsk, satisfies(verdict), criterion))
}

// explainCall is why a tool call the bot made got verdict on criterion: a
// one-sentence reason, or "" when none was given. The ask names the call and
// its arguments, and the conversation so far is context.
func (e *explainer) explainCall(
	ctx context.Context, transcript []transcriptEntry, name string, args map[string]any, criterion string,
	verdict JudgeVerdict,
) string {
	encoded, err := ordered.Marshal(argsOrEmpty(args))
	if err != nil {
		encoded = []byte("{}")
	}
	return e.reason(ctx, transcript, fmt.Sprintf(explainCallAsk, name, encoded, satisfies(verdict), criterion))
}

// reasonSchema is the shape a reason is asked for in: an object holding only
// the reason.
const reasonSchema = `{"type":"object","properties":{"reason":{"type":"string"}},` +
	`"required":["reason"],"additionalProperties":false}`

// reason is the reason the explainer gives to ask after the spoken
// conversation, asked once more when the first answer gives none.
func (e *explainer) reason(ctx context.Context, transcript []transcriptEntry, ask string) string {
	messages := make([]frames.Message, 0, len(transcript))
	for _, t := range transcript {
		switch t.role {
		case roleUser:
			messages = append(messages, frames.Message{Role: frames.RoleUser, Text: t.content})
		case roleAssistant:
			messages = append(messages, frames.Message{Role: frames.RoleAssistant, Text: t.content})
		}
	}
	key := cacheKey(ask, messageList(messages))
	e.mu.Lock()
	cached, ok := e.cache[key]
	e.mu.Unlock()
	if ok {
		return cached
	}

	reason := ""
	for range 2 {
		if response, ok := e.ask(ctx, messages, ask); ok {
			reason = parseReason(response)
		}
		if reason != "" {
			break
		}
	}
	e.mu.Lock()
	e.cache[key] = reason
	e.mu.Unlock()
	return reason
}

// ask is the explainer's raw answer, in the reason's shape, reporting false
// when the call failed or said nothing.
func (e *explainer) ask(ctx context.Context, messages []frames.Message, ask string) (string, bool) {
	// Copy the conversation and append the transient ask, so neither the ask
	// nor the answer ever lands in the conversation the judge keeps.
	convo := frames.NewLLMContext("")
	convo.SetMessages(messages)
	convo.AddUserMessage(ask)

	callCtx, cancel := context.WithTimeout(ctx, explainerTimeout)
	defer cancel()
	response, err := e.inf.RunInference(callCtx, convo, llm.InferenceOptions{
		MaxTokens:         e.maxTokens,
		SystemInstruction: explainSystemInstruction,
		ResponseSchema:    json.RawMessage(reasonSchema),
	})
	if err != nil {
		slog.ErrorContext(ctx, "eval: explainer call failed", "error", err)
		return "", false
	}
	if response == "" {
		slog.ErrorContext(ctx, "eval: explainer returned an empty response")
		return "", false
	}
	return response, true
}

// messageList is the messages as a cache key takes them.
func messageList(messages []frames.Message) []ordered.Object {
	out := make([]ordered.Object, 0, len(messages))
	for _, m := range messages {
		out = append(out, ordered.Object{{Key: "role", Value: string(m.Role)}, {Key: "content", Value: m.Text}})
	}
	return out
}

// satisfies is a yes or a no as the ask words it.
func satisfies(verdict JudgeVerdict) string {
	if verdict.Passed() {
		return "satisfies"
	}
	return "does not satisfy"
}

// parseReason is the reason in the explainer's answer: its JSON reason, or the
// answer itself when it is plain text.
func parseReason(response string) string {
	if reason := strings.TrimSpace(textOf(jsonObject(response)["reason"])); reason != "" {
		return reason
	}
	return strings.TrimSpace(response)
}

// jsonObject is the JSON object in an answer, or nil when there is none. It
// tolerates code fences and reads the first object it finds, ignoring anything
// around it: some models ignore "respond ONLY with JSON" and wrap the answer in
// prose.
func jsonObject(response string) map[string]any {
	cleaned := strings.TrimSpace(response)
	if strings.HasPrefix(cleaned, "```") {
		cleaned = strings.TrimSpace(fenced.ReplaceAllString(cleaned, ""))
	}
	start := strings.Index(cleaned, "{")
	if start < 0 {
		return nil
	}
	var obj map[string]any
	if json.NewDecoder(strings.NewReader(cleaned[start:])).Decode(&obj) != nil {
		return nil
	}
	return obj
}

// textOf writes a JSON value as the text it holds.
func textOf(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	default:
		return fmt.Sprint(t)
	}
}

var _ Judge = (*EvalJudge)(nil)
