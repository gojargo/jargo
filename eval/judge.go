package eval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync"

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
	// Reason is a one-sentence justification.
	Reason string
	// RawResponse is the judge's raw answer, for diagnostics.
	RawResponse string
	// Confidence is how sure the judge is of the verdict, from 0 to 1; nil when
	// it did not say.
	Confidence *float64
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

// defaultJudgeMaxTokens caps the explainer's answer: enough for a verdict and a
// short reason, and no room for the model to start explaining itself at length.
const defaultJudgeMaxTokens = 200

// defaultExplainBelow is the confidence under which a yes is explained too.
const defaultExplainBelow = 0.75

// judgeSystemInstruction is what the explainer is told when it judges a reply.
const judgeSystemInstruction = "You are a strict but fair judge evaluating a conversation between a user and a " +
	"bot under test. The 'user' messages are the user; the 'assistant' messages are " +
	"the bot's replies. Judge only the bot's most recent reply, which may have " +
	"arrived as several consecutive 'assistant' messages, against the given " +
	"criterion, using the earlier turns only as context. The reply may still be " +
	"streaming in. " +
	"When the bot spoke its reply, the 'assistant' text is an automatic speech-to-text " +
	"transcription, so it may contain homophones, misspellings, split or merged words, and " +
	"missing punctuation. Always judge it by the intended spoken meaning, never by its exact " +
	"spelling. In particular, treat a number as the same value whether it is spelled out, " +
	"written as a digit, or transcribed as a homophone: 'for' and 'fore' mean 'four' (4), and " +
	"'to' and 'too' mean 'two' (2). Never answer 'no' solely because of a transcription error " +
	"when the intended spoken meaning satisfies the criterion. " +
	"Respond ONLY with a JSON object on a single line containing two fields: " +
	`{"verdict": "yes" | "no" | "continue", "reason": "<one short sentence>"}. ` +
	`Use "yes" if the reply satisfies the criterion. ` +
	`Use "continue" if the bot has not given its answer yet: it says it is checking, ` +
	"looking something up, fetching, working on something, or that it will report back. " +
	"The answer is still coming, so there is nothing to judge yet. This holds however " +
	`long and however fluent the reply is: "The system is checking the current ` +
	`conditions for you right now." is waiting, not answering. A greeting or an ` +
	`obviously incomplete fragment is also "continue". ` +
	`Use "no" only when the bot has given its answer and that answer fails the ` +
	`criterion. If the bot has not answered yet, always use "continue", never "no". ` +
	"Do not include any other text, explanation, or markdown."

// judgeAsk is the transient final user message appended for the explainer's
// call. The conversation it refers to is the one the judge kept; this only poses
// the question, and is never stored in that conversation.
const judgeAsk = "Does the bot's most recent reply satisfy this criterion?\n\n" +
	"Criterion: %s\n\n" +
	"Answer yes, no, or continue."

// judgeFinalSystemInstruction is what the explainer is told when a reply may not
// be judged continue: every judged reply is a final answer, so the verdict is
// yes or no.
const judgeFinalSystemInstruction = "You are a strict but fair judge evaluating a conversation between a user and a " +
	"bot under test. The 'user' messages are the user; the 'assistant' messages are " +
	"the bot's replies. Judge only the bot's most recent reply, which may have " +
	"arrived as several consecutive 'assistant' messages, against the given " +
	"criterion, using the earlier turns only as context. The reply is the bot's final " +
	"answer. " +
	"When the bot spoke its reply, the 'assistant' text is an automatic speech-to-text " +
	"transcription, so it may contain homophones, misspellings, split or merged words, and " +
	"missing punctuation. Always judge it by the intended spoken meaning, never by its exact " +
	"spelling. In particular, treat a number as the same value whether it is spelled out, " +
	"written as a digit, or transcribed as a homophone: 'for' and 'fore' mean 'four' (4), and " +
	"'to' and 'too' mean 'two' (2). Never answer 'no' solely because of a transcription error " +
	"when the intended spoken meaning satisfies the criterion. " +
	"Respond ONLY with a JSON object on a single line containing two fields: " +
	`{"verdict": "yes" | "no", "reason": "<one short sentence>"}. ` +
	`Use "yes" if the reply satisfies the criterion and "no" if it does not. ` +
	"Do not include any other text, explanation, or markdown."

// judgeFinalAsk is the ask that goes with judgeFinalSystemInstruction.
const judgeFinalAsk = "Does the bot's most recent reply satisfy this criterion?\n\n" +
	"Criterion: %s\n\n" +
	"Answer yes or no."

// judgeCallSystemInstruction is what the explainer is told for a tool call. The
// call is the subject and the conversation is context for it, so the verdict is
// yes or no: a call is not a partial reply, and there is nothing to wait for.
const judgeCallSystemInstruction = "You are a strict but fair judge evaluating a function call made by a bot " +
	"under test in a conversation with a user. The 'user' messages are the user; the " +
	"'assistant' messages are the bot's replies so far, given only as context for the " +
	"call. Judge only the call you are asked about, by its name and its arguments, " +
	"against the given criterion. " +
	"When the bot spoke its replies, the 'assistant' text is an automatic speech-to-text " +
	"transcription, so it may contain homophones, misspellings, split or merged words, and " +
	"missing punctuation; judge it by the intended spoken meaning. " +
	"Respond ONLY with a JSON object on a single line containing two fields: " +
	`{"verdict": "yes" | "no", "reason": "<one short sentence>"}. ` +
	`Use "yes" if the call satisfies the criterion and "no" if it does not. ` +
	"Do not include any other text, explanation, or markdown."

// judgeCallAsk is the ask for a criterion on a tool call. It names the call and
// gives its arguments as JSON, so the verdict is about that call rather than
// about what the bot said around it.
const judgeCallAsk = "The bot called the function `%s` with arguments `%s`. " +
	"Does this call satisfy this criterion?\n\n" +
	"Criterion: %s\n\n" +
	"Answer yes or no."

// The reasons a verdict carries when there was nothing better to give.
const (
	// explainerFailed is the reason a verdict carries when the explainer's call
	// failed.
	explainerFailed = "explainer call failed"
	// noReason is the reason a verdict carries when the judge gave the verdict
	// without one.
	noReason = "(no reason given)"
	// judgeFailed is the reason a verdict carries when its question failed.
	judgeFailed = "judge call failed"
)

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
	// MaxTokens caps the explainer's answer; zero uses 200, enough for a verdict
	// and a short reason.
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
// ExplainBelow, but never for a continue. The explainer makes its own judgement
// over the same conversation and never changes a verdict: when it agrees, its
// reason is used; when it disagrees, the reason says so.
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
		j.explainer = newExplainer(cfg.Explainer, maxTokens, j.allowContinue)
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
			verdict = explained(verdict, j.explainer.explain(ctx, all, criterion))
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
			verdict = explained(verdict, j.explainer.explainCall(ctx, all, name, args, criterion))
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
		Verdict:     answer.Choice,
		Reason:      describeProbabilities(probabilities),
		RawResponse: string(raw),
		Confidence:  &confidence,
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
		Verdict:     verdict,
		Reason:      fmt.Sprintf("P(yes)=%.2f", p),
		RawResponse: string(raw),
		Confidence:  &confidence,
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

// explained is the verdict with the explainer's reason, noting when the
// explainer disagreed. The classifier's verdict always stands.
func explained(verdict, explanation JudgeVerdict) JudgeVerdict {
	var reason string
	if explanation.Verdict == verdict.Verdict {
		reason = verdict.Reason
		if explanation.Reason != "" && explanation.Reason != noReason {
			reason = fmt.Sprintf("%s (%s)", explanation.Reason, verdict.Reason)
		}
	} else {
		reason = fmt.Sprintf("%s; the explainer judged %s: %s", verdict.Reason, explanation.Verdict, explanation.Reason)
	}
	return JudgeVerdict{
		Verdict:     verdict.Verdict,
		Reason:      reason,
		RawResponse: verdict.RawResponse,
		Confidence:  verdict.Confidence,
	}
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

// explainer asks an LLM the question a classifier answered, for the reason
// behind it. Its answers are cached by question and conversation, so explaining
// the same verdict twice costs one call.
type explainer struct {
	inf           llm.Inferencer
	maxTokens     int
	allowContinue bool

	mu    sync.Mutex
	cache map[string]JudgeVerdict
}

func newExplainer(inf llm.Inferencer, maxTokens int, allowContinue bool) *explainer {
	return &explainer{inf: inf, maxTokens: maxTokens, allowContinue: allowContinue, cache: map[string]JudgeVerdict{}}
}

// explain judges whether the bot's latest reply satisfies criterion, and says
// why.
func (e *explainer) explain(ctx context.Context, transcript []transcriptEntry, criterion string) JudgeVerdict {
	if e.allowContinue {
		return e.evaluate(ctx, transcript, judgeSystemInstruction, fmt.Sprintf(judgeAsk, criterion))
	}
	v := e.evaluate(ctx, transcript, judgeFinalSystemInstruction, fmt.Sprintf(judgeFinalAsk, criterion))
	if v.Verdict == VerdictContinue {
		// An answer that ignored the yes/no instructions counts as a no.
		return JudgeVerdict{Verdict: VerdictNo, Reason: v.Reason, RawResponse: v.RawResponse}
	}
	return v
}

// explainCall judges whether a tool call the bot made satisfies criterion, and
// says why. The ask names the call and its arguments, and the conversation so
// far is context.
func (e *explainer) explainCall(
	ctx context.Context, transcript []transcriptEntry, name string, args map[string]any, criterion string,
) JudgeVerdict {
	encoded, err := ordered.Marshal(argsOrEmpty(args))
	if err != nil {
		encoded = []byte("{}")
	}
	return e.evaluate(ctx, transcript, judgeCallSystemInstruction, fmt.Sprintf(judgeCallAsk, name, encoded, criterion))
}

// evaluate asks the question over the spoken conversation, a reply being judged
// on what was said.
func (e *explainer) evaluate(
	ctx context.Context, transcript []transcriptEntry, instruction, ask string,
) JudgeVerdict {
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
	v, ok := e.cache[key]
	e.mu.Unlock()
	if ok {
		return v
	}

	response, ok := e.ask(ctx, messages, instruction, ask)
	if ok {
		v = parseVerdict(response)
	} else {
		v = JudgeVerdict{Verdict: VerdictNo, Reason: explainerFailed}
	}
	e.mu.Lock()
	e.cache[key] = v
	e.mu.Unlock()
	return v
}

// ask is the explainer's raw answer, reporting false when the call failed or
// said nothing.
func (e *explainer) ask(
	ctx context.Context, messages []frames.Message, instruction, ask string,
) (string, bool) {
	// Copy the conversation and append the transient ask, so neither the ask
	// nor the answer ever lands in the conversation the judge keeps.
	convo := frames.NewLLMContext("")
	convo.SetMessages(messages)
	convo.AddUserMessage(ask)

	response, err := e.inf.RunInference(ctx, convo, llm.InferenceOptions{
		MaxTokens:         e.maxTokens,
		SystemInstruction: instruction,
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

// verdictJSON is the object the explainer is asked to answer with.
type verdictJSON struct {
	Verdict any `json:"verdict"`
	Reason  any `json:"reason"`
}

// parseVerdict reads a judgement out of the explainer's response. It tolerates
// extra whitespace and code fences, and reads the first JSON object it finds,
// ignoring anything around it: some models ignore "respond ONLY with JSON" and
// wrap the verdict in prose. Absent a verdict it reports a no, so an
// unparseable judgement never silently passes.
func parseVerdict(response string) JudgeVerdict {
	cleaned := strings.TrimSpace(response)
	if strings.HasPrefix(cleaned, "```") {
		cleaned = strings.TrimSpace(fenced.ReplaceAllString(cleaned, ""))
	}

	if start := strings.Index(cleaned, "{"); start >= 0 {
		var obj verdictJSON
		dec := json.NewDecoder(strings.NewReader(cleaned[start:]))
		if err := dec.Decode(&obj); err == nil {
			verdict := strings.ToLower(strings.TrimSpace(textOf(obj.Verdict)))
			switch verdict {
			case VerdictYes, VerdictNo, VerdictContinue:
			default:
				verdict = VerdictNo
			}
			reason := strings.TrimSpace(textOf(obj.Reason))
			if reason == "" {
				reason = noReason
			}
			return JudgeVerdict{Verdict: verdict, Reason: reason, RawResponse: response}
		}
	}

	// Fall back to scanning for a verdict keyword in the raw text.
	lowered := strings.ToLower(cleaned)
	hasYes, hasNo := strings.Contains(lowered, VerdictYes), strings.Contains(lowered, VerdictNo)
	switch {
	case strings.Contains(lowered, VerdictContinue):
		return JudgeVerdict{Verdict: VerdictContinue, Reason: "(unstructured continue)", RawResponse: response}
	case hasYes && !hasNo:
		return JudgeVerdict{Verdict: VerdictYes, Reason: "(unstructured yes)", RawResponse: response}
	case hasNo && !hasYes:
		return JudgeVerdict{Verdict: VerdictNo, Reason: "(unstructured no)", RawResponse: response}
	}
	return JudgeVerdict{
		Verdict:     VerdictNo,
		Reason:      fmt.Sprintf("could not parse explainer response: %q", response),
		RawResponse: response,
	}
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
