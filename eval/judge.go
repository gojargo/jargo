package eval

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"regexp"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
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
	// VerdictContinue means the reply so far is only an interim or filler
	// utterance, and the criterion should be judged again once more text arrives.
	VerdictContinue = "continue"
)

// JudgeVerdict is the outcome of a single judge call.
type JudgeVerdict struct {
	// Verdict is VerdictYes, VerdictNo or VerdictContinue.
	Verdict string
	// Reason is a one-sentence justification.
	Reason string
	// RawResponse is the judge model's raw text, for diagnostics.
	RawResponse string
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
// treats it as optional, so a scenario with no `judge:` assertion needs none.
type Judge interface {
	// AddUserMessage records a user turn in the conversation.
	AddUserMessage(text string)
	// AddAssistantMessage appends a segment of the bot's current reply.
	AddAssistantMessage(text string)
	// Evaluate judges the conversation so far against criterion. A judge that
	// cannot answer reports VerdictNo with the reason, rather than failing: an
	// unavailable judge is a failed assertion, not a broken run.
	Evaluate(ctx context.Context, criterion string) JudgeVerdict
}

// judgeSystemInstruction steers the grading model toward a parseable verdict,
// and tells it what the conversation it is reading actually is.
// defaultJudgeMaxTokens caps a verdict: enough for the verdict word and a short
// reason, and no room for the model to start explaining itself at length.
const defaultJudgeMaxTokens = 200

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
	`Use "yes" if the reply satisfies the criterion. Use "no" if the reply gives a ` +
	`substantive answer that fails it. Use "continue" if the reply so far is only an ` +
	`interim or filler utterance (e.g. "Let me check on that.", a greeting, or an ` +
	"obviously incomplete fragment) that does not yet contain enough to decide, more " +
	"text is expected. Do not include any other text, explanation, or markdown."

// judgeAsk is the transient final user message appended for the judge call. The
// conversation it refers to is the one the harness built up; this only poses the
// question, and is never stored in that conversation.
const judgeAsk = "Does the bot's most recent reply satisfy this criterion?\n\n" +
	"Criterion: %s\n\n" +
	"Answer yes, no, or continue."

// LLMJudge evaluates criteria with an LLM. It runs each judgement as a one-shot
// generation off to the side of the pipeline (like the summarizer), so give it
// its own service instance, ideally a small, fast model. Verdicts are cached by
// (criterion, conversation) so re-runs are stable and a repeated assertion over
// an unchanged conversation pays only one round-trip.
type LLMJudge struct {
	inf       llm.Inferencer
	maxTokens int

	// convo is the conversation the judge evaluates against, grown by the
	// harness over the scenario.
	convo *frames.LLMContext

	mu    sync.Mutex
	cache map[string]JudgeVerdict
}

// NewLLMJudge builds a judge backed by inf, an LLM service that can answer a
// conversation once, e.g. eval.NewLLMJudge(chat.NewLLM(chat.LLMConfig{APIKey: key})).
func NewLLMJudge(inf llm.Inferencer) *LLMJudge {
	return &LLMJudge{
		inf:       inf,
		maxTokens: defaultJudgeMaxTokens,
		convo:     frames.NewLLMContext(judgeSystemInstruction),
		cache:     make(map[string]JudgeVerdict),
	}
}

// AddUserMessage records a user turn, so a later reply is judged in context
// (a terse "that's four" answering "what is two plus two?", say).
func (j *LLMJudge) AddUserMessage(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	j.convo.AddUserMessage(text)
}

// AddAssistantMessage appends a streamed segment of the bot's current reply. The
// reply may arrive in several segments, and each is added as its own message, so
// the accumulated conversation is exactly what the judge sees: there is no
// separate commit step.
func (j *LLMJudge) AddAssistantMessage(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	j.convo.AddAssistantMessage(text)
}

// Evaluate judges whether the bot's most recent reply satisfies criterion. The
// judge's own answer is never written back into the conversation.
func (j *LLMJudge) Evaluate(ctx context.Context, criterion string) JudgeVerdict {
	messages := j.convo.Messages()
	key := cacheKey(criterion, messages)

	j.mu.Lock()
	v, cached := j.cache[key]
	j.mu.Unlock()
	if cached {
		return v
	}

	v = j.callJudge(ctx, criterion, messages)

	j.mu.Lock()
	j.cache[key] = v
	j.mu.Unlock()

	return v
}

// callJudge is a single round-trip over the conversation plus a verdict ask.
func (j *LLMJudge) callJudge(ctx context.Context, criterion string, messages []frames.Message) JudgeVerdict {
	// Copy the conversation and append a transient verdict ask, so neither the
	// ask nor the judge's answer ever lands in the persistent one.
	convo := frames.NewLLMContext(judgeSystemInstruction)
	convo.SetMessages(messages)
	convo.AddUserMessage(fmt.Sprintf(judgeAsk, criterion))

	out, err := j.inf.RunInference(ctx, convo, llm.InferenceOptions{
		MaxTokens:         j.maxTokens,
		SystemInstruction: judgeSystemInstruction,
	})
	if err != nil {
		slog.Error("eval: judge call failed", "err", err)
		return JudgeVerdict{Verdict: VerdictNo, Reason: "judge call failed: " + err.Error()}
	}
	if out == "" {
		return JudgeVerdict{Verdict: VerdictNo, Reason: "judge returned empty response"}
	}
	return parseVerdict(out)
}

// cacheKey hashes a (criterion, conversation) pair for cache lookup. The
// conversation a judge holds is only ever the turns and reply segments it was
// fed, so a message's role and text identify it; the length prefix is what keeps
// two different splits of the same text apart.
func cacheKey(criterion string, messages []frames.Message) string {
	h := sha256.New()
	_, _ = fmt.Fprintf(h, "%d:%s", len(criterion), criterion)
	for _, m := range messages {
		_, _ = fmt.Fprintf(h, "\x00%s\x00%d:%s", m.Role, len(m.Text), m.Text)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// verdictJSON is the object the judge is asked to answer with.
type verdictJSON struct {
	Verdict string `json:"verdict"`
	Reason  string `json:"reason"`
}

// parseVerdict reads a judgement out of the model's response. It tolerates extra
// whitespace and code fences, and reads the first JSON object it finds, ignoring
// anything around it: some judge models ignore "respond ONLY with JSON" and wrap
// the verdict in prose. Absent a verdict it reports a no, so an unparseable
// judgement never silently passes.
func parseVerdict(response string) JudgeVerdict {
	cleaned := strings.TrimSpace(response)
	if strings.HasPrefix(cleaned, "```") {
		cleaned = strings.TrimSpace(fenced.ReplaceAllString(cleaned, ""))
	}

	if start := strings.Index(cleaned, "{"); start >= 0 {
		var obj verdictJSON
		dec := json.NewDecoder(strings.NewReader(cleaned[start:]))
		if err := dec.Decode(&obj); err == nil {
			verdict := strings.ToLower(strings.TrimSpace(obj.Verdict))
			switch verdict {
			case VerdictYes, VerdictNo, VerdictContinue:
			default:
				verdict = VerdictNo
			}
			reason := strings.TrimSpace(obj.Reason)
			if reason == "" {
				reason = "(no reason given)"
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
		Reason:      fmt.Sprintf("could not parse judge response: %q", response),
		RawResponse: response,
	}
}
