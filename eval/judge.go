package eval

import (
	"context"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// Judge decides whether a bot reply satisfies a natural-language criterion — it
// backs a scenario's `judge:` assertion. Implementations typically run a
// one-shot LLM inference; the harness treats the judge as optional, so a
// scenario without any `judge:` assertions needs none.
type Judge interface {
	// Evaluate reports whether reply meets criterion, with a short reason to
	// surface when it does not.
	Evaluate(ctx context.Context, criterion, reply string) (pass bool, reason string, err error)
}

// judgeInstruction steers the grading model toward a terse, parseable verdict.
const judgeInstruction = "You grade a voice assistant's reply against a single criterion. " +
	"Decide whether the reply satisfies the criterion. Start your response with the word " +
	"PASS if it does or FAIL if it does not, then give a brief reason. Judge only against the " +
	"stated criterion; do not invent extra requirements."

// LLMJudge evaluates criteria with an LLM. It runs each judgement as a one-shot
// generation off to the side of the pipeline (like the summarizer), so give it
// its own service instance — ideally a small, fast model. Verdicts are cached by
// (criterion, reply) so re-runs are stable and a repeated assertion pays only
// one round-trip.
type LLMJudge struct {
	gen llm.Generator

	mu    sync.Mutex
	cache map[string]verdict
}

// verdict is a cached judgement.
type verdict struct {
	pass   bool
	reason string
}

// NewLLMJudge builds a judge backed by gen. Any jargo LLM service works, e.g.
// eval.NewLLMJudge(chat.NewLLM(chat.LLMConfig{APIKey: key})).
func NewLLMJudge(gen llm.Generator) *LLMJudge {
	return &LLMJudge{gen: gen, cache: make(map[string]verdict)}
}

// Evaluate grades reply against criterion, caching the result.
func (j *LLMJudge) Evaluate(ctx context.Context, criterion, reply string) (bool, string, error) {
	key := criterion + "\x00" + reply

	j.mu.Lock()
	if v, ok := j.cache[key]; ok {
		j.mu.Unlock()
		return v.pass, v.reason, nil
	}
	j.mu.Unlock()

	convo := frames.NewLLMContext(judgeInstruction)
	convo.AddUserMessage("Criterion: " + criterion + "\n\nReply to evaluate:\n" + reply)

	var b strings.Builder
	if err := j.gen.Generate(ctx, convo, func(text string) error {
		b.WriteString(text)
		return nil
	}); err != nil {
		return false, "", err
	}
	pass, reason := parseVerdict(b.String())

	j.mu.Lock()
	j.cache[key] = verdict{pass: pass, reason: reason}
	j.mu.Unlock()

	return pass, reason, nil
}

// parseVerdict reads a PASS/FAIL judgement out of the model's response. It trusts
// a verdict word leading the first line, and otherwise falls back to whichever of
// PASS/FAIL appears first anywhere in the text. The full response is returned as
// the reason. Absent any verdict word it reports a failure, so an unparseable
// judgement never silently passes.
func parseVerdict(out string) (bool, string) {
	out = strings.TrimSpace(out)
	if out == "" {
		return false, "judge returned no output"
	}

	firstLine, _, _ := strings.Cut(out, "\n")
	upper := strings.ToUpper(strings.TrimSpace(firstLine))
	switch {
	case strings.HasPrefix(upper, "PASS"), strings.HasPrefix(upper, "YES"):
		return true, out
	case strings.HasPrefix(upper, "FAIL"), strings.HasPrefix(upper, "NO"):
		return false, out
	}

	all := strings.ToUpper(out)
	pi, fi := strings.Index(all, "PASS"), strings.Index(all, "FAIL")
	pass := pi >= 0 && (fi < 0 || pi < fi)
	return pass, out
}
