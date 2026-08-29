package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	errs "github.com/gojargo/jargo/utils/errors"
)

// mustParams converts the conversation and fails the test if the conversion
// did.
func (s *Service) mustParams(t *testing.T, convo *frames.LLMContext) sdk.MessageNewParams {
	t.Helper()
	p, err := s.newParams(convo, adapter.Options{})
	if err != nil {
		t.Fatalf("newParams: %v", err)
	}
	return p
}

func TestSupportsPrefill(t *testing.T) {
	cases := map[string]bool{
		// Direct model ids that still support prefill.
		"claude-haiku-4-5":           true,
		"claude-haiku-4-5-20251001":  true,
		"claude-sonnet-4-5":          true,
		"claude-3-5-sonnet-20241022": true,
		"claude-opus-4-1":            true,
		// Direct ids that dropped prefill.
		"claude-opus-4-8":   false,
		"claude-sonnet-4-6": false,
		// Bedrock ids (region-prefixed) are matched as substrings.
		"us.anthropic.claude-3-5-haiku-20241022-v1:0": true,
		"us.anthropic.claude-sonnet-4-6-v1:0":         false,
		"us.anthropic.claude-opus-4-8-v1:0":           false,
		// Non-Claude models are unaffected (nothing is injected).
		"amazon.titan-text-express-v1": true,
		"":                             true,
	}
	for model, want := range cases {
		if got := supportsPrefill(model); got != want {
			t.Errorf("supportsPrefill(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestNewParamsPrefillFixup(t *testing.T) {
	convo := frames.NewLLMContext("system")
	convo.AddUserMessage("hi")
	convo.AddAssistantMessage("hello") // context ends on an assistant message

	// A no-prefill model gets a trailing user message injected.
	noPrefill := NewLLM(Config{Model: "claude-opus-4-8"}).mustParams(t, convo).Messages
	if n := len(noPrefill); n != 3 || noPrefill[n-1].Role != sdk.MessageParamRoleUser {
		t.Fatalf("no-prefill model: want 3 messages ending in a user turn, got %d", n)
	}

	// A prefill-supported model keeps the assistant message last.
	prefill := NewLLM(Config{Model: "claude-haiku-4-5"}).mustParams(t, convo).Messages
	if n := len(prefill); n != 2 || prefill[n-1].Role != sdk.MessageParamRoleAssistant {
		t.Fatalf("prefill model: want the assistant message to stay last, got %d messages", n)
	}
}

func TestNewParamsAppliesSampling(t *testing.T) {
	temp, topP := 0.3, 0.9
	topK := int64(40)
	s := NewLLM(Config{Temperature: &temp, TopP: &topP, TopK: &topK})
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")
	b, err := json.Marshal(s.mustParams(t, convo))
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, want := range []string{`"temperature":0.3`, `"top_p":0.9`, `"top_k":40`} {
		if !strings.Contains(got, want) {
			t.Fatalf("params JSON %s missing %q", got, want)
		}
	}
}

func TestNewParamsOmitsUnsetSampling(t *testing.T) {
	s := NewLLM(Config{})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	b, err := json.Marshal(s.mustParams(t, convo))
	if err != nil {
		t.Fatal(err)
	}
	got := string(b)
	for _, absent := range []string{"temperature", "top_p", "top_k"} {
		if strings.Contains(got, absent) {
			t.Fatalf("params JSON %s should omit %q when unset", got, absent)
		}
	}
}

func TestNewParamsAppliesThinking(t *testing.T) {
	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")

	for mode, want := range map[string]string{
		"disabled": `"thinking":{"type":"disabled"}`,
		"adaptive": `"thinking":{"type":"adaptive"}`,
	} {
		s := NewLLM(Config{Thinking: &ThinkingConfig{Type: mode}})
		b, err := json.Marshal(s.mustParams(t, convo))
		if err != nil {
			t.Fatal(err)
		}
		if got := string(b); !strings.Contains(got, want) {
			t.Fatalf("thinking %q: params JSON %s missing %q", mode, got, want)
		}
	}

	// "enabled" carries the token budget.
	s := NewLLM(Config{MaxTokens: 4096, Thinking: &ThinkingConfig{Type: "enabled", BudgetTokens: 2048}})
	b, err := json.Marshal(s.mustParams(t, convo))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"type":"enabled"`, `"budget_tokens":2048`} {
		if !strings.Contains(string(b), want) {
			t.Fatalf("enabled thinking missing %q:\n%s", want, string(b))
		}
	}
}

func TestNewParamsOmitsThinkingWhenUnset(t *testing.T) {
	s := NewLLM(Config{})
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	b, err := json.Marshal(s.mustParams(t, convo))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "thinking") {
		t.Fatalf("params JSON should omit thinking when unset:\n%s", string(b))
	}
}

func TestToUsageMapsFields(t *testing.T) {
	u := toUsage(sdk.Usage{
		InputTokens:              100,
		OutputTokens:             20,
		CacheReadInputTokens:     80,
		CacheCreationInputTokens: 10,
	})
	// The input count is net of the cache here, so the total adds the cached
	// tokens back: 100 + 10 + 80 + 20.
	cacheRead, readOK := frames.TokenCount(u.CacheReadTokens)
	cacheWrite, writeOK := frames.TokenCount(u.CacheCreationTokens)
	if u.PromptTokens != 100 || u.CompletionTokens != 20 || u.TotalTokens != 210 ||
		!readOK || cacheRead != 80 || !writeOK || cacheWrite != 10 {
		t.Fatalf("usage = %+v (cache read %d/%v, write %d/%v)",
			u, cacheRead, readOK, cacheWrite, writeOK)
	}
}

// A cached prefix is only reused while it stays byte-identical. Recalled context
// is replaced every turn, so the breakpoint has to sit before it: inside it, the
// cache would be written on every request and never read back, which costs more
// than not caching at all.
func TestNewParamsKeepsTheCachedPrefixStableAcrossRecall(t *testing.T) {
	s := NewLLM(Config{})
	convo := frames.NewLLMContext("you are a companion")
	convo.AddUserMessage("hi")

	cachedPrefix := func() string {
		params := s.mustParams(t, convo)
		var prefix []string
		for _, block := range params.System {
			if block.CacheControl.Type != "" {
				prefix = append(prefix, block.Text)
			}
		}
		return strings.Join(prefix, "\n\n")
	}

	convo.SetRecall("I recall: the user has a cat")
	first := cachedPrefix()
	convo.SetRecall("I recall: the user has a dog")
	second := cachedPrefix()

	if first == "" {
		t.Fatal("nothing was marked for caching")
	}
	if first != second {
		t.Fatalf("the cached prefix changed when recall did:\n first: %q\nsecond: %q", first, second)
	}
	if strings.Contains(first, "cat") || strings.Contains(first, "dog") {
		t.Fatalf("recalled context is inside the cached prefix: %q", first)
	}
}

// Recall still has to reach the model — it just travels outside the breakpoint.
func TestNewParamsStillSendsRecall(t *testing.T) {
	s := NewLLM(Config{})
	convo := frames.NewLLMContext("you are a companion")
	convo.AddUserMessage("hi")
	convo.SetRecall("I recall: the user has a cat")

	b, err := json.Marshal(s.mustParams(t, convo))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "the user has a cat") {
		t.Fatalf("recall missing from the request: %s", b)
	}
}

// With caching off the system prompt travels as one uncached block.
func TestNewParamsWithoutCaching(t *testing.T) {
	off := false
	s := NewLLM(Config{EnablePromptCaching: &off})
	convo := frames.NewLLMContext("you are a companion")
	convo.AddUserMessage("hi")
	convo.SetRecall("I recall: the user has a cat")

	params := s.mustParams(t, convo)
	if len(params.System) != 1 {
		t.Fatalf("got %d system blocks, want 1", len(params.System))
	}
	if params.System[0].CacheControl.Type != "" {
		t.Fatal("cache control set although caching is off")
	}
	if !strings.Contains(params.System[0].Text, "the user has a cat") {
		t.Fatal("recall missing from the system prompt")
	}
}

// TestSDKRefusalsClassify checks a refusal from the SDK is classified by the
// status it carries, without this provider classifying it itself: the SDK
// reports the status on a field of its error, which the shared classification
// reads.
func TestSDKRefusalsClassify(t *testing.T) {
	cases := map[string]struct {
		status int
		want   errs.Category
	}{
		"rejected key":  {http.StatusUnauthorized, errs.Authentication},
		"unknown model": {http.StatusNotFound, errs.InvalidRequest},
		"rate limited":  {http.StatusTooManyRequests, errs.RateLimit},
		"a bad moment":  {http.StatusServiceUnavailable, errs.Server},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			err := fmt.Errorf("generate: %w", &sdk.Error{StatusCode: c.status})
			if got := errs.ClassifyError(err); got != c.want {
				t.Errorf("category = %q, want %q", got, c.want)
			}
		})
	}
}

// TestSonnetThinksByDefault covers which models get thinking turned off when the
// caller configured none. Sonnet 5 and later think adaptively unless told not
// to, which costs seconds before the first answer token; every other model is
// left at Anthropic's own default.
//
// Ported from upstream's _sonnet_generation, whose pattern is unanchored so that
// a Bedrock or Vertex id, which prefixes the name, still matches.
func TestSonnetThinksByDefault(t *testing.T) {
	for _, tc := range []struct {
		model string
		want  bool
	}{
		{"claude-sonnet-5", true},
		{"claude-sonnet-5-20260101", true},
		{"anthropic.claude-sonnet-5", true},
		{"claude-sonnet-12", true},
		{"claude-sonnet-4-5", false},
		{"claude-3-5-sonnet-20241022", false},
		{"claude-haiku-4-5", false},
		{"claude-opus-5", false},
		{"claude-fable-5", false},
		{"", false},
	} {
		if got := sonnetThinksByDefault(tc.model); got != tc.want {
			t.Errorf("sonnetThinksByDefault(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}
