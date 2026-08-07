package langchain_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/langchain"
)

// errChain is the failure the fake chains report.
//
//nolint:gochecknoglobals // sentinel error
var errChain = errors.New("chain failed")

// recorder captures the frames a chain's output produces, in order.
type recorder struct {
	mu     sync.Mutex
	names  []string
	text   strings.Builder
	errMsg string
}

func (r *recorder) down(f frames.Frame) {
	r.mu.Lock()
	defer r.mu.Unlock()
	switch fr := f.(type) {
	case *frames.LLMFullResponseStartFrame:
		r.names = append(r.names, "start")
	case *frames.LLMTextFrame:
		r.names = append(r.names, "text")
		r.text.WriteString(fr.Text)
	case *frames.LLMFullResponseEndFrame:
		r.names = append(r.names, "end")
	}
}

func (r *recorder) up(f frames.Frame) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if ef, ok := f.(*frames.ErrorFrame); ok {
		r.errMsg = ef.Error
	}
}

func (r *recorder) snapshot() ([]string, string, string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.names...), r.text.String(), r.errMsg
}

// runChain drives the given frames through a langchain processor and returns
// what came out.
func runChain(t *testing.T, chain langchain.Chain, in ...frames.Frame) *recorder {
	t.Helper()
	rec := &recorder{}
	p := langchain.New(chain)
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{
		OnReachedDownstream: rec.down,
		OnReachedUpstream:   rec.up,
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrames(in)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not finish")
	}
	return rec
}

// contextWith builds an LLM context frame carrying the given user message.
func contextWith(text string) *frames.LLMContextFrame {
	c := frames.NewLLMContext("be brief")
	c.AddUserMessage(text)
	return frames.NewLLMContextFrame(c)
}

// echo is a chain that streams its input back one word at a time.
func echo(_ context.Context, input string, emit func(string) error) error {
	for w := range strings.FieldsSeq(input) {
		if err := emit(w); err != nil {
			return err
		}
	}
	return nil
}

// TestBracketsChainOutput is the contract that lets a chain stand in for an LLM:
// the streamed tokens must be wrapped in the response start/end frames the
// assistant aggregator and TTS expect.
func TestBracketsChainOutput(t *testing.T) {
	rec := runChain(t, echo, contextWith("hello there world"))

	names, text, _ := rec.snapshot()
	want := []string{"start", "text", "text", "text", "end"}
	if !equalStrings(names, want) {
		t.Errorf("frames = %v, want %v", names, want)
	}
	if text != "hellothereworld" {
		t.Errorf("streamed text = %q, want the tokens in order", text)
	}
}

// TestPassesInputToChain checks the chain receives the latest user message.
func TestPassesInputToChain(t *testing.T) {
	got := make(chan string, 1)
	capture := func(_ context.Context, input string, _ func(string) error) error {
		got <- input
		return nil
	}
	runChain(t, capture, contextWith("  book me a table  "))

	select {
	case in := <-got:
		// The text is trimmed before the chain sees it.
		if in != "book me a table" {
			t.Errorf("chain input = %q, want the trimmed user message", in)
		}
	default:
		t.Fatal("the chain was never invoked")
	}
}

// TestUsesLatestUserMessage checks the most recent user turn wins, not the first.
func TestUsesLatestUserMessage(t *testing.T) {
	c := frames.NewLLMContext("")
	c.AddUserMessage("first question")
	c.AddAssistantMessage("an answer")
	c.AddUserMessage("second question")

	got := make(chan string, 1)
	capture := func(_ context.Context, input string, _ func(string) error) error {
		got <- input
		return nil
	}
	runChain(t, capture, frames.NewLLMContextFrame(c))

	if in := <-got; in != "second question" {
		t.Errorf("chain input = %q, want the most recent user message", in)
	}
}

// TestSkipsToolResultTurns checks a tool-result message — stored under the user
// role but not something a person said — is not fed to the chain as input.
func TestSkipsToolResultTurns(t *testing.T) {
	c := frames.NewLLMContext("")
	c.AddUserMessage("weather in Paris?")
	c.AddAssistantToolCall(frames.ToolCall{ID: "a", Name: "get_weather"})
	c.AddToolResult(frames.ToolResult{ID: "a", Name: "get_weather", Content: "sunny"})

	got := make(chan string, 1)
	capture := func(_ context.Context, input string, _ func(string) error) error {
		got <- input
		return nil
	}
	runChain(t, capture, frames.NewLLMContextFrame(c))

	if in := <-got; in != "weather in Paris?" {
		t.Errorf("chain input = %q, want the real user question, not the tool result", in)
	}
}

// TestNoUserMessageForwardsFrame checks a context with nothing to answer is
// passed along untouched rather than invoking the chain with empty input.
func TestNoUserMessageForwardsFrame(t *testing.T) {
	called := false
	chain := func(context.Context, string, func(string) error) error {
		called = true
		return nil
	}

	c := frames.NewLLMContext("be brief")
	c.AddAssistantMessage("hello")
	rec := runChain(t, chain, frames.NewLLMContextFrame(c))

	if called {
		t.Error("the chain must not run without a user message")
	}
	if names, _, _ := rec.snapshot(); len(names) != 0 {
		t.Errorf("frames = %v, want no LLM response frames", names)
	}
}

// TestEmptyTokensSkipped checks the chain can emit empty strings without
// producing empty text frames downstream.
func TestEmptyTokensSkipped(t *testing.T) {
	chain := func(_ context.Context, _ string, emit func(string) error) error {
		for _, tok := range []string{"a", "", "b", ""} {
			if err := emit(tok); err != nil {
				return err
			}
		}
		return nil
	}
	rec := runChain(t, chain, contextWith("go"))

	names, text, _ := rec.snapshot()
	want := []string{"start", "text", "text", "end"}
	if !equalStrings(names, want) {
		t.Errorf("frames = %v, want %v; empty tokens should not become frames", names, want)
	}
	if text != "ab" {
		t.Errorf("streamed text = %q, want %q", text, "ab")
	}
}

// TestChainErrorIsReported checks a failing chain surfaces as a pipeline error —
// and that the response is still closed, so downstream stages are not left
// waiting for an end frame that never comes.
func TestChainErrorIsReported(t *testing.T) {
	chain := func(_ context.Context, _ string, emit func(string) error) error {
		if err := emit("partial"); err != nil {
			return err
		}
		return errChain
	}
	rec := runChain(t, chain, contextWith("go"))

	names, _, errMsg := rec.snapshot()
	if errMsg == "" {
		t.Error("a chain failure should be reported upstream as an error frame")
	}
	if len(names) == 0 || names[len(names)-1] != "end" {
		t.Errorf("frames = %v, want the response closed with an end frame", names)
	}
}

// TestOtherFramesPassThrough checks frames that are not LLM contexts are
// forwarded untouched.
func TestOtherFramesPassThrough(t *testing.T) {
	seen := make(chan struct{}, 1)
	p := langchain.New(echo)
	task := pipeline.NewTask(pipeline.New(p), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.TTSSpeakFrame); ok {
				select {
				case seen <- struct{}{}:
				default:
				}
			}
		},
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTTSSpeakFrame("hello"))
	select {
	case <-seen:
	case <-time.After(5 * time.Second):
		t.Fatal("an unrelated frame was not forwarded")
	}
	task.StopWhenDone()
	<-done
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}
