package flows

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/provider/mistral"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/provider/openai/responses"
)

// personaReader is an LLM service as the flows manager meets it: a processor
// the persona is sent to, which then speaks under it.
type personaReader interface {
	ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error
	SystemInstruction() string
}

// TestNodePersonaReachesEveryLLM checks the persona update the manager queues
// for a node is taken up by the LLM services that keep settings of their own
// type, not only by one holding the shared settings. The manager sends the
// shared type, and these services refused it as a different type, so a flow
// ran with no persona on every one of them while the manager's own tests,
// which never hand the frame to a service, passed.
func TestNodePersonaReachesEveryLLM(t *testing.T) {
	fm, enq := newManager(t)
	n := &NodeConfig{Name: "start", RoleMessage: "be nice", TaskMessages: []frames.Message{}}
	if err := fm.Initialize(context.Background(), n); err != nil {
		t.Fatalf("Initialize: %v", err)
	}
	var update *frames.LLMUpdateSettingsFrame
	for _, f := range enq.frames() {
		if u, ok := f.(*frames.LLMUpdateSettingsFrame); ok {
			update = u
		}
	}
	if update == nil {
		t.Fatal("the manager queued no persona update")
	}

	services := map[string]personaReader{
		"anthropic":      anthropic.NewLLM(anthropic.Config{APIKey: "k", Model: "claude-sonnet-4-5"}),
		"openai chat":    chat.NewLLM(chat.LLMConfig{APIKey: "k", Model: "gpt-4o"}),
		"mistral":        mistral.NewLLM(chat.LLMConfig{APIKey: "k", Model: "mistral-small-latest"}),
		"responses http": responses.NewHTTPLLM(responses.Config{APIKey: "k", Model: "gpt-4.1"}),
		"responses ws":   responses.NewLLM(responses.Config{APIKey: "k", Model: "gpt-4.1"}),
		"gemini":         gemini.NewLLM(gemini.Config{APIKey: "k", Model: "gemini-2.5-flash"}),
	}
	for name, svc := range services {
		t.Run(name, func(t *testing.T) {
			if err := svc.ProcessFrame(context.Background(), update, processor.Downstream); err != nil {
				t.Fatalf("ProcessFrame: %v", err)
			}
			if got := svc.SystemInstruction(); got != "be nice" {
				t.Errorf("system instruction = %q, want the node's persona", got)
			}
		})
	}
}
