package mcp

import (
	"context"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// countingGenerator is the least an LLM service needs to exist. Nothing here
// generates anything; what is under test is what the service releases when it
// is cleaned up.
type countingGenerator struct{}

func (countingGenerator) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

// newLLM builds a bare LLM service to register tools on.
func newLLM(name string) *llm.Base { return llm.New(name, countingGenerator{}) }

// alive reports whether the MCP session is still usable, which is how a closed
// session is told from an open one without reaching into the SDK.
func alive(t *testing.T, c *Client) bool {
	t.Helper()
	_, err := c.session.ListTools(context.Background(), nil)
	return err == nil
}

// Registering a server's tools is what has its connection closed: a pipeline
// that used an MCP server does not leave it connected once it tears down, and
// the developer who registered the tools did not have to wire that up.
func TestCleanupClosesTheRegisteredServer(t *testing.T) {
	c, _ := searchClient(t, Config{})
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	svc := newLLM("LLM")
	convo := frames.NewLLMContext("")
	convo.SetTools(tools)
	svc.SyncToolHandlers(context.Background(), convo)

	if !alive(t, c) {
		t.Fatal("the session was closed before anything asked for it to be")
	}
	if err := svc.Cleanup(context.Background()); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if alive(t, c) {
		t.Error("the session outlived the service whose tools it answered")
	}
}

// Cleaning up twice releases the connection once, so a second teardown does not
// close a session something else may since have opened.
func TestCleanupTwiceReleasesOnce(t *testing.T) {
	c, _ := searchClient(t, Config{})
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	svc := newLLM("LLM")
	convo := frames.NewLLMContext("")
	convo.SetTools(tools)
	svc.SyncToolHandlers(context.Background(), convo)

	for range 2 {
		if err := svc.Cleanup(context.Background()); err != nil {
			t.Fatalf("Cleanup: %v", err)
		}
	}
	if alive(t, c) {
		t.Error("the session outlived the service")
	}
}

// Two services advertising the same server's tools, as a switcher's members
// would, each release the connection and it closes once.
func TestTwoServicesSharingAServerCloseItOnce(t *testing.T) {
	c, _ := searchClient(t, Config{})
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}

	convo := frames.NewLLMContext("")
	convo.SetTools(tools)
	first, second := newLLM("First"), newLLM("Second")
	first.SyncToolHandlers(context.Background(), convo)
	second.SyncToolHandlers(context.Background(), convo)

	if err := first.Cleanup(context.Background()); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if err := second.Cleanup(context.Background()); err != nil {
		t.Fatalf("second Cleanup: %v", err)
	}
	if alive(t, c) {
		t.Error("the session outlived both services")
	}
}

// Every tool of one server names the same connection, so registering all of them
// records one thing to release rather than one per tool.
func TestOneServerIsReleasedOncePerService(t *testing.T) {
	c, _ := searchClient(t, Config{})
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(tools) < 2 {
		t.Fatalf("the server offered %d tools, want several", len(tools))
	}

	svc := newLLM("LLM")
	convo := frames.NewLLMContext("")
	convo.SetTools(tools)
	svc.SyncToolHandlers(context.Background(), convo)

	if got := svc.ToolCleanupCount(); got != 1 {
		t.Errorf("recorded %d resources for one server, want 1", got)
	}
}

// A server nothing ever registered tools from is not closed by a service that
// never learned about it. It is the caller's to close, as it always was.
func TestAServerNoServiceKnowsAboutIsLeftAlone(t *testing.T) {
	c, _ := searchClient(t, Config{})
	if _, err := c.Tools(context.Background()); err != nil {
		t.Fatalf("Tools: %v", err)
	}

	svc := newLLM("LLM")
	if err := svc.Cleanup(context.Background()); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if !alive(t, c) {
		t.Error("a service that never saw the server still closed it")
	}
}

// The hand registration releases the connection too, not only the listing.
func TestRegisterAlsoReleasesTheServer(t *testing.T) {
	c, _ := searchClient(t, Config{})
	svc := newLLM("LLM")
	convo := frames.NewLLMContext("")
	if err := c.Register(context.Background(), svc, convo); err != nil {
		t.Fatalf("Register: %v", err)
	}

	if err := svc.Cleanup(context.Background()); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if alive(t, c) {
		t.Error("the session outlived the service its tools were registered on")
	}
}
