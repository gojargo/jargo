package mcp

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Failed tool calls: what the model is told, and what happens to a session whose
// transport has gone.

// serverPool runs a fresh MCP server on every dial, so a client can connect
// again once the session it held has died. It records the server sessions it
// handed out, which is how a test kills a connection from the far end.
type serverPool struct {
	t *testing.T

	mu       sync.Mutex
	sessions []*mcpsdk.ServerSession
	calls    int
}

// dial builds one transport, with a server already running on the other end.
func (p *serverPool) dial(ctx context.Context) mcpsdk.Transport {
	serverT, clientT := mcpsdk.NewInMemoryTransports()
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "test", Version: "1"}, nil)
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	server.AddTool(&mcpsdk.Tool{Name: "greet", Description: "Return a greeting", InputSchema: schema},
		func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			p.mu.Lock()
			p.calls++
			p.mu.Unlock()
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "hello there"}}}, nil
		})
	server.AddTool(&mcpsdk.Tool{Name: "boom", Description: "Fail", InputSchema: schema},
		func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			return nil, testError("upstream unavailable")
		})
	session, err := server.Connect(ctx, serverT, nil)
	if err != nil {
		p.t.Errorf("server connect: %v", err)
		return clientT
	}
	p.mu.Lock()
	p.sessions = append(p.sessions, session)
	p.mu.Unlock()
	return clientT
}

func (p *serverPool) dials() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.sessions)
}

func (p *serverPool) served() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls
}

// kill closes the server's end of the current connection, which is what a server
// that restarts or times a session out looks like from here.
func (p *serverPool) kill() {
	p.mu.Lock()
	session := p.sessions[len(p.sessions)-1]
	p.mu.Unlock()
	_ = session.Close()
}

// pooledClient is a client connected to a pool that can serve it again.
func pooledClient(t *testing.T) (*Client, *serverPool) {
	t.Helper()
	pool := &serverPool{t: t}
	c, err := connect(t.Context(), pool.dial, Config{})
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c, pool
}

// held reports the session the client currently holds.
func held(c *Client) *mcpsdk.ClientSession {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.session
}

// A session outlives its transport, so without dropping it every later call runs
// on a dead one and the server is unreachable for the rest of the conversation.
// The call itself is not run again: a request that never left and an answer that
// was lost arrive here as the same error, and running a tool twice is the more
// expensive mistake.
func TestALostConnectionDropsTheSessionAndRunsTheToolOnce(t *testing.T) {
	c, pool := pooledClient(t)
	if _, err := c.call(t.Context(), "greet", nil); err != nil {
		t.Fatalf("call: %v", err)
	}
	pool.kill()

	out, err := c.call(t.Context(), "greet", nil)
	if err != nil {
		t.Fatalf("a lost connection should reach the model, not the service: %v", err)
	}
	if !strings.Contains(out, "failed") {
		t.Errorf("result = %q, want the failure named", out)
	}
	if s := held(c); s != nil {
		t.Error("the dead session is still held, so every later call runs on it")
	}
	if got := pool.served(); got != 1 {
		t.Errorf("the tool ran %d times, want the one call that reached the server", got)
	}
	if got := pool.dials(); got != 1 {
		t.Errorf("dials = %d, want the failed call not to have reconnected on its own", got)
	}
}

// The model holds the tool loop: it reads the failure and decides whether to
// call again. That call is the one that connects.
func TestACallAfterALostConnectionConnectsAgain(t *testing.T) {
	c, pool := pooledClient(t)
	if _, err := c.call(t.Context(), "greet", nil); err != nil {
		t.Fatalf("call: %v", err)
	}
	pool.kill()
	if _, err := c.call(t.Context(), "greet", nil); err != nil {
		t.Fatalf("call: %v", err)
	}

	out, err := c.call(t.Context(), "greet", nil)
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if out != "hello there" {
		t.Errorf("result = %q, want the tool's own output", out)
	}
	if got := pool.dials(); got != 2 {
		t.Errorf("dials = %d, want a second connection", got)
	}
}

// A handler that returns an error has the service tell the model only that the
// function failed, so the reason has to travel as the result instead.
func TestAFailedCallTellsTheModelWhy(t *testing.T) {
	c, _ := pooledClient(t)

	out, err := c.call(t.Context(), "boom", nil)
	if err != nil {
		t.Fatalf("a tool's own failure should reach the model: %v", err)
	}
	if !strings.Contains(out, "upstream unavailable") {
		t.Errorf("result = %q, want the cause the server reported", out)
	}
}

// A server that refuses a call is not a server that has gone, so the session it
// answered on is the one the next call runs on.
func TestAnErrorThatIsNotALostConnectionKeepsTheSession(t *testing.T) {
	c, pool := pooledClient(t)
	before := held(c)

	if _, err := c.call(t.Context(), "boom", nil); err != nil {
		t.Fatalf("call: %v", err)
	}
	if held(c) != before {
		t.Error("the session was dropped over an error the server itself reported")
	}
	if got := pool.dials(); got != 1 {
		t.Errorf("dials = %d, want the one connection", got)
	}
}

// Tool calls run at the same time on one client, so a second call can still be
// holding the dead session the first one has already replaced. Dropping it must
// not take the live one with it.
func TestDroppingASessionAnotherCallReplacedClosesNothing(t *testing.T) {
	c, pool := pooledClient(t)
	replaced := held(c)
	pool.kill()
	if _, err := c.call(t.Context(), "greet", nil); err != nil { // drops the dead session
		t.Fatalf("call: %v", err)
	}
	if _, err := c.call(t.Context(), "greet", nil); err != nil { // connects again
		t.Fatalf("call: %v", err)
	}
	live := held(c)

	if err := c.dropSession(replaced); err != nil {
		t.Fatalf("dropSession: %v", err)
	}
	if held(c) != live {
		t.Error("dropping a session already replaced took the live one with it")
	}
}

// A closed client is nobody's to hold open, so a call on one fails rather than
// quietly connecting again behind the owner that closed it.
func TestAToolCallOnAClosedClientFails(t *testing.T) {
	c, pool := pooledClient(t)
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := c.call(t.Context(), "greet", nil); !errors.Is(err, errNotConnected) {
		t.Errorf("err = %v, want it to report holding no session", err)
	}
	if got := pool.dials(); got != 1 {
		t.Errorf("dials = %d, want no connection behind the close", got)
	}
}

// Asking for the tools is asking for a connection, so it opens one again.
func TestToolsAfterCloseConnectsAgain(t *testing.T) {
	c, pool := pooledClient(t)
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	tools, err := c.Tools(t.Context())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(tools) != 2 {
		t.Errorf("got %d tools, want the server's two", len(tools))
	}
	if got := pool.dials(); got != 2 {
		t.Errorf("dials = %d, want a second connection", got)
	}
}

// The session may be named as the resource behind tools registered on more than
// one service, and each of those releases it when it is cleaned up.
func TestCloseTwiceIsSafe(t *testing.T) {
	c, _ := pooledClient(t)
	if err := c.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Errorf("second Close: %v", err)
	}
}

// The line goes into the conversation, so it is cut to size, and an error with
// nothing to say names its type rather than leaving the line ending at a colon.
func TestErrorDetail(t *testing.T) {
	if got := errorDetail(testError("plain")); got != "plain" {
		t.Errorf("errorDetail = %q, want the message as it stands", got)
	}

	long := errorDetail(testError(strings.Repeat("é", maxErrorDetail+50)))
	if want := strings.Repeat("é", maxErrorDetail) + "..."; long != want {
		t.Errorf("a long error was cut to %d runes, want %d and an ellipsis",
			len([]rune(long)), maxErrorDetail)
	}

	// An error with nothing to say, as a closed-stream error can be.
	if got := errorDetail(testError("")); got != "mcp.testError" {
		t.Errorf("errorDetail = %q, want the type of an error carrying no message", got)
	}
}

// testError carries the message a test needs, which a sentinel cannot.
type testError string

func (e testError) Error() string { return string(e) }
