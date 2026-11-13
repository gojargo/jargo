package mcp

import (
	"context"
	"errors"
	"io"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/jsonrpc"
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

	// onCall runs once, inside the next call of the greet tool, before it
	// answers.
	onCall func()
	// greeting is what the greet tool answers with, or nil for its greeting.
	greeting []mcpsdk.Content
	// failDials is how many of the next dials fail to connect.
	failDials int
	// onClose runs once, as the next client connection is closed.
	onClose func()
	// closeErr is what closing a client connection reports, or nil for a clean
	// close.
	closeErr error
	// loseResponses has a client connection lose the next answer it reads, as
	// a connection that ends with the answer on its way does.
	loseResponses bool
	// conns are the client connections handed out, and closed how many of them
	// were closed.
	conns, closed int
}

// errConnectFailed is what a dial the pool was told to fail reports.
const errConnectFailed = testError("connect failed")

// dial builds one transport, with a server already running on the other end.
func (p *serverPool) dial(ctx context.Context) mcpsdk.Transport {
	p.mu.Lock()
	fail := p.failDials > 0
	if fail {
		p.failDials--
	}
	p.mu.Unlock()
	if fail {
		return failingTransport{}
	}
	serverT, clientT := mcpsdk.NewInMemoryTransports()
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "test", Version: "1"}, nil)
	schema := map[string]any{"type": "object", "properties": map[string]any{}}
	server.AddTool(&mcpsdk.Tool{Name: "greet", Description: "Return a greeting", InputSchema: schema},
		func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			p.mu.Lock()
			p.calls++
			onCall, greeting := p.onCall, p.greeting
			p.onCall = nil
			p.mu.Unlock()
			if onCall != nil {
				onCall()
			}
			if greeting == nil {
				greeting = []mcpsdk.Content{&mcpsdk.TextContent{Text: "hello there"}}
			}
			return &mcpsdk.CallToolResult{Content: greeting}, nil
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
	return pooledTransport{Transport: clientT, pool: p}
}

// failingTransport is a server that cannot be reached.
type failingTransport struct{}

func (failingTransport) Connect(context.Context) (mcpsdk.Connection, error) {
	return nil, errConnectFailed
}

// pooledTransport hands out connections the pool watches being closed.
type pooledTransport struct {
	mcpsdk.Transport
	pool *serverPool
}

func (t pooledTransport) Connect(ctx context.Context) (mcpsdk.Connection, error) {
	conn, err := t.Transport.Connect(ctx)
	if err != nil {
		return nil, err
	}
	t.pool.mu.Lock()
	t.pool.conns++
	t.pool.mu.Unlock()
	return &pooledConn{Connection: conn, pool: t.pool}, nil
}

// pooledConn is a client connection that reports its close to the pool. The SDK
// closes a connection whose read or write fails as well as one it is told to
// close, so it may be closed more than once.
type pooledConn struct {
	mcpsdk.Connection
	pool *serverPool
	once sync.Once
}

func (c *pooledConn) Read(ctx context.Context) (jsonrpc.Message, error) {
	msg, err := c.Connection.Read(ctx)
	if _, answer := msg.(*jsonrpc.Response); answer {
		c.pool.mu.Lock()
		lose := c.pool.loseResponses
		c.pool.loseResponses = false
		c.pool.mu.Unlock()
		if lose {
			return nil, io.ErrUnexpectedEOF
		}
	}
	return msg, err
}

func (c *pooledConn) Close() error {
	err := c.Connection.Close()
	c.once.Do(func() {
		c.pool.mu.Lock()
		c.pool.closed++
		onClose := c.pool.onClose
		c.pool.onClose = nil
		c.pool.mu.Unlock()
		if onClose != nil {
			onClose()
		}
	})
	c.pool.mu.Lock()
	closeErr := c.pool.closeErr
	c.pool.mu.Unlock()
	if closeErr != nil {
		return closeErr
	}
	return err
}

// openConnections is how many client connections are still open.
func (p *serverPool) openConnections() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.conns - p.closed
}

// set changes the pool's behavior under its lock.
func (p *serverPool) set(change func(p *serverPool)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	change(p)
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
	if !strings.HasPrefix(out, "Error calling mcp tool greet: ") {
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

// Asking for the tools of a client that holds a session runs on that session
// rather than connecting again.
func TestToolsIsIdempotentOnConnection(t *testing.T) {
	c, pool := pooledClient(t)

	for range 2 {
		if _, err := c.Tools(t.Context()); err != nil {
			t.Fatalf("Tools: %v", err)
		}
	}
	if got := pool.dials(); got != 1 {
		t.Errorf("dials = %d, want the one connection", got)
	}
}

// A client closed while a call is in flight stays closed, and the call reports
// the connection it lost, which differs from the error of a call on a client
// that holds no session.
func TestAClosedClientReportsTheLostConnection(t *testing.T) {
	c, pool := pooledClient(t)
	inCall, release := make(chan struct{}), make(chan struct{})
	pool.set(func(p *serverPool) {
		p.onCall = func() {
			close(inCall)
			<-release
		}
	})
	type result struct {
		out string
		err error
	}
	called := make(chan result, 1)
	go func() {
		out, err := c.call(t.Context(), "greet", nil)
		called <- result{out, err}
	}()
	<-inCall

	// The close waits for the call in flight, holding the client while it does.
	closed := make(chan error, 1)
	go func() { closed <- c.Close() }()
	for c.mu.TryLock() {
		c.mu.Unlock()
		runtime.Gosched()
	}
	// The connection goes with the answer still on its way.
	pool.set(func(p *serverPool) { p.loseResponses = true })
	close(release)

	r := <-called
	<-closed
	out, err := r.out, r.err
	if err != nil {
		t.Fatalf("the lost connection should reach the model, not the service: %v", err)
	}
	if want := "Error calling mcp tool greet: "; !strings.HasPrefix(out, want) {
		t.Errorf("result = %q, want it to start %q", out, want)
	}
	if held(c) != nil {
		t.Error("the closed client holds a session")
	}
	if got := pool.dials(); got != 1 {
		t.Errorf("dials = %d, want the closed client not to connect again", got)
	}
}

// A teardown that meets a call dropping its dead session waits the drop out and
// finds the session already released, so nothing is left open.
func TestATeardownMeetingADropLeavesNothingOpen(t *testing.T) {
	c, pool := pooledClient(t)
	teardown := make(chan struct{})
	pool.set(func(p *serverPool) {
		// Once the call is in flight, its connection goes with the answer on
		// its way, and the owner tears the client down as it does.
		p.onCall = func() {
			pool.set(func(p *serverPool) {
				p.loseResponses = true
				p.onClose = func() {
					go func() {
						_ = c.Close() // the connection is already gone, so how it closes is moot
						close(teardown)
					}()
				}
			})
		}
	})

	out, err := c.call(t.Context(), "greet", nil)
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	<-teardown

	if want := "Error calling mcp tool greet: "; !strings.HasPrefix(out, want) {
		t.Errorf("result = %q, want it to start %q", out, want)
	}
	if got := pool.openConnections(); got != 0 {
		t.Errorf("%d connections are still open", got)
	}
	if held(c) != nil {
		t.Error("the client still holds a session")
	}
}

// A dead session whose transport fails as it is dropped still leaves the model
// reading the cause of its own call, and the client holding no session, so the
// next call connects.
func TestADropWhoseTransportFailsKeepsTheCauseOfTheCall(t *testing.T) {
	c, pool := pooledClient(t)
	const closeFailure = testError("transport exit failed")
	pool.set(func(p *serverPool) { p.closeErr = closeFailure })
	pool.kill()

	out, err := c.call(t.Context(), "greet", nil)
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if want := "Error calling mcp tool greet: "; !strings.HasPrefix(out, want) ||
		strings.Contains(out, string(closeFailure)) {
		t.Errorf("result = %q, want the cause of the call and not the failed close", out)
	}
	if held(c) != nil {
		t.Error("the dead session is still held")
	}

	pool.set(func(p *serverPool) { p.closeErr = nil })
	if out, err := c.call(t.Context(), "greet", nil); err != nil || out != "hello there" {
		t.Errorf("result = %q, err = %v, want the tool's own output", out, err)
	}
}

// The connect a call runs after a drop can fail too. The model reads why, and
// the call after it connects and runs.
func TestAFailedConnectAfterADropReportsItsErrorAndHeals(t *testing.T) {
	c, pool := pooledClient(t)
	pool.kill()
	if _, err := c.call(t.Context(), "greet", nil); err != nil {
		t.Fatalf("call: %v", err)
	}

	pool.set(func(p *serverPool) { p.failDials = 1 })
	out, err := c.call(t.Context(), "greet", nil)
	if err != nil {
		t.Fatalf("the failed connect should reach the model, not the service: %v", err)
	}
	if want := "Error calling mcp tool greet: " + string(errConnectFailed); out != want {
		t.Errorf("result = %q, want %q", out, want)
	}

	if out, err := c.call(t.Context(), "greet", nil); err != nil || out != "hello there" {
		t.Errorf("result = %q, err = %v, want the tool's own output", out, err)
	}
}

// A call that produced no text reads as the stock line, whether the result was
// empty or of another content type.
func TestAResultWithNoTextReadsAsTheStockLine(t *testing.T) {
	for name, greeting := range map[string][]mcpsdk.Content{
		"empty":                {},
		"another content type": {&mcpsdk.ImageContent{Data: []byte{1, 2, 3}, MIMEType: "image/png"}},
	} {
		t.Run(name, func(t *testing.T) {
			c, pool := pooledClient(t)
			pool.set(func(p *serverPool) { p.greeting = greeting })

			out, err := c.call(t.Context(), "greet", nil)
			if err != nil {
				t.Fatalf("call: %v", err)
			}
			if want := "Sorry, could not call the mcp tool"; out != want {
				t.Errorf("result = %q, want %q", out, want)
			}
		})
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
