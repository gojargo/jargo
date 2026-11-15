package mcp

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Tests for the connection parameters: what each transport is configured with,
// and what reaches the server because of it.

func TestValidateConnectionOptions(t *testing.T) {
	second := time.Second
	zero := time.Duration(0)
	keep := false
	tests := []struct {
		name  string
		cfg   Config
		valid bool
	}{
		{"stdio with env and cwd", Config{Command: []string{"x"}, Env: map[string]string{"A": "1"}, Cwd: "/tmp"}, true},
		{"env without a command", Config{HTTPURL: "http://x", Env: map[string]string{"A": "1"}}, false},
		{"cwd without a command", Config{SSEURL: "http://x", Cwd: "/tmp"}, false},
		{"streamable HTTP with every option", Config{
			HTTPURL: "http://x", Headers: map[string]string{"Authorization": "Bearer t"},
			Timeout: &second, SSEReadTimeout: &second, TerminateOnClose: &keep,
		}, true},
		{"SSE with headers and timeouts", Config{
			SSEURL: "http://x", Headers: map[string]string{"A": "1"}, Timeout: &second, SSEReadTimeout: &second,
		}, true},
		{"headers for a stdio server", Config{Command: []string{"x"}, Headers: map[string]string{"A": "1"}}, false},
		{"a timeout for a stdio server", Config{Command: []string{"x"}, Timeout: &second}, false},
		{"a zero timeout", Config{HTTPURL: "http://x", Timeout: &zero}, false},
		{"a zero read timeout", Config{HTTPURL: "http://x", SSEReadTimeout: &zero}, false},
		{"terminate on close over SSE", Config{SSEURL: "http://x", TerminateOnClose: &keep}, false},
		{"a URL that is not one", Config{HTTPURL: "not a url"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.cfg.Validate()
			if tt.valid && err != nil {
				t.Errorf("rejected: %v", err)
			}
			if !tt.valid && err == nil {
				t.Error("accepted")
			}
		})
	}
}

// recorder is an MCP server behind an HTTP handler that records the requests
// it was sent.
type recorder struct {
	mu       sync.Mutex
	requests []*http.Request
}

func (r *recorder) wrap(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		r.mu.Lock()
		r.requests = append(r.requests, req.Clone(context.Background()))
		r.mu.Unlock()
		h.ServeHTTP(w, req)
	})
}

func (r *recorder) seen() []*http.Request {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.requests)
}

// greetServer is an MCP server exposing one tool.
func greetServer() *mcpsdk.Server {
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "test", Version: "1"}, nil)
	server.AddTool(&mcpsdk.Tool{
		Name:        "greet",
		Description: "Return a greeting",
		InputSchema: map[string]any{"type": "object"},
	}, func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
		return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "hello there"}}}, nil
	})
	return server
}

func streamableServer(t *testing.T) (string, *recorder) {
	t.Helper()
	rec := &recorder{}
	server := greetServer()
	h := mcpsdk.NewStreamableHTTPHandler(func(*http.Request) *mcpsdk.Server { return server }, nil)
	srv := httptest.NewServer(rec.wrap(h))
	t.Cleanup(srv.Close)
	return srv.URL, rec
}

// connectAndCall connects with cfg, lists the tools and calls the one there is.
func connectAndCall(t *testing.T, cfg Config) *Client {
	t.Helper()
	ctx := context.Background()
	c, err := Connect(ctx, cfg)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if _, err := c.Tools(ctx); err != nil {
		t.Fatalf("Tools: %v", err)
	}
	out, err := c.call(ctx, "greet", nil)
	if err != nil || out != "hello there" {
		t.Fatalf("call = %q, %v; want the tool's output", out, err)
	}
	return c
}

func TestHeadersReachAStreamableHTTPServer(t *testing.T) {
	endpoint, rec := streamableServer(t)
	c := connectAndCall(t, Config{
		HTTPURL: endpoint,
		Headers: map[string]string{"Authorization": "Bearer token", "X-Tenant": "acme"},
	})
	_ = c.Close()

	requests := rec.seen()
	if len(requests) == 0 {
		t.Fatal("the server was never called")
	}
	for _, r := range requests {
		if r.Header.Get("Authorization") != "Bearer token" || r.Header.Get("X-Tenant") != "acme" {
			t.Errorf("%s %s carried %v, want the configured headers", r.Method, r.URL, r.Header)
		}
	}
}

func TestHeadersReachAnSSEServer(t *testing.T) {
	rec := &recorder{}
	server := greetServer()
	h := mcpsdk.NewSSEHandler(func(*http.Request) *mcpsdk.Server { return server }, nil)
	srv := httptest.NewServer(rec.wrap(h))
	t.Cleanup(srv.Close)

	c := connectAndCall(t, Config{SSEURL: srv.URL, Headers: map[string]string{"Authorization": "Bearer token"}})
	_ = c.Close()

	var methods []string
	for _, r := range rec.seen() {
		methods = append(methods, r.Method)
		if r.Header.Get("Authorization") != "Bearer token" {
			t.Errorf("%s %s carried no Authorization", r.Method, r.URL)
		}
	}
	// The stream is opened with a GET and the messages go out as POSTs: both
	// are authenticated.
	if !slices.Contains(methods, http.MethodGet) || !slices.Contains(methods, http.MethodPost) {
		t.Errorf("methods = %v, want the stream and the messages", methods)
	}
}

// deletes counts the DELETE requests that end a session.
func deletes(requests []*http.Request) int {
	n := 0
	for _, r := range requests {
		if r.Method == http.MethodDelete {
			n++
		}
	}
	return n
}

func TestClosingEndsTheStreamableHTTPSessionByDefault(t *testing.T) {
	endpoint, rec := streamableServer(t)
	c := connectAndCall(t, Config{HTTPURL: endpoint})
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if deletes(rec.seen()) != 1 {
		t.Errorf("the session was not ended on the server")
	}
}

func TestTerminateOnCloseFalseLeavesTheSession(t *testing.T) {
	endpoint, rec := streamableServer(t)
	keep := false
	c := connectAndCall(t, Config{HTTPURL: endpoint, TerminateOnClose: &keep})
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if n := deletes(rec.seen()); n != 0 {
		t.Errorf("%d DELETE sent, want the session left to the server", n)
	}
}

// TestAProtocolHeaderKeepsItsValue checks a configured header does not
// override one the transport set itself.
func TestAProtocolHeaderKeepsItsValue(t *testing.T) {
	var got http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) { got = r.Header.Clone() }))
	t.Cleanup(srv.Close)

	client := Config{HTTPURL: srv.URL, Headers: map[string]string{"Content-Type": "text/plain", "X-A": "1"}}.
		httpClient(DefaultHTTPTimeout)
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodPost, srv.URL, strings.NewReader("{}"))
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	_ = resp.Body.Close()
	if got.Get("Content-Type") != "application/json" || got.Get("X-A") != "1" {
		t.Errorf("headers = %v, want the request's own Content-Type and the configured X-A", got)
	}
	if req.Header.Get("X-A") != "" {
		t.Error("the caller's request was modified")
	}
}

func TestAuthorizationIsNotCarriedAcrossOrigins(t *testing.T) {
	var got http.Header
	other := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) { got = r.Header.Clone() }))
	t.Cleanup(other.Close)
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL, http.StatusFound)
	}))
	t.Cleanup(first.Close)

	client := Config{HTTPURL: first.URL, Headers: map[string]string{"Authorization": "Bearer t", "X-A": "1"}}.
		httpClient(DefaultHTTPTimeout)
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, first.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	_ = resp.Body.Close()
	if got == nil {
		t.Fatal("the redirect was not followed")
	}
	if got.Get("Authorization") != "" {
		t.Error("Authorization was carried to another origin")
	}
	if got.Get("X-A") != "1" {
		t.Errorf("X-A = %q, want the other headers carried", got.Get("X-A"))
	}
}

func TestSameOrigin(t *testing.T) {
	tests := []struct {
		from, to string
		same     bool
	}{
		{"http://a.test/x", "http://a.test/y", true},
		{"http://a.test", "https://a.test", true},
		{"http://a.test:8080", "https://a.test", false},
		{"https://a.test", "http://a.test", false},
		{"https://a.test", "https://b.test", false},
		{"https://a.test", "https://a.test:8443", false},
	}
	for _, tt := range tests {
		from, to := mustParse(t, tt.from), mustParse(t, tt.to)
		if got := sameOrigin(from, to); got != tt.same {
			t.Errorf("sameOrigin(%s, %s) = %v, want %v", tt.from, tt.to, got, tt.same)
		}
	}
}

func mustParse(t *testing.T, raw string) *url.URL {
	t.Helper()
	u, err := url.Parse(raw)
	if err != nil {
		t.Fatalf("parse %q: %v", raw, err)
	}
	return u
}

func TestAStalledResponseFailsAfterTheReadTimeout(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "data: first\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-release
	}))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })

	read := 50 * time.Millisecond
	client := Config{HTTPURL: srv.URL, SSEReadTimeout: &read}.httpClient(DefaultHTTPTimeout)
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer func() { _ = resp.Body.Close() }()

	buf := make([]byte, 64)
	if n, err := resp.Body.Read(buf); err != nil || !strings.Contains(string(buf[:n]), "first") {
		t.Fatalf("first read = %q, %v; want the event the server sent", buf[:n], err)
	}
	start := time.Now()
	_, err = resp.Body.Read(buf)
	if !errors.Is(err, errReadTimeout) {
		t.Fatalf("stalled read err = %v, want the read timeout", err)
	}
	if waited := time.Since(start); waited > 2*time.Second {
		t.Errorf("the stalled read took %v", waited)
	}
}

func TestStalledResponseHeadersFailAfterTheReadTimeout(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-release }))
	t.Cleanup(srv.Close)
	t.Cleanup(func() { close(release) })

	read := 50 * time.Millisecond
	client := Config{HTTPURL: srv.URL, SSEReadTimeout: &read}.httpClient(DefaultHTTPTimeout)
	req, _ := http.NewRequestWithContext(context.Background(), http.MethodGet, srv.URL, nil)
	resp, err := client.Do(req)
	if err == nil {
		_ = resp.Body.Close()
		t.Fatal("a server that never answered was waited on past the read timeout")
	}
}

func TestStdioEnvironment(t *testing.T) {
	t.Setenv("HOME", "/home/someone")
	t.Setenv("PATH", "/usr/bin")
	t.Setenv("SHELL", "() { :; }; echo")
	t.Setenv("JARGO_TEST_SECRET", "s3cret")

	env := stdioEnvironment(map[string]string{"MEMORY_FILE_PATH": "/tmp/m.jsonl", "PATH": "/opt/bin"})

	has := func(kv string) bool { return slices.Contains(env, kv) }
	if !has("HOME=/home/someone") {
		t.Errorf("env = %v, want HOME inherited", env)
	}
	if !has("MEMORY_FILE_PATH=/tmp/m.jsonl") {
		t.Errorf("env = %v, want the configured variable", env)
	}
	if !has("PATH=/opt/bin") {
		t.Errorf("env = %v, want the configured PATH over the inherited one", env)
	}
	for _, kv := range env {
		if strings.HasPrefix(kv, "JARGO_TEST_SECRET=") {
			t.Errorf("env = %v, want variables outside the default set left out", env)
		}
		if strings.HasPrefix(kv, "SHELL=") {
			t.Errorf("env = %v, want a shell function left out", env)
		}
	}
}

func TestStdioCommandIsConfigured(t *testing.T) {
	cfg := Config{Command: []string{"server", "--flag"}, Env: map[string]string{"A": "1"}, Cwd: "/srv"}
	tr, ok := cfg.transport(context.Background()).(*mcpsdk.CommandTransport)
	if !ok {
		t.Fatal("a command did not build a stdio transport")
	}
	if tr.Command.Dir != "/srv" {
		t.Errorf("Dir = %q, want the configured directory", tr.Command.Dir)
	}
	if !slices.Contains(tr.Command.Env, "A=1") {
		t.Errorf("Env = %v, want the configured variable", tr.Command.Env)
	}
	if !slices.Equal(tr.Command.Args, []string{"server", "--flag"}) {
		t.Errorf("Args = %v", tr.Command.Args)
	}
}

func TestHTTPTransportsDefaultTheirTimeouts(t *testing.T) {
	for _, tt := range []struct {
		cfg  Config
		want time.Duration
	}{
		{Config{SSEURL: "http://x"}, DefaultSSETimeout},
		{Config{HTTPURL: "http://x"}, DefaultHTTPTimeout},
	} {
		var client *http.Client
		switch tr := tt.cfg.transport(context.Background()).(type) {
		case *mcpsdk.SSEClientTransport:
			client = tr.HTTPClient
		case *mcpsdk.StreamableClientTransport:
			client = tr.HTTPClient
		}
		rt, ok := client.Transport.(*roundTripper)
		if !ok {
			t.Fatalf("%+v: no configured client", tt.cfg)
		}
		base, _ := rt.base.(*http.Transport)
		if base.TLSHandshakeTimeout != tt.want || base.ResponseHeaderTimeout != DefaultSSEReadTimeout ||
			rt.readTimeout != DefaultSSEReadTimeout {
			t.Errorf("%+v: timeouts %v/%v/%v, want %v and the default read timeout",
				tt.cfg, base.TLSHandshakeTimeout, base.ResponseHeaderTimeout, rt.readTimeout, tt.want)
		}
	}
}
