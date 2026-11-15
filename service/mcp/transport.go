package mcp

import (
	"errors"
	"io"
	"maps"
	"net"
	"net/http"
	"net/url"
	"os"
	"runtime"
	"slices"
	"strings"
	"sync"
	"time"
)

// The connection defaults, as the MCP transports define them.
const (
	// DefaultSSETimeout bounds connecting to an SSE server.
	DefaultSSETimeout = 5 * time.Second
	// DefaultHTTPTimeout bounds connecting to a streamable-HTTP server.
	DefaultHTTPTimeout = 30 * time.Second
	// DefaultSSEReadTimeout is how long a request to either waits for the server
	// to send anything.
	DefaultSSEReadTimeout = 5 * time.Minute
)

// errReadTimeout is a request that waited longer than the read timeout for the
// server to send anything.
//
//nolint:gochecknoglobals // sentinel error
var errReadTimeout = errors.New("mcp: timed out waiting for the server")

// defaultInheritedEnv are the variables a stdio server is given from this
// process's environment: what a program needs to run, and nothing a secret
// is likely to be kept in.
func defaultInheritedEnv() []string {
	if runtime.GOOS == "windows" {
		return []string{
			"APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT",
			"PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "USERNAME", "USERPROFILE",
		}
	}
	return []string{"HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"}
}

// stdioEnvironment is the environment a stdio server runs with: the default
// inherited variables, with env laid over them. A variable holding a shell
// function is not passed on, since it is code the server's shell would run.
func stdioEnvironment(env map[string]string) []string {
	merged := map[string]string{}
	for _, key := range defaultInheritedEnv() {
		value, ok := os.LookupEnv(key)
		if !ok || strings.HasPrefix(value, "()") {
			continue
		}
		merged[key] = value
	}
	maps.Copy(merged, env)
	out := make([]string, 0, len(merged))
	for _, key := range slices.Sorted(maps.Keys(merged)) {
		out = append(out, key+"="+merged[key])
	}
	return out
}

// httpClient is the client an SSE or streamable-HTTP transport runs on. Timeout
// bounds connecting, defaultTimeout standing in when it is unset, and the read
// timeout bounds every wait for the server to send something: the response
// headers, and each read of the body after them.
func (c Config) httpClient(defaultTimeout time.Duration) *http.Client {
	timeout := defaultTimeout
	if c.Timeout != nil {
		timeout = *c.Timeout
	}
	read := DefaultSSEReadTimeout
	if c.SSEReadTimeout != nil {
		read = *c.SSEReadTimeout
	}
	base, _ := http.DefaultTransport.(*http.Transport)
	if base == nil {
		base = &http.Transport{}
	}
	tr := base.Clone()
	tr.DialContext = (&net.Dialer{Timeout: timeout, KeepAlive: 30 * time.Second}).DialContext
	tr.TLSHandshakeTimeout = timeout
	tr.ResponseHeaderTimeout = read
	return &http.Client{Transport: &roundTripper{
		base:        tr,
		headers:     c.Headers,
		readTimeout: read,
		// The session is ended on a streamable-HTTP server unless the
		// configuration says to leave it.
		keepSession: c.HTTPURL != "" && c.TerminateOnClose != nil && !*c.TerminateOnClose,
	}}
}

// roundTripper adds the configured headers to every request, bounds each read
// of a response by the read timeout, and holds back the DELETE that ends a
// session the configuration says to leave open.
type roundTripper struct {
	base        http.RoundTripper
	headers     map[string]string
	readTimeout time.Duration
	keepSession bool
}

func (t *roundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if t.keepSession && req.Method == http.MethodDelete {
		// Closing the client sends the DELETE that ends the session; answered
		// here, the server keeps the session until it expires it.
		if req.Body != nil {
			_ = req.Body.Close()
		}
		return &http.Response{
			Status: "204 No Content", StatusCode: http.StatusNoContent,
			Proto: "HTTP/1.1", ProtoMajor: 1, ProtoMinor: 1,
			Header: http.Header{}, Body: http.NoBody, Request: req,
		}, nil
	}
	if len(t.headers) > 0 {
		req = req.Clone(req.Context())
		crossOrigin := redirectedAcrossOrigins(req)
		for key, value := range t.headers {
			if req.Header.Get(key) != "" {
				// The protocol's own header keeps its value.
				continue
			}
			if crossOrigin && http.CanonicalHeaderKey(key) == "Authorization" {
				continue
			}
			req.Header.Set(key, value)
		}
	}
	resp, err := t.base.RoundTrip(req)
	if err != nil || resp.Body == nil || t.readTimeout <= 0 {
		return resp, err
	}
	resp.Body = newIdleTimeoutBody(resp.Body, t.readTimeout)
	return resp, nil
}

// redirectedAcrossOrigins reports whether req follows a redirect that left the
// origin the first request went to. A redirect from http to https on the same
// host stays within it.
func redirectedAcrossOrigins(req *http.Request) bool {
	first := req
	for first.Response != nil && first.Response.Request != nil {
		first = first.Response.Request
	}
	if first == req {
		return false
	}
	return !sameOrigin(first.URL, req.URL)
}

// sameOrigin reports whether to is on the origin from is, counting an upgrade
// from http to https on the default ports as the same.
func sameOrigin(from, to *url.URL) bool {
	if from.Hostname() != to.Hostname() {
		return false
	}
	if from.Scheme == to.Scheme && port(from) == port(to) {
		return true
	}
	return from.Scheme == "http" && port(from) == "80" && to.Scheme == "https" && port(to) == "443"
}

// port is the URL's port, or the scheme's default when it names none.
func port(u *url.URL) string {
	if p := u.Port(); p != "" {
		return p
	}
	if u.Scheme == "https" {
		return "443"
	}
	return "80"
}

// idleTimeoutBody fails a read that waits longer than timeout for data,
// closing the body under it, which is what unblocks the read.
type idleTimeoutBody struct {
	body    io.ReadCloser
	timeout time.Duration

	mu       sync.Mutex
	timedOut bool
}

func newIdleTimeoutBody(body io.ReadCloser, timeout time.Duration) *idleTimeoutBody {
	return &idleTimeoutBody{body: body, timeout: timeout}
}

func (b *idleTimeoutBody) Read(p []byte) (int, error) {
	timer := time.AfterFunc(b.timeout, func() {
		b.mu.Lock()
		b.timedOut = true
		b.mu.Unlock()
		_ = b.body.Close()
	})
	n, err := b.body.Read(p)
	timer.Stop()
	b.mu.Lock()
	timedOut := b.timedOut
	b.mu.Unlock()
	if timedOut {
		return n, errReadTimeout
	}
	return n, err
}

func (b *idleTimeoutBody) Close() error { return b.body.Close() }
