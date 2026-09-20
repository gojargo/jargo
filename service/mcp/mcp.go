// Package mcp connects a jargo LLM to Model Context Protocol tool servers. It
// lists a server's tools, exposes them on an LLMContext, and registers a handler
// per tool that proxies calls to the server, so an MCP server's tools become
// ordinary function calls the model can make.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"net"
	"os/exec"
	"strings"
	"sync"
	"syscall"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Sentinel errors.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errNoTransport  = errors.New("mcp: config must set exactly one of Command, SSEURL or HTTPURL")
	errNotConnected = errors.New("mcp: client holds no session; Connect it before calling its tools")
)

// maxErrorDetail is how much of a failure's text the model is told. The line
// goes into the conversation, where a stack trace or a page of a server's output
// costs more than it tells the model.
const maxErrorDetail = 200

// The lines the model reads when a call produced nothing it can use. A failure
// names its cause, because the model is the thing that decides whether to call
// again, differently, or not at all.
const (
	callFailedTemplate = "The MCP tool %q failed: %s"
	noResultTemplate   = "The MCP tool %q returned no result."
)

// Config selects and configures a single MCP server. Exactly one transport must
// be set.
type Config struct {
	// Command runs an MCP server over stdio, e.g.
	// {"npx", "-y", "@modelcontextprotocol/server-filesystem", "/data"}.
	Command []string
	// SSEURL connects to an MCP server over SSE.
	SSEURL string
	// HTTPURL connects to an MCP server over streamable HTTP.
	HTTPURL string
	// ToolsFilter, when non-empty, restricts the exposed tools to these names.
	ToolsFilter []string
	// ToolsArguments fixes some of a tool's arguments, by tool name. The fixed
	// values are merged into every call of that tool, overriding anything the
	// model supplied, and the parameters they fill are taken out of the schema
	// the tool is advertised with, so the model never sees them.
	//
	// It is how a server's tool is bound to this session: a tenant id, a
	// directory the file tools are confined to, an API the model has no business
	// choosing. A name here that the server does not advertise is reported and
	// otherwise ignored.
	ToolsArguments map[string]map[string]any
	// ToolsOutputFilters reshapes a tool's result before the model sees it, by
	// tool name. A server's output is written for whatever consumer it had in
	// mind, and a model reading a page of it will do worse than one reading the
	// line that matters.
	//
	// A filter that panics is treated as one that produced nothing, and the
	// model is told the call could not be made rather than being handed output
	// the filter was meant to have reshaped.
	ToolsOutputFilters map[string]func(result string) string
}

// Validate reports whether exactly one transport is configured.
func (c Config) Validate() error {
	n := 0
	if len(c.Command) > 0 {
		n++
	}
	if c.SSEURL != "" {
		n++
	}
	if c.HTTPURL != "" {
		n++
	}
	if n != 1 {
		return errNoTransport
	}
	return nil
}

func (c Config) transport(ctx context.Context) mcpsdk.Transport {
	switch {
	case len(c.Command) > 0:
		//nolint:gosec // launching the user-configured MCP server is the intended behavior
		return &mcpsdk.CommandTransport{Command: exec.CommandContext(ctx, c.Command[0], c.Command[1:]...)}
	case c.SSEURL != "":
		return &mcpsdk.SSEClientTransport{Endpoint: c.SSEURL}
	default:
		return &mcpsdk.StreamableClientTransport{Endpoint: c.HTTPURL}
	}
}

// Client is a connected MCP session.
//
// A tool call that finds its connection gone drops that session and tells the
// model what happened, and the call after it connects again. A client that was
// never connected, or whose owner has closed it, fails the call instead.
type Client struct {
	filter  map[string]bool
	fixed   map[string]map[string]any
	filters map[string]func(string) string

	// dial builds the transport to connect over, and is called again for a
	// session opened to replace a dead one.
	dial func(context.Context) mcpsdk.Transport
	// baseCtx scopes the connection, and is the context every session is opened
	// under. It is the one Connect was given rather than a caller's own, so a
	// reconnect does not tie the server to the tool call that happened to
	// trigger it: a stdio server is a subprocess, and would be killed with it.
	baseCtx context.Context //nolint:containedctx // the connection's own lifetime, not a call's

	// mu owns the session's lifetime: opening one, closing it, dropping a dead
	// one, and the reconnect a tool call does.
	mu      sync.Mutex
	session *mcpsdk.ClientSession
	// wanted records that an owner has asked this client for a session, which is
	// what tells a session that was dropped, and which a tool call may reconnect,
	// from one never opened or since closed, which fails the call.
	wanted bool
}

// Connect dials the configured MCP server and initializes the session.
//
// ctx scopes the connection rather than this call: it is the context every
// session is opened under, including one opened to replace a session that died,
// so canceling it ends the server this client talks to.
func Connect(ctx context.Context, cfg Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return connect(ctx, cfg.transport, cfg)
}

func connect(ctx context.Context, dial func(context.Context) mcpsdk.Transport, cfg Config) (*Client, error) {
	c := &Client{
		dial:    dial,
		baseCtx: ctx,
		fixed:   cfg.ToolsArguments,
		filters: cfg.ToolsOutputFilters,
	}
	if len(cfg.ToolsFilter) > 0 {
		c.filter = make(map[string]bool, len(cfg.ToolsFilter))
		for _, name := range cfg.ToolsFilter {
			c.filter[name] = true
		}
	}
	if _, err := c.openSession(true); err != nil {
		return nil, err
	}
	return c, nil
}

// openSession returns the session the caller runs on, opening one if this client
// holds none. The lock is held across the open and the read, so a close cannot
// take the session away in between.
//
// asOwner is for a caller that owns the connection. A tool call passes false and
// runs on the session an owner asked for: it may reconnect one that was dropped,
// but it neither opens a client nobody connected nor reopens one its owner
// closed.
func (c *Client) openSession(asOwner bool) (*mcpsdk.ClientSession, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if asOwner {
		c.wanted = true
	}
	if c.wanted && c.session == nil {
		client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "jargo", Version: "0.0.1"}, nil)
		session, err := client.Connect(c.baseCtx, c.dial(c.baseCtx), nil)
		if err != nil {
			return nil, err
		}
		c.session = session
	}
	if c.session == nil {
		return nil, errNotConnected
	}
	return c.session, nil
}

// dropSession closes the session a failed call ran on, so a later call connects
// again. A session outlives its transport, so without this every later call runs
// on a dead one. Tool calls run at the same time on a single client, so a caller
// whose session another call has already replaced drops nothing.
func (c *Client) dropSession(session *mcpsdk.ClientSession) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.session != session {
		return nil
	}
	c.session = nil
	return session.Close()
}

// Tools lists the server's tools converted to jargo tools, honoring the filter.
//
// Each carries its own handler, so putting the result on an LLMContext is all it
// takes: the LLM service registers what the context advertises and drops it
// again when the toolset changes, and the tools and the code answering them stay
// the same set.
func (c *Client) Tools(ctx context.Context) ([]frames.Tool, error) {
	session, err := c.openSession(true)
	if err != nil {
		return nil, err
	}
	res, err := session.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}
	advertised := make(map[string]bool, len(res.Tools))
	var tools []frames.Tool
	for _, t := range res.Tools {
		advertised[t.Name] = true
		if c.filter != nil && !c.filter[t.Name] {
			continue
		}
		schema, err := json.Marshal(t.InputSchema)
		if err != nil {
			continue
		}
		if fixed := c.fixed[t.Name]; len(fixed) > 0 {
			schema = withoutFixedArguments(schema, fixed)
		}
		tools = append(tools, frames.Tool{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  schema,
			Handler:     c.handlerFor(t.Name),
			// Every tool here works through this one session, so registering any
			// of them is what closes it at teardown, and registering all of them
			// closes it once.
			Cleanup: c,
		})
	}
	c.warnUnknownFixedArguments(advertised)
	return tools, nil
}

// handlerFor answers calls of one tool by proxying them to the server. What the
// call produced is reported as the result, a failure included, since a handler
// that returns an error has the service tell the model only that the function
// failed and keeps the reason from it.
func (c *Client) handlerFor(name string) llm.FunctionCallHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) error {
		result, err := c.call(ctx, name, params.Arguments)
		if err != nil {
			return err
		}
		return params.Result(ctx, result, nil)
	}
}

// warnUnknownFixedArguments reports arguments fixed for a tool the server does
// not offer, which is a configuration that will never take effect.
func (c *Client) warnUnknownFixedArguments(advertised map[string]bool) {
	for name := range c.fixed {
		if !advertised[name] {
			slog.Warn("arguments are fixed for a tool the MCP server does not offer", "tool", name)
		}
	}
}

// withoutFixedArguments takes the parameters this session fills in for the model
// out of the schema the tool is advertised with. A parameter the model cannot
// choose has no business being described to it, and describing it invites the
// model to supply a value that is then overwritten.
//
// A schema this cannot read is returned as it was: advertising a parameter that
// is ignored is a smaller fault than advertising a tool with no schema at all.
func withoutFixedArguments(schema json.RawMessage, fixed map[string]any) json.RawMessage {
	var doc map[string]any
	if err := json.Unmarshal(schema, &doc); err != nil {
		return schema
	}
	if props, ok := doc["properties"].(map[string]any); ok {
		for name := range fixed {
			delete(props, name)
		}
	}
	if required, ok := doc["required"].([]any); ok {
		kept := make([]any, 0, len(required))
		for _, r := range required {
			if name, ok := r.(string); ok {
				if _, isFixed := fixed[name]; isFixed {
					continue
				}
			}
			kept = append(kept, r)
		}
		doc["required"] = kept
	}
	out, err := json.Marshal(doc)
	if err != nil {
		return schema
	}
	return out
}

// Register lists the server's tools, adds them to convo, and registers a handler
// per tool on base that proxies calls to the MCP server. Existing tools on convo
// are kept.
func (c *Client) Register(ctx context.Context, base *llm.Base, convo *frames.LLMContext) error {
	tools, err := c.Tools(ctx)
	if err != nil {
		return err
	}
	for _, t := range tools {
		base.RegisterFunction(t.Name, c.handlerFor(t.Name), llm.WithToolCleanup(c))
	}
	convo.SetTools(append(convo.Tools(), tools...))
	return nil
}

// call proxies one tool invocation to the MCP server and returns what the model
// is told: the tool's text output, or, when the call produced none, the reason.
//
// The failure goes to the model rather than back to the service, because the
// model holds the tool loop: it is the thing that decides whether to call again,
// call differently, or answer without the tool. The call is not retried here,
// since a request that never left and an answer that was lost reach this as one
// error, and running the tool twice is the more expensive mistake.
//
// A Go error is returned only for a call this client cannot make at all: one
// whose arguments will not parse, or one on a client holding no session.
func (c *Client) call(ctx context.Context, name string, args json.RawMessage) (string, error) {
	var argMap map[string]any
	if len(args) > 0 {
		if err := json.Unmarshal(args, &argMap); err != nil {
			return "", err
		}
	}
	// What this session fixed wins over what the model supplied. The model was
	// never shown these parameters, so a value here is one it invented.
	if fixed := c.fixed[name]; len(fixed) > 0 {
		if argMap == nil {
			argMap = make(map[string]any, len(fixed))
		}
		maps.Copy(argMap, fixed)
	}
	session, err := c.openSession(false)
	if err != nil {
		return "", err
	}

	var failure string
	res, err := session.CallTool(ctx, &mcpsdk.CallToolParams{Name: name, Arguments: argMap})
	if err != nil {
		failure = fmt.Sprintf(callFailedTemplate, name, errorDetail(err))
		slog.Error("an MCP tool call failed", "tool", name, "err", err)
		if isConnectionLost(err) {
			slog.Warn("dropping the MCP session a tool call found gone", "tool", name)
			// The model still gets the cause of its own call, so a transport that
			// fails on the way out goes to the log and no further.
			if dropErr := c.dropSession(session); dropErr != nil {
				slog.Error("an MCP session failed to close after it was dropped", "err", dropErr)
			}
		}
	}

	var sb strings.Builder
	if res != nil {
		for _, content := range res.Content {
			if tc, ok := content.(*mcpsdk.TextContent); ok {
				sb.WriteString(tc.Text)
			}
		}
	}
	if out := c.filterResult(name, sb.String()); out != "" {
		return out, nil
	}
	if failure != "" {
		return failure, nil
	}
	return fmt.Sprintf(noResultTemplate, name), nil
}

// isConnectionLost tells a transport that has gone from an error the MCP server
// itself reported. The first call after a transport dies fails on the stream
// itself, and only once the connection has noticed does the SDK report its own
// closed-connection error, so both have to count: otherwise the session is
// dropped one call later than it could be, or not at all.
func isConnectionLost(err error) bool {
	return errors.Is(err, mcpsdk.ErrConnectionClosed) || // the SDK, once it knows
		errors.Is(err, io.ErrClosedPipe) || // a stdio server's pipes, closed
		errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) || // either, ended
		errors.Is(err, net.ErrClosed) || // an HTTP or SSE socket, closed
		errors.Is(err, syscall.EPIPE) || errors.Is(err, syscall.ECONNRESET) // or broken
}

// errorDetail names the cause of a failed call, in the one line the model reads.
// An error carrying no message names its type instead, so the line does not end
// at the colon, and a long one is cut: it goes into the conversation.
func errorDetail(err error) string {
	detail := err.Error()
	if detail == "" {
		return fmt.Sprintf("%T", err)
	}
	if r := []rune(detail); len(r) > maxErrorDetail {
		return string(r[:maxErrorDetail]) + "..."
	}
	return detail
}

// filterResult reshapes a tool's output for the model, when this session was
// configured to. A filter that panics is treated as one that produced nothing:
// handing the model the unfiltered output would defeat a filter that exists to
// cut something out of it.
func (c *Client) filterResult(name, result string) (out string) {
	filter, ok := c.filters[name]
	if !ok || filter == nil {
		return result
	}
	defer func() {
		if r := recover(); r != nil {
			slog.Error("an MCP tool's output filter failed", "tool", name, "panic", r)
			out = ""
		}
	}()
	return filter(result)
}

// Close ends the MCP session. Calling it again does nothing: the session may
// have been named as the resource behind tools registered on more than one
// service, and each of those releases it when it is cleaned up.
//
// The client stays closed until Tools asks it for a session again. A tool call
// on a closed client fails rather than reconnecting, since the connection is no
// longer anybody's to hold open.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.wanted = false
	session := c.session
	c.session = nil
	if session == nil {
		return nil
	}
	return session.Close()
}

// CloseTools ends the session when the LLM service that registered this
// server's tools is cleaned up, so a pipeline that used an MCP server does not
// leave it connected. It implements llm.ToolCleanup.
func (c *Client) CloseTools(context.Context) error { return c.Close() }
