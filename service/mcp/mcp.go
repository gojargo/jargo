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
	"log/slog"
	"maps"
	"os/exec"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Sentinel errors.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errNoTransport = errors.New("mcp: config must set exactly one of Command, SSEURL or HTTPURL")
	errCallFailed  = errors.New("mcp: tool call failed")
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
type Client struct {
	session *mcpsdk.ClientSession
	filter  map[string]bool
	fixed   map[string]map[string]any
	filters map[string]func(string) string
	// closeOnce keeps the session's teardown to one, however many services named
	// it as the resource their tools work through.
	closeOnce sync.Once
}

// Connect dials the configured MCP server and initializes the session.
func Connect(ctx context.Context, cfg Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return connect(ctx, cfg.transport(ctx), cfg)
}

func connect(ctx context.Context, tr mcpsdk.Transport, cfg Config) (*Client, error) {
	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "jargo", Version: "0.0.1"}, nil)
	session, err := client.Connect(ctx, tr, nil)
	if err != nil {
		return nil, err
	}
	c := &Client{session: session, fixed: cfg.ToolsArguments, filters: cfg.ToolsOutputFilters}
	if len(cfg.ToolsFilter) > 0 {
		c.filter = make(map[string]bool, len(cfg.ToolsFilter))
		for _, name := range cfg.ToolsFilter {
			c.filter[name] = true
		}
	}
	return c, nil
}

// Tools lists the server's tools converted to jargo tools, honoring the filter.
//
// Each carries its own handler, so putting the result on an LLMContext is all it
// takes: the LLM service registers what the context advertises and drops it
// again when the toolset changes, and the tools and the code answering them stay
// the same set.
func (c *Client) Tools(ctx context.Context) ([]frames.Tool, error) {
	res, err := c.session.ListTools(ctx, nil)
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

// handlerFor answers calls of one tool by proxying them to the server.
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

// call proxies one tool invocation to the MCP server and returns the joined text
// content. A tool-level error is returned as text (with the error content) so the
// model can see it and self-correct; only transport failures return a Go error.
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
	res, err := c.session.CallTool(ctx, &mcpsdk.CallToolParams{Name: name, Arguments: argMap})
	if err != nil {
		return "", fmt.Errorf("%w: %s: %w", errCallFailed, name, err)
	}
	var sb strings.Builder
	for _, content := range res.Content {
		if tc, ok := content.(*mcpsdk.TextContent); ok {
			sb.WriteString(tc.Text)
		}
	}
	return c.filterResult(name, sb.String()), nil
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
func (c *Client) Close() error {
	var err error
	c.closeOnce.Do(func() { err = c.session.Close() })
	return err
}

// CloseTools ends the session when the LLM service that registered this
// server's tools is cleaned up, so a pipeline that used an MCP server does not
// leave it connected. It implements llm.ToolCleanup.
func (c *Client) CloseTools(context.Context) error { return c.Close() }
