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
	"os/exec"
	"strings"

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
}

// Connect dials the configured MCP server and initializes the session.
func Connect(ctx context.Context, cfg Config) (*Client, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return connect(ctx, cfg.transport(ctx), cfg.ToolsFilter)
}

func connect(ctx context.Context, tr mcpsdk.Transport, filter []string) (*Client, error) {
	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "jargo", Version: "0.0.1"}, nil)
	session, err := client.Connect(ctx, tr, nil)
	if err != nil {
		return nil, err
	}
	c := &Client{session: session}
	if len(filter) > 0 {
		c.filter = make(map[string]bool, len(filter))
		for _, name := range filter {
			c.filter[name] = true
		}
	}
	return c, nil
}

// Tools lists the server's tools converted to jargo tools, honoring the filter.
func (c *Client) Tools(ctx context.Context) ([]frames.Tool, error) {
	res, err := c.session.ListTools(ctx, nil)
	if err != nil {
		return nil, err
	}
	var tools []frames.Tool
	for _, t := range res.Tools {
		if c.filter != nil && !c.filter[t.Name] {
			continue
		}
		schema, err := json.Marshal(t.InputSchema)
		if err != nil {
			continue
		}
		tools = append(tools, frames.Tool{Name: t.Name, Description: t.Description, Parameters: schema})
	}
	return tools, nil
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
		name := t.Name
		base.RegisterFunction(name, func(ctx context.Context, args json.RawMessage) (string, error) {
			return c.call(ctx, name, args)
		})
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
	return sb.String(), nil
}

// Close ends the MCP session.
func (c *Client) Close() error { return c.session.Close() }
