package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

func TestValidate(t *testing.T) {
	if err := (Config{Command: []string{"x"}}).Validate(); err != nil {
		t.Errorf("stdio config rejected: %v", err)
	}
	if err := (Config{SSEURL: "http://x"}).Validate(); err != nil {
		t.Errorf("sse config rejected: %v", err)
	}
	if err := (Config{}).Validate(); err == nil {
		t.Error("empty config should be rejected")
	}
	if err := (Config{Command: []string{"x"}, HTTPURL: "http://x"}).Validate(); err == nil {
		t.Error("two transports should be rejected")
	}
}

// testClient spins up an in-process MCP server exposing one "greet" tool and
// returns a jargo MCP client connected to it over an in-memory transport.
func testClient(t *testing.T, filter []string) *Client {
	t.Helper()
	ctx := context.Background()
	serverT, clientT := mcpsdk.NewInMemoryTransports()

	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "test", Version: "1"}, nil)
	server.AddTool(&mcpsdk.Tool{
		Name:        "greet",
		Description: "Return a greeting",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"name": map[string]any{"type": "string"}},
		},
	}, func(_ context.Context, _ *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "hello there"}},
		}, nil
	})
	go func() { _ = server.Run(ctx, serverT) }()

	c, err := connect(ctx, clientT, filter)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestToolsAndCall(t *testing.T) {
	ctx := context.Background()
	c := testClient(t, nil)

	tools, err := c.Tools(ctx)
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "greet" || tools[0].Description != "Return a greeting" {
		t.Fatalf("tools = %+v, want one greet tool", tools)
	}
	var schema map[string]any
	if err := json.Unmarshal(tools[0].Parameters, &schema); err != nil {
		t.Fatalf("tool schema is not valid JSON: %v", err)
	}
	if schema["type"] != "object" {
		t.Errorf("schema type = %v, want object", schema["type"])
	}

	out, err := c.call(ctx, "greet", json.RawMessage(`{"name":"ada"}`))
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if !strings.Contains(out, "hello there") {
		t.Fatalf("call result = %q, want to contain 'hello there'", out)
	}
}

func TestToolsFilter(t *testing.T) {
	ctx := context.Background()
	if tools, _ := testClient(t, []string{"greet"}).Tools(ctx); len(tools) != 1 {
		t.Fatalf("allowlisted tool missing: %v", tools)
	}
	if tools, _ := testClient(t, []string{"other"}).Tools(ctx); len(tools) != 0 {
		t.Fatalf("filtered tool leaked: %v", tools)
	}
}

type stubGen struct{}

func (stubGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

func TestRegisterAddsToolsToContext(t *testing.T) {
	ctx := context.Background()
	c := testClient(t, nil)
	base := llm.New("test", stubGen{})
	convo := frames.NewLLMContext("system")

	if err := c.Register(ctx, base, convo); err != nil {
		t.Fatalf("Register: %v", err)
	}
	tools := convo.Tools()
	if len(tools) != 1 || tools[0].Name != "greet" {
		t.Fatalf("convo tools = %+v, want greet", tools)
	}
}
