package mcp

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// recordingServer is an MCP server that remembers the arguments each tool was
// called with, so a test can see what actually reached it.
type recordingServer struct {
	mu    sync.Mutex
	calls []recordedCall
}

type recordedCall struct {
	name string
	args map[string]any
}

func (r *recordingServer) record(name string, args map[string]any) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls = append(r.calls, recordedCall{name: name, args: args})
}

func (r *recordingServer) recorded() []recordedCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]recordedCall(nil), r.calls...)
}

// searchClient connects to a server offering a "search" tool with two
// parameters and an unrelated "other" tool, which is what tells a setting that
// names one tool from one that reaches every tool.
func searchClient(t *testing.T, cfg Config) (*Client, *recordingServer) {
	t.Helper()
	ctx := context.Background()
	serverT, clientT := mcpsdk.NewInMemoryTransports()
	rec := &recordingServer{}

	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "test", Version: "1"}, nil)
	handler := func(name string) func(context.Context, *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
		return func(_ context.Context, req *mcpsdk.CallToolRequest) (*mcpsdk.CallToolResult, error) {
			var args map[string]any
			if len(req.Params.Arguments) > 0 {
				_ = json.Unmarshal(req.Params.Arguments, &args)
			}
			rec.record(name, args)
			return &mcpsdk.CallToolResult{
				Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: name + "-result"}},
			}, nil
		}
	}
	server.AddTool(&mcpsdk.Tool{
		Name:        "search",
		Description: "Search for something",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
				"mode":  map[string]any{"type": "string"},
			},
			"required": []any{"query", "mode"},
		},
	}, handler("search"))
	server.AddTool(&mcpsdk.Tool{
		Name:        "other",
		Description: "Something else",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"x": map[string]any{"type": "string"}},
			"required":   []any{"x"},
		},
	}, handler("other"))
	go func() { _ = server.Run(ctx, serverT) }()

	c, err := connect(ctx, clientT, cfg)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c, rec
}

// byName indexes the listed tools.
func byName(t *testing.T, c *Client) map[string]frames.Tool {
	t.Helper()
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatalf("Tools: %v", err)
	}
	out := make(map[string]frames.Tool, len(tools))
	for _, tool := range tools {
		out[tool.Name] = tool
	}
	return out
}

// schemaOf decodes a tool's advertised parameter schema.
func schemaOf(t *testing.T, tool frames.Tool) map[string]any {
	t.Helper()
	var doc map[string]any
	if err := json.Unmarshal(tool.Parameters, &doc); err != nil {
		t.Fatalf("schema for %s: %v", tool.Name, err)
	}
	return doc
}

// propertyNames are the parameters a tool is advertised as taking, sorted.
func propertyNames(t *testing.T, tool frames.Tool) string {
	t.Helper()
	props, _ := schemaOf(t, tool)["properties"].(map[string]any)
	names := make([]string, 0, len(props))
	for name := range props {
		names = append(names, name)
	}
	return joinSorted(names)
}

// requiredNames are the parameters a tool is advertised as requiring, sorted.
func requiredNames(t *testing.T, tool frames.Tool) string {
	t.Helper()
	required, _ := schemaOf(t, tool)["required"].([]any)
	names := make([]string, 0, len(required))
	for _, r := range required {
		if name, ok := r.(string); ok {
			names = append(names, name)
		}
	}
	return joinSorted(names)
}

func joinSorted(names []string) string {
	for i := range names {
		for j := i + 1; j < len(names); j++ {
			if names[j] < names[i] {
				names[i], names[j] = names[j], names[i]
			}
		}
	}
	return strings.Join(names, ",")
}

// callThrough invokes a tool through the handler its own listing carried, the
// way the LLM service does, and returns the result the model would be given.
func callThrough(t *testing.T, tool frames.Tool, args string) string {
	t.Helper()
	handler, ok := tool.Handler.(llm.FunctionCallHandler)
	if !ok {
		t.Fatalf("tool %s carries no handler (%T)", tool.Name, tool.Handler)
	}
	var got string
	err := handler(context.Background(), llm.FunctionCallParams{
		FunctionName: tool.Name,
		ToolCallID:   "call-1",
		Arguments:    json.RawMessage(args),
		Result: func(_ context.Context, result string, _ *frames.FunctionCallResultProperties) error {
			got = result
			return nil
		},
	})
	if err != nil {
		t.Fatalf("calling %s: %v", tool.Name, err)
	}
	return got
}

// A listed tool carries the code that answers it, so putting the listing on a
// conversation is all it takes: the LLM service registers what the conversation
// advertises, and a tool with no handler would be a tool nothing answers.
func TestListedToolsCarryTheirHandler(t *testing.T) {
	c, rec := searchClient(t, Config{})
	tools := byName(t, c)

	if got := callThrough(t, tools["search"], `{"query":"news","mode":"fast"}`); got != "search-result" {
		t.Errorf("result = %q, want %q", got, "search-result")
	}
	calls := rec.recorded()
	if len(calls) != 1 || calls[0].name != "search" {
		t.Fatalf("server saw %v, want one call to search", calls)
	}
}

// A tool's output is reshaped for the model when the session was configured to.
func TestOutputFilterReshapesWhatTheModelSees(t *testing.T) {
	c, _ := searchClient(t, Config{
		ToolsOutputFilters: map[string]func(string) string{
			"search": strings.ToUpper,
		},
	})
	tools := byName(t, c)

	if got := callThrough(t, tools["search"], `{"query":"news","mode":"fast"}`); got != "SEARCH-RESULT" {
		t.Errorf("result = %q, want it filtered", got)
	}
	if got := callThrough(t, tools["other"], `{"x":"y"}`); got != "other-result" {
		t.Errorf("result = %q, want the unfiltered tool untouched", got)
	}
}

// A parameter this session fills in is taken out of the schema the tool is
// advertised with: the model cannot choose it, so describing it only invites a
// value that is then overwritten. Other tools are left alone.
func TestFixedArgumentsAreHiddenFromTheAdvertisedSchema(t *testing.T) {
	c, _ := searchClient(t, Config{
		ToolsArguments: map[string]map[string]any{"search": {"mode": "realtime"}},
	})
	tools := byName(t, c)

	if got := propertyNames(t, tools["search"]); got != "query" {
		t.Errorf("search advertises %q, want %q", got, "query")
	}
	if got := requiredNames(t, tools["search"]); got != "query" {
		t.Errorf("search requires %q, want %q", got, "query")
	}
	if got := propertyNames(t, tools["other"]); got != "x" {
		t.Errorf("other advertises %q, want it untouched", got)
	}
	if got := requiredNames(t, tools["other"]); got != "x" {
		t.Errorf("other requires %q, want it untouched", got)
	}
}

// What the session fixed wins over what the model supplied. The model was never
// shown the parameter, so a value it produced for it is one it invented.
func TestFixedArgumentsWinOverTheModel(t *testing.T) {
	c, rec := searchClient(t, Config{
		ToolsArguments: map[string]map[string]any{"search": {"mode": "realtime"}},
	})
	tools := byName(t, c)

	callThrough(t, tools["search"], `{"query":"news","mode":"model-supplied"}`)

	calls := rec.recorded()
	if len(calls) != 1 {
		t.Fatalf("server saw %d calls, want 1", len(calls))
	}
	if calls[0].args["mode"] != "realtime" || calls[0].args["query"] != "news" {
		t.Errorf("server saw %v, want the fixed mode and the model's query", calls[0].args)
	}
}

// A call the model made with no arguments at all still carries what the session
// fixed.
func TestFixedArgumentsAreSentWhenTheModelSuppliedNone(t *testing.T) {
	c, rec := searchClient(t, Config{
		ToolsArguments: map[string]map[string]any{"search": {"mode": "realtime"}},
	})
	tools := byName(t, c)

	callThrough(t, tools["search"], ``)

	calls := rec.recorded()
	if len(calls) != 1 || calls[0].args["mode"] != "realtime" {
		t.Errorf("server saw %v, want the fixed mode", calls)
	}
}

// Fixing a parameter the server's schema never had takes nothing out of that
// schema, but the value is still sent: a server may accept more than it
// advertises.
func TestFixedArgumentAbsentFromTheSchemaIsStillSent(t *testing.T) {
	c, rec := searchClient(t, Config{
		ToolsArguments: map[string]map[string]any{"other": {"hidden": float64(1)}},
	})
	tools := byName(t, c)

	if got := propertyNames(t, tools["other"]); got != "x" {
		t.Errorf("other advertises %q, want it untouched", got)
	}
	callThrough(t, tools["other"], `{"x":"y"}`)

	calls := rec.recorded()
	if len(calls) != 1 || calls[0].args["hidden"] != float64(1) || calls[0].args["x"] != "y" {
		t.Errorf("server saw %v, want both the model's argument and the fixed one", calls)
	}
}

// The filter still applies to a tool reached through a hand registration, not
// only through the listing, since both go through the same call.
func TestFilterAppliesThroughEitherRegistration(t *testing.T) {
	c, _ := searchClient(t, Config{
		ToolsOutputFilters: map[string]func(string) string{"search": strings.ToUpper},
	})

	got, err := c.call(context.Background(), "search", json.RawMessage(`{"query":"n","mode":"f"}`))
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if got != "SEARCH-RESULT" {
		t.Errorf("result = %q, want it filtered", got)
	}
}
