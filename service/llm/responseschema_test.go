package llm_test

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm"
)

// plainGen is a generator that says nothing about response schemas.
type plainGen struct{}

func (plainGen) Generate(context.Context, *frames.LLMContext, llm.Emit) error { return nil }

// schemaGen enforces a schema on every model but "old".
type schemaGen struct{ plainGen }

func (schemaGen) SupportsResponseSchema() bool { return true }

func (schemaGen) ModelSupportsResponseSchema(model string) bool { return model != "old" }

// A schema is sent only by a service whose provider and model can enforce
// one; otherwise it is left off with a warning.
func TestCheckResponseSchema(t *testing.T) {
	schema := json.RawMessage(`{"type":"object"}`)
	cases := []struct {
		name  string
		gen   llm.Generator
		model string
		want  bool
		warn  string
	}{
		{"unsupported service", plainGen{}, "new", false, "is not supported and is ignored"},
		{"unsupported model", schemaGen{}, "old", false, "not supported by the model"},
		{"supported", schemaGen{}, "new", true, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			svc := llm.New("FakeLLM", c.gen)
			svc.SetModel(c.model)
			buf, restore := captureDebug(t)
			got := svc.CheckResponseSchema(t.Context(), schema)
			restore()
			if (got != nil) != c.want {
				t.Fatalf("schema sent = %v, want %v", got != nil, c.want)
			}
			if c.warn != "" && !strings.Contains(buf.String(), c.warn) {
				t.Errorf("log = %q, want a warning containing %q", buf.String(), c.warn)
			}
		})
	}
	if llm.New("FakeLLM", schemaGen{}).CheckResponseSchema(t.Context(), nil) != nil {
		t.Error("no schema asked for, yet one was returned")
	}
}
