package main

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/rime/realtime"
	"github.com/gojargo/jargo/service/llm"
)

// TestToolsAreDescribed checks every tool's schema is valid JSON naming its
// required parameters, and carries its handler.
func TestToolsAreDescribed(t *testing.T) {
	for _, tool := range []frames.Tool{transferTool(), invoiceTool()} {
		var schema struct {
			Properties map[string]any `json:"properties"`
			Required   []string       `json:"required"`
		}
		if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
			t.Fatalf("%s: schema is not valid JSON: %v", tool.Name, err)
		}
		for _, r := range schema.Required {
			if _, ok := schema.Properties[r]; !ok {
				t.Fatalf("%s: required %q is not a property", tool.Name, r)
			}
		}
		if _, ok := tool.Handler.(llm.FunctionCallHandler); !ok {
			t.Fatalf("%s: handler is %T", tool.Name, tool.Handler)
		}
	}
}

// TestSettingsAreValid checks both the opening settings and the billing ones
// are settings Prism takes.
func TestSettingsAreValid(t *testing.T) {
	for name, s := range map[string]realtime.Settings{
		"support": supportSettings(),
		"billing": *billingSettings(),
	} {
		if err := (realtime.Config{APIKey: "test", Settings: s}).Validate(); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// TestBillingReplacesTheTools checks the handoff advertises the invoice lookup
// only, with a filler for it.
func TestBillingReplacesTheTools(t *testing.T) {
	delta := billingSettings()
	tools, ok := delta.Tools.Value()
	if !ok || len(tools.Standard) != 1 || tools.Standard[0].Name != "lookup_invoice" {
		t.Fatalf("billing tools = %+v, want lookup_invoice", tools)
	}
	fillers, _ := delta.ToolFillers.Value()
	filler, ok := fillers["lookup_invoice"]
	if !ok || filler.Type != "text" || filler.Text == nil || *filler.Text == "" {
		t.Errorf("lookup_invoice filler = %+v, want a text line", filler)
	}
	if instruction, _ := delta.SystemInstruction.Value(); instruction == "" {
		t.Error("billing system instruction is empty")
	}
}

// TestTransferNeedsAPipelineWorker checks the transfer reports nothing when it
// has no pipeline to send the settings update through.
func TestTransferNeedsAPipelineWorker(t *testing.T) {
	reported := false
	err := transferToBilling(t.Context(), llm.FunctionCallParams{
		Result: func(context.Context, string, *frames.FunctionCallResultProperties) error {
			reported = true
			return nil
		},
	})
	if !errors.Is(err, errNoPipelineWorker) {
		t.Errorf("err = %v, want errNoPipelineWorker", err)
	}
	if reported {
		t.Error("transfer reported a result without a pipeline worker")
	}
}

// TestLookupInvoiceReportsStatus checks the invoice lookup answers with the
// invoice asked about.
func TestLookupInvoiceReportsStatus(t *testing.T) {
	var result string
	err := lookupInvoice(t.Context(), llm.FunctionCallParams{
		Arguments: json.RawMessage(`{"invoice_id": "INV-1001"}`),
		Result: func(_ context.Context, r string, _ *frames.FunctionCallResultProperties) error {
			result = r
			return nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	if err = json.Unmarshal([]byte(result), &got); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}
	if got["invoice_id"] != "INV-1001" || got["status"] != "paid" {
		t.Errorf("result = %v", got)
	}
}
