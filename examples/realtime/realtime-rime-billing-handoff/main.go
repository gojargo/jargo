// Command realtime-rime-billing-handoff is a voice agent on Rime's Prism, a
// speech-to-speech model, that hands the conversation off to billing support
// mid-call.
//
// A general support assistant can transfer the caller to billing. The transfer
// tool updates the Prism session's instructions, tools and tool fillers with
// an LLMUpdateSettingsFrame, so the same conversation continues with the
// billing configuration. A Prism tool filler speaks while the invoice lookup
// runs.
//
//	RIME_API_KEY=… go run ./examples/realtime/realtime-rime-billing-handoff
//
// RIME_REALTIME_ENDPOINT, when set, overrides the Prism endpoint.
//
// Then open http://localhost:7860, connect, and ask about an invoice.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/observers"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/rime/realtime"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// errNoPipelineWorker is reported when the transfer runs outside a pipeline
// worker, which is what carries the settings update.
var errNoPipelineWorker = errors.New("the tool runs outside a pipeline worker")

// lookupInvoice reads an invoice's payment status.
func lookupInvoice(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		InvoiceID string `json:"invoice_id"`
	}
	_ = json.Unmarshal(params.Arguments, &args)
	result, err := json.Marshal(map[string]any{
		"invoice_id": args.InvoiceID,
		"status":     "paid",
		"total":      42,
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// transferToBilling transfers the caller to billing support.
func transferToBilling(ctx context.Context, params llm.FunctionCallParams) error {
	worker, ok := params.PipelineWorker.(*pipeline.Worker)
	if !ok {
		return errNoPipelineWorker
	}
	worker.QueueFrame(frames.NewLLMUpdateSettingsFrame(billingSettings()))
	return params.Result(ctx, `{"department": "billing"}`, nil)
}

// transferTool describes transferToBilling, the general assistant's tool.
func transferTool() frames.Tool {
	return frames.Tool{
		Name:        "transfer_to_billing",
		Description: "Transfer the caller to billing support.",
		Parameters:  json.RawMessage(`{"type": "object", "properties": {}, "required": []}`),
		Handler:     llm.FunctionCallHandler(transferToBilling),
	}
}

// invoiceTool describes lookupInvoice, the billing assistant's tool.
func invoiceTool() frames.Tool {
	return frames.Tool{
		Name:        "lookup_invoice",
		Description: "Read an invoice's payment status.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"invoice_id": {
					"type": "string",
					"description": "Invoice identifier."
				}
			},
			"required": ["invoice_id"]
		}`),
		Handler: llm.FunctionCallHandler(lookupInvoice),
	}
}

// supportSettings are the general support assistant's settings, which the
// conversation opens with.
func supportSettings() realtime.Settings {
	return realtime.Settings{
		SystemInstruction: settings.Set("You are a voice support assistant. Keep answers short. " +
			"Transfer the caller to billing for any question about invoices or payments."),
		Tools: settings.Set(frames.ToolsSchema{Standard: []frames.Tool{transferTool()}}),
	}
}

// billingSettings are the changes that turn the session into billing support.
func billingSettings() *realtime.Settings {
	return &realtime.Settings{
		SystemInstruction: settings.Set("You handle billing. Begin each answer with 'Billing support'. " +
			"Keep answers short. Use lookup_invoice for invoice status."),
		Tools: settings.Set(frames.ToolsSchema{Standard: []frames.Tool{invoiceTool()}}),
		ToolFillers: settings.Set(map[string]realtime.Filler{
			"lookup_invoice": {Type: "text", Text: new("Checking your invoice.")},
		}),
	}
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

// outputSampleRate is the rate Prism speaks at, which the output transport
// runs at.
const outputSampleRate = 24000

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	// The tools are configured on the service, so the conversation advertises
	// none and a settings update can replace them.
	s2s := realtime.New(realtime.Config{
		APIKey:   os.Getenv("RIME_API_KEY"),
		Endpoint: os.Getenv("RIME_REALTIME_ENDPOINT"),
		Settings: supportSettings(),
	})

	convo := frames.NewLLMContext("")
	convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Greet the caller and offer help."})
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(),
		agg.User(),
		s2s,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
			AudioOutSampleRate: outputSampleRate,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		Observers:               []pipeline.Observer{observers.NewTranscriptionLog(observers.TranscriptionLogConfig{})},
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	return r.Run(ctx, workers.RunOptions{})
}

func bot(ctx context.Context, args runner.Arguments) error {
	t, err := runner.CreateTransport(ctx, args, transportParams)
	if err != nil {
		return err
	}
	return runBot(ctx, t, args)
}

func main() {
	runner.Main(bot)
}
