// Command eval is a tiny demo bot for the jargo eval harness. It serves an eval
// endpoint over a plain WebSocket — no WebRTC, no API keys — so you can drive it
// with the CLI or from a Go test.
//
// Run it and eval it from the command line:
//
//	go run ./examples/eval    # serves ws://localhost:8080
//	go run ./cmd/jargo eval run examples/eval/scenarios/greeting.yaml --bot-url ws://localhost:8080
//
// Or run the same scenario in-process:
//
//	go test ./examples/eval
//
// A real bot would wire an STT/LLM/TTS in buildBot; this one uses a canned,
// deterministic responder so the example is self-contained and its assertions
// are stable.
package main

import (
	"context"
	"encoding/json"
	"log"
	"log/slog"
	"net/http"
	"strings"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
)

func main() {
	const addr = ":8080"
	http.Handle("/", eval.Handler(buildBot))
	slog.Info("jargo eval demo bot listening", "url", "ws://localhost"+addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

// buildBot assembles the demo pipeline around the harness-provided transport
// endpoints. The RTVI processor the worker adds is what lets the harness drive
// the bot and observe its events.
func buildBot(in, out processor.Processor) *pipeline.Worker {
	agg := aggregators.New(frames.NewLLMContext("You are a friendly demo assistant."))
	// The worker adds the RTVI processor and its observer itself, which is what
	// lets the harness drive the bot and watch what it does.
	return pipeline.NewWorker(pipeline.New(
		in, agg.User(), newDemoLLM(), out, agg.Assistant(),
	), pipeline.WorkerConfig{})
}

// demoLLM stands in for an LLM service: it answers each user turn with a canned
// reply, so the example runs deterministically and without a provider.
type demoLLM struct {
	*processor.Base
}

func newDemoLLM() *demoLLM {
	m := &demoLLM{}
	m.Base = processor.New("DemoLLM", m)
	return m
}

func (m *demoLLM) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := m.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	cf, ok := f.(*frames.LLMContextFrame)
	if !ok {
		return m.PushFrame(ctx, f, dir)
	}

	last := ""
	if msgs := cf.Context.Messages(); len(msgs) > 0 {
		last = msgs[len(msgs)-1].Text
	}
	if err := m.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	// An order places the order rather than answering in text, so the example
	// shows a scenario asserting on a tool call and its arguments.
	if size, ok := order(last); ok {
		args := json.RawMessage(`{"drink":"latte","size":"` + size + `"}`)
		call := frames.NewFunctionCallInProgressFrame("call-1", "place_order", args, true, "g1")
		if err := m.PushFrame(ctx, call, processor.Downstream); err != nil {
			return err
		}
	} else if err := m.PushFrame(ctx, frames.NewLLMTextFrame(reply(last)), processor.Downstream); err != nil {
		return err
	}
	return m.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}

// reply is the demo's canned response logic.
func reply(userText string) string {
	lower := strings.ToLower(userText)
	switch {
	case strings.Contains(lower, "hello"), strings.Contains(lower, "hey"):
		return "Hello! How can I help you today?"
	case strings.Contains(lower, "coffee"):
		return "Sure, what kind of coffee would you like?"
	default:
		return "You said: " + userText
	}
}

// order reads a drink size out of the user's turn, reporting whether the turn
// is an order at all.
func order(userText string) (string, bool) {
	lower := strings.ToLower(userText)
	if !strings.Contains(lower, "latte") {
		return "", false
	}
	if strings.Contains(lower, "large") {
		return "large", true
	}
	return "small", true
}
