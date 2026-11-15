// Command local-handoff-two-agents-tts is two LLM workers with a voice of
// their own each.
//
// It has the shape of local-handoff-two-agents, but each child worker runs its
// own TTS with a distinct voice. The main worker has no TTS: the audio comes
// from the child workers over the bus and is played by the main worker's
// transport. The workers announce the transfer ("let me connect you with...")
// before handing off.
//
// The architecture:
//
//	Main worker (no TTS):
//	  transport.in → STT → user_agg → BusBridge → transport.out → assistant_agg
//
//	Child worker (with TTS):
//	  bridge_in → LLM → TTS → bridge_out
//
//	OPENAI_API_KEY=… DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/multi-worker/local-handoff/local-handoff-two-agents-tts
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
)

const mainName = "acme"

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	WebRTC:  transport.DefaultParams,
}

// acmeTTSWorker is a child worker with an LLM and a TTS of its own, bridged to
// the main worker.
//
// Each child wraps the standard pipeline of its LLM with an extra TTS
// processor, so the audio is produced locally by each child and shipped to the
// main worker over the bus.
type acmeTTSWorker struct {
	*llmworker.Worker
}

// newAcmeTTSWorker builds a child worker named name, running model and
// speaking with the Cartesia voice voiceID.
func newAcmeTTSWorker(name string, model *chat.LLMService, voiceID string) *acmeTTSWorker {
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: voiceID,
	})
	w := &acmeTTSWorker{}
	w.Worker = llmworker.New(llmworker.Config{
		Name:     name,
		LLM:      model,
		Pipeline: pipeline.New(model, tts),
		Tools:    w.tools(),
		Bridged:  []string{},
	})
	return w
}

// tools are the worker's tools, each carrying its handler.
func (w *acmeTTSWorker) tools() []frames.Tool {
	return []frames.Tool{
		{
			Name:        "transfer_to_agent",
			Description: "Transfer the user to another agent.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"agent": {
						"type": "string",
						"description": "The agent to transfer to (e.g. 'greeter', 'support')."
					},
					"reason": {
						"type": "string",
						"description": "Why the user is being transferred."
					}
				},
				"required": ["agent", "reason"]
			}`),
			Handler:              llm.FunctionCallHandler(w.transferToAgent),
			CancelOnInterruption: new(false),
		},
		{
			Name:        "end_conversation",
			Description: "End the conversation when the user says goodbye.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"reason": {
						"type": "string",
						"description": "Why the conversation is ending."
					}
				},
				"required": ["reason"]
			}`),
			Handler: llm.FunctionCallHandler(w.endConversation),
		},
	}
}

// transferToAgent transfers the user to another agent.
func (w *acmeTTSWorker) transferToAgent(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Agent  string `json:"agent"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Info(fmt.Sprintf("Worker '%s': transferring to '%s' (%s)", w.Name(), args.Agent, args.Reason))
	if err := params.Result(ctx, fmt.Sprintf("Transferring to %s agent.", args.Agent), nil); err != nil {
		return err
	}
	message := fmt.Sprintf("The %s agent transferred the user to you. Take over the "+
		"conversation and help them. Reason for the transfer: %s", w.Name(), args.Reason)
	w.ActivateWorker(ctx, args.Agent, workers.ActivateOptions{
		Args: llmworker.ActivationArgs{
			Messages: []map[string]any{{"role": "developer", "content": message}},
		},
		DeactivateSelf: true,
	})
	return nil
}

// endConversation ends the conversation when the user says goodbye.
func (w *acmeTTSWorker) endConversation(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Info(fmt.Sprintf("Worker '%s': ending conversation (%s)", w.Name(), args.Reason))
	if err := params.Result(ctx, args.Reason, nil); err != nil {
		return err
	}
	w.End(ctx, args.Reason)
	return nil
}

// buildGreeter builds the greeter, which routes the user to support when they
// pick a product.
func buildGreeter() *acmeTTSWorker {
	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction("You are a friendly greeter for Acme Corp. The available products " +
		"are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme " +
		"Tornado Kit. Ask which one they'd like to learn more about. " +
		"When the user picks a product or asks a question about one, " +
		"call the transfer_to_agent tool with agent 'support'. " +
		"Do not answer product questions yourself. If the user says goodbye, " +
		"call the end_conversation tool. Keep responses brief: this is a " +
		"voice conversation.")
	return newAcmeTTSWorker("greeter", model, "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc") // Jacqueline
}

// buildSupport builds the support worker, which answers product questions
// and can hand back to the greeter.
func buildSupport() *acmeTTSWorker {
	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction("You are a support agent for Acme Corp. You know about three " +
		"products: Acme Rocket Boots (jet-powered boots, $299, run up " +
		"to 60 mph), Acme Invisible Paint (makes anything invisible for " +
		"24 hours, $49 per can), and Acme Tornado Kit (portable tornado " +
		"generator, $199, batteries included). Answer the user's questions " +
		"about these products. If the user wants to browse other products " +
		"or start over, call the transfer_to_agent tool with agent " +
		"'greeter'. If the user says goodbye, call the end_conversation " +
		"tool. Keep responses brief: this is a voice conversation.")
	return newAcmeTTSWorker("support", model, "a167e0f3-df7e-4d52-a9c3-f949145efdab") // Blake
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting two-agents-with-tts bot")

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	// The main worker has no TTS. The audio comes from the children over the
	// bus; the main bridge tees the user context out and pushes the incoming
	// audio and text frames back into the local pipeline.
	bridge := bus.NewBridgeProcessor(bus.BridgeConfig{
		Bus:        r.Bus(),
		WorkerName: mainName,
	})

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		bridge,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Name: mainName,
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	r.AddWorkers(ctx, buildGreeter(), buildSupport(), worker)

	events.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		worker.ActivateWorker(ctx, "greeter", workers.ActivateOptions{
			Args: llmworker.ActivationArgs{
				Messages: []map[string]any{{
					"role": "developer",
					"content": "Welcome the user to Acme Corp, mention the available products " +
						"and ask how you can help.",
				}},
			},
		})
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
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
