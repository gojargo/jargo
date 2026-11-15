// Command parallel-debate is a parallel debate run through a job group.
//
// A voice bot receives a topic from the user and fans out to three workers in
// parallel through the worker's JobGroup. Each worker runs its own LLM context,
// so it remembers the previous topics across debate rounds. The bot collects
// the three perspectives and the main worker's LLM synthesizes a balanced
// answer.
//
// The architecture:
//
//	Main worker (transport + LLM + debate tool)
//	  └── JobGroup(advocate, critic, analyst)
//	        └── debateWorker (an LLM context worker, one per role)
//
//	OPENAI_API_KEY=… DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/multi-worker/parallel-debate/parallel-debate
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/jobcontext"
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

// roles are the debate's roles, in the order they are asked and reported.
//
//nolint:gochecknoglobals // the fixed set of roles
var roles = []string{"advocate", "critic", "analyst"}

// rolePrompts are the system instructions of each role.
//
//nolint:gochecknoglobals // the fixed prompts of the roles
var rolePrompts = map[string]string{
	"advocate": "You argue IN FAVOR of the topic. Present the strongest case for why " +
		"this is a good idea, with concrete benefits. Be persuasive but honest. " +
		"Be concise, just 2-3 sentences.",
	"critic": "You argue AGAINST the topic. Present the strongest concerns, risks, " +
		"and downsides. Be critical but fair. Be concise, just 2-3 sentences.",
	"analyst": "You provide a BALANCED, NEUTRAL analysis. Weigh both sides objectively " +
		"and highlight the key trade-offs. Be concise, just 2-3 sentences.",
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	WebRTC:  transport.DefaultParams,
}

// debateWorker generates a perspective using its own LLM context.
//
// Each worker keeps its own conversation, so it remembers the previous topics
// across debate rounds. A job request appends the new topic and runs the LLM;
// the assistant aggregator captures the full reply, which is sent back as the
// job response.
type debateWorker struct {
	*llmworker.ContextWorker

	role string

	mu           sync.Mutex
	currentJobID string
}

// newDebateWorker builds the worker for role, which is one of "advocate",
// "critic" and "analyst": it is used as the worker's name and selects the
// system prompt.
func newDebateWorker(role string) *debateWorker {
	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(rolePrompts[role])

	d := &debateWorker{role: role}
	d.ContextWorker = llmworker.NewContext(llmworker.ContextConfig{Name: role, LLM: model})
	// The base reaches OnJobRequest through the worker it is bound to.
	d.Bind(d)

	events.On(d.AssistantAggregator().Events(), aggregators.EventAssistantTurnStopped,
		func(ctx context.Context, message aggregators.AssistantTurnStopped) {
			text := message.Content
			slog.Info(fmt.Sprintf("Worker '%s': completed (%d chars)", d.Name(), len(text)))
			d.mu.Lock()
			jobID := d.currentJobID
			d.currentJobID = ""
			d.mu.Unlock()
			if jobID == "" {
				return
			}
			if err := d.SendJobResponse(ctx, jobID, map[string]any{"role": d.role, "text": text},
				workers.JobResponseOptions{}); err != nil {
				slog.Warn("could not answer the debate job", "worker", d.Name(), "err", err)
			}
		})
	return d
}

// OnJobRequest puts the topic into the conversation and runs the LLM.
func (d *debateWorker) OnJobRequest(ctx context.Context, m *bus.JobRequestMessage) {
	d.ContextWorker.OnJobRequest(ctx, m)
	d.mu.Lock()
	d.currentJobID = m.JobID
	d.mu.Unlock()
	f := frames.NewLLMMessagesAppendFrame([]frames.Message{
		{Role: frames.RoleDeveloper, Text: fmt.Sprintf("Topic: %v", m.Payload["topic"])},
	})
	f.RunLLM = true
	d.QueueFrame(ctx, f)
}

// errNoPipelineWorker is reported when the debate tool runs outside a pipeline
// worker, which is what launches the job group.
var errNoPipelineWorker = errors.New("the debate tool runs outside a pipeline worker")

// debate analyzes a topic from multiple perspectives (advocate, critic,
// analyst).
func debate(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Topic string `json:"topic"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	worker, ok := params.PipelineWorker.(*pipeline.Worker)
	if !ok {
		return errNoPipelineWorker
	}
	slog.Info(fmt.Sprintf("Starting debate on '%s'", args.Topic))
	tg, err := worker.JobGroup(ctx, roles, jobcontext.Request{
		Payload: map[string]any{"topic": args.Topic},
		Timeout: 30 * time.Second,
	}, nil)
	if err != nil {
		return err
	}
	responses := tg.Responses()
	parts := make([]string, 0, len(responses))
	for _, role := range roles {
		r, ok := responses[role]
		if !ok {
			continue
		}
		parts = append(parts, fmt.Sprintf("%s: %v", strings.ToUpper(fmt.Sprint(r["role"])), r["text"]))
	}
	slog.Info("Debate complete, synthesizing")
	return params.Result(ctx, strings.Join(parts, "\n\n"), nil)
}

// tools describes the debate tool to the LLM, carrying its handler. It is not
// canceled when the user interrupts, and may take a minute.
func tools() []frames.Tool {
	return []frames.Tool{{
		Name:        "debate",
		Description: "Analyze a topic from multiple perspectives (advocate, critic, analyst).",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"topic": {
					"type": "string",
					"description": "The topic or question to debate."
				}
			},
			"required": ["topic"]
		}`),
		Handler:              llm.FunctionCallHandler(debate),
		CancelOnInterruption: new(false),
		TimeoutSecs:          new(60.0),
	}}
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting parallel-debate bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", // Jacqueline
	})
	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction("You are a debate moderator in a voice conversation. When the user " +
		"gives you a topic, call the debate tool to gather perspectives from " +
		"three viewpoints (advocate, critic, analyst). Then synthesize the " +
		"results into a clear, balanced summary for the user. Keep your " +
		"responses concise and natural for speaking.")

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		model,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Name: "parallel-debate",
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})

	r.AddWorkers(ctx,
		newDebateWorker("advocate"),
		newDebateWorker("critic"),
		newDebateWorker("analyst"),
		worker,
	)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Greet the user and tell them you can moderate a debate on any " +
				"topic. Ask what they'd like to explore.",
		})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
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
