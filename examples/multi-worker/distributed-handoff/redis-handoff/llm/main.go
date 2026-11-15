// Command llm is the LLM worker of the redis-handoff example, run on machine B
// or locally alongside the main command.
//
// It is a standalone process that runs one LLM worker (greeter or support)
// attached to the same Redis-backed worker bus as the main worker. Several
// instances can run on different machines.
//
// The example is a module of its own, so only it depends on a Redis client.
// From this directory's parent:
//
//	OPENAI_API_KEY=… go run ./llm greeter -redis-url redis://localhost:6379
//	OPENAI_API_KEY=… go run ./llm support -redis-url redis://localhost:6379
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"

	redisbus "github.com/gojargo/jargo/bus/network/redis"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
	"github.com/joho/godotenv"
	goredis "github.com/redis/go-redis/v9"
)

// workerConfig is how one of the LLM workers is set up.
type workerConfig struct {
	systemInstruction string
	watch             []string
}

// The names of the workers this command can run.
const (
	greeterName = "greeter"
	supportName = "support"
)

// workerNames are the workers this command can run, in the order they are
// listed.
//
//nolint:gochecknoglobals // the fixed set of workers
var workerNames = []string{greeterName, supportName}

// workerConfigs are the workers this command can run.
//
//nolint:gochecknoglobals // the fixed set of workers
var workerConfigs = map[string]workerConfig{
	greeterName: {
		systemInstruction: "You are a friendly greeter for Acme Corp. The available products " +
			"are: the Acme Rocket Boots, the Acme Invisible Paint, and the Acme " +
			"Tornado Kit. Ask which one they'd like to learn more about. " +
			"When the user picks a product or asks a question about one, " +
			"immediately call the transfer_to_agent tool with agent 'support'. " +
			"Do not answer product questions yourself. If the user says goodbye, " +
			"call the end_conversation tool. Do not mention transferring; just do it " +
			"seamlessly. Keep responses brief: this is a voice conversation.",
		watch: []string{supportName},
	},
	supportName: {
		systemInstruction: "You are a support agent for Acme Corp. You know about three " +
			"products: Acme Rocket Boots (jet-powered boots, $299, run up " +
			"to 60 mph), Acme Invisible Paint (makes anything invisible for " +
			"24 hours, $49 per can), and Acme Tornado Kit (portable tornado " +
			"generator, $199, batteries included). Answer the user's questions " +
			"about these products. If the user wants to browse other products " +
			"or start over, call the transfer_to_agent tool with agent " +
			"'greeter'. If the user says goodbye, call the end_conversation " +
			"tool. Do not mention transferring; just do it seamlessly. " +
			"Keep responses brief: this is a voice conversation.",
		watch: []string{greeterName},
	},
}

// acmeLLMWorker is an LLM worker for Acme Corp with transfer and end tools.
type acmeLLMWorker struct {
	*llmworker.Worker
}

// newAcmeLLMWorker builds the worker named name ("greeter" or "support"),
// running with systemInstruction. watch names the sibling workers it watches
// through the registry, so it knows when they become available for a handoff.
func newAcmeLLMWorker(name, systemInstruction string, watch []string) *acmeLLMWorker {
	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	w := &acmeLLMWorker{}
	w.Worker = llmworker.New(llmworker.Config{
		Name:    name,
		LLM:     model,
		Tools:   w.tools(),
		Bridged: []string{},
	})
	// The sibling workers are watched once the worker has started.
	events.On(w.Events(), pipeline.EventPipelineStarted, func(ctx context.Context, _ *frames.StartFrame) {
		w.WatchWorkers(ctx, watch...)
	})
	return w
}

// tools are the worker's tools, each carrying its handler.
func (w *acmeLLMWorker) tools() []frames.Tool {
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
func (w *acmeLLMWorker) transferToAgent(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Agent  string `json:"agent"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Info(fmt.Sprintf("Worker '%s': transferring to '%s' (%s)", w.Name(), args.Agent, args.Reason))
	if err := params.Result(ctx, "", nil); err != nil {
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
func (w *acmeLLMWorker) endConversation(ctx context.Context, params llm.FunctionCallParams) error {
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

// errUsage is reported when the command is not told which worker to run.
var errUsage = errors.New("usage: llm {" + strings.Join(workerNames, ",") + "} [-redis-url URL] [-channel NAME]")

func run(ctx context.Context) error {
	_ = godotenv.Overload()

	redisURL := flag.String("redis-url", "redis://localhost:6379", "Redis URL")
	channel := flag.String("channel", "jargo:acme", "Redis pub/sub channel")
	flag.Parse()
	// The worker's name comes first, and the options may follow it.
	positional := flag.Args()
	if len(positional) == 0 || !slices.Contains(workerNames, positional[0]) {
		return errUsage
	}
	name := positional[0]
	if err := flag.CommandLine.Parse(positional[1:]); err != nil {
		return err
	}
	if flag.NArg() > 0 {
		return errUsage
	}

	redisOpts, err := goredis.ParseURL(*redisURL)
	if err != nil {
		return err
	}
	b := redisbus.NewBus(redisbus.Config{Client: goredis.NewClient(redisOpts), Channel: *channel})

	cfg := workerConfigs[name]
	worker := newAcmeLLMWorker(name, cfg.systemInstruction, cfg.watch)

	r := workers.NewRunner(workers.RunnerConfig{Bus: b.Bus})
	slog.Info(fmt.Sprintf("Starting %s worker, waiting for activation...", name))
	r.AddWorkers(ctx, worker)
	return r.Run(ctx, workers.RunOptions{})
}

func main() {
	if err := run(context.Background()); err != nil {
		slog.Error("llm worker failed", "err", err)
		os.Exit(1) //nolint:forbidigo // the program's entry point
	}
}
