package llmworker

import (
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
)

// ContextConfig configures a context worker. It is Config without the pipeline,
// which the worker builds itself around the conversation it owns.
type ContextConfig struct {
	// Name is what other workers address this one by.
	Name string
	// LLM is the language model service the worker runs.
	LLM ToolRegistrar
	// Tools are the tools the model may call.
	Tools []frames.Tool
	// Active reports whether the worker starts active; nil leaves it inactive.
	Active *bool
	// Bridged names the bridges this worker's pipeline exchanges frames over.
	Bridged []string
	// DeferToolFrames says whether what a tool handler queues is held until the
	// calls in flight finish; nil holds it.
	DeferToolFrames *bool
	// Context is the conversation to run. Nil starts an empty one.
	Context *frames.LLMContext
	// AggregatorOptions configure the aggregator pair the worker builds.
	AggregatorOptions []aggregators.Option
	// WorkerConfig is the rest of the pipeline worker's configuration.
	WorkerConfig pipeline.WorkerConfig
}

// ContextWorker is an LLM worker that owns its conversation.
//
// It is for a worker that tracks a conversation of its own rather than sharing
// the one a transport pipeline carries: the pipeline is built as user
// aggregator, model, assistant aggregator, so the worker keeps its own history
// without the caller wiring it up.
type ContextWorker struct {
	*Worker

	convo       *frames.LLMContext
	aggregators *aggregators.Pair
}

// NewContext builds a worker that owns its conversation.
func NewContext(cfg ContextConfig) *ContextWorker {
	convo := cfg.Context
	if convo == nil {
		convo = frames.NewLLMContext("")
	}
	pair := aggregators.New(convo, cfg.AggregatorOptions...)

	c := &ContextWorker{convo: convo, aggregators: pair}
	c.Worker = New(Config{
		Name:            cfg.Name,
		LLM:             cfg.LLM,
		Pipeline:        pipeline.New(pair.User(), cfg.LLM, pair.Assistant()),
		Tools:           cfg.Tools,
		Active:          cfg.Active,
		Bridged:         cfg.Bridged,
		DeferToolFrames: cfg.DeferToolFrames,
		WorkerConfig:    cfg.WorkerConfig,
	})
	return c
}

// Context is the conversation this worker owns.
func (c *ContextWorker) Context() *frames.LLMContext { return c.convo }

// UserAggregator is the half that gathers what the user said.
func (c *ContextWorker) UserAggregator() *aggregators.UserAggregator {
	return c.aggregators.User()
}

// AssistantAggregator is the half that gathers what the bot said, and is where
// the events reporting a finished turn are raised.
func (c *ContextWorker) AssistantAggregator() *aggregators.AssistantAggregator {
	return c.aggregators.Assistant()
}
