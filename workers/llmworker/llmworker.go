package llmworker

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/workers"
)

// inToolCall marks a context as running inside one of a worker's own tool
// handlers. It carries the worker rather than a flag, because a handler may
// reach another worker and only the one running the call will release what it
// held: a worker that held a frame for a call it is not running would never let
// it go.
type inToolCallKey struct{}

// Config configures an LLM worker.
type Config struct {
	// Name is what other workers address this one by.
	Name string
	// LLM is the language model service the worker runs. The tools are
	// registered on it.
	LLM ToolRegistrar
	// Pipeline is the pipeline to run. Nil runs the model on its own, which is
	// what a worker doing nothing but answering needs.
	Pipeline processor.Processor
	// Tools are the tools the model may call. Each is registered on the service
	// with the call options it carries, wrapped so that what its handler queues
	// is held until the call finishes.
	Tools []frames.Tool
	// Active reports whether the worker starts active. Nil leaves it inactive,
	// because a worker with a model behind it is almost always started by
	// something else deciding it is time.
	Active *bool
	// Bridged names the bridges this worker's pipeline exchanges frames over.
	// Non-nil wraps the pipeline in bus edges, and turns off the RTVI processor
	// a standalone worker gets, since a bridged worker has no client of its own.
	Bridged []string
	// DeferToolFrames says whether what a tool handler queues is held until
	// every call in flight has finished. Nil holds it, which is what keeps a
	// tool's own frames behind the result the model is waiting for.
	DeferToolFrames *bool
	// WorkerConfig is the rest of the pipeline worker's configuration. Name,
	// Active, Bridged and the RTVI setting are taken from the fields above.
	WorkerConfig pipeline.WorkerConfig
}

// ToolRegistrar is what an LLM worker needs of the service it runs: somewhere to
// register the tools. The LLM service base satisfies it.
type ToolRegistrar interface {
	processor.Processor
	RegisterFunction(name string, h llm.FunctionCallHandler, opts ...llm.RegisterOption)
}

// Worker runs a language model as a worker on the bus, and holds back what its
// tools do until the calls that asked for it have finished.
//
// A tool handler runs while the model is still waiting for its result. Anything
// the handler does to the pipeline in the meantime, appending to the
// conversation, ending the session, handing over to another worker, would land
// ahead of that result and put the turn out of order. What the handler queues is
// therefore held, and released once the last call in flight is done; ending and
// handing over wait for the same moment.
type Worker struct {
	*pipeline.Worker

	llm             ToolRegistrar
	tools           []frames.Tool
	deferToolFrames bool

	mu sync.Mutex
	// inflight counts the tool calls running now, so what was held is released
	// only once the last of them is done.
	inflight int
	// deferred are the frames held for those calls, in the order they were
	// queued.
	deferred []deferredFrame
	// handover is what a tool asked for that must wait for its own call to
	// finish: ending the session, or activating another worker.
	handover func(ctx context.Context)
	// closing is set once the worker is ending, after which nothing more is
	// held: there is no later moment to release it in.
	closing bool
}

// deferredFrame is one frame held for the tool call that queued it.
type deferredFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

// New builds an LLM worker.
func New(cfg Config) *Worker {
	w := &Worker{
		llm:             cfg.LLM,
		tools:           cfg.Tools,
		deferToolFrames: cfg.DeferToolFrames == nil || *cfg.DeferToolFrames,
	}

	// The tools are registered before the pipeline is built, so a call arriving
	// on the first turn already has a handler behind it.
	w.registerTools()

	wc := cfg.WorkerConfig
	wc.Name = cfg.Name
	wc.Active = cfg.Active
	if wc.Active == nil {
		inactive := false
		wc.Active = &inactive
	}
	wc.Bridged = cfg.Bridged
	if wc.EnableRTVI == nil {
		// A bridged worker exchanges frames with another worker rather than with
		// a client, so there is nobody for the protocol to talk to.
		bridged := cfg.Bridged != nil
		enable := !bridged
		wc.EnableRTVI = &enable
	}
	// Metrics are on unless the caller turned them on itself: a worker running a
	// model is where the cost and the latency are, and both are worth having by
	// default.
	if !wc.Params.EnableMetrics && !wc.Params.EnableUsageMetrics {
		wc.Params.EnableMetrics = true
		wc.Params.EnableUsageMetrics = true
	}

	pipe := cfg.Pipeline
	if pipe == nil {
		pipe = pipeline.New(cfg.LLM)
	}
	w.Worker = pipeline.NewWorker(pipe, wc)
	return w
}

// LLM is the service this worker runs.
func (w *Worker) LLM() ToolRegistrar { return w.llm }

// ToolCallActive reports whether any of this worker's tools is running.
func (w *Worker) ToolCallActive() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.inflight > 0
}

// Tools are the tools the worker registered, which is what it advertises when
// it activates.
func (w *Worker) Tools() []frames.Tool { return w.tools }

// OnActivated advertises the tools and puts the activation's messages into the
// conversation.
func (w *Worker) OnActivated(ctx context.Context, args map[string]any) {
	w.Worker.OnActivated(ctx, args)

	if len(w.tools) > 0 {
		w.QueueFrame(ctx, frames.NewLLMSetToolsFrame(w.tools))
	}

	activation := ActivationArgsFrom(args)
	if len(activation.Messages) == 0 {
		return
	}
	messages := make([]frames.Message, 0, len(activation.Messages))
	for _, raw := range activation.Messages {
		messages = append(messages, messageFrom(raw))
	}
	f := frames.NewLLMMessagesAppendFrame(messages)
	f.RunLLM = activation.runLLM()
	w.QueueFrame(ctx, f)
}

// QueueFrame queues a frame, holding it when one of this worker's own tool
// handlers is what queued it.
//
// A frame queued from inside a handler, or from anything that handler passes its
// context to, is held and delivered once the last call finishes. Everything else
// is queued at once: the worker's own traffic, what arrives over the bus, and
// what a handler on a different worker queues here, which this worker would
// never release.
//
// Reaching past this to the embedded worker's QueueFrame queues without holding,
// which is what a caller wants when it is deliberately acting inside a call.
func (w *Worker) QueueFrame(ctx context.Context, f frames.Frame, dir ...processor.Direction) {
	direction := processor.Downstream
	if len(dir) > 0 {
		direction = dir[0]
	}

	w.mu.Lock()
	hold := w.deferToolFrames && !w.closing && ctx.Value(inToolCallKey{}) == w
	if hold {
		w.deferred = append(w.deferred, deferredFrame{frame: f, dir: direction})
	}
	w.mu.Unlock()

	if !hold {
		w.Worker.QueueFrame(f, direction)
	}
}

// End brings the session to a close once the tool call asking for it is done.
//
// A tool that ends the session is still running when it asks, and ending the
// worker underneath it would leave the rest of that call to nobody. Deliver the
// call's result first, so what the model says about it is spoken before the
// session goes.
func (w *Worker) End(ctx context.Context, reason string) {
	w.mu.Lock()
	w.closing = true
	w.mu.Unlock()
	w.afterToolCalls(ctx, func(ctx context.Context) { w.Worker.End(ctx, reason) })
}

// ActivateWorker hands over to another worker once the tool call asking for it
// is done, for the reason End waits.
func (w *Worker) ActivateWorker(ctx context.Context, name string, opts workers.ActivateOptions) {
	w.afterToolCalls(ctx, func(ctx context.Context) { w.Worker.ActivateWorker(ctx, name, opts) })
}

// ProcessDeferredFrames is called with the frames held during a run of tool
// calls, just before they are queued, and returns the ones to queue. The
// default returns them as they are; override it to inspect, reorder or drop
// them.
func (w *Worker) ProcessDeferredFrames(_ context.Context, held []deferredFrame) []deferredFrame {
	return held
}

// registerTools registers each tool on the service, wrapped so the worker knows
// when its handlers are running.
//
// A tool with no handler is advertised and not registered: something else
// answers it, and wrapping a handler that is not there would report calls this
// worker never ran.
func (w *Worker) registerTools() {
	for _, t := range w.tools {
		handler, ok := t.Handler.(llm.FunctionCallHandler)
		if !ok {
			if t.Handler != nil {
				slog.Warn("llmworker: a tool's handler is not a function-call handler, "+
					"so the tool is advertised without one", "tool", t.Name)
			}
			continue
		}
		opts := []llm.RegisterOption{}
		if t.CancelOnInterruption != nil {
			opts = append(opts, llm.WithCancelOnInterruption(*t.CancelOnInterruption))
		}
		if t.TimeoutSecs != nil {
			opts = append(opts, llm.WithTimeout(secondsToDuration(*t.TimeoutSecs)))
		}
		w.llm.RegisterFunction(t.Name, w.track(handler), opts...)
	}
}

// track wraps a handler so the worker knows a call of its own is running, and
// releases what was held for it once the last one finishes.
func (w *Worker) track(handler llm.FunctionCallHandler) llm.FunctionCallHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) error {
		w.mu.Lock()
		w.inflight++
		w.mu.Unlock()

		// The context carries the worker, so anything the handler passes it to
		// is recognized as part of the call. A handler that starts work without
		// it is outside the call as far as this is concerned, which is the same
		// answer as for work started on another worker.
		err := handler(context.WithValue(ctx, inToolCallKey{}, w), params)

		w.mu.Lock()
		w.inflight--
		last := w.inflight <= 0
		if last {
			w.inflight = 0
		}
		closing := w.closing
		w.mu.Unlock()

		if !last {
			return err
		}
		if !closing {
			w.flushDeferred(ctx)
		}
		w.runHandover(ctx)
		return err
	}
}

// afterToolCalls runs what a tool asked for, once no call of this worker's is in
// flight. Outside a call there is nothing to wait for, so it runs now.
func (w *Worker) afterToolCalls(ctx context.Context, run func(context.Context)) {
	w.mu.Lock()
	waiting := w.inflight > 0
	if waiting {
		w.handover = run
	}
	w.mu.Unlock()

	if !waiting {
		run(ctx)
	}
}

// runHandover runs what a tool asked to happen after its call, if anything did.
func (w *Worker) runHandover(ctx context.Context) {
	w.mu.Lock()
	run := w.handover
	w.handover = nil
	w.mu.Unlock()
	if run != nil {
		run(ctx)
	}
}

// flushDeferred queues what was held, in the order it was queued.
//
// The pipeline is drained first, so the held frames go in behind the result the
// model was waiting for rather than ahead of it. With nothing held there is
// nothing to order and the round trip would buy nothing.
func (w *Worker) flushDeferred(ctx context.Context) {
	w.mu.Lock()
	held := w.deferred
	w.deferred = nil
	w.mu.Unlock()

	if len(held) == 0 {
		return
	}
	if err := w.Flush(ctx); err != nil {
		slog.WarnContext(ctx, "llmworker: the pipeline did not settle before releasing "+
			"what a tool queued, so it may land ahead of the call's result",
			"worker", w.Name(), "err", err)
	}
	for _, d := range w.ProcessDeferredFrames(ctx, held) {
		w.Worker.QueueFrame(d.frame, d.dir)
	}
}

// secondsToDuration is a tool's timeout as a duration. A tool carries it in
// seconds, which is the unit the model's own configuration uses.
func secondsToDuration(secs float64) time.Duration {
	return time.Duration(secs * float64(time.Second))
}

// messageFrom reads a conversation message out of the shape it travels in on
// the bus.
func messageFrom(raw map[string]any) frames.Message {
	m := frames.Message{}
	if role, ok := raw["role"].(string); ok {
		m.Role = frames.Role(role)
	}
	if content, ok := raw["content"].(string); ok {
		m.Text = content
	}
	return m
}
