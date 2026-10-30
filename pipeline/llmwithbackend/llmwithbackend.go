package llmwithbackend

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/adapter"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/jobcontext"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
)

// Frontend is what LLMWithBackend needs of the frontend LLM service. The LLM
// service base satisfies all of it but the adapter, which a service converting
// through one exposes (see llm.AdapterHolder).
type Frontend interface {
	processor.Processor
	llm.AdapterHolder
	llm.LLMWithBackendRoleObjector
	// RegisterFunction registers a tool's handler.
	RegisterFunction(name string, h llm.FunctionCallHandler, opts ...llm.RegisterOption)
	// AppendSystemInstruction adds text to the model's system instruction.
	AppendSystemInstruction(instruction string)
	// ServiceMetadataFrame describes the service, saying whether it is
	// speech-to-speech.
	ServiceMetadataFrame() frames.ServiceMetadata
}

// ErrNoBuiltinTools is reported when the frontend cannot carry a tool of its
// own beside the conversation's: the delegate tool would never reach its model.
var ErrNoBuiltinTools = errors.New("the frontend cannot carry built-in tools")

// ErrFrontendDeclined is reported when the frontend service cannot be the
// frontend; the error names the service's own reason.
var ErrFrontendDeclined = errors.New("the LLM service declines the frontend role")

// ErrToolWithoutHandler is reported when a tool the connector builds carries no
// handler to install on the frontend.
var ErrToolWithoutHandler = errors.New("the tool carries no handler")

// Config configures an LLMWithBackend.
type Config struct {
	// Frontend is the conversational LLM service, text or speech-to-speech.
	Frontend Frontend
	// Backend is the backend worker. It is added to the pipeline worker at
	// setup, so the app never wires it up. Leave it nil and set BackendName to
	// use a backend registered elsewhere (in the app, or in another process on
	// a shared bus).
	Backend *llmworker.BackendWorker
	// BackendName is the name of a backend registered elsewhere, read when
	// Backend is nil.
	BackendName string
	// Connector is how delegation works. A default BackendConnector when nil.
	Connector *BackendConnector
	// PredictiveDelegation speeds up responses that need backend involvement:
	// its classifier determines whether to wait for the frontend's inference or
	// go ahead and trigger an early backend delegation. Only for a text
	// frontend. None when nil.
	PredictiveDelegation *PredictiveDelegation
}

// workerHost is what the pipeline worker the frontend runs in offers: adding
// the local backend as a child, and the jobs the session with it runs on.
type workerHost interface {
	jobcontext.Requester
	AddWorkers(ctx context.Context, children ...workers.Worker)
}

// LLMWithBackend is a conversational LLM, kept fast and light, with a backend
// that does the work it hands off.
//
// Alpha: the API is unstable; names, arguments, the strategies and the job
// contract are likely to change between releases.
//
// Put it where the LLM goes in a pipeline. It is a pipeline, not an LLM service:
// functions, settings updates and event handlers go on Frontend. It wraps the
// frontend service, installs the connector's tools on it as built-in tools,
// appends the connector's guidance to the frontend's system instruction, adds a
// local backend worker to the pipeline worker so the app never wires it up, and
// holds the session with the backend for the pipeline's life. A built-in tool is
// sent on every inference beside whatever tools the frontend has, and never
// enters the context's tool set, so a tool change announced to the model never
// mentions it. The tools are the backend's; the frontend has delegate and no
// more, though a tool it must keep, in its context or configured on the service,
// stays.
//
// The guidance is the connector's pairing instruction (how to work with a
// backend: when and how to delegate, and what to do with the backend's messages)
// with the strategy's (how the request is worded) in place. The frontend's own
// system instruction is the place to say what the backend is for, in plain
// words, as the backend's own instruction does: no tool names, nothing to update
// when a tool changes.
//
//	backend, _ := llmworker.NewBackend(llmworker.BackendConfig{LLM: anthropicLLM, Context: convo})
//	llm, _ := llmwithbackend.New(llmwithbackend.Config{Frontend: openaiLLM, Backend: backend})
//	pipe := pipeline.New(input, stt, agg.User(), llm, tts, output, agg.Assistant())
type LLMWithBackend struct {
	*pipeline.Pipeline

	frontend    Frontend
	backend     *llmworker.BackendWorker
	backendName string
	connector   *BackendConnector
	predictive  *PredictiveDelegation

	mu      sync.Mutex
	cancel  context.CancelFunc
	session chan struct{}
}

// New builds an LLMWithBackend. It reports an error when the frontend service
// declines the role, when it cannot carry built-in tools, when the connector's
// request strategy reads the frontend's context and the frontend is
// speech-to-speech, or when predictive delegation is given with a
// speech-to-speech frontend or with a request strategy whose delegate call needs
// arguments.
func New(cfg Config) (*LLMWithBackend, error) {
	slog.Warn("LLMWithBackend is alpha: its API is likely to change between releases")
	frontend := cfg.Frontend
	if objection := frontend.LLMWithBackendRoleObjection(llm.RoleFrontend); objection != "" {
		return nil, fmt.Errorf("%w: %s", ErrFrontendDeclined, objection)
	}
	holder := frontend.LLMAdapter()
	if holder == nil {
		return nil, ErrNoBuiltinTools
	}
	name := cfg.BackendName
	if cfg.Backend != nil {
		name = cfg.Backend.Name()
	}
	connector := cfg.Connector
	if connector == nil {
		connector = NewBackendConnector(ConnectorConfig{})
	}
	realtime := isRealtime(frontend)
	if err := connector.Bind(ConnectorContext{
		BackendName:        name,
		FrontendIsRealtime: realtime,
	}); err != nil {
		return nil, err
	}
	var gated predictiveFrontend
	if predictive := cfg.PredictiveDelegation; predictive != nil {
		if realtime {
			return nil, ErrPredictiveDelegationNeedsTextFrontend
		}
		if strategy := connector.RequestStrategy(); strategy.DelegateNeedsArguments() {
			return nil, fmt.Errorf("%w, which %T cannot build a request from; use a strategy that needs "+
				"none, such as TranscriptBackendRequestStrategy", ErrPredictiveDelegationNeedsNoArguments, strategy)
		}
		var ok bool
		if gated, ok = frontend.(predictiveFrontend); !ok {
			return nil, fmt.Errorf("%w: %T is not built on the LLM service base",
				ErrPredictiveDelegationNeedsTextFrontend, frontend)
		}
	}
	if instruction := connector.FrontendInstruction(); instruction != "" {
		frontend.AppendSystemInstruction(instruction)
	}
	for _, tool := range connector.Tools() {
		h, ok := tool.Handler.(llm.FunctionCallHandler)
		if !ok {
			return nil, fmt.Errorf("%w: %s", ErrToolWithoutHandler, tool.Name)
		}
		frontend.RegisterFunction(tool.Name, h, llm.WithCancelOnInterruption(false))
		builtin := tool
		builtin.Handler = nil
		holder.SetBuiltin(adapter.Builtin{Tool: builtin})
	}
	procs := []processor.Processor{frontend}
	if predictive := cfg.PredictiveDelegation; predictive != nil {
		predictive.bind(gated, connector)
		procs = []processor.Processor{predictive, frontend}
	}
	return &LLMWithBackend{
		Pipeline:    pipeline.New(procs...),
		frontend:    frontend,
		backend:     cfg.Backend,
		backendName: name,
		connector:   connector,
		predictive:  cfg.PredictiveDelegation,
	}, nil
}

// isRealtime reports whether a service announces itself as speech-to-speech.
func isRealtime(frontend Frontend) bool {
	md, ok := frontend.ServiceMetadataFrame().(*frames.LLMServiceMetadataFrame)
	return ok && md.Realtime
}

// Frontend is the frontend LLM service.
func (l *LLMWithBackend) Frontend() Frontend { return l.frontend }

// Backend is the local backend worker, or nil for one registered elsewhere.
func (l *LLMWithBackend) Backend() *llmworker.BackendWorker { return l.backend }

// BackendName is the name of the backend worker.
func (l *LLMWithBackend) BackendName() string { return l.backendName }

// Connector is the connector joining the two layers.
func (l *LLMWithBackend) Connector() *BackendConnector { return l.connector }

// PredictiveDelegation is the PredictiveDelegation in front of the frontend, or
// nil.
func (l *LLMWithBackend) PredictiveDelegation() *PredictiveDelegation { return l.predictive }

// Tool is the delegate tool installed on the frontend.
func (l *LLMWithBackend) Tool() frames.Tool { return l.connector.Tool() }

// Setup adds a local backend worker as a child of the pipeline worker, sets the
// frontend up, and opens the session with the backend.
func (l *LLMWithBackend) Setup(ctx context.Context, s processor.Setup) error {
	host, ok := s.Running.(workerHost)
	if !ok {
		return l.Pipeline.Setup(ctx, s)
	}
	// Before setting the frontend up: it gets the StartFrame as soon as its own
	// setup is done, and can delegate from then on.
	if l.backend != nil {
		host.AddWorkers(ctx, l.backend)
	}
	if err := l.Pipeline.Setup(ctx, s); err != nil {
		return err
	}
	sessionCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	l.mu.Lock()
	l.cancel, l.session = cancel, done
	l.mu.Unlock()
	go func() {
		defer close(done)
		l.connector.RunSession(sessionCtx, host, l.frontend)
	}()
	return nil
}

// Cleanup closes the session with the backend, then cleans up the frontend.
func (l *LLMWithBackend) Cleanup(ctx context.Context) error {
	l.mu.Lock()
	cancel, done := l.cancel, l.session
	l.cancel, l.session = nil, nil
	l.mu.Unlock()
	if cancel != nil {
		cancel()
		<-done
	}
	return l.Pipeline.Cleanup(ctx)
}
