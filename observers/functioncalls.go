package observers

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// FunctionCallEventKind is where a function call has got to.
//
// A call starts when the model asks for it and is in progress once it is
// running, which are two moments rather than one: calls run one at a time unless
// the service was built to run them in parallel, so a call can wait in between,
// and a call still waiting when the conversation moves on never runs at all.
//
// A call settles in one of four ways: it returned, its handler failed, it ran
// past its deadline, or it was canceled, by an interruption or by the model
// asking for it.
type FunctionCallEventKind string

// The moments in the life of a function call.
const (
	FunctionCallStarted    FunctionCallEventKind = "function_call_started"
	FunctionCallInProgress FunctionCallEventKind = "function_call_in_progress"
	FunctionCallCompleted  FunctionCallEventKind = "function_call_completed"
	FunctionCallFailed     FunctionCallEventKind = "function_call_failed"
	FunctionCallTimedOut   FunctionCallEventKind = "function_call_timed_out"
	FunctionCallCanceled   FunctionCallEventKind = "function_call_canceled"
)

// FunctionCallEvent is one moment in the life of a function call.
//
// The moments that open a call describe it; the moment that settles it describes
// what became of it. Each names the call, so they read as a sequence without any
// of them repeating the others.
type FunctionCallEvent struct {
	// Kind is what happened to the call.
	Kind FunctionCallEventKind
	// FunctionName is the name of the function.
	FunctionName string
	// ToolCallID is the model's identifier for this call, unique within a
	// conversation.
	ToolCallID string
	// Timestamp is the wall-clock time of the moment.
	Timestamp time.Time
	// GroupID identifies the calls the model asked for in one response, which run
	// together. Set when the call goes in progress.
	GroupID string
	// Blocking reports whether the conversation waited for this call. A call that
	// does not block is answered later through a developer message, while the
	// model carries on talking. Set when the call goes in progress, and nil
	// everywhere else.
	Blocking *bool
	// Arguments is what the model passed to the function, when the observer is
	// reporting arguments. Set both when the call starts and when it goes in
	// progress, since a call can be reported at either moment without the other.
	Arguments json.RawMessage
	// StartedAt is when the call started, on the moment it goes in progress, so
	// the wait between the two reads from one record. The zero value means the
	// call started before the observer was watching.
	StartedAt time.Time
	// InProgressAt is when the call went in progress, on the moment that settles
	// it, so the time it ran reads from one record. The zero value means it began
	// running before the observer was watching.
	InProgressAt time.Time
	// Result is what the handler returned, when the observer is reporting
	// results.
	Result string
	// Error is what went wrong, on a call whose handler failed.
	Error string
}

// FunctionCallConfig configures a FunctionCalls observer.
type FunctionCallConfig struct {
	// MaxFrames is how many recent frame ids the observer remembers to
	// recognize one it has already reported; 0 uses 100.
	MaxFrames int
	// IncludeArguments reports the arguments a call was made with. Nil leaves it
	// on: they are small, and they are the reason a call is worth reading at all.
	IncludeArguments *bool
	// IncludeResults reports what a call returned. Nil leaves it off: a result is
	// whatever a provider decided to return, which can be anything at all.
	IncludeResults *bool
	// Now reads the current time. Nil uses time.Now. Supplying one lets a test
	// place moments without waiting.
	Now func() time.Time
	// OnFunctionCallEvent is called for each moment, as a FunctionCallEvent.
	OnFunctionCallEvent func(e FunctionCallEvent)
}

// FunctionCalls reports each function call a conversation makes, from start to
// outcome.
//
// A function call is the one thing a bot does rather than says, and the part of
// a turn whose duration belongs to the application's own code. A call is
// reported at each moment it reaches rather than summarized once it is over,
// because the moments can be far apart and a call need not reach all of them:
// one waiting its turn to run is dropped if the conversation moves on, and one
// the conversation does not wait for can settle long after the turn that asked
// for it.
//
// Arguments and results are where a call holds whatever the conversation was
// about, so each is a choice. Arguments travel by default and results do not.
type FunctionCalls struct {
	cfg FunctionCallConfig

	mu sync.Mutex
	dd deduper
	// startedAt and inProgressAt are when each call started and when it went in
	// progress, so the moment that follows either can carry it.
	startedAt    map[string]time.Time
	inProgressAt map[string]time.Time
}

// NewFunctionCalls builds a FunctionCalls observer.
func NewFunctionCalls(cfg FunctionCallConfig) *FunctionCalls {
	return &FunctionCalls{
		cfg:          cfg,
		dd:           newDeduper(cfg.MaxFrames),
		startedAt:    map[string]time.Time{},
		inProgressAt: map[string]time.Time{},
	}
}

// now reads the clock the observer was configured with.
func (o *FunctionCalls) now() time.Time {
	if o.cfg.Now != nil {
		return o.cfg.Now()
	}
	return time.Now()
}

// includeArguments reports whether arguments travel; they do unless turned off.
func (o *FunctionCalls) includeArguments() bool {
	return o.cfg.IncludeArguments == nil || *o.cfg.IncludeArguments
}

// includeResults reports whether results travel; they do not unless turned on.
func (o *FunctionCalls) includeResults() bool {
	return o.cfg.IncludeResults != nil && *o.cfg.IncludeResults
}

// partOfACall reports whether a frame is part of a function call's life. It is
// asked before the frame is deduplicated, so a frame that is none of the
// observer's business does not crowd one that is out of the window of recent
// ids.
func partOfACall(f frames.Frame) bool {
	switch f.(type) {
	case *frames.FunctionCallsStartedFrame, *frames.FunctionCallInProgressFrame,
		*frames.FunctionCallResultFrame, *frames.FunctionCallCancelFrame:
		return true
	default:
		return false
	}
}

// OnPushFrame implements processor.Observer. It reports the moments a frame
// represents, the first time that frame is seen.
func (o *FunctionCalls) OnPushFrame(data processor.FramePushed) {
	// These frames are broadcast, arriving as two frames with two ids, so an id
	// alone would not tell them apart. Read the downstream one.
	if skipBroadcastSibling(data.Frame, data.Direction) {
		return
	}
	if !partOfACall(data.Frame) {
		return
	}

	o.mu.Lock()
	if o.dd.seenBefore(data.Frame.ID()) {
		o.mu.Unlock()
		return
	}
	events := o.eventsFor(data.Frame)
	o.mu.Unlock()

	if o.cfg.OnFunctionCallEvent == nil {
		return
	}
	for _, e := range events {
		o.cfg.OnFunctionCallEvent(e)
	}
}

// eventsFor builds the moments a frame represents. One frame reports one moment,
// except the frame starting the calls a model response asked for, which reports
// one for each of them. The caller holds o.mu.
func (o *FunctionCalls) eventsFor(f frames.Frame) []FunctionCallEvent {
	switch f := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		at := o.now()
		events := make([]FunctionCallEvent, 0, len(f.Calls))
		for _, call := range f.Calls {
			o.startedAt[call.ID] = at
			e := FunctionCallEvent{
				Kind:         FunctionCallStarted,
				FunctionName: call.Name,
				ToolCallID:   call.ID,
				Timestamp:    at,
			}
			if o.includeArguments() {
				e.Arguments = call.Args
			}
			events = append(events, e)
		}
		return events

	case *frames.FunctionCallInProgressFrame:
		at := o.now()
		o.inProgressAt[f.ToolCallID] = at
		// A call that survives an interruption is one the conversation was never
		// waiting on.
		blocking := f.CancelOnInterruption
		e := FunctionCallEvent{
			Kind:         FunctionCallInProgress,
			FunctionName: f.ToolName,
			ToolCallID:   f.ToolCallID,
			Timestamp:    at,
			GroupID:      f.GroupID,
			Blocking:     &blocking,
			StartedAt:    o.startedAt[f.ToolCallID],
		}
		delete(o.startedAt, f.ToolCallID)
		if o.includeArguments() {
			e.Arguments = f.Args
		}
		return []FunctionCallEvent{e}

	case *frames.FunctionCallResultFrame:
		// A call that does not block may report progress before it is done, and
		// only its final result settles it.
		if !f.Properties.Final() {
			return nil
		}
		kind := FunctionCallCompleted
		if f.Error != "" {
			kind = FunctionCallFailed
		}
		e := o.settles(kind, f.ToolName, f.ToolCallID)
		e.Error = f.Error
		if o.includeResults() && f.Error == "" {
			e.Result = f.Result
		}
		return []FunctionCallEvent{e}

	case *frames.FunctionCallCancelFrame:
		// Inference is asked for by the deadline that settled the call, and by
		// nothing else that cancels one.
		kind := FunctionCallCanceled
		if f.RunLLM {
			kind = FunctionCallTimedOut
		}
		return []FunctionCallEvent{o.settles(kind, f.ToolName, f.ToolCallID)}

	default:
		return nil
	}
}

// settles records the end of a call, naming when it began running if that is
// known. A call that began before the observer was watching settles without that
// moment rather than borrowing one from another call. The caller holds o.mu.
func (o *FunctionCalls) settles(kind FunctionCallEventKind, name, toolCallID string) FunctionCallEvent {
	delete(o.startedAt, toolCallID)
	inProgressAt := o.inProgressAt[toolCallID]
	delete(o.inProgressAt, toolCallID)
	return FunctionCallEvent{
		Kind:         kind,
		FunctionName: name,
		ToolCallID:   toolCallID,
		Timestamp:    o.now(),
		InProgressAt: inProgressAt,
	}
}

// Compile-time interface check.
var _ processor.Observer = (*FunctionCalls)(nil)
