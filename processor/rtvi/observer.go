package rtvi

import (
	"context"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Observer reports pipeline events to an RTVI client. It watches every frame
// handed between processors and turns the ones a client cares about into RTVI
// messages, which it hands to a Processor to send.
//
// Only outgoing events come from here. Messages arriving from the client are
// handled by the Processor, which also carries out the handshake.
//
// Watching rather than sitting in the pipeline is what lets a client be told
// about frames that never travel the whole chain, and lets an event be reported
// from where the frame actually is: text pushed by the LLM is seen at the LLM,
// whether or not anything downstream forwards it.
type Observer struct {
	sink *Processor

	mu    sync.Mutex
	seen  map[uint64]struct{}
	order []uint64

	// paramsMu guards params, which a ConfigureObserverFrame rewrites while the
	// pipeline is running.
	paramsMu sync.Mutex
	params   ObserverParams
}

// FunctionCallReportLevel is how much of a tool call is exposed in the RTVI
// function-call events. A call's name and its arguments can carry information a
// client has no business seeing, so what is reported is a per-function setting
// rather than a fixed payload.
type FunctionCallReportLevel string

// The report levels, in increasing order of disclosure.
const (
	// ReportDisabled emits no function-call event at all for the call.
	ReportDisabled FunctionCallReportLevel = "disabled"
	// ReportNone emits the event with the tool call id only. This is the default:
	// a client learns that a call is running, and nothing more.
	ReportNone FunctionCallReportLevel = "none"
	// ReportName adds the function's name, still without arguments or result.
	ReportName FunctionCallReportLevel = "name"
	// ReportFull adds the function's name, its arguments and its result.
	ReportFull FunctionCallReportLevel = "full"
)

// ObserverParams configures what an Observer reports.
type ObserverParams struct {
	// FunctionCallReportLevel maps a function name to the level of detail its
	// events carry. The "*" key sets the level for functions not listed; when it
	// is absent too, ReportNone applies.
	FunctionCallReportLevel map[string]FunctionCallReportLevel
}

// DefaultObserverParams is the configuration NewObserver uses: function-call
// events carry the tool call id alone.
func DefaultObserverParams() ObserverParams {
	return ObserverParams{
		FunctionCallReportLevel: map[string]FunctionCallReportLevel{"*": ReportNone},
	}
}

// seenCap bounds the ids remembered for deduplication. A frame is reported once
// per handover, so each is recognized on sight, and only a bounded window has to
// be kept: by the time an id is evicted the frame has long since left.
const seenCap = 4096

// NewObserver builds an observer that sends through sink, with the default
// parameters. Use NewObserverWithParams to report more of a tool call than its
// id.
func NewObserver(sink *Processor) *Observer {
	return NewObserverWithParams(sink, DefaultObserverParams())
}

// NewObserverWithParams builds an observer that sends through sink and reports
// what params allows.
func NewObserverWithParams(sink *Processor, params ObserverParams) *Observer {
	return &Observer{
		sink:   sink,
		seen:   make(map[uint64]struct{}, seenCap),
		params: params,
	}
}

// OnPushFrame implements processor.Observer.
func (o *Observer) OnPushFrame(data processor.FramePushed) {
	f, dir := data.Frame, data.Direction
	// A broadcast frame is pushed both ways; report only the downstream copy so
	// the client is not told twice.
	if _, broadcast := f.Base().BroadcastSiblingID(); broadcast && dir != processor.Downstream {
		return
	}
	if o.alreadySeen(f.ID()) {
		return
	}
	if cfg, ok := f.(*ConfigureObserverFrame); ok {
		o.applyConfig(cfg)
		return
	}
	msg, ok := o.messageFor(f)
	if !ok {
		return
	}
	o.sink.Send(msg)
}

// applyConfig applies a runtime reconfiguration, leaving unset fields alone.
func (o *Observer) applyConfig(f *ConfigureObserverFrame) {
	if f.FunctionCallReportLevel == nil {
		return
	}
	o.paramsMu.Lock()
	o.params.FunctionCallReportLevel = f.FunctionCallReportLevel
	o.paramsMu.Unlock()
	slog.Debug("RTVI observer reconfigured", "function_call_report_level", f.FunctionCallReportLevel)
}

// reportLevelFor is the level to report a call to name at: the function's own
// entry, else the "*" default, else ReportNone.
func (o *Observer) reportLevelFor(name string) FunctionCallReportLevel {
	o.paramsMu.Lock()
	defer o.paramsMu.Unlock()
	if level, ok := o.params.FunctionCallReportLevel[name]; ok {
		return level
	}
	if level, ok := o.params.FunctionCallReportLevel["*"]; ok {
		return level
	}
	return ReportNone
}

// alreadySeen reports whether the frame has been reported, recording it when it
// has not.
func (o *Observer) alreadySeen(id uint64) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	if _, ok := o.seen[id]; ok {
		return true
	}
	o.seen[id] = struct{}{}
	o.order = append(o.order, id)
	if len(o.order) > seenCap {
		delete(o.seen, o.order[0])
		o.order = o.order[1:]
	}
	return false
}

// Send hands an RTVI message to the client. It is safe to call from any
// goroutine: the message is queued to the next processor like any other frame.
func (p *Processor) Send(msg Message) {
	ctx := p.sendContext()
	_ = p.PushFrame(ctx, frames.NewOutputTransportMessageUrgentFrame(msg), processor.Downstream)
}

// sendContext is the context an out-of-band send runs under.
func (p *Processor) sendContext() context.Context {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.baseCtx != nil {
		return p.baseCtx
	}
	return context.Background()
}

var _ processor.Observer = (*Observer)(nil)
