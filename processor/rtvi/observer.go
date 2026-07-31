package rtvi

import (
	"context"
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
}

// seenCap bounds the ids remembered for deduplication. A frame is reported once
// per handover, so each is recognized on sight, and only a bounded window has to
// be kept: by the time an id is evicted the frame has long since left.
const seenCap = 4096

// NewObserver builds an observer that sends through sink.
func NewObserver(sink *Processor) *Observer {
	return &Observer{sink: sink, seen: make(map[uint64]struct{}, seenCap)}
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
	msg, ok := messageFor(f)
	if !ok {
		return
	}
	o.sink.Send(msg)
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
