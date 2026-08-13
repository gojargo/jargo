package bus

import (
	"context"
	"reflect"
	"slices"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// The two processors here put a pipeline on the bus, in the two shapes a worker
// needs.
//
// A BridgeProcessor sits in the middle of a pipeline and takes frames off it:
// what reaches the bridge crosses to another worker instead of continuing. An
// EdgeProcessor sits at a pipeline's edge and tees: frames carry on locally and
// are copied to the bus as well.
//
// Neither lets a lifecycle frame cross. Starting, ending, stopping and
// canceling are each pipeline's own business, and a worker that received
// another's would shut down with it.

// isLifecycleFrame reports whether a frame belongs to its own pipeline's
// lifecycle and so must never cross the bus.
func isLifecycleFrame(f frames.Frame) bool {
	switch f.(type) {
	case *frames.StartFrame, *frames.EndFrame, *frames.CancelFrame, *frames.StopFrame:
		return true
	}
	return false
}

// isPassthroughFrame reports whether a frame goes straight on locally rather
// than crossing the bus. An urgent transport message has to reach the transport
// even when no child worker is active to carry it.
func isPassthroughFrame(f frames.Frame) bool {
	_, ok := f.(*frames.OutputTransportMessageUrgentFrame)
	return ok
}

// BridgeConfig configures a BridgeProcessor.
type BridgeConfig struct {
	// Bus is the bus to exchange frames over. Required.
	Bus *Bus
	// WorkerName is the owning worker's name, carried as the source of
	// everything this bridge sends.
	WorkerName string
	// TargetWorker, when set, is the only worker this bridge accepts frames
	// from.
	TargetWorker string
	// Bridge, when set, names this bridge for routing: what it sends is tagged
	// with the name, and it accepts only frames tagged with the same one.
	Bridge string
	// ExcludeFrames are frame types that never cross the bus, on top of the
	// lifecycle frames, which never do. A frame matches when it is of the same
	// concrete type as one given here.
	ExcludeFrames []frames.Frame
}

// BridgeProcessor exchanges frames with other workers over the bus, from the
// middle of a pipeline.
//
// A frame reaching it is taken off the local pipeline and sent to the bus; a
// frame arriving from the bus is pushed on locally. Lifecycle frames, urgent
// transport messages and any excluded types pass straight through instead.
type BridgeProcessor struct {
	*processor.Base
	cfg BridgeConfig
}

// NewBridgeProcessor builds a mid-pipeline bridge onto the bus.
func NewBridgeProcessor(cfg BridgeConfig) *BridgeProcessor {
	p := &BridgeProcessor{cfg: cfg}
	p.Base = processor.New("BusBridge", p)
	return p
}

// Setup subscribes the bridge to the bus.
func (p *BridgeProcessor) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	p.cfg.Bus.Subscribe(p)
	return nil
}

// Cleanup unsubscribes the bridge from the bus.
func (p *BridgeProcessor) Cleanup(ctx context.Context) error {
	p.cfg.Bus.Unsubscribe(p)
	return p.Base.Cleanup(ctx)
}

// ProcessFrame sends a frame across the bus, or forwards it locally when it is
// one that must not cross.
func (p *BridgeProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	if isLifecycleFrame(f) || isPassthroughFrame(f) || excluded(f, p.cfg.ExcludeFrames) {
		return p.PushFrame(ctx, f, dir)
	}

	m := &FrameMessage{Frame: f, Direction: dir, Bridge: p.cfg.Bridge}
	m.From = p.cfg.WorkerName
	p.cfg.Bus.Send(ctx, m)
	return nil
}

// OnBusMessage pushes a frame arriving from another worker into this pipeline.
func (p *BridgeProcessor) OnBusMessage(ctx context.Context, m Message) {
	fm, ok := m.(*FrameMessage)
	if !ok {
		return
	}
	// A bridge never takes back what it sent.
	if fm.Source() == p.cfg.WorkerName {
		return
	}
	if p.cfg.Bridge != "" && fm.Bridge != p.cfg.Bridge {
		return
	}
	if p.cfg.TargetWorker != "" && fm.Source() != p.cfg.TargetWorker {
		return
	}
	if fm.Target() != "" && fm.Target() != p.cfg.WorkerName {
		return
	}
	_ = p.PushFrame(ctx, fm.Frame, fm.Direction)
}

// BridgedWorker is the part of a pipeline worker an EdgeProcessor reads. It is
// named here rather than imported because the workers are built on this
// package; a pipeline worker satisfies it.
type BridgedWorker interface {
	// Bus is the bus the worker is attached to. It is read when the edge is set
	// up rather than when it is built, so the worker need only be attached by
	// then.
	Bus() *Bus
	// WorkerName is the worker's name, carried as the source of what the edge
	// sends.
	WorkerName() string
	// Active reports whether the worker is active. An inactive worker takes no
	// frames off the bus.
	Active() bool
	// QueueFrame hands a frame to the worker's own queue.
	QueueFrame(ctx context.Context, f frames.Frame, dir processor.Direction)
}

// EdgeConfig configures an EdgeProcessor.
type EdgeConfig struct {
	// Worker is the worker that owns the pipeline. Required.
	Worker BridgedWorker
	// Direction is the direction this edge copies to the bus. Frames arriving
	// from the bus traveling the other way are injected here.
	Direction processor.Direction
	// Bridges are the bridge names this edge accepts frames from. Empty accepts
	// every bridge.
	Bridges []string
	// ExcludeFrames are frame types that never cross the bus, on top of the
	// lifecycle frames.
	ExcludeFrames []frames.Frame
}

// EdgeProcessor tees a pipeline's edge onto the bus: frames carry on locally
// and matching ones are copied across.
//
// It is placed by a pipeline worker at the source and sink of a bridged
// pipeline. Upstream keeps it private to the worker; Go's package boundary
// makes that impossible, so it is exported and documented as the worker's own.
type EdgeProcessor struct {
	*processor.Base
	cfg EdgeConfig
}

// NewEdgeProcessor builds a pipeline-edge tee onto the bus.
func NewEdgeProcessor(cfg EdgeConfig) *EdgeProcessor {
	p := &EdgeProcessor{cfg: cfg}
	p.Base = processor.New("BusEdge", p)
	return p
}

// Setup subscribes the edge to the worker's bus.
func (p *EdgeProcessor) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	p.cfg.Worker.Bus().Subscribe(p)
	return nil
}

// Cleanup unsubscribes the edge from the worker's bus.
func (p *EdgeProcessor) Cleanup(ctx context.Context) error {
	p.cfg.Worker.Bus().Unsubscribe(p)
	return p.Base.Cleanup(ctx)
}

// ProcessFrame forwards every frame locally, and copies the ones traveling
// this edge's direction to the bus.
func (p *EdgeProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if err := p.PushFrame(ctx, f, dir); err != nil {
		return err
	}

	if dir != p.cfg.Direction || isLifecycleFrame(f) || excluded(f, p.cfg.ExcludeFrames) {
		return nil
	}

	m := &FrameMessage{Frame: f, Direction: dir}
	m.From = p.cfg.Worker.WorkerName()
	p.cfg.Worker.Bus().Send(ctx, m)
	return nil
}

// OnBusMessage injects a frame arriving from another worker into the pipeline.
func (p *EdgeProcessor) OnBusMessage(ctx context.Context, m Message) {
	fm, ok := m.(*FrameMessage)
	if !ok {
		return
	}
	if fm.Source() == p.cfg.Worker.WorkerName() {
		return
	}
	// This edge captures one direction and injects the other.
	if fm.Direction == p.cfg.Direction {
		return
	}
	if !p.cfg.Worker.Active() {
		return
	}
	if fm.Target() != "" && fm.Target() != p.cfg.Worker.WorkerName() {
		return
	}
	if len(p.cfg.Bridges) > 0 && !slices.Contains(p.cfg.Bridges, fm.Bridge) {
		return
	}
	// Through the worker's own queue rather than pushed from here, so what
	// arrives from the bus is ordered with the frames the worker queues itself.
	p.cfg.Worker.QueueFrame(ctx, fm.Frame, fm.Direction)
}

// excluded reports whether f is of the same concrete type as one of the
// excluded frames.
func excluded(f frames.Frame, exclude []frames.Frame) bool {
	for _, e := range exclude {
		if sameFrameType(f, e) {
			return true
		}
	}
	return false
}

// sameFrameType reports whether two frames are of the same concrete type.
func sameFrameType(a, b frames.Frame) bool {
	return reflect.TypeOf(a) == reflect.TypeOf(b)
}
