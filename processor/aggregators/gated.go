package aggregators

import (
	"context"
	"log/slog"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// GateFunc reports whether a frame opens or closes a gate.
type GateFunc func(frames.Frame) bool

// Gated holds frames back until something opens the gate, then releases what it
// held.
//
// The frame that opens the gate goes first, then everything accumulated behind
// it, in the order it arrived. That ordering is the point: what was held is
// meant to arrive after whatever announced it, not before. The frame that closes
// the gate is held with the rest, so it is the first thing released next time.
//
// System frames are never held, whatever the gate says, because a pipeline that
// cannot hear about an interruption while a gate is closed would have no way out
// of it. Frames traveling the other way are not this gate's business and pass
// untouched.
//
// Ending the run is not a system frame, so a closed gate holds that too and the
// processors behind it never hear the run end. A pipeline carrying a gate has to
// open it before shutting down; canceling works either way, since that is a
// system frame.
type Gated struct {
	*processor.Base
	openFn  GateFunc
	closeFn GateFunc
	open    bool
	dir     processor.Direction
	held    []heldFrame
}

// heldFrame is a frame waiting for the gate to open, and the way it was going.
type heldFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

// GatedConfig configures a Gated aggregator.
type GatedConfig struct {
	// Open reports whether a frame opens the gate. It is consulted only while
	// the gate is closed.
	Open GateFunc
	// Close reports whether a frame closes the gate. It is consulted only while
	// the gate is open.
	Close GateFunc
	// StartOpen has the gate open before any frame arrives.
	StartOpen bool
	// Direction is the way the frames this gate holds are traveling. Frames
	// going the other way pass untouched. Downstream when unset.
	Direction processor.Direction
}

// NewGated builds a gate over cfg.
func NewGated(name string, cfg GatedConfig) *Gated {
	g := &Gated{
		openFn:  cfg.Open,
		closeFn: cfg.Close,
		open:    cfg.StartOpen,
		dir:     cfg.Direction,
	}
	g.Base = processor.New(name, g, processor.WithDirectMode())
	return g
}

// ProcessFrame implements processor.Processor.
func (g *Gated) ProcessFrame(
	ctx context.Context, frame frames.Frame, dir processor.Direction,
) error {
	if err := g.Base.ProcessFrame(ctx, frame, dir); err != nil {
		return err
	}

	// A gate that swallowed these would have no way of ever being told to open
	// again, or to shut down.
	if _, ok := frame.(frames.SystemFrame); ok {
		return g.PushFrame(ctx, frame, dir)
	}
	if dir != g.dir {
		return g.PushFrame(ctx, frame, dir)
	}

	was := g.open
	if g.open {
		g.open = !g.closeFn(frame)
	} else {
		g.open = g.openFn(frame)
	}
	if was != g.open {
		state := "closed"
		if g.open {
			state = "open"
		}
		slog.Debug("gate changed", "processor", g.Name(), "state", state, "frame", frame.Name())
	}

	if !g.open {
		g.held = append(g.held, heldFrame{frame: frame, dir: dir})
		return nil
	}

	if err := g.PushFrame(ctx, frame, dir); err != nil {
		return err
	}
	held := g.held
	g.held = nil
	for _, h := range held {
		if err := g.PushFrame(ctx, h.frame, h.dir); err != nil {
			return err
		}
	}
	return nil
}
