package llm

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// OutputGate decides whether the service's output goes out: each frame it
// pushes, and its function calls.
//
// It lets output through or drops it; it never changes it. It sees each frame
// the service processes, to keep track of what the service is doing. Predictive
// delegation (see llmwithbackend.PredictiveDelegation) uses one to drop what a
// frontend's inference outputs once its turn is delegated early.
type OutputGate interface {
	// ProcessFrame is called with each frame the service processes, before the
	// service handles it.
	ProcessFrame(ctx context.Context, service *Base, f frames.Frame) error
	// AllowPush reports whether the service may push a frame.
	AllowPush(f frames.Frame) bool
	// AllowFunctionCalls reports whether the service may run its inference's
	// function calls.
	AllowFunctionCalls() bool
}

// SetOutputGate installs the gate that can drop what this service pushes or the
// function calls it runs, or removes it with nil. Set it before the service
// starts.
func (b *Base) SetOutputGate(g OutputGate) { b.outputGate = g }

// processBaseAndGate has the service base handle a frame, then shows the frame
// to the output gate, if one is set, before the LLM service handles it.
func (b *Base) processBaseAndGate(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := b.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if b.outputGate != nil {
		return b.outputGate.ProcessFrame(ctx, b, f)
	}
	return nil
}
