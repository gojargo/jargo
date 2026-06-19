// Package pipeline connects frame processors into a chain and drives them.
//
// A Pipeline links a sequence of processors between a source and a sink. A Task
// runs a pipeline for one session: it sends the StartFrame, pushes frames, and
// shuts the pipeline down on an EndFrame or CancelFrame. A Runner runs a Task
// and ends it on an interrupt signal.
package pipeline

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Pipeline is a linear chain of processors. It wraps the chain with a source
// and a sink so frames can be fed in and observed at the edges, and is itself a
// processor, so pipelines can nest.
type Pipeline struct {
	*processor.Base
	source     processor.Processor
	sink       processor.Processor
	processors []processor.Processor
}

// New builds a Pipeline from procs, connected in order. Frames pushed out of the
// chain's source and sink are forwarded to the pipeline's own neighbors.
func New(procs ...processor.Processor) *Pipeline {
	return build(nil, nil, procs)
}

// build assembles a Pipeline. When source or sink is nil a default endpoint is
// created that forwards frames to the pipeline's neighbors.
func build(source, sink processor.Processor, procs []processor.Processor) *Pipeline {
	p := &Pipeline{}
	p.Base = processor.New("Pipeline", p, processor.WithDirectMode())
	if source == nil {
		source = processor.NewSource(p.Name()+"::Source", p.PushFrame)
	}
	if sink == nil {
		sink = processor.NewSink(p.Name()+"::Sink", p.PushFrame)
	}
	p.source = source
	p.sink = sink

	all := make([]processor.Processor, 0, len(procs)+2)
	all = append(all, source)
	all = append(all, procs...)
	all = append(all, sink)
	p.processors = all

	for i := 0; i < len(all)-1; i++ {
		all[i].Link(all[i+1])
	}
	return p
}

// ProcessFrame routes a frame into the chain: downstream frames enter at the
// source, upstream frames enter at the sink.
func (p *Pipeline) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch dir {
	case processor.Downstream:
		return p.source.QueueFrame(ctx, f, dir)
	case processor.Upstream:
		return p.sink.QueueFrame(ctx, f, dir)
	}
	return nil
}

// Setup sets up the pipeline and every processor in the chain.
func (p *Pipeline) Setup(ctx context.Context, s processor.Setup) error {
	if err := p.Base.Setup(ctx, s); err != nil {
		return err
	}
	for _, proc := range p.processors {
		if err := proc.Setup(ctx, s); err != nil {
			return err
		}
	}
	return nil
}

// Cleanup cleans up the pipeline and every processor in the chain.
func (p *Pipeline) Cleanup(ctx context.Context) error {
	_ = p.Base.Cleanup(ctx)
	for _, proc := range p.processors {
		_ = proc.Cleanup(ctx)
	}
	return nil
}
