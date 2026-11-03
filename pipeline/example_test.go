package pipeline_test

import (
	"context"
	"fmt"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// printer prints the text of every TextFrame and passes every frame on.
type printer struct {
	*processor.Base
}

func newPrinter() *printer {
	p := &printer{}
	p.Base = processor.New("Printer", p)
	return p
}

func (p *printer) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if t, ok := f.(*frames.TextFrame); ok {
		fmt.Println(t.Text)
	}
	return p.PushFrame(ctx, f, dir)
}

// A pipeline is a chain of processors, and a worker runs it. Frames queued on
// the worker enter at the head; an EndFrame ends the run once everything
// before it has gone through.
func Example() {
	worker := pipeline.NewWorker(pipeline.New(newPrinter()), pipeline.WorkerConfig{})
	worker.QueueFrames([]frames.Frame{
		frames.NewTextFrame("Hello"),
		frames.NewTextFrame("world"),
		frames.NewEndFrame(),
	})
	if err := worker.Run(context.Background()); err != nil {
		fmt.Println("error:", err)
	}
	// Output:
	// Hello
	// world
}
