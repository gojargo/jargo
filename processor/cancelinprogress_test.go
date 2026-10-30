package processor_test

import (
	"context"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// cancelInProgressFrame asks the processor to cancel target if it is the frame
// in progress.
type cancelInProgressFrame struct {
	frames.BaseSystemFrame
	target frames.Frame
}

func newCancelInProgressFrame(target frames.Frame) *cancelInProgressFrame {
	return &cancelInProgressFrame{
		BaseSystemFrame: frames.NewBaseSystemFrame("CancelInProgressFrame"), target: target,
	}
}

// cancelInProgressProcessor takes 400ms over every non-system frame, and
// records what each cancel request did.
type cancelInProgressProcessor struct {
	*processor.Base
	mu       sync.Mutex
	canceled []bool
}

func newCancelInProgressProcessor() *cancelInProgressProcessor {
	p := &cancelInProgressProcessor{}
	p.Base = processor.New("CancelInProgress", p)
	return p
}

func (p *cancelInProgressProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *cancelInProgressFrame:
		canceled := p.CancelFrameInProgress(fr.target)
		p.mu.Lock()
		p.canceled = append(p.canceled, canceled)
		p.mu.Unlock()
	case frames.SystemFrame:
	default:
		select {
		case <-time.After(400 * time.Millisecond):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return p.PushFrame(ctx, f, dir)
}

func (p *cancelInProgressProcessor) results() []bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.canceled)
}

// runCancelInProgress sends each frame to the processor, sleeping where a
// duration stands in the list, and returns what came out downstream, the
// StartFrame left out, once nothing more has for a while.
func runCancelInProgress(t *testing.T, p *cancelInProgressProcessor, steps ...any) []frames.Frame {
	t.Helper()
	ctx := t.Context()
	out := newSink("Down")
	linkChain(t, ctx, processor.Setup{Clock: clock.NewSystem()}, p, out)
	queue(t, ctx, p, frames.NewStartFrame())
	for _, step := range steps {
		switch s := step.(type) {
		case time.Duration:
			time.Sleep(s)
		case frames.Frame:
			queue(t, ctx, p, s)
		}
	}
	// Long enough for any frame still in progress to finish.
	time.Sleep(600 * time.Millisecond)
	out.mu.Lock()
	defer out.mu.Unlock()
	var down []frames.Frame
	for _, f := range out.got {
		if _, ok := f.(*frames.StartFrame); !ok {
			down = append(down, f)
		}
	}
	return down
}

// texts is the text of each text frame, and the type name of every other.
func texts(fs []frames.Frame) []string {
	var got []string
	for _, f := range fs {
		switch fr := f.(type) {
		case *frames.TextFrame:
			got = append(got, fr.Text)
		case *cancelInProgressFrame:
			got = append(got, "cancel")
		case *frames.InterruptionFrame:
			got = append(got, "interruption")
		default:
			got = append(got, f.Name())
		}
	}
	return got
}

func TestCancelFrameInProgressKeepsTheQueue(t *testing.T) {
	p := newCancelInProgressProcessor()
	inProgress := frames.NewTextFrame("in progress")
	down := runCancelInProgress(t, p,
		inProgress,
		100*time.Millisecond,
		frames.NewTextFrame("queued"),
		newCancelInProgressFrame(inProgress),
	)
	if got := p.results(); !slices.Equal(got, []bool{true}) {
		t.Errorf("canceled = %v, want [true]", got)
	}
	if got := texts(down); !slices.Equal(got, []string{"cancel", "queued"}) {
		t.Errorf("down = %v, want [cancel queued]", got)
	}
}

func TestCancelFrameInProgressLeavesOtherFramesAlone(t *testing.T) {
	p := newCancelInProgressProcessor()
	queued := frames.NewTextFrame("queued")
	down := runCancelInProgress(t, p,
		frames.NewTextFrame("in progress"),
		100*time.Millisecond,
		queued,
		newCancelInProgressFrame(queued),
		// The two text frames take 800ms between them.
		400*time.Millisecond,
	)
	if got := p.results(); !slices.Equal(got, []bool{false}) {
		t.Errorf("canceled = %v, want [false]", got)
	}
	if got := texts(down); !slices.Equal(got, []string{"cancel", "in progress", "queued"}) {
		t.Errorf("down = %v, want [cancel, in progress, queued]", got)
	}
}

func TestAnInterruptionAfterCancelFrameInProgressStillInterrupts(t *testing.T) {
	p := newCancelInProgressProcessor()
	inProgress := frames.NewTextFrame("in progress")
	down := runCancelInProgress(t, p,
		inProgress,
		100*time.Millisecond,
		newCancelInProgressFrame(inProgress),
		frames.NewTextFrame("next"),
		100*time.Millisecond,
		frames.NewInterruptionFrame(),
	)
	if got := p.results(); !slices.Equal(got, []bool{true}) {
		t.Errorf("canceled = %v, want [true]", got)
	}
	if got := texts(down); !slices.Equal(got, []string{"cancel", "interruption"}) {
		t.Errorf("down = %v, want [cancel interruption]", got)
	}
}
