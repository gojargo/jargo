package pipeline_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// errBoom is the non-fatal error a tagSvc raises to trigger failover.
var errBoom = errors.New("boom")

// tagSvc replaces each downstream TextFrame with one prefixed by its tag, so
// only the active branch of a switcher produces output. When a frame's text
// equals failOn it raises a non-fatal error instead, to drive failover.
type tagSvc struct {
	*processor.Base
	tag    string
	failOn string
}

func newTagSvc(tag, failOn string) *tagSvc {
	s := &tagSvc{tag: tag, failOn: failOn}
	s.Base = processor.New("TagSvc:"+tag, s)
	return s
}

func (s *tagSvc) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if tf, ok := f.(*frames.TextFrame); ok && dir == processor.Downstream {
		if s.failOn != "" && tf.Text == s.failOn {
			s.PushError(ctx, "tag svc failed", errBoom, false)
			return nil
		}
		return s.PushFrame(ctx, frames.NewTextFrame(s.tag+tf.Text), dir)
	}
	return s.PushFrame(ctx, f, dir)
}

// runCollector runs a task over proc, returning the task (to queue frames into),
// a channel of every downstream TextFrame text, and a stop function.
func runCollector(t *testing.T, proc processor.Processor) (*pipeline.Task, <-chan string, func()) {
	t.Helper()
	out := make(chan string, 64)
	task := pipeline.NewTask(pipeline.New(proc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			if tf, ok := f.(*frames.TextFrame); ok {
				select {
				case out <- tf.Text:
				default:
				}
			}
		},
	})
	done := make(chan struct{})
	go func() { _ = task.Run(context.Background()); close(done) }()
	return task, out, func() {
		task.Cancel()
		<-done
	}
}

// wantText waits for the next collected text and checks it.
func wantText(t *testing.T, out <-chan string, want string) {
	t.Helper()
	select {
	case got := <-out:
		if got != want {
			t.Errorf("downstream text = %q, want %q", got, want)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %q", want)
	}
}

func TestFunctionFilterGatesDirection(t *testing.T) {
	// A downstream filter drops "block" but passes "ok".
	allow := func(f frames.Frame) bool {
		tf, ok := f.(*frames.TextFrame)
		return !ok || tf.Text != "block"
	}
	pipe := pipeline.New(processor.NewFunctionFilter("F", processor.Downstream, allow), newEcho())
	task, out, stop := runCollector(t, pipe)
	defer stop()

	task.QueueFrame(frames.NewTextFrame("block"))
	task.QueueFrame(frames.NewTextFrame("ok"))
	wantText(t, out, "ok") // "block" was dropped, so "ok" arrives first
}

func TestServiceSwitcherRouting(t *testing.T) {
	a := newTagSvc("A:", "")
	b := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.SwitchManual)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}
	if sw.ActiveService() != a {
		t.Error("initial active service is not the first service")
	}

	task, out, stop := runCollector(t, sw)
	defer stop()

	task.QueueFrame(frames.NewTextFrame("one"))
	wantText(t, out, "A:one") // first service is active

	sw.SwitchTo(b)
	task.QueueFrame(frames.NewTextFrame("two"))
	wantText(t, out, "B:two") // routed to the new active service
}

func TestServiceSwitcherInBandSwitch(t *testing.T) {
	a := newTagSvc("A:", "")
	b := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.SwitchManual)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	task, out, stop := runCollector(t, sw)
	defer stop()

	// A SwitchServiceFrame queued in-band switches the active service before the
	// following text is routed.
	task.QueueFrame(pipeline.NewSwitchServiceFrame(b))
	task.QueueFrame(frames.NewTextFrame("x"))
	wantText(t, out, "B:x")
}

func TestServiceSwitcherFailover(t *testing.T) {
	a := newTagSvc("A:", "FAIL") // errors on "FAIL"
	b := newTagSvc("B:", "")
	sw, err := pipeline.NewServiceSwitcher([]processor.Processor{a, b}, pipeline.SwitchFailover)
	if err != nil {
		t.Fatalf("NewServiceSwitcher: %v", err)
	}

	switched := make(chan processor.Processor, 1)
	sw.OnSwitch(func(p processor.Processor) {
		select {
		case switched <- p:
		default:
		}
	})

	task, out, stop := runCollector(t, sw)
	defer stop()

	// The active service errors on this frame, which fails over to the next.
	task.QueueFrame(frames.NewTextFrame("FAIL"))
	select {
	case p := <-switched:
		if p != b {
			t.Errorf("failed over to %v, want service b", p.Name())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for failover")
	}

	task.QueueFrame(frames.NewTextFrame("hi"))
	wantText(t, out, "B:hi") // now served by the backup
}

func TestServiceSwitcherNoServices(t *testing.T) {
	if _, err := pipeline.NewServiceSwitcher(nil, pipeline.SwitchManual); err == nil {
		t.Fatal("NewServiceSwitcher(nil): want error, got nil")
	}
}
