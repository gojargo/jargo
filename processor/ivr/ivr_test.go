package ivr_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/ivr"
)

func TestNavigatesAndStripsTags(t *testing.T) {
	var mu sync.Mutex
	var status ivr.Status
	var dtmf []frames.KeypadEntry
	var spoken string
	done := make(chan struct{})

	proc := ivr.New(ivr.Config{OnStatusChanged: func(s ivr.Status) {
		mu.Lock()
		status = s
		mu.Unlock()
	}})
	task := pipeline.NewTask(pipeline.New(proc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.OutputDTMFFrame:
				dtmf = append(dtmf, fr.Buttons...)
			case *frames.LLMTextFrame:
				spoken += fr.Text
			case *frames.LLMFullResponseEndFrame:
				select {
				case <-done:
				default:
					close(done)
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// LLM streams: detect menu, press 1 then 2, then speak.
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	for _, tok := range []string{"<ivr>de", "tected</ivr>", "Pressing <dtmf>12", "</dtmf> now.", ""} {
		task.QueueFrame(frames.NewLLMTextFrame(tok))
	}
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("response never ended")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if status != ivr.StatusDetected {
		t.Errorf("status = %q, want detected", status)
	}
	if len(dtmf) != 2 || dtmf[0] != frames.KeypadOne || dtmf[1] != frames.KeypadTwo {
		t.Errorf("dtmf = %v, want [1 2]", dtmf)
	}
	if want := "Pressing  now."; spoken != want {
		t.Errorf("spoken = %q, want %q (tags stripped)", spoken, want)
	}
}
