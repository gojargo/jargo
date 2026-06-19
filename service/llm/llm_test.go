package llm_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// fakeGen emits a fixed list of text deltas.
type fakeGen struct {
	deltas []string
	gotSys string
}

func (g *fakeGen) Generate(_ context.Context, convo *frames.LLMContext, emit llm.Emit) error {
	g.gotSys = convo.System()
	for _, d := range g.deltas {
		if err := emit(d); err != nil {
			return err
		}
	}
	return nil
}

func TestBaseBracketsResponse(t *testing.T) {
	gen := &fakeGen{deltas: []string{"Hello ", "world"}}
	svc := llm.New("FakeLLM", gen)

	var mu sync.Mutex
	var got []string
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.LLMFullResponseStartFrame:
				got = append(got, "start")
			case *frames.LLMTextFrame:
				got = append(got, "text:"+fr.Text)
			case *frames.LLMFullResponseEndFrame:
				got = append(got, "end")
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	convo := frames.NewLLMContext("be brief")
	convo.AddUserMessage("hi")
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("LLM base did not complete the response")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	want := []string{"start", "text:Hello ", "text:world", "end"}
	if len(got) != len(want) {
		t.Fatalf("frames = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("frame %d = %q, want %q (all: %v)", i, got[i], want[i], got)
		}
	}
	if gen.gotSys != "be brief" {
		t.Fatalf("generator system prompt = %q, want %q", gen.gotSys, "be brief")
	}
}
