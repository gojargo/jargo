package llm_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// Ported from upstream (test_function_call_params_receives_app_resources). A
// tool handler is given what the application shares across the session, the
// same value the worker was configured with, and the worker itself.
func TestFunctionCallParamsReceivesAppResources(t *testing.T) {
	type resources struct{ db string }
	shared := &resources{db: "orders"}

	gen := &scriptedGen{turns: [][]frames.ToolCall{{
		{ID: "call-1", Name: "lookup", Args: json.RawMessage(`{}`)},
	}}}
	svc := llm.New("FakeLLM", gen)
	got := make(chan llm.FunctionCallParams, 1)
	svc.RegisterFunction("lookup", func(ctx context.Context, p llm.FunctionCallParams) error {
		got <- p
		return p.Result(ctx, `{"ok": true}`, nil)
	})

	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{AppResources: shared})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	defer func() {
		task.Cancel(context.Background(), "")
		<-runDone
	}()
	convo := frames.NewLLMContext("sys")
	convo.AddUserMessage("Where is my order?")
	convo.SetTools([]frames.Tool{{Name: "lookup", Description: "Look the order up."}})
	task.QueueFrame(frames.NewLLMContextFrame(convo))

	select {
	case p := <-got:
		if p.AppResources != shared {
			t.Errorf("AppResources = %v, want the value the worker was configured with", p.AppResources)
		}
		if p.PipelineWorker != task {
			t.Error("PipelineWorker is not the worker running the call")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the tool was never called")
	}
}
