package llm_test

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// markedHandler builds a distinct handler that answers with marker. Each call
// returns a new closure of the same literal, which is what a toolset building
// a fresh handler per step does.
func markedHandler(marker string) llm.FunctionCallHandler {
	return func(ctx context.Context, p llm.FunctionCallParams) error {
		return p.Result(ctx, marker, nil)
	}
}

// weatherToolWith advertises get_weather carrying h.
func weatherToolWith(h llm.FunctionCallHandler) frames.Tool {
	return frames.Tool{
		Name: "get_weather", Description: "weather", Parameters: json.RawMessage(`{"type":"object"}`),
		Handler: h,
	}
}

// callWeather runs one inference over tools and returns what the handler that
// answered the call said.
func callWeather(t *testing.T, svc *llm.Base, task *pipeline.Worker, results chan string, tools []frames.Tool) string {
	t.Helper()
	convo := frames.NewLLMContext("be brief")
	convo.SetTools(tools)
	convo.AddUserMessage("weather?")
	task.QueueFrame(frames.NewLLMContextFrame(convo))
	select {
	case got := <-results:
		return got
	case <-time.After(3 * time.Second):
		t.Fatal("no handler ran")
		return ""
	}
}

// A toolset that re-declares a tool under the same name with a new handler
// has the new one answer, rather than the one registered from the toolset
// before it.
func TestAReDeclaredToolRunsItsNewHandler(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	results := make(chan string, 4)
	probe := newProbe(func(f frames.Frame) {
		if fr, ok := f.(*frames.FunctionCallResultFrame); ok {
			results <- fr.Result
		}
	})
	task := pipeline.NewWorker(pipeline.New(svc, probe), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	defer func() {
		task.StopWhenDone()
		<-runDone
	}()

	if got := callWeather(t, svc, task, results, []frames.Tool{weatherToolWith(markedHandler("A"))}); got != "A" {
		t.Fatalf("first result = %q, want %q", got, "A")
	}
	if got := callWeather(t, svc, task, results, []frames.Tool{weatherToolWith(markedHandler("B"))}); got != "B" {
		t.Fatalf("second result = %q, want the re-declared handler %q", got, "B")
	}

	// Still the toolset's, so withdrawing the tool takes the handler with it.
	svc.SyncToolHandlers(context.Background(), nil)
	if svc.HasFunction("get_weather") {
		t.Error("a rebound handler should still be dropped when its tool is withdrawn")
	}
}

// captureDebug routes the default logger to a buffer at debug level until the
// returned function is called.
func captureDebug(t *testing.T) (*bytes.Buffer, func()) {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug})))
	return &buf, func() { slog.SetDefault(prev) }
}

// The same handler advertised again, as a context sent on every turn does, is
// not registered again.
func TestTheSameHandlerAdvertisedAgainIsLeftAlone(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	h := markedHandler("A")
	ctx := context.Background()
	svc.SyncToolHandlers(ctx, []frames.Tool{weatherToolWith(h)})

	buf, restore := captureDebug(t)
	svc.SyncToolHandlers(ctx, []frames.Tool{weatherToolWith(h)})
	restore()

	if strings.Contains(buf.String(), "rebound") || strings.Contains(buf.String(), "registered the handler") {
		t.Fatalf("an unchanged handler was registered again:\n%s", buf.String())
	}
}

// A handler registered by hand is not rebound by a tool carrying another one.
func TestAHandRegisteredHandlerIsNotRebound(t *testing.T) {
	svc := llm.New("FakeToolLLM", cancelOnInterruptionGen{})
	svc.RegisterFunction("get_weather", markedHandler("by hand"))
	ctx := context.Background()

	buf, restore := captureDebug(t)
	svc.SyncToolHandlers(ctx, []frames.Tool{weatherToolWith(markedHandler("carried"))})
	svc.SyncToolHandlers(ctx, nil)
	restore()

	if strings.Contains(buf.String(), "rebound") {
		t.Fatalf("a hand-registered handler was rebound:\n%s", buf.String())
	}
	if !svc.HasFunction("get_weather") {
		t.Fatal("withdrawing the tool took a hand-registered handler with it")
	}
}
