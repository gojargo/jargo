package novasonic

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// usageEnvelope parses one wire event the way the read loop does, so a test
// exercises the same decoding a live session would.
func usageEnvelope(t *testing.T, raw string) outputEvent {
	t.Helper()
	var env outputEnvelope
	if err := json.Unmarshal([]byte(raw), &env); err != nil {
		t.Fatalf("decode: %v", err)
	}
	return env.Event
}

// runUsage feeds one event to a service wired into a running pipeline and
// returns every token usage it reported.
func runUsage(t *testing.T, ev outputEvent) []frames.LLMTokenUsage {
	t.Helper()
	s := New(Config{Region: "us-east-1", AccessKeyID: "id", SecretAccessKey: "secret"})

	got := make(chan frames.LLMTokenUsage, 4)
	started := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(s), pipeline.TaskParams{
		EnableUsageMetrics:      true,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			switch fr := f.(type) {
			case *frames.TextFrame:
				select {
				case started <- struct{}{}:
				default:
				}
			case *frames.MetricsFrame:
				for _, d := range fr.Data {
					if u, ok := d.(frames.LLMUsageMetricsData); ok {
						select {
						case got <- u.Value:
						default:
						}
					}
				}
			}
		},
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	// A frame the service passes through tells us it has started, so the event
	// below is handled by a service already able to report metrics.
	task.QueueFrame(frames.NewTextFrame("start"))
	select {
	case <-started:
	case <-time.After(2 * time.Second):
		t.Fatal("the pipeline never started")
	}

	s.handle(ev)

	task.QueueFrame(frames.NewEndFrame())
	if err := <-done; err != nil {
		t.Fatalf("task: %v", err)
	}
	close(got)

	var usage []frames.LLMTokenUsage
	for u := range got {
		usage = append(usage, u)
	}
	return usage
}

// A usage event reports what it adds, not the running total, so usage stays
// incremental the way the other realtime services report it.
func TestUsageEventReportsTheDelta(t *testing.T) {
	ev := usageEnvelope(t, `{"event":{"usageEvent":{"details":{
		"delta":{"input":{"speechTokens":0,"textTokens":3},
		         "output":{"speechTokens":20,"textTokens":0}},
		"total":{"input":{"speechTokens":288,"textTokens":3443},
		         "output":{"speechTokens":694,"textTokens":203}}}}}}`)

	usage := runUsage(t, ev)
	if len(usage) != 1 {
		t.Fatalf("reported %d usages, want 1", len(usage))
	}
	u := usage[0]
	// The prompt and completion counts are each direction's speech and text
	// added together; the breakdown is kept alongside.
	if u.PromptTokens != 3 || u.CompletionTokens != 20 || u.TotalTokens != 23 {
		t.Errorf("usage = %+v, want 3 prompt, 20 completion, 23 total", u)
	}
	if u.InputTextTokens != 3 || u.OutputAudioTokens != 20 ||
		u.InputAudioTokens != 0 || u.OutputTextTokens != 0 {
		t.Errorf("modality breakdown = %+v", u)
	}
}

// An event that accounts for nothing reports nothing, rather than an empty
// measurement.
func TestUsageEventWithoutTokensReportsNothing(t *testing.T) {
	ev := usageEnvelope(t, `{"event":{"usageEvent":{"details":{
		"delta":{"input":{"speechTokens":0,"textTokens":0},
		         "output":{"speechTokens":0,"textTokens":0}}}}}}`)

	if usage := runUsage(t, ev); len(usage) != 0 {
		t.Errorf("reported %+v for an event carrying no tokens", usage)
	}
}

// A partial event is handled without reporting anything, since a missing
// accounting is not a zero one worth emitting.
func TestUsageEventWithoutDetailsReportsNothing(t *testing.T) {
	ev := usageEnvelope(t, `{"event":{"usageEvent":{}}}`)

	if usage := runUsage(t, ev); len(usage) != 0 {
		t.Errorf("reported %+v for an event with no accounting at all", usage)
	}
}

// The service reports metrics, without which the usage above is dropped before
// it reaches the pipeline.
func TestServiceGeneratesMetrics(t *testing.T) {
	s := New(Config{Region: "us-east-1", AccessKeyID: "id", SecretAccessKey: "secret"})
	if !s.CanGenerateMetrics() {
		t.Error("the service does not report metrics, so its usage never leaves it")
	}
	var _ processor.Processor = s
}
