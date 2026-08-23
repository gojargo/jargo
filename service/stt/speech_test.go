package stt_test

import (
	"context"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/utils/events"
)

// runSpeechResults feeds the given results through a stream service and returns
// the speech-related frames that reached the pipeline, in order.
func runSpeechResults(t *testing.T, results [][]stt.Result) []string {
	t.Helper()
	conn := &fakeConnector{stream: &fakeStream{results: results}}
	svc := stt.NewStream("FakeSTT", conn, 16000)

	var mu sync.Mutex
	var seq []string
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch f.(type) {
		case *frames.ProposedUserStartedSpeakingFrame:
			seq = append(seq, "started")
		case *frames.ProposedUserStoppedSpeakingFrame:
			seq = append(seq, "stopped")
		case *frames.UserStartedSpeakingFrame, *frames.UserStoppedSpeakingFrame:
			seq = append(seq, "announced")
		case *frames.InterruptionFrame:
			seq = append(seq, "interrupted")
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// The results are delivered as fast as the service reads them; give it the
	// moment it needs before the pipeline is stopped.
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return append([]string(nil), seq...)
}

// TestSpeechBoundariesGoOutInPairs covers a provider that runs its own detection
// server-side. The pipeline is told the user started and then stopped, as a
// proposal each way, so whatever resolves them sees a turn that opens and
// closes.
func TestSpeechBoundariesGoOutInPairs(t *testing.T) {
	got := runSpeechResults(t, [][]stt.Result{
		{{Speech: stt.SpeechStarted}},
		{{Text: "hello there", Final: true, EndOfTurn: true}},
		{{Speech: stt.SpeechStopped}},
	})

	want := []string{"started", "stopped"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v", got, want)
	}
}

// TestSpeechBoundariesAreProposedNotAnnounced covers who decides. The provider
// heard where the speech begins and ends; whether that is where the turn begins
// and ends, and whether the bot is barged in on, belongs to the turn strategies
// this service recommends. So the service proposes and announces nothing.
func TestSpeechBoundariesAreProposedNotAnnounced(t *testing.T) {
	got := runSpeechResults(t, [][]stt.Result{
		{{Speech: stt.SpeechStarted}},
		{{Speech: stt.SpeechStopped}},
	})

	for _, f := range got {
		if f == "announced" {
			t.Error("the service announced the turn itself instead of proposing it")
		}
		if f == "interrupted" {
			t.Error("the service barged in itself instead of leaving it to the strategies")
		}
	}
}

// TestSpeechStartIsReportedOnce covers a provider repeating itself. Only a
// change is reported, so a second start does not stack up frames that the single
// stop will not balance.
func TestSpeechStartIsReportedOnce(t *testing.T) {
	got := runSpeechResults(t, [][]stt.Result{
		{{Speech: stt.SpeechStarted}},
		{{Speech: stt.SpeechStarted}},
		{{Speech: stt.SpeechStopped}},
	})

	want := []string{"started", "stopped"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("frames = %v, want %v: a repeated start was reported twice", got, want)
	}
}

// TestSpeechStopWithoutAStartIsIgnored covers the other direction. A stop for
// speech that never started here would close a turn the pipeline never opened.
func TestSpeechStopWithoutAStartIsIgnored(t *testing.T) {
	got := runSpeechResults(t, [][]stt.Result{
		{{Speech: stt.SpeechStopped}},
		{{Speech: stt.SpeechStopped}},
	})

	if len(got) != 0 {
		t.Errorf("frames = %v, want none: no turn had been opened", got)
	}
}

// TestATranscriptAloneReportsNoSpeech covers every provider that leaves
// detection to the pipeline. Their results say nothing about speech boundaries,
// and the service must not invent any.
func TestATranscriptAloneReportsNoSpeech(t *testing.T) {
	got := runSpeechResults(t, [][]stt.Result{
		{{Text: "hel", Final: false}},
		{{Text: "hello world", Final: true, EndOfTurn: true}},
	})

	if len(got) != 0 {
		t.Errorf("frames = %v, want none: this provider reports no speech boundaries", got)
	}
}

// TestSpeechBoundaryCanCarryText covers a result that is both: the boundary is
// reported and the transcript still reaches the pipeline.
func TestSpeechBoundaryCanCarryText(t *testing.T) {
	conn := &fakeConnector{stream: &fakeStream{results: [][]stt.Result{
		{{Text: "hello", Final: true, EndOfTurn: true, Speech: stt.SpeechStopped}},
	}}}
	svc := stt.NewStream("FakeSTT", conn, 16000)

	var mu sync.Mutex
	var text string
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if fr, ok := f.(*frames.TranscriptionFrame); ok {
			mu.Lock()
			text = fr.Text
			mu.Unlock()
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if text != "hello" {
		t.Errorf("transcript = %q, want %q: the boundary swallowed the text", text, "hello")
	}
}
