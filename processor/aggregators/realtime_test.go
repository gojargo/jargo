package aggregators_test

// Tests for the mode a pipeline runs in when a speech-to-speech service drives
// the conversation. The service answers the audio itself, so transcripts are off
// the critical path, and the transcript for what the user said arrives late,
// sometimes only once the answer has started.

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
)

// realtimeMetadata is what a realtime LLM service broadcasts about itself.
func realtimeMetadata(name string) *frames.LLMServiceMetadataFrame {
	f := frames.NewLLMServiceMetadataFrame(name)
	f.Realtime = true
	return f
}

// speechTimeoutTurns is the model-free chain, so these tests need no model.
func speechTimeoutTurns() aggregators.Option {
	return aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Stop: []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})},
		},
	})
}

// A realtime service announcing itself takes the transcript dependence out of
// the turn strategies: a transcript no longer opens a turn, and no stop strategy
// holds one open waiting for one.
func TestRealtimeServiceStripsTheTranscriptDependence(t *testing.T) {
	stop := turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})
	start := turns.NewTranscriptionStart(turns.TranscriptionStartConfig{})
	s := turns.UserTurnStrategies{
		Start: []turns.StartStrategy{turns.NewVADStart(), start},
		Stop:  []turns.StopStrategy{stop},
	}
	if !stop.WaitForTranscript() {
		t.Fatal("the speech-timeout strategy does not wait for a transcript to begin with")
	}

	dropped, flipped := s.ApplyRealtimeServiceMode()
	if len(dropped) != 1 {
		t.Errorf("dropped %v, want the transcription start strategy", dropped)
	}
	if len(s.Start) != 1 {
		t.Errorf("start chain = %d strategies, want only the VAD one left", len(s.Start))
	}
	if _, ok := s.Start[0].(*turns.VADStart); !ok {
		t.Errorf("the strategy left is %T, want the VAD one", s.Start[0])
	}
	if len(flipped) != 1 {
		t.Errorf("flipped %v, want the speech-timeout strategy", flipped)
	}
	if stop.WaitForTranscript() {
		t.Error("the speech-timeout strategy still waits for a transcript")
	}

	// Applying it again changes nothing, so a service re-announcing itself is
	// safe to act on.
	dropped, flipped = s.ApplyRealtimeServiceMode()
	if len(dropped) != 0 || len(flipped) != 0 {
		t.Errorf("re-applying changed %v / %v, want nothing", dropped, flipped)
	}
}

// runRealtimePair drives a pipeline of both halves and reports what reached the
// end of it.
func runRealtimePair(t *testing.T, opts ...aggregators.Option) (
	*pipeline.Worker, *frames.LLMContext, chan error,
) {
	t.Helper()
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, opts...)
	task := pipeline.NewWorker(
		pipeline.New(pair.User(), pair.Assistant()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, convo, runDone
}

// awaitMessages waits for the conversation to hold n messages and reports what
// it holds. The system instruction is not one of them.
func awaitMessages(t *testing.T, convo *frames.LLMContext, n int) []frames.Message {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if msgs := convo.Messages(); len(msgs) >= n {
			return msgs
		}
		time.Sleep(10 * time.Millisecond)
	}
	return convo.Messages()
}

// The user's message is written when the assistant starts answering, not when
// the turn ends: a realtime service delivers the transcript late, and writing at
// turn end would write nothing.
func TestRealtimeWritesTheUserMessageWhenTheAnswerStarts(t *testing.T) {
	task, convo, runDone := runRealtimePair(t,
		speechTimeoutTurns(), aggregators.WithRealtimeServiceMode(true))
	defer func() { task.StopWhenDone(); <-runDone }()

	// The turn happens with no transcript at all, which is the case this mode
	// exists for.
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	// The transcript lands only as the service begins answering.
	task.QueueFrame(frames.NewTranscriptionFrame("hello there", "u", "ts"))
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("hi"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	msgs := awaitMessages(t, convo, 2)
	if len(msgs) < 2 {
		t.Fatalf("conversation = %+v, want the user and assistant messages", msgs)
	}
	if msgs[0].Text != "hello there" {
		t.Errorf("user message = %q, want the transcript that arrived late", msgs[0].Text)
	}
	if msgs[1].Text != "hi" {
		t.Errorf("assistant message = %q, want it after the user's", msgs[1].Text)
	}
}

// The mode turns itself on when a realtime service says what it is, without the
// application having to configure anything.
func TestRealtimeModeTurnsOnFromTheServiceMetadata(t *testing.T) {
	task, convo, runDone := runRealtimePair(t, speechTimeoutTurns())
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(realtimeMetadata("OpenAIRealtime"))
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewTranscriptionFrame("hello there", "u", "ts"))
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame("hi"))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	msgs := awaitMessages(t, convo, 2)
	if len(msgs) < 2 {
		t.Fatalf("conversation = %+v, want the user and assistant messages", msgs)
	}
	if msgs[0].Text != "hello there" {
		t.Errorf("user message = %q, want the transcript that arrived late", msgs[0].Text)
	}
}

// Turned off deliberately, a realtime service announcing itself changes nothing:
// the application asked for the cascaded behavior and gets it.
func TestRealtimeModeCanBeRefused(t *testing.T) {
	task, convo, runDone := runRealtimePair(t,
		speechTimeoutTurns(), aggregators.WithRealtimeServiceMode(false))
	defer func() { task.StopWhenDone(); <-runDone }()

	task.QueueFrame(realtimeMetadata("OpenAIRealtime"))
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewTranscriptionFrame("hello there", "u", "ts"))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))

	// Written when the turn ends, with no answer needed to trigger it.
	msgs := awaitMessages(t, convo, 1)
	if len(msgs) < 1 {
		t.Fatalf("conversation = %+v, want the user message written at turn end", msgs)
	}
	if msgs[0].Text != "hello there" {
		t.Errorf("user message = %q, want the transcript", msgs[0].Text)
	}
}

// Whatever the user said last is written when the session ends, rather than
// being lost with the processor because no answer ever started.
func TestRealtimeCommitsWhatIsLeftAtSessionEnd(t *testing.T) {
	task, convo, runDone := runRealtimePair(t,
		speechTimeoutTurns(), aggregators.WithRealtimeServiceMode(true))

	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	task.QueueFrame(frames.NewTranscriptionFrame("one last thing", "u", "ts"))
	task.StopWhenDone()
	<-runDone

	msgs := convo.Messages()
	if len(msgs) < 1 {
		t.Fatalf("conversation = %+v, want the last thing said written", msgs)
	}
	if msgs[0].Text != "one last thing" {
		t.Errorf("user message = %q, want what was said last", msgs[0].Text)
	}
}
