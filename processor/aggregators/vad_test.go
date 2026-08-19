package aggregators_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/utils/events"
)

// Tests for the voice-activity detection a user aggregator can run itself,
// instead of taking it from a detector sitting earlier in the pipeline.

// scriptedVAD returns scripted states, one per AnalyzeAudio call, holding the
// last one once the script runs out.
type scriptedVAD struct {
	mu     sync.Mutex
	states []vad.State
	i      int
	params vad.Params
}

func newScriptedVAD(states ...vad.State) *scriptedVAD {
	return &scriptedVAD{states: states, params: vad.DefaultParams()}
}

func (f *scriptedVAD) SetSampleRate(int) error { return nil }

func (f *scriptedVAD) AnalyzeAudio([]byte) vad.State {
	f.mu.Lock()
	defer f.mu.Unlock()
	s := f.states[min(f.i, len(f.states)-1)]
	f.i++
	return s
}

func (f *scriptedVAD) Params() vad.Params     { return f.params }
func (f *scriptedVAD) SetParams(p vad.Params) { f.params = p }
func (f *scriptedVAD) Reset()                 {}
func (f *scriptedVAD) Close() error           { return nil }

// audioChunk is 20 ms of silence at 16 kHz, which is what a transport delivers.
func audioChunk() *frames.InputAudioRawFrame {
	return frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1)
}

// TestAggregatorDetectsSpeechItself checks that an aggregator given an analyzer
// raises the speaking frames itself, with no detector upstream of it.
func TestAggregatorDetectsSpeechItself(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{
		Analyzer: newScriptedVAD(vad.StateSpeaking, vad.StateSpeaking, vad.StateQuiet),
	}))

	seen := make(chan frames.Frame, 64)
	task := pipeline.NewWorker(pipeline.New(pair.User()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for range 3 {
		task.QueueFrame(audioChunk())
	}

	awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.VADUserStartedSpeakingFrame)
		return ok
	}, "the speech-started frame")
	awaitFrame(t, seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.VADUserStoppedSpeakingFrame)
		return ok
	}, "the speech-stopped frame")

	task.StopWhenDone()
	<-runDone
}

// TestDetectedSpeechDrivesTheTurn is what detecting inside the aggregator is
// for: the frames it raises are queued back into this processor, so the turn
// strategies running here see them and a turn opens on the user's voice alone.
func TestDetectedSpeechDrivesTheTurn(t *testing.T) {
	convo := frames.NewLLMContext("system")
	pair := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{
			Analyzer: newScriptedVAD(vad.StateSpeaking),
		}),
		aggregators.WithTurns(turns.Config{
			Strategies: turns.UserTurnStrategies{
				Start: []turns.StartStrategy{turns.NewVADStart()},
				Stop:  []turns.StopStrategy{turns.NewExternalCompletionStop()},
			},
		}),
	)

	started := make(chan struct{}, 1)
	events.On(pair.User().Events(), aggregators.EventUserTurnStarted,
		func(context.Context, turns.StartStrategy) {
			select {
			case started <- struct{}{}:
			default:
			}
		})

	task := pipeline.NewWorker(pipeline.New(pair.User()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for range 3 {
		task.QueueFrame(audioChunk())
	}

	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("the detected speech never opened a turn")
	}

	task.StopWhenDone()
	<-runDone
}

// TestMutedAudioNeverReachesTheDetector checks that input the user is not
// allowed to give is dropped before detection runs over it, so a muted
// microphone cannot read as speech.
func TestMutedAudioNeverReachesTheDetector(t *testing.T) {
	convo := frames.NewLLMContext("system")
	analyzer := newScriptedVAD(vad.StateSpeaking)
	pair := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: analyzer}),
		// Muted for as long as the bot is speaking, which it is throughout.
		aggregators.WithMuteStrategies(turns.NewAlwaysUserMute()),
	)

	seen := make(chan frames.Frame, 64)
	task := pipeline.NewWorker(pipeline.New(pair.User()), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case seen <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewBotStartedSpeakingFrame())
	for range 3 {
		task.QueueFrame(audioChunk())
	}

	if sawFrame(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.VADUserStartedSpeakingFrame)
		return ok
	}) {
		t.Error("muted audio was analyzed and read as speech")
	}

	task.StopWhenDone()
	<-runDone
}

// TestNoAnalyzerMeansNoDetection checks that an aggregator built without one
// runs no detection, leaving it to a detector elsewhere in the pipeline.
func TestNoAnalyzerMeansNoDetection(t *testing.T) {
	convo := frames.NewLLMContext("system")
	task, seen, runDone := runAggregator(t, convo)

	for range 3 {
		task.QueueFrame(audioChunk())
	}

	if sawFrame(seen, func(f frames.Frame) bool {
		_, ok := f.(*frames.VADUserStartedSpeakingFrame)
		return ok
	}) {
		t.Error("an aggregator with no analyzer raised a speaking frame")
	}

	task.StopWhenDone()
	<-runDone
}
