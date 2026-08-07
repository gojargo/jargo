package tts_test

// Ported from upstream's TTS frame-ordering suite. Each unit spoken has to reach
// the pipeline as one intact, ordered run: the aggregated text describing it,
// the start of its audio, its audio, and the end of it, with nothing from the
// next unit interleaved. A frame sent behind a unit has to land behind all of
// it.
//
// It is checked for both shapes a provider can take, because they populate the
// audio context by different routes: one answers inline, the other delivers on
// its own receive loop afterwards. Testing only the first is what let a whole
// class of provider go uncovered before (see issue #65).

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
)

const (
	orderingSampleRate = 16000
	orderingChunkBytes = 640
	// trailingChunks is how many chunks the trailing provider delivers, spread
	// out far enough that the end of the pipeline is reached while it is still
	// going.
	trailingChunks = 5
)

// fooFrame is a plain data frame sent behind a spoken unit, marking where that
// unit's frames must have finished.
type fooFrame struct {
	frames.BaseDataFrame
	label string
}

func newFooFrame(label string) *fooFrame {
	return &fooFrame{BaseDataFrame: frames.NewBaseDataFrame("FooFrame"), label: label}
}

// inlineSynth answers within the call, the way an HTTP provider does: the audio
// is on the context before anything else is processed.
type inlineSynth struct{}

func (s *inlineSynth) SampleRate() int { return orderingSampleRate }

func (s *inlineSynth) RunTTS(_ context.Context, _, contextID string, yield func(frames.Frame) error) error {
	return yield(frames.NewTTSAudioRawFrame(make([]byte, orderingChunkBytes), orderingSampleRate, 1))
}

// asyncSynthOrdering answers on its own receive loop, the way a WebSocket
// provider does: the call returns having yielded nothing, and the audio and the
// end of it are appended to the context afterwards.
type asyncSynthOrdering struct {
	mu   sync.Mutex
	host tts.AudioContextHost
	wg   sync.WaitGroup
}

func (s *asyncSynthOrdering) SampleRate() int { return orderingSampleRate }

func (s *asyncSynthOrdering) SetAudioContextHost(h tts.AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *asyncSynthOrdering) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	s.mu.Lock()
	host := s.host
	s.mu.Unlock()
	if host == nil {
		return nil
	}
	s.wg.Go(func() {
		time.Sleep(10 * time.Millisecond)
		host.AppendToAudioContext(contextID,
			frames.NewTTSAudioRawFrame(make([]byte, orderingChunkBytes), orderingSampleRate, 1))
		host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
		host.RemoveAudioContext(contextID)
	})
	return nil
}

// trailingAsyncSynth answers on its own receive loop and takes its time about
// it, the way a streaming provider does: the call returns having yielded
// nothing, and the audio lands on the context chunk by chunk long after the
// frame that asked for it was processed.
type trailingAsyncSynth struct {
	mu   sync.Mutex
	host tts.AudioContextHost
	wg   sync.WaitGroup
}

func (s *trailingAsyncSynth) SampleRate() int { return orderingSampleRate }

func (s *trailingAsyncSynth) SetAudioContextHost(h tts.AudioContextHost) {
	s.mu.Lock()
	s.host = h
	s.mu.Unlock()
}

func (s *trailingAsyncSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	s.mu.Lock()
	host := s.host
	s.mu.Unlock()
	if host == nil {
		return nil
	}
	s.wg.Go(func() {
		for range trailingChunks {
			time.Sleep(40 * time.Millisecond)
			host.AppendToAudioContext(contextID,
				frames.NewTTSAudioRawFrame(make([]byte, orderingChunkBytes), orderingSampleRate, 1))
		}
		host.AppendToAudioContext(contextID, frames.NewTTSStoppedFrame())
		host.RemoveAudioContext(contextID)
	})
	return nil
}

// TestEndFrameWaitsForAudioStillInFlight covers a pipeline stopped right after
// queueing speech. The frames arrive in order, the audio does not: a streaming
// provider delivers it on its own receive loop, after the frame that asked for
// it has been processed. The end of the pipeline must wait for it, or the last
// utterance is cut off halfway through.
func TestEndFrameWaitsForAudioStillInFlight(t *testing.T) {
	var mu sync.Mutex
	var got []frames.Frame
	synth := &trailingAsyncSynth{}
	base := tts.New("EndTTS", synth)
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	speak := frames.NewTTSSpeakFrame("Goodbye, take care.")
	speak.AppendToContext = false
	task.QueueFrame(speak)
	// Queued right behind the speech, the way a bot that says its farewell and
	// hangs up does it.
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
	synth.wg.Wait()

	mu.Lock()
	defer mu.Unlock()

	audio, end, stopped := 0, -1, -1
	for i, f := range got {
		switch f.(type) {
		case *frames.TTSAudioRawFrame:
			audio++
			if end >= 0 {
				t.Errorf("audio arrived after the pipeline had ended\nframes: %s", names(got))
			}
		case *frames.TTSStoppedFrame:
			stopped = i
		case *frames.EndFrame:
			end = i
		}
	}
	if end < 0 {
		t.Fatalf("the pipeline never ended\nframes: %s", names(got))
	}
	if audio != trailingChunks {
		t.Errorf("heard %d of %d audio chunks: the utterance was cut short\nframes: %s",
			audio, trailingChunks, names(got))
	}
	if stopped < 0 || stopped > end {
		t.Errorf("the end of the audio did not arrive before the end of the pipeline\nframes: %s",
			names(got))
	}
}

// assertGroupOrdering checks that the frames arrived as one ordered run per
// spoken unit, delimited by the marker sent behind each.
func assertGroupOrdering(t *testing.T, got []frames.Frame, labels []string) {
	t.Helper()

	// Only the frames describing a spoken unit, and the markers between them.
	var relevant []frames.Frame
	for _, f := range got {
		switch f.(type) {
		case *frames.AggregatedTextFrame, *frames.TTSStartedFrame,
			*frames.TTSAudioRawFrame, *frames.TTSStoppedFrame, *fooFrame:
			relevant = append(relevant, f)
		}
	}

	var groups [][]frames.Frame
	start := 0
	for i, f := range relevant {
		if _, ok := f.(*fooFrame); ok {
			groups = append(groups, relevant[start:i+1])
			start = i + 1
		}
	}
	if len(groups) != len(labels) {
		t.Fatalf("saw %d marked groups, want %d\nframes: %s", len(groups), len(labels), names(relevant))
	}

	for g, group := range groups {
		aggregated, started, audio, stopped, foo := -1, -1, -1, -1, -1
		for i, f := range group {
			switch fr := f.(type) {
			case *frames.AggregatedTextFrame:
				if aggregated < 0 {
					aggregated = i
				}
			case *frames.TTSStartedFrame:
				if started < 0 {
					started = i
				}
			case *frames.TTSAudioRawFrame:
				if audio < 0 {
					audio = i
				}
			case *frames.TTSStoppedFrame:
				if stopped < 0 {
					stopped = i
				}
			case *fooFrame:
				foo = i
				if fr.label != labels[g] {
					t.Errorf("group %d marker = %q, want %q", g, fr.label, labels[g])
				}
			}
		}

		for what, idx := range map[string]int{
			"AggregatedTextFrame": aggregated,
			"TTSStartedFrame":     started,
			"TTSAudioRawFrame":    audio,
			"TTSStoppedFrame":     stopped,
		} {
			if idx < 0 {
				t.Fatalf("group %d is missing a %s\nframes: %s", g, what, names(group))
			}
		}
		if started > stopped {
			t.Errorf("group %d: the audio ended before it started\nframes: %s", g, names(group))
		}
		if stopped > foo {
			t.Errorf("group %d: a frame sent behind the unit arrived before the unit finished\nframes: %s",
				g, names(group))
		}
		// Nothing from another unit may sit inside this one's audio.
		for _, f := range group[started+1 : stopped] {
			switch f.(type) {
			case *frames.TTSAudioRawFrame, *frames.TTSTextFrame:
			default:
				t.Errorf("group %d: %s came between the start and end of the audio\nframes: %s",
					g, f.Name(), names(group))
			}
		}
	}
}

func names(fs []frames.Frame) string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = f.Name()
	}
	return fmt.Sprint(out)
}

// runOrdering speaks two units back to back, each followed by a marker, and
// returns everything that reached the end of the pipeline.
func runOrdering(t *testing.T, synth tts.Synthesizer) []frames.Frame {
	t.Helper()

	var mu sync.Mutex
	var got []frames.Frame
	base := tts.New("OrderingTTS", synth)
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	for _, label := range []string{"1", "2"} {
		speak := frames.NewTTSSpeakFrame("test " + label)
		speak.AppendToContext = false
		task.QueueFrame(speak)
		task.QueueFrame(newFooFrame(label))
	}

	time.Sleep(600 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return got
}

// TestFrameOrderingInlineProvider covers a provider answering within the call.
func TestFrameOrderingInlineProvider(t *testing.T) {
	assertGroupOrdering(t, runOrdering(t, &inlineSynth{}), []string{"1", "2"})
}

// TestFrameOrderingAsyncProvider covers a provider answering on its own receive
// loop afterwards. The units must still come out whole and in order, rather than
// the second one's audio overtaking the first one's end.
func TestFrameOrderingAsyncProvider(t *testing.T) {
	synth := &asyncSynthOrdering{}
	got := runOrdering(t, synth)
	synth.wg.Wait()
	assertGroupOrdering(t, got, []string{"1", "2"})
}

// runTurns drives the service with the given frames and returns everything that
// reached the end of the pipeline.
func runTurns(t *testing.T, synth tts.Synthesizer, send []frames.Frame) []frames.Frame {
	t.Helper()

	var mu sync.Mutex
	var got []frames.Frame
	base := tts.New("TurnsTTS", synth)
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		},
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	for _, f := range send {
		task.QueueFrame(f)
	}

	time.Sleep(700 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	return got
}

// TestResponseEndFollowsAllSpokenText covers the end of a turn arriving behind
// everything the turn said. A consumer building the assistant's message from the
// spoken text closes its turn on that frame, so anything still to come after it
// is text the conversation loses.
func TestResponseEndFollowsAllSpokenText(t *testing.T) {
	got := runTurns(t, &inlineSynth{}, []frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		// The first sentence ends on a period, so it is spoken at once; the
		// second has no terminator and is flushed by the end of the turn.
		frames.NewLLMTextFrame("Hello there. "),
		frames.NewLLMTextFrame("How are you?"),
		frames.NewLLMFullResponseEndFrame(),
	})

	spoken, end := 0, -1
	for i, f := range got {
		switch f.(type) {
		case *frames.TTSTextFrame:
			spoken++
			if end >= 0 {
				t.Errorf("spoken text arrived after the turn had ended\nframes: %s", names(got))
			}
		case *frames.LLMFullResponseEndFrame:
			end = i
		}
	}
	if spoken != 2 {
		t.Errorf("saw %d spoken units, want 2\nframes: %s", spoken, names(got))
	}
	if end < 0 {
		t.Fatalf("the turn never ended\nframes: %s", names(got))
	}
}

// TestSecondTurnDoesNotOvertakeFirst covers a turn beginning while the one
// before it is still being delivered. With a provider answering on its own
// receive loop nothing pauses the pipeline, so the next turn's start can be
// processed while the previous turn's audio is still draining. It must still
// come out behind it: a consumer builds the assistant's turns from this order,
// and two turns interleaved are two turns merged.
func TestSecondTurnDoesNotOvertakeFirst(t *testing.T) {
	synth := &asyncSynthOrdering{}
	got := runTurns(t, synth, []frames.Frame{
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("Hello there."),
		frames.NewLLMFullResponseEndFrame(),
		frames.NewLLMFullResponseStartFrame(),
		frames.NewLLMTextFrame("World."),
		frames.NewLLMFullResponseEndFrame(),
	})
	synth.wg.Wait()

	var seq []string
	for _, f := range got {
		switch f.(type) {
		case *frames.LLMFullResponseStartFrame:
			seq = append(seq, "start")
		case *frames.TTSStoppedFrame:
			seq = append(seq, "stopped")
		case *frames.LLMFullResponseEndFrame:
			seq = append(seq, "end")
		}
	}

	want := []string{"start", "stopped", "end", "start", "stopped", "end"}
	if fmt.Sprint(seq) != fmt.Sprint(want) {
		t.Errorf("turn order = %v, want %v: the second turn overtook the first\nframes: %s",
			seq, want, names(got))
	}
}

// Every boundary frame names the synthesis it belongs to. A turn opens one
// context and its start, its audio and its stop all carry that context's id, so
// a consumer watching the stream can tell two overlapping syntheses apart
// instead of guessing from arrival order.
func TestBoundaryFramesCarryTheirContext(t *testing.T) {
	var mu sync.Mutex
	var announced string
	var started, stopped []string
	base := tts.New("ContextTTS", &inlineSynth{})
	task := pipeline.NewTask(pipeline.New(base), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.AggregatedTextFrame:
				announced = fr.ContextID
			case *frames.TTSStartedFrame:
				started = append(started, fr.ContextID)
			case *frames.TTSStoppedFrame:
				stopped = append(stopped, fr.ContextID)
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	speak := frames.NewTTSSpeakFrame("Hello there.")
	speak.AppendToContext = false
	task.QueueFrame(speak)
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not finish")
	}

	mu.Lock()
	defer mu.Unlock()
	if announced == "" {
		t.Fatal("the aggregated text frame named no context")
	}
	if len(started) != 1 || started[0] != announced {
		t.Fatalf("started frame contexts = %q, want one carrying %q", started, announced)
	}
	if len(stopped) != 1 || stopped[0] != announced {
		t.Fatalf("stopped frame contexts = %q, want one carrying %q", stopped, announced)
	}
}
