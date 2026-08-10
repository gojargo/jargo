package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// fakeSynth records the text it was asked to speak and emits fixed PCM.
type fakeSynth struct {
	rate   int
	chunk  []byte
	spoken chan string
}

func (s *fakeSynth) SampleRate() int { return s.rate }

func (s *fakeSynth) RunTTS(_ context.Context, text, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, s.SampleRate())
	s.spoken <- text
	return emit(s.chunk)
}

// runTTS wires a fake synthesizer into a task and records the downstream frame
// sequence and audio sample rates.
func runTTS(t *testing.T, syn *fakeSynth, feed func(task *pipeline.Task)) []string {
	t.Helper()
	var mu sync.Mutex
	var seq []string
	stopped := make(chan struct{}, 4)
	task := pipeline.NewTask(pipeline.New(tts.New("FakeTTS", syn)), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.TTSStartedFrame:
				seq = append(seq, "started")
			case *frames.TTSAudioRawFrame:
				seq = append(seq, "audio")
				if fr.SampleRate != syn.rate {
					t.Errorf("audio sample rate = %d, want %d", fr.SampleRate, syn.rate)
				}
			case *frames.TTSStoppedFrame:
				seq = append(seq, "stopped")
				select {
				case stopped <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	feed(task)

	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("TTS base did not finish a synthesis")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	return seq
}

func TestSynthesizesCompletedSentence(t *testing.T) {
	syn := &fakeSynth{rate: 24000, chunk: []byte{1, 2, 3, 4}, spoken: make(chan string, 1)}
	seq := runTTS(t, syn, func(task *pipeline.Task) {
		// Split across frames; synthesis fires once the sentence terminates.
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewLLMTextFrame("Hello "))
		task.QueueFrame(frames.NewLLMTextFrame("world."))
		// The turn's context closes when the LLM response ends, which is what
		// brackets the audio with a stop frame.
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	if got := <-syn.spoken; got != "Hello world." {
		t.Fatalf("synthesized text = %q, want %q", got, "Hello world.")
	}
	want := []string{"started", "audio", "stopped"}
	if !equal(seq, want) {
		t.Fatalf("frame sequence = %v, want %v", seq, want)
	}
}

// describingSynth reports a model and voice, so the base can label the span.
type describingSynth struct {
	*fakeSynth
}

func (s *describingSynth) Metadata() tts.Metadata {
	return tts.Metadata{Model: "eleven_flash_v2_5", VoiceID: "XB0fDUnXU5powFXDhCwa"}
}

// TestSynthesisReportsCharacterUsage checks that a synthesis is priceable: it
// carries the provider model and its character count. The count is in runes,
// so accented text is not billed twice.
func TestSynthesisReportsCharacterUsage(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	// 11 runes, 13 bytes in UTF-8 (Ç and è take two bytes each).
	const text = "Ça va très."
	syn := &describingSynth{&fakeSynth{rate: 24000, chunk: []byte{1, 2}, spoken: make(chan string, 1)}}
	svc := tts.New("FakeTTS", syn)

	stopped := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.TTSStoppedFrame); ok {
				select {
				case stopped <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewLLMFullResponseStartFrame())
	task.QueueFrame(frames.NewLLMTextFrame(text))
	task.QueueFrame(frames.NewLLMFullResponseEndFrame())

	select {
	case <-stopped:
	case <-time.After(3 * time.Second):
		t.Fatal("TTS base did not finish a synthesis")
	}
	task.StopWhenDone()
	<-runDone

	var span sdktrace.ReadOnlySpan
	for _, s := range rec.Ended() {
		if s.Name() == "tts" {
			span = s
		}
	}
	if span == nil {
		t.Fatal("no tts span recorded")
	}
	attrs := map[string]string{}
	for _, kv := range span.Attributes() {
		attrs[string(kv.Key)] = kv.Value.String()
	}
	want := map[string]string{
		"tts.chars":                          "11",
		"gen_ai.request.model":               "eleven_flash_v2_5",
		"gen_ai.request.voice":               "XB0fDUnXU5powFXDhCwa",
		"gen_ai.output.type":                 "speech",
		"langfuse.observation.type":          "generation",
		"langfuse.observation.usage_details": `{"characters":11}`,
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
}

func TestFlushSynthesizesTrailingText(t *testing.T) {
	syn := &fakeSynth{rate: 16000, chunk: []byte{9}, spoken: make(chan string, 1)}
	seq := runTTS(t, syn, func(task *pipeline.Task) {
		// No sentence terminator: only the end-of-response flush speaks it.
		task.QueueFrame(frames.NewLLMFullResponseStartFrame())
		task.QueueFrame(frames.NewTextFrame("no period here"))
		task.QueueFrame(frames.NewLLMFullResponseEndFrame())
	})

	if got := <-syn.spoken; got != "no period here" {
		t.Fatalf("synthesized text = %q, want %q", got, "no period here")
	}
	want := []string{"started", "audio", "stopped"}
	if !equal(seq, want) {
		t.Fatalf("frame sequence = %v, want %v", seq, want)
	}
}

func equal(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// closingSynth records whether the base released it at teardown.
type closingSynth struct {
	*fakeSynth
	closed chan struct{}
}

func (s *closingSynth) Close() error {
	close(s.closed)
	return nil
}

// TestCleanupClosesSynthesizer checks a Synthesizer holding a resource open
// across syntheses (a reused connection, say) is released when the pipeline
// tears down, rather than leaking for the life of the process.
func TestCleanupClosesSynthesizer(t *testing.T) {
	syn := &closingSynth{
		fakeSynth: &fakeSynth{rate: 24000, chunk: []byte{1}, spoken: make(chan string, 1)},
		closed:    make(chan struct{}),
	}
	base := tts.New("ClosingTTS", syn)

	if err := base.Cleanup(context.Background()); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	select {
	case <-syn.closed:
	case <-time.After(time.Second):
		t.Fatal("Cleanup did not close the synthesizer")
	}
}

// startingSynth records that it was given the chance to set up before the first
// synthesis.
type startingSynth struct {
	*fakeSynth
	started chan struct{}
}

func (s *startingSynth) Start(context.Context) {
	select {
	case s.started <- struct{}{}:
	default:
	}
}

// TestStartStartsSynthesizer checks a Synthesizer with setup to do is asked to do
// it on start. The vendor handshake is the slowest part of a session's first
// synthesis, and it belongs in the window where the transport is still
// negotiating rather than in front of the bot's first words.
func TestStartStartsSynthesizer(t *testing.T) {
	syn := &startingSynth{
		fakeSynth: &fakeSynth{rate: 24000, chunk: []byte{1}, spoken: make(chan string, 1)},
		started:   make(chan struct{}, 1),
	}
	task := pipeline.NewTask(pipeline.New(tts.New("StartingTTS", syn)), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// StartFrame alone: no text has been queued and nothing has been synthesized.
	select {
	case <-syn.started:
	case <-time.After(3 * time.Second):
		t.Fatal("the synthesizer was not started on start")
	}

	task.StopWhenDone()
	<-runDone
}

// TestCleanupWithoutCloserIsFine checks a Synthesizer that holds nothing open is
// unaffected by the teardown hook.
func TestCleanupWithoutCloserIsFine(t *testing.T) {
	syn := &fakeSynth{rate: 24000, chunk: []byte{1}, spoken: make(chan string, 1)}
	if err := tts.New("PlainTTS", syn).Cleanup(context.Background()); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
}
