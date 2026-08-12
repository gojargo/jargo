package stt_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

func TestWAVHeader(t *testing.T) {
	pcm := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	out := stt.WAV(pcm, 16000, 1)

	if !bytes.Equal(out[0:4], []byte("RIFF")) || !bytes.Equal(out[8:12], []byte("WAVE")) {
		t.Fatalf("missing RIFF/WAVE markers: % x", out[0:12])
	}
	if !bytes.Equal(out[12:16], []byte("fmt ")) || !bytes.Equal(out[36:40], []byte("data")) {
		t.Fatalf("missing fmt/data chunks")
	}
	if rate := binary.LittleEndian.Uint32(out[24:28]); rate != 16000 {
		t.Fatalf("sample rate = %d, want 16000", rate)
	}
	if bits := binary.LittleEndian.Uint16(out[34:36]); bits != 16 {
		t.Fatalf("bits per sample = %d, want 16", bits)
	}
	if dataLen := binary.LittleEndian.Uint32(out[40:44]); int(dataLen) != len(pcm) {
		t.Fatalf("data length = %d, want %d", dataLen, len(pcm))
	}
	if !bytes.Equal(out[44:], pcm) {
		t.Fatalf("payload mismatch")
	}
}

// fakeStream replays canned results then blocks until the session is canceled.
type fakeStream struct {
	results [][]stt.Result
	idx     int
	ctx     context.Context
}

func (s *fakeStream) Send([]byte) error { return nil }

func (s *fakeStream) Recv() ([]stt.Result, error) {
	if s.idx < len(s.results) {
		r := s.results[s.idx]
		s.idx++
		return r, nil
	}
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *fakeStream) Close() error { return nil }

type fakeConnector struct {
	stream *fakeStream
	model  string
}

func (c *fakeConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

func (c *fakeConnector) Metadata() stt.Metadata { return stt.Metadata{Model: c.model} }

func TestStreamServiceEmitsInterimAndFinal(t *testing.T) {
	conn := &fakeConnector{stream: &fakeStream{results: [][]stt.Result{
		{{Text: "hel", Final: false}},
		{{Text: "hello world", Final: true, EndOfTurn: true, Language: "en"}},
	}}}
	svc := stt.NewStream("FakeSTT", conn, 16000)

	var mu sync.Mutex
	var seq []string
	var finalized bool
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			mu.Lock()
			defer mu.Unlock()
			switch fr := f.(type) {
			case *frames.InterimTranscriptionFrame:
				seq = append(seq, "interim:"+fr.Text)
			case *frames.TranscriptionFrame:
				seq = append(seq, "final:"+fr.Text)
				finalized = fr.Finalized
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("stream service did not emit a final transcription")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	want := []string{"interim:hel", "final:hello world"}
	if len(seq) != len(want) || seq[0] != want[0] || seq[1] != want[1] {
		t.Fatalf("frames = %v, want %v", seq, want)
	}
	if !finalized {
		t.Fatal("final transcription not marked finalized (end of turn)")
	}
}

// fakeTranscriber records the audio it received and returns fixed text.
type fakeTranscriber struct {
	text string
	got  chan []byte
}

func (tr *fakeTranscriber) Transcribe(_ context.Context, audio []byte, _ int) (string, error) {
	tr.got <- append([]byte(nil), audio...)
	return tr.text, nil
}

func TestSegmentServiceTranscribesBufferedSpeech(t *testing.T) {
	tr := &fakeTranscriber{text: "buffered words", got: make(chan []byte, 1)}
	svc := stt.NewSegment("FakeSegmentSTT", tr, 16000)

	var captured string
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if fr, ok := f.(*frames.TranscriptionFrame); ok {
				captured = fr.Text
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	pcm := []byte{1, 2, 3, 4}
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	task.QueueFrame(frames.NewInputAudioRawFrame(pcm, 16000, 1))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	select {
	case got := <-tr.got:
		if !bytes.Equal(got, pcm) {
			t.Fatalf("transcriber audio = % x, want % x", got, pcm)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("segment service did not call the transcriber")
	}

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("segment service did not emit a transcription")
	}
	task.StopWhenDone()
	<-runDone

	if captured != "buffered words" {
		t.Fatalf("transcription = %q, want %q", captured, "buffered words")
	}
}

// TestStreamServiceReportsAudioUsage checks that the audio streamed to the
// provider is reported as usage. Streaming STT is billed on connection time, so
// silence counts too, and the whole connection is measured rather than the
// segments cut out of it.
func TestStreamServiceReportsAudioUsage(t *testing.T) {
	conn := &fakeConnector{
		model:  "nova-3",
		stream: &fakeStream{results: [][]stt.Result{{{Text: "hello", Final: true, EndOfTurn: true}}}},
	}
	svc := stt.NewStream("FakeSTT", conn, 16000)

	done := make(chan struct{}, 1)
	var usage []frames.STTUsageMetricsData
	var usageMu sync.Mutex
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableUsageMetrics:      true,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if mf, ok := f.(*frames.MetricsFrame); ok {
				usageMu.Lock()
				for _, d := range mf.Data {
					if u, ok := d.(frames.STTUsageMetricsData); ok {
						usage = append(usage, u)
					}
				}
				usageMu.Unlock()
			}
			if _, ok := f.(*frames.TranscriptionFrame); ok {
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("stream service did not emit a transcription")
	}
	// 16000 bytes of 16-bit mono at 16 kHz is 500 ms of audio.
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 16000), 16000, 1))
	task.StopWhenDone()
	<-runDone

	usageMu.Lock()
	defer usageMu.Unlock()
	var total float64
	for _, u := range usage {
		total += u.Value.AudioSeconds
		if u.Model != "nova-3" {
			t.Errorf("usage model = %q, want nova-3", u.Model)
		}
	}
	if total != 0.5 {
		t.Errorf("reported audio = %vs, want the 0.5s streamed (reports: %+v)", total, usage)
	}
}

// TestStreamServiceSpansOneSegment checks that a transcription span covers the
// segment it transcribed rather than the connection it arrived on: it is
// anchored where the speech began, carries the transcript and the model, and
// closes on the transcript that finalizes it.
func TestStreamServiceSpansOneSegment(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	conn := &fakeConnector{
		model:  "nova-3",
		stream: &fakeStream{results: [][]stt.Result{{{Text: "hello", Final: true, EndOfTurn: true}}}},
	}
	svc := stt.NewStream("FakeSTT", conn, 16000)

	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableTracing:           true,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.TranscriptionFrame); ok {
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	speechStart := time.Now()
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, speechStart.Add(200*time.Millisecond)))

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("stream service did not emit a transcription")
	}
	task.StopWhenDone()
	<-runDone

	var span sdktrace.ReadOnlySpan
	for _, s := range rec.Ended() {
		if s.Name() == "stt" {
			span = s
		}
	}
	if span == nil {
		t.Fatal("no stt span recorded")
	}
	attrs := map[string]string{}
	for _, kv := range span.Attributes() {
		attrs[string(kv.Key)] = kv.Value.String()
	}
	want := map[string]string{
		"gen_ai.provider.name":  "fake",
		"gen_ai.request.model":  "nova-3",
		"gen_ai.operation.name": "stt",
		"transcript":            "hello",
		"is_final":              "true",
		"vad_enabled":           "true",
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
	// Anchored where the speech began, which is earlier than the VAD's
	// determination by the delay it took to confirm it.
	if got := span.StartTime(); got.After(speechStart.Add(10 * time.Millisecond)) {
		t.Errorf("span starts at %v, want it anchored at the speech start %v", got, speechStart)
	}
}

// finalizingStream records that it was told the speech ended.
type finalizingStream struct {
	fakeStream
	finalized chan struct{}
}

func (s *finalizingStream) Finalize() error {
	select {
	case s.finalized <- struct{}{}:
	default:
	}
	return nil
}

// finalizingConnector opens the one stream the test watches.
type finalizingConnector struct{ stream *finalizingStream }

func (c *finalizingConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// A provider that flushes an utterance only when told is told as soon as the VAD
// reports the speech ended, rather than waiting on its own endpointing.
func TestStreamServiceFinalizesOnVADStop(t *testing.T) {
	stream := &finalizingStream{finalized: make(chan struct{}, 1)}
	svc := stt.NewStream("FinalizingSTT", &finalizingConnector{stream: stream}, 16000)

	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	select {
	case <-stream.finalized:
	case <-time.After(3 * time.Second):
		t.Fatal("the session was never told the speech ended")
	}

	task.StopWhenDone()
	<-runDone
}

// answeringStream says nothing until it is told the speech ended, then answers
// that finalize the way a provider that confirms its flushes does.
type answeringStream struct {
	told chan struct{}
	sent bool
	ctx  context.Context //nolint:containedctx // the session context, set on dial
}

func (s *answeringStream) Send([]byte) error { return nil }

func (s *answeringStream) Close() error { return nil }

func (s *answeringStream) Finalize() error {
	select {
	case s.told <- struct{}{}:
	default:
	}
	return nil
}

func (s *answeringStream) Recv() ([]stt.Result, error) {
	if !s.sent {
		select {
		case <-s.told:
		case <-s.ctx.Done():
			return nil, s.ctx.Err()
		}
		s.sent = true
		return []stt.Result{{Text: "hello world", Final: true, FromFinalize: true}}, nil
	}
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

type answeringConnector struct{ stream *answeringStream }

func (c *answeringConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

// The transcript answering a finalize the provider confirms is the last one for
// the utterance, and the frame says so. A provider calling a result final is not
// the same claim: it means the words will not change, not that the turn is over.
func TestStreamServiceMarksTheAnswerToAFinalizeFinal(t *testing.T) {
	stream := &answeringStream{told: make(chan struct{}, 1)}
	svc := stt.NewStream("ConfirmingSTT", &answeringConnector{stream: stream}, 16000)

	var mu sync.Mutex
	var finalized bool
	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			fr, ok := f.(*frames.TranscriptionFrame)
			if !ok {
				return
			}
			mu.Lock()
			finalized = fr.Finalized
			mu.Unlock()
			select {
			case done <- struct{}{}:
			default:
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("no transcription arrived after the finalize")
	}
	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()
	if !finalized {
		t.Fatal("the transcript answering a confirmed finalize was not marked final")
	}
}

// TestSegmentServiceSpansOneSegment checks that a segmented service records the
// segment it transcribed: the transcript it produced, the audio it was billed
// for, and the model it ran against.
func TestSegmentServiceSpansOneSegment(t *testing.T) {
	rec := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(rec))
	prev := otel.GetTracerProvider()
	otel.SetTracerProvider(tp)
	defer otel.SetTracerProvider(prev)

	tr := &fakeTranscriber{text: "buffered words", got: make(chan []byte, 1)}
	svc := stt.NewSegment("FakeSegmentSTT", tr, 16000)

	done := make(chan struct{}, 1)
	task := pipeline.NewTask(pipeline.New(svc), pipeline.TaskParams{
		EnableTracing:           true,
		ReachedDownstreamFilter: pipeline.AnyFrame,
		OnReachedDownstream: func(f frames.Frame) {
			if _, ok := f.(*frames.TranscriptionFrame); ok {
				select {
				case done <- struct{}{}:
				default:
				}
			}
		},
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	speechStart := time.Now()
	task.QueueFrame(frames.NewVADUserStartedSpeakingFrame(0.2, speechStart.Add(200*time.Millisecond)))
	task.QueueFrame(frames.NewUserStartedSpeakingFrame())
	// 16000 bytes of 16-bit mono at 16 kHz is 500 ms of audio.
	task.QueueFrame(frames.NewInputAudioRawFrame(make([]byte, 16000), 16000, 1))
	task.QueueFrame(frames.NewUserStoppedSpeakingFrame())

	select {
	case <-tr.got:
	case <-time.After(3 * time.Second):
		t.Fatal("segment service did not call the transcriber")
	}
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("segment service did not emit a transcription")
	}
	task.StopWhenDone()
	<-runDone

	var span sdktrace.ReadOnlySpan
	for _, s := range rec.Ended() {
		if s.Name() == "stt" {
			span = s
		}
	}
	if span == nil {
		t.Fatal("no stt span recorded")
	}
	attrs := map[string]string{}
	for _, kv := range span.Attributes() {
		attrs[string(kv.Key)] = kv.Value.String()
	}
	want := map[string]string{
		"gen_ai.provider.name":  "fakesegment",
		"gen_ai.operation.name": "stt",
		"transcript":            "buffered words",
		"is_final":              "true",
		"metrics.audio_seconds": "0.5",
	}
	for k, v := range want {
		if attrs[k] != v {
			t.Errorf("attr %q = %q, want %q (all: %v)", k, attrs[k], v, attrs)
		}
	}
	if got := span.StartTime(); got.After(speechStart.Add(10 * time.Millisecond)) {
		t.Errorf("span starts at %v, want it anchored at the speech start %v", got, speechStart)
	}
}
