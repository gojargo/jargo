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

type fakeConnector struct{ stream *fakeStream }

func (c *fakeConnector) Connect(ctx context.Context, _ int) (stt.Stream, error) {
	c.stream.ctx = ctx
	return c.stream, nil
}

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
