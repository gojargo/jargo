package rime

import (
	"bytes"
	"context"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

// waitTimeout bounds every wait in these tests.
const waitTimeout = 3 * time.Second

// fakeStream is one stream the fake client was asked for: the options it was
// opened with, the audio it was sent, and the updates the test hands it.
type fakeStream struct {
	opts      streamOptions
	requestID string
	automatic bool

	mu     sync.Mutex
	chunks [][]byte

	// updates are what the stream reports, in order: a transcriptUpdate, a
	// transcriptFinal, an error, or nil to end it.
	updates   chan any
	uploaded  chan struct{}
	exhausted chan struct{}
	closed    chan struct{}
}

// audio is everything the stream was sent.
func (s *fakeStream) audio() []byte {
	s.mu.Lock()
	defer s.mu.Unlock()
	return bytes.Join(s.chunks, nil)
}

// finish ends the stream with a final transcript.
func (s *fakeStream) finish(text, language string) {
	s.updates <- transcriptFinal{text: text, language: language}
	s.updates <- nil
}

// fakeRime stands in for Rime's client, letting a test script each stream.
type fakeRime struct {
	automatic bool
	closed    atomic.Bool
	newStream chan *fakeStream

	mu      sync.Mutex
	streams []*fakeStream
}

func newFakeRime() *fakeRime { return &fakeRime{newStream: make(chan *fakeStream, 16)} }

func (f *fakeRime) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.streams)
}

func (f *fakeRime) close() error {
	f.closed.Store(true)
	return nil
}

func (f *fakeRime) stream(
	ctx context.Context, opts streamOptions, source audioSource, onUpdate func(transcriptUpdate),
) (transcriptFinal, error) {
	f.mu.Lock()
	s := &fakeStream{
		opts:      opts,
		requestID: fmt.Sprintf("request-%d", len(f.streams)),
		automatic: f.automatic,
		updates:   make(chan any, 16),
		uploaded:  make(chan struct{}),
		exhausted: make(chan struct{}),
		closed:    make(chan struct{}),
	}
	f.streams = append(f.streams, s)
	f.mu.Unlock()
	f.newStream <- s
	defer close(s.closed)

	uploadCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		var once sync.Once
		for {
			chunk, ok, err := source.next(uploadCtx)
			if err != nil {
				return
			}
			if !ok {
				break
			}
			s.mu.Lock()
			s.chunks = append(s.chunks, chunk)
			s.mu.Unlock()
			once.Do(func() { close(s.uploaded) })
		}
		close(s.exhausted)
		if s.automatic {
			s.finish("segment "+s.requestID, "en")
		}
	}()

	var final *transcriptFinal
	for {
		select {
		case u := <-s.updates:
			switch v := u.(type) {
			case transcriptUpdate:
				onUpdate(v)
			case transcriptFinal:
				final = &v
			case error:
				return transcriptFinal{}, v
			case nil:
				if final == nil {
					return transcriptFinal{}, newRimeError(kindStream, "The service ended without a final transcript")
				}
				return *final, nil
			}
		case <-ctx.Done():
			return transcriptFinal{}, newRimeError(kindCancelled, "Transcription canceled")
		}
	}
}

// sttRun is a Rime STT service in a pipeline, recording the frames it pushes
// downstream and the ones that reach the start of the pipeline.
type sttRun struct {
	c      *sttConnector
	client *fakeRime
	task   *pipeline.Worker
	done   chan error
	keys   []string

	mu   sync.Mutex
	down []frames.Frame
	up   []frames.Frame
}

// runSTT starts a service with cfg on client, followed by downstream, and waits
// for its session.
func runSTT(t *testing.T, cfg STTConfig, client *fakeRime, downstream ...processor.Processor) *sttRun {
	t.Helper()
	r := &sttRun{client: client}
	return r.run(t, cfg, func(apiKey string) (transcriber, error) {
		r.mu.Lock()
		r.keys = append(r.keys, apiKey)
		r.mu.Unlock()
		return client, nil
	}, downstream...)
}

// runSTTWith starts a service with cfg whose sessions create their client with
// newClient.
func runSTTWith(
	t *testing.T, cfg STTConfig, newClient func(string) (transcriber, error), downstream ...processor.Processor,
) *sttRun {
	t.Helper()
	return (&sttRun{}).run(t, cfg, newClient, downstream...)
}

func (r *sttRun) run(
	t *testing.T, cfg STTConfig, newClient func(string) (transcriber, error), downstream ...processor.Processor,
) *sttRun {
	t.Helper()
	r.done = make(chan error, 1)
	if cfg.APIKey == "" {
		cfg.APIKey = "test"
	}
	r.c = newSTT(cfg, newClient)
	r.start(t, downstream...)
	return r
}

// apiKeys are the keys the service's clients were created with.
func (r *sttRun) apiKeys() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.keys)
}

func (r *sttRun) start(t *testing.T, downstream ...processor.Processor) {
	t.Helper()
	// The frames the service pushes are recorded as they leave it, ahead of
	// whatever consumes them downstream.
	tap := processor.NewFrameFilter("tap", func(f frames.Frame) bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.down = append(r.down, f)
		return true
	})
	procs := append([]processor.Processor{r.c.svc, tap}, downstream...)
	noInitialMetrics := false
	r.task = pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:           true,
			EnableUsageMetrics:      true,
			SendInitialEmptyMetrics: &noInitialMetrics,
		},
		IdleTimeout:           -1,
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	event.On(&r.task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.up = append(r.up, f)
	})
	go func() { r.done <- r.task.Run(context.Background()) }()
	t.Cleanup(func() {
		r.task.Cancel(context.Background(), "test over")
		select {
		case <-r.done:
		case <-time.After(waitTimeout):
			t.Error("the pipeline did not finish")
		}
	})
}

// waitConnected waits for the service to hold its session.
func (r *sttRun) waitConnected(t *testing.T) {
	t.Helper()
	waitFor(t, "the service never held its session", r.c.svc.Connected)
}

// send queues frames into the pipeline, in order.
func (r *sttRun) send(fs ...frames.Frame) {
	for _, f := range fs {
		r.task.QueueFrame(f)
	}
}

// speak queues one stretch of speech: the VAD start, the frames, the VAD stop.
func (r *sttRun) speak(fs ...frames.Frame) {
	r.send(vadStart())
	r.send(fs...)
	r.send(vadStop(0))
}

func downstreamOf[T frames.Frame](r *sttRun) []T {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []T
	for _, f := range r.down {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

func upstreamOf[T frames.Frame](r *sttRun) []T {
	r.mu.Lock()
	defer r.mu.Unlock()
	var out []T
	for _, f := range r.up {
		if v, ok := f.(T); ok {
			out = append(out, v)
		}
	}
	return out
}

func (r *sttRun) transcripts() []*frames.TranscriptionFrame {
	return downstreamOf[*frames.TranscriptionFrame](r)
}

func (r *sttRun) interims() []string {
	var texts []string
	for _, f := range downstreamOf[*frames.InterimTranscriptionFrame](r) {
		texts = append(texts, f.Text)
	}
	return texts
}

func (r *sttRun) errors() []*frames.ErrorFrame { return upstreamOf[*frames.ErrorFrame](r) }

// waitTranscripts waits for n transcripts to reach the end of the pipeline.
func (r *sttRun) waitTranscripts(t *testing.T, n int) []*frames.TranscriptionFrame {
	t.Helper()
	waitFor(t, fmt.Sprintf("%d transcripts never arrived", n), func() bool { return len(r.transcripts()) >= n })
	return r.transcripts()
}

func (r *sttRun) waitInterims(t *testing.T, n int) []string {
	t.Helper()
	waitFor(t, fmt.Sprintf("%d interim transcripts never arrived", n), func() bool { return len(r.interims()) >= n })
	return r.interims()
}

func (r *sttRun) waitErrors(t *testing.T, n int) []*frames.ErrorFrame {
	t.Helper()
	waitFor(t, fmt.Sprintf("%d errors never arrived", n), func() bool { return len(r.errors()) >= n })
	return r.errors()
}

// nextStream waits for the client to be asked for its next stream.
func nextStream(t *testing.T, client *fakeRime) *fakeStream {
	t.Helper()
	select {
	case s := <-client.newStream:
		return s
	case <-time.After(waitTimeout):
		t.Fatal("no stream was started")
		return nil
	}
}

// waitClosed waits for ch to close.
func waitClosed(t *testing.T, what string, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(waitTimeout):
		t.Fatal(what)
	}
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(waitTimeout)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatal(what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func vadStart() *frames.VADUserStartedSpeakingFrame {
	return frames.NewVADUserStartedSpeakingFrame(0, time.Now())
}

func vadStop(stopSecs float64) *frames.VADUserStoppedSpeakingFrame {
	return frames.NewVADUserStoppedSpeakingFrame(stopSecs, time.Now())
}

// pcm is 10 ms of 16 kHz audio, every sample b.
func pcm(b byte) []byte { return bytes.Repeat([]byte{b, 0}, 160) }

func audioFrame(data []byte, rate int) *frames.InputAudioRawFrame {
	return frames.NewInputAudioRawFrame(data, rate, 1)
}

func partial(text string) transcriptUpdate { return transcriptUpdate{kind: updatePartial, text: text} }

func delta(text string) transcriptUpdate { return transcriptUpdate{kind: updateDelta, text: text} }

func texts(fs []*frames.TranscriptionFrame) []string {
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		out = append(out, f.Text)
	}
	return out
}

func finalized(fs []*frames.TranscriptionFrame) []bool {
	out := make([]bool, 0, len(fs))
	for _, f := range fs {
		out = append(out, f.Finalized)
	}
	return out
}
