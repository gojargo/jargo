package transport_test

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/transport"
)

// The output filter runs on the audio the pipeline sends, before it is
// written.

// filterTestRate gives one-chunk frames of filterChunk bytes: 10 ms at 48 kHz
// is 960 bytes, written two at a time.
const (
	filterTestRate = 48000
	filterChunk    = 1920
)

// invertingFilter inverts every byte and records its calls in order: a chunk
// by its fingerprint, a control frame by its name.
type invertingFilter struct {
	rate  atomic.Int32
	stops atomic.Int32
	// empties makes Filter return no audio, as a filter still buffering does.
	empties bool

	mu    sync.Mutex
	calls []string
}

func (f *invertingFilter) Start(_ context.Context, sampleRate int) error {
	f.rate.Store(int32(sampleRate))
	return nil
}

func (f *invertingFilter) Stop(context.Context) error {
	f.stops.Add(1)
	return nil
}

func (f *invertingFilter) ProcessFrame(_ context.Context, fr frames.FilterControlFrame) error {
	f.record(fr.Name())
	return nil
}

func (f *invertingFilter) Filter(_ context.Context, pcm []byte) ([]byte, error) {
	f.record(fingerprint(pcm))
	if f.empties {
		return []byte{}, nil
	}
	return inverted(pcm), nil
}

func (f *invertingFilter) record(call string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, call)
}

func (f *invertingFilter) recorded() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return slices.Clone(f.calls)
}

func inverted(pcm []byte) []byte {
	out := make([]byte, len(pcm))
	for i, b := range pcm {
		out[i] = b ^ 0xFF
	}
	return out
}

// fingerprint names a chunk by its length and hash, so a failed comparison
// prints short.
func fingerprint(pcm []byte) string {
	return fmt.Sprintf("%d:%x", len(pcm), sha256.Sum256(pcm))[:20]
}

// filterAudio is one chunk of audio, every byte set to fill.
func filterAudio(fill byte) *frames.OutputAudioRawFrame {
	return frames.NewOutputAudioRawFrame(bytes.Repeat([]byte{fill}, filterChunk), filterTestRate, 1)
}

func filterParams(f *invertingFilter) transport.Params {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = filterTestRate
	params.AudioOutFilter = f
	return params
}

func runOutput(t *testing.T, o *fakeOutput) (*pipeline.Worker, func()) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, func() {
		task.Cancel(context.Background(), "")
		select {
		case <-runDone:
		case <-time.After(5 * time.Second):
			t.Error("the output task did not shut down")
		}
	}
}

func TestBaseOutputWritesTheFilteredAudio(t *testing.T) {
	filter := &invertingFilter{}
	o := newFakeOutput(filterParams(filter))
	task, stop := runOutput(t, o)

	frame := filterAudio(1)
	audio := slices.Clone(frame.Audio)
	task.QueueFrame(frame)
	if got := recvWrite(t, o); !bytes.Equal(got, inverted(audio)) {
		t.Errorf("written %s, want the filtered audio %s", fingerprint(got), fingerprint(inverted(audio)))
	}
	stop()

	if got := filter.rate.Load(); got != filterTestRate {
		t.Errorf("filter started at %d Hz, want the transport's %d", got, filterTestRate)
	}
	if got := filter.stops.Load(); got != 1 {
		t.Errorf("filter stopped %d times, want once", got)
	}
}

func TestBaseOutputSkipsAChunkTheFilterEmpties(t *testing.T) {
	filter := &invertingFilter{empties: true}
	o := newFakeOutput(filterParams(filter))
	task, stop := runOutput(t, o)
	defer stop()

	task.QueueFrame(filterAudio(1))
	time.Sleep(100 * time.Millisecond)

	if got := len(filter.recorded()); got != 1 {
		t.Errorf("filter called %d times, want once", got)
	}
	select {
	case got := <-o.writes:
		t.Errorf("a chunk the filter emptied was written: %s", fingerprint(got))
	default:
	}
}

// recordingMixer passes its audio through and records each chunk it mixes.
type recordingMixer struct {
	mu    sync.Mutex
	mixed []string
}

func (*recordingMixer) Start(context.Context, int) error { return nil }
func (*recordingMixer) Stop(context.Context) error       { return nil }

func (m *recordingMixer) Mix(_ context.Context, pcm []byte) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.mixed = append(m.mixed, fingerprint(pcm))
	return pcm, nil
}

func (*recordingMixer) ProcessFrame(context.Context, frames.MixerControlFrame) error { return nil }

func (m *recordingMixer) recorded() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.mixed)
}

func TestBaseOutputFilterRunsBeforeTheMixerAndSkipsMixerAudio(t *testing.T) {
	filter := &invertingFilter{}
	mixer := &recordingMixer{}
	params := filterParams(filter)
	params.AudioOutMixer = mixer
	o := newFakeOutput(params)
	task, stop := runOutput(t, o)
	defer stop()

	frame := filterAudio(1)
	audio := slices.Clone(frame.Audio)
	task.QueueFrame(frame)
	deadline := time.Now().Add(3 * time.Second)
	for !slices.Contains(mixer.recorded(), fingerprint(inverted(audio))) {
		if time.Now().After(deadline) {
			t.Fatal("the mixer never mixed the filtered audio")
		}
		// The writes pace the loop, so they have to be taken for it to go on.
		select {
		case <-o.writes:
		case <-time.After(10 * time.Millisecond):
		}
	}

	// The mixer mixes a silent chunk on each pass with an empty queue, and the
	// filter gets none of them.
	if got := filter.recorded(); !slices.Equal(got, []string{fingerprint(audio)}) {
		t.Errorf("filter calls = %q, want only the queued audio", got)
	}
}

func TestBaseOutputFilterControlFramesReachTheFilterInOrderWithTheAudio(t *testing.T) {
	filter := &invertingFilter{}
	o := newGatedOutput(filterParams(filter))
	task, _, stop := runLogged(t, o)
	defer stop()

	first, second, third := filterAudio(1), filterAudio(2), filterAudio(3)
	audio := []string{fingerprint(first.Audio), fingerprint(second.Audio), fingerprint(third.Audio)}
	controls := []frames.Frame{
		frames.NewFilterEnableFrame(false),
		frames.NewFilterUpdateSettingsFrame(map[string]any{}),
	}

	// The first chunk holds the write, so the second chunk is still queued when
	// the control frames arrive.
	task.QueueFrame(first)
	waitEntered(t, o.entered)
	task.QueueFrames(append(append([]frames.Frame{second}, controls...), third))
	time.Sleep(100 * time.Millisecond)
	close(o.gate)

	want := []string{audio[0], audio[1], controls[0].Name(), controls[1].Name(), audio[2]}
	deadline := time.Now().Add(3 * time.Second)
	for !slices.Equal(filter.recorded(), want) {
		if time.Now().After(deadline) {
			t.Fatalf("filter calls = %q, want %q", filter.recorded(), want)
		}
		time.Sleep(10 * time.Millisecond)
	}
}
