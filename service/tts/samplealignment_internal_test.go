package tts

// Keeping TTS audio frames aligned to whole 16-bit samples. A provider may cut
// its PCM stream at any byte, so a chunk can end mid-sample. The base holds the
// partial sample back and puts it in front of the context's next frame, so
// every audio frame it emits holds whole samples.

import (
	"bytes"
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/utils/events"
)

const alignSampleRate = 16000

// testPCM is the audio the providers below deliver, cut into chunks.
func testPCM() []byte {
	b := make([]byte, 4096)
	for i := range b {
		b[i] = byte(i % 251)
	}
	return b
}

func splitPCM(data []byte, sizes ...int) [][]byte {
	chunks := make([][]byte, 0, len(sizes))
	pos := 0
	for _, size := range sizes {
		chunks = append(chunks, data[pos:pos+size])
		pos += size
	}
	return chunks
}

// chunkSynth yields each utterance as the given chunks, the way an HTTP
// provider streaming a response body does.
type chunkSynth struct {
	chunks      [][]byte
	numChannels int
	delay       time.Duration

	mu      sync.Mutex
	yielded []*frames.TTSAudioRawFrame
}

func (s *chunkSynth) SampleRate() int { return alignSampleRate }

func (s *chunkSynth) RunTTS(ctx context.Context, _, contextID string, yield func(frames.Frame) error) error {
	for _, chunk := range s.chunks {
		f := frames.NewTTSAudioRawFrame(append([]byte(nil), chunk...), alignSampleRate, max(s.numChannels, 1))
		f.ContextID = contextID
		s.mu.Lock()
		s.yielded = append(s.yielded, f)
		s.mu.Unlock()
		if err := yield(f); err != nil {
			return err
		}
		if s.delay > 0 {
			select {
			case <-time.After(s.delay):
			case <-ctx.Done():
				return nil
			}
		}
	}
	return nil
}

// speakAligned speaks one utterance through a base driving syn, then queues
// after behind it, and returns every frame that reached the end of the
// pipeline.
func speakAligned(t *testing.T, syn *chunkSynth, after ...frames.Frame) (*Base, []frames.Frame) {
	t.Helper()
	base := New("AlignTTS", syn)
	task := pipeline.NewWorker(pipeline.New(base), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	var mu sync.Mutex
	var down []frames.Frame
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTTSSpeakFrame("Hello."))
	for _, f := range after {
		if sleep, ok := f.(*sleepFrame); ok {
			time.Sleep(sleep.d)
			continue
		}
		task.QueueFrame(f)
	}
	task.StopWhenDone()
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return base, append([]frames.Frame(nil), down...)
}

// sleepFrame is not sent: it holds the frames queued after it back for d.
type sleepFrame struct {
	frames.BaseDataFrame
	d time.Duration
}

func sleepFor(d time.Duration) *sleepFrame { return &sleepFrame{d: d} }

func audioOf(down []frames.Frame) []*frames.TTSAudioRawFrame {
	var out []*frames.TTSAudioRawFrame
	for _, f := range down {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, a)
		}
	}
	return out
}

func joinAudio(fs []*frames.TTSAudioRawFrame) []byte {
	var out []byte
	for _, f := range fs {
		out = append(out, f.Audio...)
	}
	return out
}

func remaindersOf(b *Base) int {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return len(b.audioRemainders)
}

func TestSplitSamplesAreRealigned(t *testing.T) {
	alignPCM := testPCM()
	// Odd chunk sizes, including a single byte, totaling an even length.
	syn := &chunkSynth{chunks: splitPCM(alignPCM, 1023, 1, 2047, 1, 1024)}

	_, down := speakAligned(t, syn)

	audio := audioOf(down)
	for _, f := range audio {
		if len(f.Audio)%2 != 0 {
			t.Fatalf("frame of %d bytes is not whole samples", len(f.Audio))
		}
	}
	if !bytes.Equal(joinAudio(audio), alignPCM) {
		t.Fatal("realigned audio differs from what the provider delivered")
	}
}

func TestAlignedFramesPassThroughUntouched(t *testing.T) {
	alignPCM := testPCM()
	syn := &chunkSynth{chunks: splitPCM(alignPCM, 1024, 2048, 1024)}

	_, down := speakAligned(t, syn)

	audio := audioOf(down)
	if len(audio) != len(syn.yielded) {
		t.Fatalf("got %d audio frames, want %d", len(audio), len(syn.yielded))
	}
	for i, f := range audio {
		if f != syn.yielded[i] || !bytes.Equal(f.Audio, syn.chunks[i]) {
			t.Fatalf("frame %d was not passed through as is", i)
		}
	}
}

func TestStereoFramesAlignToWholeSamplePairs(t *testing.T) {
	alignPCM := testPCM()
	syn := &chunkSynth{chunks: splitPCM(alignPCM, 1022, 3, 2049, 1022), numChannels: 2}

	_, down := speakAligned(t, syn)

	audio := audioOf(down)
	for _, f := range audio {
		if len(f.Audio)%4 != 0 {
			t.Fatalf("stereo frame of %d bytes is not whole sample pairs", len(f.Audio))
		}
	}
	if !bytes.Equal(joinAudio(audio), alignPCM) {
		t.Fatal("realigned audio differs from what the provider delivered")
	}
}

func TestTrailingPartialSampleIsPaddedAtEndOfContext(t *testing.T) {
	alignPCM := testPCM()
	syn := &chunkSynth{chunks: splitPCM(alignPCM[:2047], 1023, 1024)}

	base, down := speakAligned(t, syn)

	audio := audioOf(down)
	for _, f := range audio {
		if len(f.Audio)%2 != 0 {
			t.Fatalf("frame of %d bytes is not whole samples", len(f.Audio))
		}
	}
	want := append(append([]byte(nil), alignPCM[:2047]...), 0)
	if !bytes.Equal(joinAudio(audio), want) {
		t.Fatal("trailing partial sample was not zero-padded onto the audio")
	}
	// The padded sample is part of the utterance, so it plays before the stop
	// frame.
	lastAudio, stopped := -1, -1
	for i, f := range down {
		switch f.(type) {
		case *frames.TTSAudioRawFrame:
			lastAudio = i
		case *frames.TTSStoppedFrame:
			if stopped < 0 {
				stopped = i
			}
		}
	}
	if stopped < 0 || lastAudio > stopped {
		t.Fatalf("last audio at %d, stop frame at %d: the padded sample played after the stop", lastAudio, stopped)
	}
	if n := remaindersOf(base); n != 0 {
		t.Fatalf("%d partial samples still held", n)
	}
}

func TestContextTimeoutDropsPartialSample(t *testing.T) {
	alignPCM := testPCM()
	// The response stalls mid-sample until its audio context times out, and the
	// drain loop forgets the context.
	base := New("AlignTTS", &chunkSynth{})
	base.CreateAudioContext("ctx")
	f := frames.NewTTSAudioRawFrame(append([]byte(nil), alignPCM[:1023]...), alignSampleRate, 1)
	base.AppendToAudioContext("ctx", f)
	if n := remaindersOf(base); n != 1 {
		t.Fatalf("%d partial samples held, want 1", n)
	}

	base.deleteAudioContext("ctx")

	if n := remaindersOf(base); n != 0 {
		t.Fatalf("%d partial samples still held", n)
	}
}

func TestInterruptionDiscardsPartialSample(t *testing.T) {
	alignPCM := testPCM()
	syn := &chunkSynth{chunks: splitPCM(alignPCM, 1023, 1024, 2049), delay: 50 * time.Millisecond}

	base, _ := speakAligned(t, syn,
		sleepFor(20*time.Millisecond), frames.NewInterruptionFrame(), sleepFor(200*time.Millisecond))

	if n := remaindersOf(base); n != 0 {
		t.Fatalf("%d partial samples still held", n)
	}
}
