package transport_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"slices"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
)

// A barge-in keeps the send loop running and only drains what it had queued of
// the turn that was cut off. The tests below cover what that preserves: the
// write already in flight, the audio and frames marked uninterruptible, and the
// bot's hold on the floor while that audio is still to play.

// gatedOutput holds every write until the test opens the gate, and records how
// each one ended and what it carried.
type gatedOutput struct {
	*transport.BaseOutput
	entered chan struct{}
	gate    chan struct{}

	finished atomic.Int32
	aborted  atomic.Int32

	mu      sync.Mutex
	written []byte
}

func newGatedOutput(p transport.Params) *gatedOutput {
	o := &gatedOutput{entered: make(chan struct{}, 64), gate: make(chan struct{})}
	o.BaseOutput = transport.NewBaseOutput("GatedOutput", p, o)
	return o
}

func (o *gatedOutput) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	select {
	case o.entered <- struct{}{}:
	default:
	}
	select {
	case <-o.gate:
	case <-ctx.Done():
		o.aborted.Add(1)
		return false, ctx.Err()
	}
	o.finished.Add(1)
	o.mu.Lock()
	o.written = append(o.written, f.AudioData().Audio...)
	o.mu.Unlock()
	return true, nil
}

func (o *gatedOutput) writtenAudio() []byte {
	o.mu.Lock()
	defer o.mu.Unlock()
	return slices.Clone(o.written)
}

// frameLog records the frames of the given kinds that reach the end of the
// pipeline, in order.
type frameLog struct {
	mu   sync.Mutex
	seen []frames.Frame
}

func (l *frameLog) add(f frames.Frame) {
	l.mu.Lock()
	l.seen = append(l.seen, f)
	l.mu.Unlock()
}

func (l *frameLog) botStopped() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := 0
	for _, f := range l.seen {
		if _, ok := f.(*frames.BotStoppedSpeakingFrame); ok {
			n++
		}
	}
	return n
}

func (l *frameLog) texts() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	var out []string
	for _, f := range l.seen {
		if tf, ok := f.(*frames.TTSTextFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

// runLogged runs out alone in a pipeline, logging every frame that reaches its
// end, and returns a stop function that cancels the run.
func runLogged(t *testing.T, out processor.Processor) (*pipeline.Worker, *frameLog, func()) {
	t.Helper()
	log := &frameLog{}
	task := pipeline.NewWorker(pipeline.New(out), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		log.add(f)
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, log, func() {
		task.Cancel(context.Background(), "")
		select {
		case <-runDone:
		case <-time.After(5 * time.Second):
			t.Error("the output task did not shut down")
		}
	}
}

func waitEntered(t *testing.T, entered <-chan struct{}) {
	t.Helper()
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("the transport was never asked to write")
	}
}

// TestBaseOutputUninterruptibleWriteFinishesAfterInterruption covers a write of
// uninterruptible audio that is in flight when the barge-in lands: it has to be
// allowed to finish rather than be cut short.
func TestBaseOutputUninterruptibleWriteFinishesAfterInterruption(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000 // 640-byte chunks

	o := newGatedOutput(params)
	task, _, stop := runLogged(t, o)
	defer stop()

	f := frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 640), 16000, 1)
	f.SetInterruptible(false)
	task.QueueFrame(f)
	waitEntered(t, o.entered)

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(100 * time.Millisecond)
	close(o.gate)
	time.Sleep(100 * time.Millisecond)

	if got := o.aborted.Load(); got != 0 {
		t.Errorf("the barge-in cut short %d uninterruptible writes, want 0", got)
	}
	if got := o.finished.Load(); got != 1 {
		t.Errorf("finished %d writes, want 1", got)
	}
}

// TestBaseOutputInterruptionWithoutMixerKeepsAudioLoop covers the send loop
// without a mixer. A barge-in drains its queue but does not stop it, so the
// write it lands on is left to finish instead of being canceled. Canceling that
// write would be the barge-in breaking the transport: a WebSocket, for one,
// closes the whole connection on a write abandoned halfway. And nothing about
// the barge-in may be reported as a failure.
func TestBaseOutputInterruptionWithoutMixerKeepsAudioLoop(t *testing.T) {
	logged := captureErrorLogs(t)

	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000 // 640-byte chunks

	o := newGatedOutput(params)
	task, _, stop := runLogged(t, o)
	defer stop()

	// Interruptible audio: the first chunk blocks in the write, the rest queues.
	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 640*4), 16000, 1))
	waitEntered(t, o.entered)

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(100 * time.Millisecond)
	if got := o.aborted.Load(); got != 0 {
		t.Fatalf("the barge-in canceled %d writes in flight, want 0", got)
	}
	close(o.gate)
	time.Sleep(100 * time.Millisecond)

	if got := o.finished.Load(); got != 1 {
		t.Errorf("finished %d writes, want 1: the audio queued behind the barge-in was still sent", got)
	}
	if got := logged(); len(got) > 0 {
		t.Errorf("the barge-in reported itself as a failure: %v", got)
	}

	// The loop is the same one, still running: the next turn goes out.
	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x02}, 640), 16000, 1))
	deadline := time.Now().Add(3 * time.Second)
	for o.finished.Load() < 2 {
		if time.Now().After(deadline) {
			t.Fatal("the send loop stopped running after the barge-in")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// TestBaseOutputInterruptionPreservesWaitingAndQueuedUninterruptibleText covers
// the frames waiting on the clock. A barge-in drops the interruptible ones,
// including the one the clock is waiting on, and keeps the uninterruptible
// ones, which are still delivered at their time.
func TestBaseOutputInterruptionPreservesWaitingAndQueuedUninterruptibleText(t *testing.T) {
	for _, interruptible := range []bool{true, false} {
		name := "first interruptible"
		if !interruptible {
			name = "first uninterruptible"
		}
		t.Run(name, func(t *testing.T) {
			params := transport.DefaultParams()
			params.AudioOutSampleRate = 16000

			o := newFakeOutput(params)
			task, log, stop := runLogged(t, o)
			defer stop()

			// Let the clock start before reading it.
			time.Sleep(50 * time.Millisecond)
			now := int64(o.Clock().Time())

			first := frames.NewTTSTextFrame("first", frames.TextTypeSentence)
			first.SetInterruptible(interruptible)
			first.SetPTS(now + int64(200*time.Millisecond))
			kept := frames.NewTTSTextFrame("kept", frames.TextTypeSentence)
			kept.SetInterruptible(false)
			kept.SetPTS(now + int64(240*time.Millisecond))
			dropped := frames.NewTTSTextFrame("dropped", frames.TextTypeSentence)
			dropped.SetPTS(now + int64(260*time.Millisecond))
			for _, f := range []frames.Frame{first, kept, dropped} {
				task.QueueFrame(f)
			}
			time.Sleep(50 * time.Millisecond)

			task.QueueFrame(frames.NewInterruptionFrame())
			time.Sleep(400 * time.Millisecond)

			want := []string{"kept"}
			if !interruptible {
				want = []string{"first", "kept"}
			}
			if got := log.texts(); !slices.Equal(got, want) {
				t.Errorf("delivered %v, want %v", got, want)
			}
		})
	}
}

// TestBaseOutputInterruptionKeepsUninterruptibleLeftoverAudioAndBotSpeaking
// covers the audio still waiting to fill a chunk when the barge-in lands. The
// uninterruptible part of it is kept and the bot keeps the floor; it plays once
// its speech ends, padded to a chunk, and only then does the bot stop speaking.
func TestBaseOutputInterruptionKeepsUninterruptibleLeftoverAudioAndBotSpeaking(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000
	const chunk = 640

	o := newGatedOutput(params)
	close(o.gate)
	task, log, stop := runLogged(t, o)
	defer stop()

	// A chunk and a half of uninterruptible audio, then a bit of normal audio:
	// one chunk plays and the rest waits in the buffer.
	uninterruptible := frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, chunk*3/2), 16000, 1)
	uninterruptible.SetInterruptible(false)
	normal := frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x02}, chunk/4), 16000, 1)
	task.QueueFrame(uninterruptible)
	task.QueueFrame(normal)
	time.Sleep(100 * time.Millisecond)

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(100 * time.Millisecond)

	// The normal audio is dropped, the uninterruptible audio is kept, and the bot
	// is still speaking.
	if got := log.botStopped(); got != 0 {
		t.Fatalf("the bot stopped speaking %d times with uninterruptible audio still to play", got)
	}

	// The uninterruptible speech ends: its leftover audio plays and the bot
	// stops speaking.
	task.QueueFrame(frames.NewTTSStoppedFrame())
	time.Sleep(200 * time.Millisecond)

	want := append(bytes.Repeat([]byte{0x01}, chunk*3/2), make([]byte, chunk/2)...)
	if got := o.writtenAudio(); !bytes.Equal(got, want) {
		t.Errorf("wrote %d bytes %x, want the uninterruptible audio padded to %d bytes",
			len(got), got, len(want))
	}
	if got := log.botStopped(); got == 0 {
		t.Error("the bot never stopped speaking once the uninterruptible audio ended")
	}
}

// TestBaseOutputInterruptionStopsBotSpeakingWithOnlyUninterruptibleControlFrames
// covers a queue left holding uninterruptible frames that carry no audio. They
// are kept, but there is nothing left for the bot to say, so it gives up the
// floor at once.
func TestBaseOutputInterruptionStopsBotSpeakingWithOnlyUninterruptibleControlFrames(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000

	o := newGatedOutput(params)
	defer close(o.gate)
	task, log, stop := runLogged(t, o)
	defer stop()

	// The bot is speaking, with more audio and a function call (an
	// uninterruptible control frame) queued behind it.
	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 640*2), 16000, 1))
	task.QueueFrame(frames.NewFunctionCallInProgressFrame("1", "get_weather", nil, false, ""))
	waitEntered(t, o.entered)
	time.Sleep(50 * time.Millisecond)

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(100 * time.Millisecond)

	if got := log.botStopped(); got == 0 {
		t.Error("the bot kept the floor with no audio left to play")
	}
}

// TestBaseOutputInterruptionDoesNotReplayAbortedAudio covers audio the barge-in
// cut short at a rate the transport converts. What the resampler was still
// holding of it is dropped, not carried into what the bot says next.
func TestBaseOutputInterruptionDoesNotReplayAbortedAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 8000
	params.AudioOutEndSilenceSecs = 0

	o := newFakeOutput(params)
	task, stop := startFakeOutput(t, o)
	defer stop()

	constant := func(level int16, samples int) []byte {
		pcm := make([]byte, samples*2)
		for i := range samples {
			binary.LittleEndian.PutUint16(pcm[i*2:], uint16(level))
		}
		return pcm
	}
	queue := func(pcm []byte) {
		const frameBytes = 640 * 2
		for off := 0; off < len(pcm); off += frameBytes {
			task.QueueFrame(frames.NewTTSAudioRawFrame(pcm[off:min(off+frameBytes, len(pcm))], 16000, 1))
		}
	}

	queue(constant(8000, 8000))
	time.Sleep(100 * time.Millisecond)
	task.QueueFrame(frames.NewInterruptionFrame())
	drainWrites(t, o, 200*time.Millisecond)

	// The next turn is entirely negative, so anything clearly positive in it is
	// left over from the turn that was cut short, whose level is 8000. The
	// filter rings either side of a step, by up to about a tenth of it, so a
	// sample that far above zero is the conversion, not the old turn.
	queue(constant(-8000, 8000))
	task.QueueFrame(frames.NewTTSStoppedFrame())
	after := drainStream(t, o, 300*time.Millisecond)

	if len(after) == 0 {
		t.Fatal("nothing was written for the next turn")
	}
	for i := 0; i+1 < len(after); i += 2 {
		if s := int16(binary.LittleEndian.Uint16(after[i:])); s > 2000 {
			t.Fatalf("sample %d of the next turn is %d: audio from the interrupted turn was replayed", i/2, s)
		}
	}
}
