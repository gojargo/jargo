package transport_test

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
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

// fakeInput is an input transport whose StartReading emits a fixed number of
// audio frames.
type fakeInput struct {
	*transport.BaseInput
	frames int

	// streaming records calls to StartAudioStreaming.
	streaming chan struct{}
}

func newFakeInput(p transport.Params, n int) *fakeInput {
	in := &fakeInput{frames: n, streaming: make(chan struct{}, 4)}
	in.BaseInput = transport.NewBaseInput("FakeInput", p, in)
	return in
}

func (in *fakeInput) StartAudioStreaming(context.Context) error {
	in.streaming <- struct{}{}
	return nil
}

func (in *fakeInput) StartReading(ctx context.Context) error {
	go func() {
		for range in.frames {
			f := frames.NewInputAudioRawFrame(make([]byte, 1920), 48000, 1)
			in.PushAudioFrame(ctx, f)
		}
	}()
	return nil
}

func TestBaseInputPushesAudioDownstream(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000

	var got atomic.Int32
	done := make(chan struct{})
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	}

	in := newFakeInput(params, 3)
	task := pipeline.NewWorker(pipeline.New(in), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.InputAudioRawFrame); ok {
			if got.Add(1) == 3 {
				close(done)
			}
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatalf("only %d of 3 audio frames reached downstream", got.Load())
	}

	task.StopWhenDone()
	<-runDone
}

// TestBaseInputRoutesUpstreamAudioThroughFilter covers audio pushed into the
// input from elsewhere in the pipeline (by the RTVI processor, say). It has to go
// down the same filtering path as audio read from the transport itself, rather
// than being forwarded on untouched as a plain frame.
func TestBaseInputRoutesUpstreamAudioThroughFilter(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000
	filter := &fakeFilter{}
	params.AudioInFilter = filter

	got := make(chan []byte, 4)
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	}

	// No frames of its own: the only audio is the one queued below.
	in := newFakeInput(params, 0)
	task := pipeline.NewWorker(pipeline.New(in), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, fr frames.Frame) {
		if af, ok := fr.(*frames.InputAudioRawFrame); ok {
			got <- af.Audio
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{1, 2, 3, 4}, 48000, 1))

	select {
	case audio := <-got:
		// fakeFilter increments every byte, so filtered audio proves the frame
		// went through the audio path instead of straight downstream.
		if !bytes.Equal(audio, []byte{2, 3, 4, 5}) {
			t.Errorf("audio = %v, want %v (upstream audio bypassed the filter)", audio, []byte{2, 3, 4, 5})
		}
	case <-time.After(3 * time.Second):
		t.Fatal("upstream audio never reached downstream")
	}

	task.StopWhenDone()
	<-runDone
}

// TestBaseInputStartAudioStreamingFrame checks that the frame reaches the
// transport's streaming hook, so starting the stream stays frame-based rather
// than a direct call across processors.
func TestBaseInputStartAudioStreamingFrame(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000

	in := newFakeInput(params, 0)
	task := pipeline.NewWorker(pipeline.New(in), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewInputTransportStartAudioStreamingFrame())

	select {
	case <-in.streaming:
	case <-time.After(3 * time.Second):
		t.Fatal("InputTransportStartAudioStreamingFrame did not reach StartAudioStreaming")
	}

	task.StopWhenDone()
	<-runDone
}

// TestBaseInputStopFramePausesAudio covers the pause: a StopFrame stops the
// pipeline while leaving its processors running, so the transport keeps reading
// but what it receives must no longer reach the pipeline. Upstream has no test
// for this; it covers the ported behavior.
func TestBaseInputStopFramePausesAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000

	// The filter counter is what makes this observable: a StopFrame ends the
	// task, so nothing reaches the end of the pipeline afterwards either way,
	// but the filter only runs for audio the input actually accepted.
	filter := &fakeFilter{}
	params.AudioInFilter = filter

	in := newFakeInput(params, 0)
	task := pipeline.NewWorker(pipeline.New(in), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	ctx := context.Background()
	audio := func() *frames.InputAudioRawFrame {
		return frames.NewInputAudioRawFrame(make([]byte, 320), 48000, 1)
	}

	// Settle first, so the StartFrame has been processed and the input is
	// actually accepting audio.
	if err := task.Flush(ctx); err != nil {
		t.Fatalf("flush after start: %v", err)
	}

	in.PushAudioFrame(ctx, audio())
	deadline := time.Now().Add(3 * time.Second)
	for filter.filtered.Load() == 0 {
		if time.Now().After(deadline) {
			t.Fatal("audio was not filtered before the stop")
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Stopping the pipeline pauses the input: the transport keeps reading, but
	// what it receives no longer enters the pipeline. The StopFrame is pushed on
	// before the pause runs, so give the pause a moment to land.
	task.QueueFrame(frames.NewStopFrame())
	<-runDone
	time.Sleep(50 * time.Millisecond)

	before := filter.filtered.Load()
	in.PushAudioFrame(ctx, audio())
	time.Sleep(300 * time.Millisecond)
	if got := filter.filtered.Load(); got != before {
		t.Errorf("filter ran %d more times: audio was not dropped while paused", got-before)
	}
}

// TestBaseInputRoutesFilterSettings checks that a filter settings frame reaches
// the filter, so it can be retuned at runtime without being torn down.
func TestBaseInputRoutesFilterSettings(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000
	filter := &fakeFilter{controls: make(chan *frames.FilterUpdateSettingsFrame, 4)}
	params.AudioInFilter = filter

	in := newFakeInput(params, 0)
	task := pipeline.NewWorker(pipeline.New(in), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewFilterUpdateSettingsFrame(map[string]any{"level": 3}))

	select {
	case settings := <-filter.controls:
		if settings.Settings["level"] != 3 {
			t.Errorf("settings = %v, want level 3", settings.Settings)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("FilterUpdateSettingsFrame never reached the filter")
	}

	task.StopWhenDone()
	<-runDone
}

// fakeOutput is an output transport that records the audio chunks it is asked
// to send.
type fakeOutput struct {
	*transport.BaseOutput
	writes chan []byte
}

func newFakeOutput(p transport.Params) *fakeOutput {
	o := &fakeOutput{writes: make(chan []byte, 64)}
	o.BaseOutput = transport.NewBaseOutput("FakeOutput", p, o)
	return o
}

func (o *fakeOutput) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	pcm := f.AudioData().Audio
	cp := make([]byte, len(pcm))
	copy(cp, pcm)
	// Recording the write must not outlast the send loop it runs on: a test
	// that stops reading leaves this channel full, and a send that ignored ctx
	// would hold the shutdown open the way a real transport never may.
	select {
	case o.writes <- cp:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// fakeFilter is an AudioFilter that increments every byte and records its
// lifecycle calls, so a test can assert the base wired it in.
type fakeFilter struct {
	started  atomic.Int32
	stopped  atomic.Int32
	rate     atomic.Int32
	filtered atomic.Int32

	// controls records the settings frames routed to the filter at runtime.
	controls chan *frames.FilterUpdateSettingsFrame
}

func (f *fakeFilter) ProcessFrame(_ context.Context, fr frames.FilterControlFrame) error {
	if settings, ok := fr.(*frames.FilterUpdateSettingsFrame); ok && f.controls != nil {
		f.controls <- settings
	}
	return nil
}

func (f *fakeFilter) Start(_ context.Context, sampleRate int) error {
	f.rate.Store(int32(sampleRate))
	f.started.Add(1)
	return nil
}

func (f *fakeFilter) Stop(context.Context) error {
	f.stopped.Add(1)
	return nil
}

func (f *fakeFilter) Filter(_ context.Context, pcm []byte) ([]byte, error) {
	f.filtered.Add(1)
	out := make([]byte, len(pcm))
	for i, b := range pcm {
		out[i] = b + 1
	}
	return out, nil
}

func TestBaseInputAppliesFilter(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000
	filter := &fakeFilter{}
	params.AudioInFilter = filter

	var got atomic.Int32
	done := make(chan struct{})
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	}

	in := newFakeInput(params, 3)
	task := pipeline.NewWorker(pipeline.New(in), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, fr frames.Frame) {
		af, ok := fr.(*frames.InputAudioRawFrame)
		if !ok {
			return
		}
		// The fake input emits all-zero audio; the filter increments every
		// byte, so every downstream sample must be 1.
		for _, b := range af.Audio {
			if b != 1 {
				t.Errorf("downstream byte = %d, want 1 (filtered)", b)
				break
			}
		}
		if got.Add(1) == 3 {
			close(done)
		}
	})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatalf("only %d of 3 filtered frames reached downstream", got.Load())
	}

	task.StopWhenDone()
	<-runDone

	if filter.started.Load() != 1 {
		t.Errorf("Start called %d times, want 1", filter.started.Load())
	}
	if filter.rate.Load() != 48000 {
		t.Errorf("Start sample rate = %d, want 48000", filter.rate.Load())
	}
	if filter.filtered.Load() < 3 {
		t.Errorf("Filter called %d times, want >= 3", filter.filtered.Load())
	}
	if filter.stopped.Load() != 1 {
		t.Errorf("Stop called %d times, want 1", filter.stopped.Load())
	}
}

func TestBaseOutputChunksAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // chunk size = 480 samples/10ms * 2 * 2 = 1920 bytes

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})

	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Two chunks worth of audio in a single frame.
	task.QueueFrame(frames.NewOutputAudioRawFrame(make([]byte, 3840), 48000, 1))

	for i := range 2 {
		select {
		case chunk := <-o.writes:
			if len(chunk) != 1920 {
				t.Fatalf("chunk %d = %d bytes, want 1920", i, len(chunk))
			}
		case <-time.After(3 * time.Second):
			t.Fatalf("timed out waiting for chunk %d", i)
		}
	}

	task.StopWhenDone()
	<-runDone
}

// pacedOutput records how many chunks it was asked to write, sleeping on each so
// a drain must wait for the queued backlog to finish.
type pacedOutput struct {
	*transport.BaseOutput
	writes atomic.Int32
}

func newPacedOutput(p transport.Params) *pacedOutput {
	o := &pacedOutput{}
	o.BaseOutput = transport.NewBaseOutput("PacedOutput", p, o)
	return o
}

func (o *pacedOutput) WriteAudio(context.Context, frames.OutputAudioFrame) (bool, error) {
	time.Sleep(3 * time.Millisecond) // stand in for real-time pacing
	o.writes.Add(1)
	return true, nil
}

// TestBaseOutputEndFrameDrainsAudio is the regression test for the farewell
// cutoff: a graceful EndFrame must let the queued audio finish before the
// pipeline stops, unlike a CancelFrame which stops at once.
func TestBaseOutputEndFrameDrainsAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks
	// This is about the queued audio surviving the end, so leave the closing
	// silence out of the count; TestBaseOutputSendsClosingSilence covers that.
	params.AudioOutEndSilenceSecs = 0

	o := newPacedOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	const chunks = 15
	task.QueueFrame(frames.NewOutputAudioRawFrame(make([]byte, 1920*chunks), 48000, 1))
	task.StopWhenDone() // EndFrame right behind the audio

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not end")
	}
	if got := o.writes.Load(); got != chunks {
		t.Fatalf("wrote %d/%d chunks — EndFrame cut off queued audio", got, chunks)
	}
}

// TestBaseOutputFlushesTailOnTurnEnd is the regression test for the turn-end
// cutoff: audio that does not fill a whole chunk must still play out when the
// bot's turn ends, padded to a full chunk so downstream whole-frame encoders
// emit it instead of stranding it until the next turn.
func TestBaseOutputFlushesTailOnTurnEnd(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// One full chunk plus a half-chunk tail, marked with 0x01 so the padding is
	// distinguishable from the audio.
	audio := bytes.Repeat([]byte{0x01}, 1920+960)
	task.QueueFrame(frames.NewTTSAudioRawFrame(audio, 48000, 1))

	// First write is the full chunk, unpadded.
	first := recvWrite(t, o)
	if len(first) != 1920 {
		t.Fatalf("first write = %d bytes, want 1920", len(first))
	}

	// The turn ends: the TTS stop flushes the padded tail ahead of itself, so
	// the last of the audio plays before the stop is handled.
	task.QueueFrame(frames.NewTTSStoppedFrame())
	tail := recvWrite(t, o)
	if len(tail) != 1920 {
		t.Fatalf("tail write = %d bytes, want 1920 (padded)", len(tail))
	}
	if !bytes.Equal(tail[:960], bytes.Repeat([]byte{0x01}, 960)) {
		t.Errorf("tail audio was not preserved")
	}
	if !bytes.Equal(tail[960:], make([]byte, 960)) {
		t.Errorf("tail was not zero-padded with silence")
	}

	task.StopWhenDone()
	<-runDone
}

// TestBaseOutputShortTurnSignalsBotSpeaking covers a turn whose whole audio
// never fills one chunk. It has to flush as the frame type it was buffered from,
// so the bot-speaking bookkeeping (which dispatches on that type) still fires for
// it instead of skipping the turn's start and stop events.
func TestBaseOutputShortTurnSignalsBotSpeaking(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	var mu sync.Mutex
	var down, up []frames.Frame
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		ReachedUpstreamFilter:   pipeline.AnyFrame,
	}

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		down = append(down, f)
		mu.Unlock()
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		up = append(up, f)
		mu.Unlock()
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Audio shorter than a single chunk is never queued for playback on its own;
	// it only ever sits in the buffer until the turn ends.
	partial := bytes.Repeat([]byte{0x03, 0x04}, 1920/4)
	task.QueueFrame(frames.NewTTSAudioRawFrame(partial, 48000, 1))
	task.QueueFrame(frames.NewTTSStoppedFrame())

	got := recvWrite(t, o)
	if len(got) != 1920 {
		t.Fatalf("flushed write = %d bytes, want 1920 (padded)", len(got))
	}
	if !bytes.Equal(got[:len(partial)], partial) {
		t.Errorf("flushed audio was not preserved")
	}
	if !bytes.Equal(got[len(partial):], make([]byte, 1920-len(partial))) {
		t.Errorf("flush was not zero-padded with silence")
	}

	task.StopWhenDone()
	<-runDone

	mu.Lock()
	defer mu.Unlock()

	// The flushed chunk must stay a TTSAudioRawFrame rather than being flattened
	// to a plain OutputAudioRawFrame, or the bookkeeping never recognizes it.
	var sawTTSAudio bool
	for _, f := range down {
		if _, ok := f.(*frames.TTSAudioRawFrame); ok {
			sawTTSAudio = true
		}
	}
	if !sawTTSAudio {
		t.Errorf("flushed chunk did not reach downstream as a TTSAudioRawFrame")
	}

	// Both speaking events must have fired even though no full chunk was queued,
	// and both must be broadcast in each direction.
	for _, dir := range []struct {
		name string
		got  []frames.Frame
	}{{"downstream", down}, {"upstream", up}} {
		var started, stopped bool
		for _, f := range dir.got {
			switch f.(type) {
			case *frames.BotStartedSpeakingFrame:
				started = true
			case *frames.BotStoppedSpeakingFrame:
				stopped = true
			}
		}
		if !started {
			t.Errorf("no BotStartedSpeakingFrame reached %s", dir.name)
		}
		if !stopped {
			t.Errorf("no BotStoppedSpeakingFrame reached %s", dir.name)
		}
	}
}

// TestBaseOutputSendsClosingSilence covers the run of silence sent after the
// last of the audio when the pipeline ends, so the closing words are not clipped
// by whatever closes on top of them.
func TestBaseOutputSendsClosingSilence(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks
	params.AudioOutEndSilenceSecs = 1

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 1920), 48000, 1))
	task.StopWhenDone()

	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("pipeline did not end")
	}
	close(o.writes)

	var last []byte
	writes := 0
	for w := range o.writes {
		last = w
		writes++
	}
	if writes < 2 {
		t.Fatalf("saw %d writes, want the turn's audio and the closing silence after it", writes)
	}

	// One second at 48 kHz of 16-bit samples.
	want := 48000 * 2
	if len(last) != want {
		t.Fatalf("closing write = %d bytes, want %d", len(last), want)
	}
	if !bytes.Equal(last, make([]byte, want)) {
		t.Error("the closing write was not silence")
	}
}

// uninterruptibleMarkerFrame is a test-only control frame that must survive an
// interruption, used to check it is still delivered when the audio around it is
// dropped.
type uninterruptibleMarkerFrame struct {
	frames.BaseControlFrame
	frames.UninterruptibleMixin
}

func newUninterruptibleMarkerFrame() *uninterruptibleMarkerFrame {
	return &uninterruptibleMarkerFrame{
		BaseControlFrame: frames.NewBaseControlFrame("UninterruptibleMarkerFrame"),
	}
}

// blockingOutput holds each write until the test releases it, so a test can
// leave frames queued behind an in-flight write.
type blockingOutput struct {
	*transport.BaseOutput
	entered chan struct{}
	release chan struct{}
	writes  atomic.Int32
}

func newBlockingOutput(p transport.Params) *blockingOutput {
	o := &blockingOutput{entered: make(chan struct{}, 8), release: make(chan struct{})}
	o.BaseOutput = transport.NewBaseOutput("BlockingOutput", p, o)
	return o
}

func (o *blockingOutput) WriteAudio(ctx context.Context, _ frames.OutputAudioFrame) (bool, error) {
	o.writes.Add(1)
	select {
	case o.entered <- struct{}{}:
	default:
	}
	select {
	case <-o.release:
		return true, nil
	case <-ctx.Done():
		// What a real transport reports when the send loop is stopped under it,
		// which is how a barge-in cuts short the write it interrupts.
		return false, ctx.Err()
	}
}

// TestBaseOutputKeepsUninterruptibleFramesThroughBargeIn covers the frames that
// must be delivered even when the bot is cut off. A barge-in drops the audio
// still queued, but a frame marked uninterruptible stays and is still forwarded.
// Upstream has no test for this at the output; the marker-frame technique is
// borrowed from its ordering tests.
func TestBaseOutputKeepsUninterruptibleFramesThroughBargeIn(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	seen := make(chan struct{}, 4)
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	}

	o := newBlockingOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*uninterruptibleMarkerFrame); ok {
			seen <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Five chunks: the first is taken up by the loop and blocks in the write,
	// leaving four queued behind it.
	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 1920*5), 48000, 1))
	select {
	case <-o.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("the transport was never asked to write")
	}

	// Queued behind the audio that is about to be dropped. It has to really be
	// on the queue when the barge-in lands, or the queue holds nothing that must
	// survive and this tests the wrong branch. An interruption is a system frame
	// and would overtake it on the way here, so the two are separated by a
	// settle. The flush probe cannot be used for that here: it is queued behind
	// the audio too, and this test is holding that audio up on purpose.
	task.QueueFrame(newUninterruptibleMarkerFrame())
	time.Sleep(200 * time.Millisecond)

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(200 * time.Millisecond)

	close(o.release)

	select {
	case <-seen:
	case <-time.After(3 * time.Second):
		t.Fatal("the uninterruptible frame was dropped by the barge-in")
	}

	// The audio behind the in-flight write belonged to the turn that was cut
	// off, so it must not have been sent.
	if got := o.writes.Load(); got != 1 {
		t.Errorf("wrote %d chunks, want 1: audio queued before the barge-in was still sent", got)
	}

	task.Cancel(t.Context(), "")
	<-runDone
}

// markerFrame is a plain downstream data frame carrying no audio and no
// timestamp, used to check where it lands relative to the audio around it.
type markerFrame struct {
	frames.BaseDataFrame
}

func newMarkerFrame() *markerFrame {
	return &markerFrame{BaseDataFrame: frames.NewBaseDataFrame("MarkerFrame")}
}

// TestBaseOutputOrdersSyncFramesWithAudio covers a frame that carries neither
// audio nor a timestamp. It has to wait behind the audio queued before it, so it
// is forwarded in step with what is being heard, instead of overtaking it by
// however much audio is buffered. Upstream has no test for this at the output;
// the marker-frame technique is borrowed from its TTS ordering tests.
func TestBaseOutputOrdersSyncFramesWithAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	var mu sync.Mutex
	var order []string
	seen := make(chan struct{}, 8)
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	}

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		switch f.(type) {
		case *frames.TTSAudioRawFrame:
			mu.Lock()
			order = append(order, "audio")
			mu.Unlock()
		case *markerFrame:
			mu.Lock()
			order = append(order, "marker")
			mu.Unlock()
			seen <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Drain the transport's writes so the audio loop keeps pacing.
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		for range o.writes {
		}
	}()

	// Three whole chunks of audio, then a frame with nothing to play.
	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 1920*3), 48000, 1))
	task.QueueFrame(newMarkerFrame())

	select {
	case <-seen:
	case <-time.After(3 * time.Second):
		t.Fatal("the marker frame never reached downstream")
	}

	mu.Lock()
	got := append([]string(nil), order...)
	mu.Unlock()

	if len(got) == 0 || got[len(got)-1] != "marker" {
		t.Fatalf("order = %v, want the marker last: it overtook the audio it was queued behind", got)
	}
	audio := 0
	for _, s := range got {
		if s == "audio" {
			audio++
		}
	}
	if audio != 3 {
		t.Errorf("saw %d audio chunks before the marker, want 3", audio)
	}

	task.StopWhenDone()
	<-runDone
}

// readyOutput records the order of its lifecycle callbacks, so a test can check
// the media path is opened before anything is written to it.
type readyOutput struct {
	*transport.BaseOutput
	mu    sync.Mutex
	calls []string
}

func newReadyOutput(p transport.Params) *readyOutput {
	o := &readyOutput{}
	o.BaseOutput = transport.NewBaseOutput("ReadyOutput", p, o)
	return o
}

func (o *readyOutput) record(name string) {
	o.mu.Lock()
	o.calls = append(o.calls, name)
	o.mu.Unlock()
}

func (o *readyOutput) StartWriting(context.Context) error {
	o.record("StartWriting")
	return nil
}

func (o *readyOutput) WriteAudio(context.Context, frames.OutputAudioFrame) (bool, error) {
	o.record("WriteAudio")
	return true, nil
}

// TestBaseOutputReportsReady covers the readiness handshake: the transport's own
// media path has to be open before the output queues anything for it, and once it
// is the pipeline is told upstream, so a producer that must not speak into a
// connection that is not up yet can wait for it.
func TestBaseOutputReportsReady(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000

	ready := make(chan struct{}, 4)
	taskParams := pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	}

	o := newReadyOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.OutputTransportReadyFrame); ok {
			ready <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case <-ready:
	case <-time.After(3 * time.Second):
		t.Fatal("no OutputTransportReadyFrame reached upstream")
	}

	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 1920), 48000, 1))
	if err := task.Flush(context.Background()); err != nil {
		t.Fatalf("flush: %v", err)
	}

	task.StopWhenDone()
	<-runDone

	o.mu.Lock()
	defer o.mu.Unlock()
	if len(o.calls) == 0 || o.calls[0] != "StartWriting" {
		t.Fatalf("calls = %v, want StartWriting first: audio was queued before the media path was open", o.calls)
	}
}

// addressedWrite is one chunk the transport was asked to send, with the outgoing
// stream it was addressed to.
type addressedWrite struct {
	destination string
	pcm         []byte
}

// multiDestOutput records which destination each chunk was written to, and which
// destinations it was asked to open.
type multiDestOutput struct {
	*transport.BaseOutput
	writes     chan addressedWrite
	registered chan string
}

func newMultiDestOutput(p transport.Params) *multiDestOutput {
	o := &multiDestOutput{
		writes:     make(chan addressedWrite, 64),
		registered: make(chan string, 8),
	}
	o.BaseOutput = transport.NewBaseOutput("MultiDestOutput", p, o)
	return o
}

func (o *multiDestOutput) RegisterAudioDestination(_ context.Context, destination string) error {
	o.registered <- destination
	return nil
}

func (o *multiDestOutput) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	pcm := f.AudioData().Audio
	cp := make([]byte, len(pcm))
	copy(cp, pcm)
	select {
	case o.writes <- addressedWrite{destination: f.Base().TransportDestination(), pcm: cp}:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	}
}

// TestBaseOutputRoutesByDestination covers a transport carrying more than one
// outgoing stream: each frame has to reach the stream it names, and reach the
// transport still carrying that name, or the transport cannot tell the streams
// apart when it sends.
func TestBaseOutputRoutesByDestination(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000
	params.AudioOutDestinations = []string{"side"}

	o := newMultiDestOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case got := <-o.registered:
		if got != "side" {
			t.Errorf("registered destination %q, want \"side\"", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the named destination was never registered with the transport")
	}

	// One full chunk to the default stream, one to the named stream.
	def := frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 1920), 48000, 1)
	side := frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x02}, 1920), 48000, 1)
	side.Base().SetTransportDestination("side")
	task.QueueFrame(def)
	task.QueueFrame(side)

	seen := map[string]byte{}
	for range 2 {
		select {
		case w := <-o.writes:
			if len(w.pcm) == 0 {
				t.Fatalf("empty write for destination %q", w.destination)
			}
			seen[w.destination] = w.pcm[0]
		case <-time.After(3 * time.Second):
			t.Fatalf("only saw writes for %v", seen)
		}
	}

	if seen[""] != 0x01 {
		t.Errorf("default stream carried %#x, want 0x01", seen[""])
	}
	if seen["side"] != 0x02 {
		t.Errorf("named stream carried %#x, want 0x02: audio reached the wrong stream", seen["side"])
	}

	task.Cancel(t.Context(), "")
	<-runDone
}

// passthroughMixer returns the audio it is given, so the only thing a test can
// read from it is whether the output asked it to mix at all.
type passthroughMixer struct{}

func (passthroughMixer) Start(context.Context, int) error { return nil }
func (passthroughMixer) Stop(context.Context) error       { return nil }

func (passthroughMixer) Mix(_ context.Context, pcm []byte) ([]byte, error) { return pcm, nil }

func (passthroughMixer) ProcessFrame(context.Context, frames.MixerControlFrame) error { return nil }

// TestBaseOutputMixerFillsGaps covers auxiliary audio between turns: with a
// mixer configured, mixer-only audio has to keep flowing while nothing is
// queued, and keep flowing across an interruption. Otherwise background audio
// would only ever be audible underneath the bot's speech and would cut out the
// moment it stopped.
func TestBaseOutputMixerFillsGaps(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000
	params.AudioOutMixer = passthroughMixer{}

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Nothing is queued at any point: every write below is the mixer's own audio.
	for i := range 3 {
		if got := recvWrite(t, o); len(got) == 0 {
			t.Fatalf("mixer write %d was empty", i)
		}
	}

	task.QueueFrame(frames.NewInterruptionFrame())

	// The mixer keeps playing across a barge-in: it is not the bot's turn that
	// was interrupted, it is the background.
	for i := range 3 {
		if got := recvWrite(t, o); len(got) == 0 {
			t.Fatalf("mixer write %d after the interruption was empty", i)
		}
	}

	task.Cancel(t.Context(), "")
	<-runDone
}

// botAudioByte marks a chunk as the bot's own speech. The mixer's gap-filling
// audio is silence, so a chunk carrying this byte can only have come from the
// turn that was queued.
const botAudioByte = 0x01

// mixerBargeInOutput holds its first write until the test releases it, leaving
// the send loop stuck in flight so audio can be queued behind it, and records
// whether any of the bot's own audio was written once it resumes.
type mixerBargeInOutput struct {
	*transport.BaseOutput
	entered chan struct{}
	release chan struct{}
	hold    sync.Once

	mu        sync.Mutex
	writes    int
	botChunks int
}

func newMixerBargeInOutput(p transport.Params) *mixerBargeInOutput {
	o := &mixerBargeInOutput{entered: make(chan struct{}, 1), release: make(chan struct{})}
	o.BaseOutput = transport.NewBaseOutput("MixerBargeInOutput", p, o)
	return o
}

func (o *mixerBargeInOutput) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	o.hold.Do(func() {
		select {
		case o.entered <- struct{}{}:
		default:
		}
		select {
		case <-o.release:
		case <-ctx.Done():
		}
	})
	if ctx.Err() != nil {
		return false, ctx.Err()
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	o.writes++
	if bytes.IndexByte(f.AudioData().Audio, botAudioByte) >= 0 {
		o.botChunks++
	}
	return true, nil
}

func (o *mixerBargeInOutput) counts() (writes, botChunks int) {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.writes, o.botChunks
}

// TestBaseOutputMixerBargeInStillDropsQueuedBotAudio covers the half of the
// mixer branch that TestBaseOutputMixerFillsGaps does not. A mixer keeps the
// send loop running through a barge-in, since canceling it would silence the
// background audio too, but the bot's own audio queued behind the interruption
// still belongs to the turn that was cut off and must never be played.
func TestBaseOutputMixerBargeInStillDropsQueuedBotAudio(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks
	params.AudioOutMixer = passthroughMixer{}

	o := newMixerBargeInOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// The loop blocks in its first gap-filling write, so nothing queued behind
	// it can be consumed before the barge-in lands.
	select {
	case <-o.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("the transport was never asked to write")
	}

	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{botAudioByte}, 1920*4), 48000, 1))
	time.Sleep(200 * time.Millisecond)

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(200 * time.Millisecond)

	close(o.release)

	// Give the loop time to drain anything the barge-in failed to drop, and to
	// show it is still running.
	time.Sleep(300 * time.Millisecond)

	writes, botChunks := o.counts()
	if botChunks != 0 {
		t.Errorf("the barge-in played %d chunks of the interrupted turn, want 0", botChunks)
	}
	if writes == 0 {
		t.Error("the mixer stopped playing across the barge-in: the loop was torn down with the turn")
	}

	task.Cancel(t.Context(), "")
	<-runDone
}

// waitForFullWrites blocks until nothing more can be recorded, which is what
// leaves the send loop stuck inside a write.
func waitForFullWrites(t *testing.T, o *fakeOutput) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for len(o.writes) < cap(o.writes) {
		select {
		case <-time.After(2 * time.Millisecond):
		case <-deadline:
			t.Fatalf("the recording buffer never filled: %d of %d", len(o.writes), cap(o.writes))
		}
	}
	// Full is not yet blocked; give the loop the moment it needs to enter the
	// write that has nowhere to go.
	time.Sleep(20 * time.Millisecond)
}

// TestBaseOutputShutdownIsNotHeldByAWriteInFlight is the regression test for a
// shutdown that never finished. With a mixer filling the gaps, the send loop
// writes whether or not anything is queued, so at shutdown there is nearly
// always a write in flight. Stopping cancels the loop's context and then waits
// for the loop to finish, so cancellation has to reach inside that write: a
// transport that blocks past it holds the pipeline open for good.
func TestBaseOutputShutdownIsNotHeldByAWriteInFlight(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000
	params.AudioOutMixer = passthroughMixer{}

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// Nobody reads the writes, so the mixer's gap-filling audio fills the
	// recording buffer and the loop ends up blocked mid-write.
	waitForFullWrites(t, o)

	task.Cancel(t.Context(), "")
	select {
	case <-runDone:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline never stopped: a write in flight held the shutdown open")
	}
}

// TestBaseOutputInterruptionDiscardsTail checks the other half of the contract:
// a barge-in must drop the pending sub-chunk tail, not play it after the fact.
func TestBaseOutputInterruptionDiscardsTail(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	audio := bytes.Repeat([]byte{0x01}, 1920+960)
	task.QueueFrame(frames.NewOutputAudioRawFrame(audio, 48000, 1))

	// Wait for the full chunk so the tail is buffered, then barge in before the
	// debounce can flush it.
	if len(recvWrite(t, o)) != 1920 {
		t.Fatal("did not receive the full chunk")
	}
	task.QueueFrame(frames.NewInterruptionFrame())

	// The tail must never be written — not even after the debounce window.
	select {
	case extra := <-o.writes:
		t.Fatalf("interruption failed to discard tail: got a %d-byte write", len(extra))
	case <-time.After(2 * botStopDebounceTest):
	}

	task.StopWhenDone()
	<-runDone
}

// botStopDebounceTest mirrors the base output's internal bot-stop debounce so
// the interruption test waits past a would-be tail flush.
const botStopDebounceTest = 250 * time.Millisecond

// recvWrite returns the next chunk the output was asked to write, failing if
// none arrives in time.
func recvWrite(t *testing.T, o *fakeOutput) []byte {
	t.Helper()
	select {
	case chunk := <-o.writes:
		return chunk
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for an audio write")
		return nil
	}
}

// TestBaseOutputForwardsWordFramesInOrder checks that a word-aligned
// TTSTextFrame is forwarded downstream in step with the audio it was queued
// between — after the preceding audio chunk, before the following one.
func TestBaseOutputForwardsWordFramesInOrder(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	// Audio and timed frames now reach downstream from separate goroutines, so
	// the recorded sequence needs its own lock.
	var mu struct {
		sync.Mutex
		seq []string
	}
	wordSeen := make(chan struct{}, 1)
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	}

	o := newFakeOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch fr := f.(type) {
		case *frames.OutputAudioRawFrame:
			mu.seq = append(mu.seq, "audio")
		case *frames.TTSTextFrame:
			mu.seq = append(mu.seq, "word:"+fr.Text)
			select {
			case wordSeen <- struct{}{}:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewOutputAudioRawFrame(make([]byte, 1920), 48000, 1))
	// The word carries the moment it is spoken, which is what holds it back
	// until the audio around it has gone out.
	word := frames.NewTTSTextFrame("hello")
	word.SetPTS(int64(20 * time.Millisecond))
	task.QueueFrame(word)
	task.QueueFrame(frames.NewOutputAudioRawFrame(make([]byte, 1920), 48000, 1))

	select {
	case <-wordSeen:
	case <-time.After(3 * time.Second):
		t.Fatal("word frame never reached downstream")
	}
	task.StopWhenDone()
	<-runDone

	// The word frame must appear, and not before the audio queued ahead of it.
	mu.Lock()
	defer mu.Unlock()
	idxWord, idxFirstAudio := -1, -1
	for i, s := range mu.seq {
		if s == "audio" && idxFirstAudio < 0 {
			idxFirstAudio = i
		}
		if s == "word:hello" {
			idxWord = i
		}
	}
	if idxWord < 0 {
		t.Fatalf("word frame missing from downstream sequence %v", mu.seq)
	}
	if idxFirstAudio < 0 || idxWord < idxFirstAudio {
		t.Fatalf("word frame at %d preceded its audio at %d (seq %v)", idxWord, idxFirstAudio, mu.seq)
	}
}

// TestBaseOutputInterruptionDropsUnplayedWordFrames checks that a barge-in drops
// word-aligned text whose moment has not arrived, so the assistant context never
// records words that were not actually spoken.
func TestBaseOutputInterruptionDropsUnplayedWordFrames(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	lateSeen := make(chan string, 4)
	taskParams := pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	}

	o := newPacedOutput(params) // ~3ms per chunk write
	task := pipeline.NewWorker(pipeline.New(o), taskParams)
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if fr, ok := f.(*frames.TTSTextFrame); ok {
			lateSeen <- fr.Text
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// A long run of audio with a word frame queued behind it: the word's audio is
	// nowhere near played when the interruption arrives.
	task.QueueFrame(frames.NewOutputAudioRawFrame(make([]byte, 1920*40), 48000, 1))
	unspoken := frames.NewTTSTextFrame("unspoken")
	unspoken.SetPTS(int64(700 * time.Millisecond)) // far beyond the interruption
	task.QueueFrame(unspoken)
	time.Sleep(10 * time.Millisecond) // a couple of chunks play
	task.QueueFrame(frames.NewInterruptionFrame())

	select {
	case got := <-lateSeen:
		t.Fatalf("interruption failed to drop unplayed word frame: got %q", got)
	case <-time.After(300 * time.Millisecond):
	}

	task.StopWhenDone()
	<-runDone
}

// ender pushes an EndWorkerFrame upstream once the pipeline has started.
type ender struct {
	*processor.Base
}

func newEnder() *ender {
	e := &ender{}
	e.Base = processor.New("Ender", e)
	return e
}

func (e *ender) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		if err := e.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return e.PushFrame(ctx, frames.NewEndWorkerFrame(), processor.Upstream)
	}
	return e.PushFrame(ctx, f, dir)
}

// TestEndWorkerFrameEndsTask checks a processor can end the run gracefully by
// pushing an EndWorkerFrame upstream — the Task turns it into an EndFrame.
func TestEndWorkerFrameEndsTask(t *testing.T) {
	task := pipeline.NewWorker(pipeline.New(newEnder()), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	select {
	case err := <-runDone:
		if err != nil {
			t.Fatalf("run: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("task did not end on EndWorkerFrame")
	}
}

// errClosedForTest stands in for a connection that has gone away.
//
//nolint:gochecknoglobals // sentinel error
var errClosedForTest = errors.New("connection closed")

// failingMessageOutput is an output whose message sending always fails, the way
// a closed connection does.
type failingMessageOutput struct {
	*transport.BaseOutput
}

func newFailingMessageOutput(params transport.Params) *failingMessageOutput {
	o := &failingMessageOutput{}
	o.BaseOutput = transport.NewBaseOutput("FailingOutput", params, o)
	return o
}

func (o *failingMessageOutput) SendMessage(context.Context, frames.OutputTransportMessage) error {
	return errClosedForTest
}

func (o *failingMessageOutput) WriteAudio(context.Context, frames.OutputAudioFrame) (bool, error) {
	return true, nil
}

// A message that cannot be sent must not become an error frame. Anything that
// reports errors to the client would turn that into another message to send,
// and with the connection down the pipeline would feed itself errors without
// end.
func TestBaseOutputDoesNotEscalateSendFailures(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000

	errs := make(chan string, 8)
	o := newFailingMessageOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		if fr, ok := f.(*frames.ErrorFrame); ok {
			select {
			case errs <- fr.Error:
			default:
			}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewOutputTransportMessageUrgentFrame("hello"))
	time.Sleep(200 * time.Millisecond)
	task.StopWhenDone()
	<-runDone

	select {
	case got := <-errs:
		t.Fatalf("a failed send produced an error frame (%q); it would feed itself", got)
	default:
	}
}

// errorLogHandler records the messages logged at ERROR or above.
type errorLogHandler struct {
	mu   sync.Mutex
	msgs []string
}

func (h *errorLogHandler) Enabled(_ context.Context, level slog.Level) bool {
	return level >= slog.LevelError
}

func (h *errorLogHandler) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.msgs = append(h.msgs, r.Message)
	return nil
}

func (h *errorLogHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *errorLogHandler) WithGroup(string) slog.Handler      { return h }

func (h *errorLogHandler) logged() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.msgs)
}

// captureErrorLogs redirects the default logger for the rest of the test and
// returns what has been logged at ERROR or above so far.
func captureErrorLogs(t *testing.T) func() []string {
	t.Helper()
	h := &errorLogHandler{}
	prev := slog.Default()
	slog.SetDefault(slog.New(h))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return h.logged
}

// TestBaseOutputBargeInIsNotReportedAsAFailure covers the send the barge-in
// itself cuts short. Restarting the audio loop cancels the send in flight on
// purpose, so the interruption does not have to wait behind it, and the
// transport reports that cancellation back. It is the interruption working as
// designed, so nothing about it may be logged as a failure.
//
// Upstream gets this for free: canceling its audio task raises
// asyncio.CancelledError, which derives from BaseException and so passes
// straight through the handler that logs a failed send, unwinding the task.
func TestBaseOutputBargeInIsNotReportedAsAFailure(t *testing.T) {
	logged := captureErrorLogs(t)

	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	o := newBlockingOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	// The first chunk is taken up by the loop and blocks in the write, so the
	// barge-in lands with a write in flight for it to cut short.
	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 1920*2), 48000, 1))
	select {
	case <-o.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("the transport was never asked to write")
	}

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(200 * time.Millisecond)

	if got := logged(); len(got) > 0 {
		t.Errorf("the barge-in reported itself as a failure: %v", got)
	}

	close(o.release)
	task.Cancel(t.Context(), "")
	<-runDone
}

// decliningOutput reports every chunk as unsent without failing, the way a
// transport reports audio it has nowhere to put: its track not live yet, its
// stream closed, its serializer producing nothing for the frame.
type decliningOutput struct {
	*transport.BaseOutput
	asked atomic.Int32
}

func newDecliningOutput(p transport.Params) *decliningOutput {
	o := &decliningOutput{}
	o.BaseOutput = transport.NewBaseOutput("DecliningOutput", p, o)
	return o
}

func (o *decliningOutput) WriteAudio(context.Context, frames.OutputAudioFrame) (bool, error) {
	o.asked.Add(1)
	return false, nil
}

// TestBaseOutputDoesNotForwardAudioTheTransportDeclined covers the third thing a
// transport can say about a chunk, alongside sending it and failing. A transport
// with nowhere to put the audio reports it as unsent without an error: nothing
// went wrong, so there is nothing to log, but the audio will not be heard, so it
// is not forwarded downstream as though it had been.
func TestBaseOutputDoesNotForwardAudioTheTransportDeclined(t *testing.T) {
	logged := captureErrorLogs(t)

	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000 // 1920-byte chunks

	forwarded := make(chan struct{}, 4)
	o := newDecliningOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.TTSAudioRawFrame); ok {
			forwarded <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{0x01}, 1920), 48000, 1))
	time.Sleep(300 * time.Millisecond)

	if got := o.asked.Load(); got == 0 {
		t.Fatal("the transport was never asked to write, so nothing was declined")
	}
	select {
	case <-forwarded:
		t.Error("audio the transport did not send was forwarded downstream as though it had been heard")
	default:
	}
	if got := logged(); len(got) > 0 {
		t.Errorf("declining to send is not a failure, but it was reported as one: %v", got)
	}

	task.Cancel(t.Context(), "")
	<-runDone
}

// blockingSyncFrameOutput holds every frame that carries no audio inside
// WriteTransportFrame until the transport is stopped, so a test can leave the
// audio loop inside that call when an interruption lands.
type blockingSyncFrameOutput struct {
	*transport.BaseOutput
	entered chan struct{}
}

func newBlockingSyncFrameOutput(p transport.Params) *blockingSyncFrameOutput {
	o := &blockingSyncFrameOutput{entered: make(chan struct{}, 8)}
	o.BaseOutput = transport.NewBaseOutput("BlockingSyncFrameOutput", p, o)
	return o
}

func (o *blockingSyncFrameOutput) WriteTransportFrame(ctx context.Context, _ frames.Frame) error {
	select {
	case o.entered <- struct{}{}:
	default:
	}
	<-ctx.Done()
	return ctx.Err()
}

// TestBaseOutputBargeInDropsAFrameItCutShort covers the same cancellation one
// frame type over. A frame carrying no audio waits in the same queue and is
// handed to the transport the same way, so a barge-in can cut that call short
// too. The frame it caught belongs to the turn that was interrupted: it is
// neither reported as a failure nor forwarded downstream as though the client
// had received it.
func TestBaseOutputBargeInDropsAFrameItCutShort(t *testing.T) {
	logged := captureErrorLogs(t)

	params := transport.DefaultParams()
	params.AudioOutSampleRate = 48000

	forwarded := make(chan struct{}, 4)
	o := newBlockingSyncFrameOutput(params)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*markerFrame); ok {
			forwarded <- struct{}{}
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(newMarkerFrame())
	select {
	case <-o.entered:
	case <-time.After(3 * time.Second):
		t.Fatal("the transport was never handed the frame")
	}

	task.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(200 * time.Millisecond)

	if got := logged(); len(got) > 0 {
		t.Errorf("the barge-in reported itself as a failure: %v", got)
	}
	select {
	case <-forwarded:
		t.Error("a frame the transport never finished sending was forwarded downstream")
	default:
	}

	task.Cancel(t.Context(), "")
	<-runDone
}

// TestBaseInputFilteredAudioKeepsItsFrame checks the frame that travels on is
// the one the transport produced. The filter replaces the audio on it, so what
// the frame carries about itself — the source it names, the moment it was
// captured — is not lost with the samples.
func TestBaseInputFilteredAudioKeepsItsFrame(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000
	params.AudioInFilter = &fakeFilter{}

	got := make(chan frames.Frame, 4)
	in := newFakeInput(params, 0)
	task := pipeline.NewWorker(pipeline.New(in), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, fr frames.Frame) {
		if af, ok := fr.(*frames.InputAudioRawFrame); ok {
			got <- af
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	f := frames.NewInputAudioRawFrame([]byte{1, 2, 3, 4}, 48000, 1)
	f.SetTransportSource("mic-1")
	task.QueueFrame(f)

	select {
	case out := <-got:
		if src := out.Base().TransportSource(); src != "mic-1" {
			t.Errorf("TransportSource = %q, want %q: filtering lost what the frame said about itself", src, "mic-1")
		}
		af, ok := out.(*frames.InputAudioRawFrame)
		if !ok {
			t.Fatalf("got a %T, want an InputAudioRawFrame", out)
		}
		if !bytes.Equal(af.Audio, []byte{2, 3, 4, 5}) {
			t.Errorf("audio = %v, want %v (the filter did not run)", af.Audio, []byte{2, 3, 4, 5})
		}
	case <-time.After(3 * time.Second):
		t.Fatal("no audio reached downstream")
	}

	task.StopWhenDone()
	<-runDone
}

// TestBaseInputFiltersWithPassthroughOff checks the filter sees every frame
// whatever passthrough is set to. It is stateful, so feeding it only the audio
// that happens to be forwarded would leave it working from a signal with holes
// in it.
func TestBaseInputFiltersWithPassthroughOff(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = 48000
	params.AudioInPassthrough = false
	filter := &fakeFilter{}
	params.AudioInFilter = filter

	in := newFakeInput(params, 0)
	task := pipeline.NewWorker(pipeline.New(in), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewInputAudioRawFrame([]byte{1, 2, 3, 4}, 48000, 1))

	deadline := time.Now().Add(3 * time.Second)
	for filter.filtered.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if filter.filtered.Load() == 0 {
		t.Error("the input filter never saw the audio: passthrough gated filtering")
	}

	task.StopWhenDone()
	<-runDone
}

// wedgedOutput never returns from a write, which is what a peer that has stopped
// reading looks like: the connection is up and nothing fails, so the only signal
// available is that the write does not come back.
type wedgedOutput struct {
	*transport.BaseOutput
	writes atomic.Int32
	// delay is how long each write takes; zero never returns.
	delay time.Duration
}

func newWedgedOutput(p transport.Params, delay time.Duration) *wedgedOutput {
	o := &wedgedOutput{delay: delay}
	o.BaseOutput = transport.NewBaseOutput("WedgedOutput", p, o)
	return o
}

func (o *wedgedOutput) WriteAudio(ctx context.Context, _ frames.OutputAudioFrame) (bool, error) {
	o.writes.Add(1)
	if o.delay > 0 {
		select {
		case <-time.After(o.delay):
			return true, nil
		case <-ctx.Done():
			return false, ctx.Err()
		}
	}
	<-ctx.Done()
	return false, ctx.Err()
}

// oneSecondOfAudio is 16 kHz 16-bit mono, which the sender splits into chunks.
func oneSecondOfAudio() *frames.TTSAudioRawFrame {
	return frames.NewTTSAudioRawFrame(make([]byte, 16000*2), 16000, 1)
}

// TestOutputWritesOffAPeerThatStoppedReading covers the bounded write. Paying the
// timeout per queued chunk would hang the session by a slower route, so the peer
// is written off once however much audio is queued behind it.
//
// Ported from upstream's test_peer_is_written_off_once.
func TestOutputWritesOffAPeerThatStoppedReading(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000
	params.AudioOutEndSilenceSecs = 0
	params.AudioOutWriteTimeout = 300 * time.Millisecond

	o := newWedgedOutput(params, 0)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(oneSecondOfAudio())
	time.Sleep(time.Second)

	if got := o.writes.Load(); got != 1 {
		t.Errorf("writes = %d, want 1: the peer was written off per chunk", got)
	}
	if o.Usable() {
		t.Error("the transport is still usable after a write that never returned")
	}

	task.Cancel(context.Background(), "")
	<-runDone
}

// TestOutputDoesNotCutShortASlowWrite is the other half: long playout is
// legitimately slow, and only a stalled write is written off.
//
// Ported from upstream's test_slow_write_within_the_bound_is_not_cut_short.
func TestOutputDoesNotCutShortASlowWrite(t *testing.T) {
	params := transport.DefaultParams()
	params.AudioOutSampleRate = 16000
	params.AudioOutEndSilenceSecs = 0
	params.AudioOutWriteTimeout = time.Second

	o := newWedgedOutput(params, 10*time.Millisecond)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()

	task.QueueFrame(oneSecondOfAudio())
	time.Sleep(time.Second)

	// A second of audio is many chunks; a written-off peer would show one, so
	// the exact count does not matter.
	if got := o.writes.Load(); got < 10 {
		t.Errorf("writes = %d, want a slow write left to carry on", got)
	}
	if !o.Usable() {
		t.Error("a slow but progressing write cost the transport its usability")
	}

	task.Cancel(context.Background(), "")
	<-runDone
}
