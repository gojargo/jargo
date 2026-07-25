package transport

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// botStopDebounce is how long after the last audio chunk the bot is considered
// to have stopped speaking.
const botStopDebounce = 250 * time.Millisecond

// BaseOutput is the tail of a pipeline: it buffers OutputAudioRawFrames, slices
// them into fixed-size chunks, and hands each chunk to the concrete transport
// to send. Chunking into small, uniform pieces keeps output latency low and
// makes interruptions responsive. A concrete transport embeds it and implements
// OutputDriver to send the audio.
type BaseOutput struct {
	*processor.Base
	params Params
	self   OutputDriver

	sampleRate int
	channels   int
	chunkSize  int

	resampler  *resample.Resampler
	resampleIn int

	bufMu  sync.Mutex
	buffer []byte

	audioOut    chan frames.Frame
	audioCtx    context.Context
	audioCancel context.CancelFunc
	audioWG     sync.WaitGroup
	// drainWait, set under bufMu before a drain marker is queued, is closed by
	// the audio loop once it has paced out everything ahead of the marker.
	drainWait chan struct{}
	// flushFrame holds the padded trailing chunk emitted when a turn ends. The
	// audio loop plays it but skips the bot-speaking bookkeeping, so flushing a
	// turn's tail does not re-arm speaking on a floor the bot has already given up.
	flushFrame atomic.Pointer[frames.OutputAudioRawFrame]

	// Bot-speaking detection: a BotStartedSpeakingFrame is emitted upstream when
	// audio starts flowing and a BotStoppedSpeakingFrame after it drains, so the
	// turn and idle controllers know when the bot holds the floor.
	botMu         sync.Mutex
	botSpeaking   bool
	botStopCancel func()
}

// NewBaseOutput builds a BaseOutput. self is the embedding transport, used to
// dispatch WriteAudio and to process frames.
func NewBaseOutput(name string, params Params, self OutputDriver) *BaseOutput {
	bo := &BaseOutput{params: params, self: self}
	bo.Base = processor.New(name, self)
	return bo
}

// SampleRate is the output sample rate in Hz, set when the transport starts.
func (bo *BaseOutput) SampleRate() int { return bo.sampleRate }

// Params returns the transport parameters.
func (bo *BaseOutput) Params() Params { return bo.params }

// WriteAudio is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) WriteAudio(context.Context, []byte) error { return nil }

// SendMessage is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) SendMessage(context.Context, []byte) error { return nil }

// ProcessFrame handles the transport lifecycle and routes audio.
func (bo *BaseOutput) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := bo.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		// Initialize before forwarding so the chunk size is set before any
		// audio frame can be processed. Nothing downstream of the output
		// transport needs the StartFrame ahead of this.
		bo.startStreaming(ctx, fr)
		return bo.PushFrame(ctx, f, dir)
	case *frames.EndFrame:
		bo.drainAudio(ctx)
		bo.stopStreaming(ctx)
		return bo.PushFrame(ctx, f, dir)
	case *frames.CancelFrame:
		bo.stopStreaming(ctx)
		return bo.PushFrame(ctx, f, dir)
	case *frames.InterruptionFrame:
		if err := bo.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		bo.handleInterruption()
		bo.stopBotSpeaking(ctx)
		return nil
	case *frames.OutputTransportMessageFrame, *frames.OutputTransportMessageUrgentFrame:
		return bo.handleTransportMessage(ctx, f, dir)
	case frames.MixerControlFrame:
		return bo.handleMixerControl(ctx, fr, dir)
	case *frames.TTSAudioRawFrame, *frames.OutputAudioRawFrame:
		if dir == processor.Downstream {
			audio, rate, channels := outputAudio(f)
			bo.handleAudioFrame(audio, rate, channels)
			return nil
		}
		return bo.PushFrame(ctx, f, dir)
	case *frames.TTSTextFrame:
		// Release word-aligned text in step with the audio it belongs to: queue
		// it behind the buffered audio so the audio loop forwards it downstream
		// only once that audio has played, and an interruption drops it along
		// with the audio that never played.
		return bo.enqueueWordFrame(ctx, fr, dir)
	default:
		return bo.PushFrame(ctx, f, dir)
	}
}

// Cleanup stops the audio goroutine and the processor.
func (bo *BaseOutput) Cleanup(ctx context.Context) error {
	bo.stopStreaming(ctx)
	// Free the soxr handle after the base stops the process goroutine, so no
	// in-flight resample touches a freed native resampler.
	err := bo.Base.Cleanup(ctx)
	if bo.resampler != nil {
		bo.resampler.Close()
		bo.resampler = nil
	}
	return err
}

func (bo *BaseOutput) startStreaming(ctx context.Context, f *frames.StartFrame) {
	bo.sampleRate = pick(bo.params.AudioOutSampleRate, f.AudioOutSampleRate)
	bo.channels = bo.params.AudioOutChannels
	if bo.channels == 0 {
		bo.channels = 1
	}
	chunks := bo.params.AudioOut10msChunks
	if chunks == 0 {
		chunks = 2
	}
	bytesPer10ms := bo.sampleRate / 100 * bo.channels * 2
	bo.chunkSize = bytesPer10ms * chunks

	bo.bufMu.Lock()
	bo.buffer = nil
	bo.bufMu.Unlock()

	if bo.params.AudioOutMixer != nil {
		_ = bo.params.AudioOutMixer.Start(ctx, bo.sampleRate)
	}

	bo.audioCtx, bo.audioCancel = context.WithCancel(ctx)
	bo.audioOut = make(chan frames.Frame, audioFrameChanCap)
	bo.audioWG.Add(1)
	go bo.audioLoop(bo.audioCtx)
}

func (bo *BaseOutput) stopStreaming(ctx context.Context) {
	bo.stopBotSpeaking(ctx)
	if bo.params.AudioOutMixer != nil {
		_ = bo.params.AudioOutMixer.Stop(ctx)
	}
	cancel := bo.audioCancel
	bo.audioCancel = nil
	if cancel != nil {
		cancel()
		bo.audioWG.Wait()
	}
}

// drainMarker is a sentinel queued on audioOut to detect when the audio loop has
// paced out everything ahead of it. It is compared by identity and never played.
//
//nolint:gochecknoglobals // sentinel value
var drainMarker = &frames.OutputAudioRawFrame{}

// drainAudio blocks until the audio loop has paced out everything it has queued,
// so a graceful EndFrame lets the bot finish speaking (a farewell, say) instead
// of cutting off. A CancelFrame or an interruption skips it and stops at once. It
// queues a marker behind the buffered audio and waits for the loop to reach it.
func (bo *BaseOutput) drainAudio(ctx context.Context) {
	bo.bufMu.Lock()
	ac := bo.audioCtx
	if ac == nil {
		bo.bufMu.Unlock()
		return
	}
	if len(bo.buffer) > 0 { // pad and flush the sub-chunk tail so it plays too
		chunk := padChunk(bo.buffer, bo.chunkSize)
		bo.buffer = nil
		sendAudio(ac, bo.audioOut, frames.Frame(frames.NewOutputAudioRawFrame(chunk, bo.sampleRate, bo.channels)))
	}
	done := make(chan struct{})
	bo.drainWait = done
	bo.bufMu.Unlock()

	sendAudio(ac, bo.audioOut, frames.Frame(drainMarker))
	select {
	case <-done:
	case <-ac.Done():
	case <-ctx.Done():
	}
}

// handleMixerControl applies a mixer control frame to the output mixer and
// forwards it. The mixer's settings map is its whole control surface, so
// enabling is expressed as the "enabled" setting.
func (bo *BaseOutput) handleMixerControl(
	ctx context.Context, f frames.MixerControlFrame, dir processor.Direction,
) error {
	if bo.params.AudioOutMixer != nil {
		var settings map[string]any
		switch fr := f.(type) {
		case *frames.MixerUpdateSettingsFrame:
			settings = fr.Settings
		case *frames.MixerEnableFrame:
			settings = map[string]any{"enabled": fr.Enable}
		}
		if settings != nil {
			_ = bo.params.AudioOutMixer.Control(ctx, settings)
		}
	}
	return bo.PushFrame(ctx, f, dir)
}

// handleTransportMessage sends an application message to the client and forwards
// the frame on. It serves both message frames: the ordered one, which reaches
// here in step with the surrounding audio, and the urgent one, which is a system
// frame and so arrives ahead of anything queued.
func (bo *BaseOutput) handleTransportMessage(
	ctx context.Context, f frames.Frame, dir processor.Direction,
) error {
	var message any
	switch fr := f.(type) {
	case *frames.OutputTransportMessageFrame:
		message = fr.Message
	case *frames.OutputTransportMessageUrgentFrame:
		message = fr.Message
	}
	if err := bo.sendMessage(ctx, message); err != nil {
		return err
	}
	return bo.PushFrame(ctx, f, dir)
}

// sendMessage serializes a transport message payload to JSON and hands it to the
// concrete transport. It serves both the ordered and the urgent message frames.
func (bo *BaseOutput) sendMessage(ctx context.Context, message any) error {
	data, err := json.Marshal(message)
	if err != nil {
		return err
	}
	return bo.self.SendMessage(ctx, data)
}

// outputAudio extracts the PCM, sample rate and channel count from a bot audio
// frame (a TTSAudioRawFrame or a plain OutputAudioRawFrame).
func outputAudio(f frames.Frame) (audio []byte, sampleRate, channels int) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels
	case *frames.OutputAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels
	}
	return nil, 0, 0
}

// handleAudioFrame resamples incoming audio to the output rate, buffers it, and
// emits fixed-size chunks.
func (bo *BaseOutput) handleAudioFrame(audio []byte, sampleRate, channels int) {
	if !bo.params.AudioOutEnabled || bo.chunkSize == 0 {
		return
	}
	audio = bo.resample(audio, sampleRate, channels)

	bo.bufMu.Lock()
	bo.buffer = append(bo.buffer, audio...)
	var chunks [][]byte
	for len(bo.buffer) >= bo.chunkSize {
		chunk := make([]byte, bo.chunkSize)
		copy(chunk, bo.buffer[:bo.chunkSize])
		chunks = append(chunks, chunk)
		bo.buffer = bo.buffer[bo.chunkSize:]
	}
	ctx := bo.audioCtx
	bo.bufMu.Unlock()

	for _, chunk := range chunks {
		out := frames.NewOutputAudioRawFrame(chunk, bo.sampleRate, bo.channels)
		sendAudio(ctx, bo.audioOut, frames.Frame(out))
	}
}

// enqueueWordFrame queues a downstream word-aligned TTSTextFrame on the audio
// channel so the audio loop forwards it downstream in playback order with the
// surrounding audio. When audio output is disabled or streaming has not started
// there is no pacing to align to, so it is forwarded immediately; an upstream
// frame is forwarded as-is.
func (bo *BaseOutput) enqueueWordFrame(ctx context.Context, fr *frames.TTSTextFrame, dir processor.Direction) error {
	bo.bufMu.Lock()
	ac := bo.audioCtx
	out := bo.audioOut
	bo.bufMu.Unlock()
	if dir != processor.Downstream || ac == nil || out == nil || !bo.params.AudioOutEnabled {
		return bo.PushFrame(ctx, fr, dir)
	}
	sendAudio(ac, out, frames.Frame(fr))
	return nil
}

// padChunk copies b into a fresh chunk of size bytes, zero-padding the remainder
// with silence (a zero S16LE sample is silent). A turn's trailing sub-chunk is
// always shorter than a full chunk; padding it lets it flush through downstream
// encoders that only emit whole frames (Opus), instead of stalling there until
// the next turn supplies the missing bytes. If b already fills a chunk it is
// returned unchanged.
func padChunk(b []byte, size int) []byte {
	if len(b) >= size {
		return b
	}
	chunk := make([]byte, size)
	copy(chunk, b)
	return chunk
}

// resample converts audio at sampleRate to the transport output rate. The
// resampler is created lazily and reused across frames; it is only touched on
// the process goroutine, so it needs no lock.
func (bo *BaseOutput) resample(audio []byte, sampleRate, channels int) []byte {
	if sampleRate == bo.sampleRate {
		return audio
	}
	if bo.resampler == nil || bo.resampleIn != sampleRate {
		if bo.resampler != nil {
			bo.resampler.Close()
			bo.resampler = nil
		}
		r, err := resample.New(sampleRate, bo.sampleRate, channels)
		if err != nil {
			slog.Error("transport: create resampler", "from", sampleRate, "to", bo.sampleRate, "err", err)
			return audio
		}
		bo.resampler = r
		bo.resampleIn = sampleRate
	}
	return bo.resampler.Process(audio)
}

// handleInterruption drops buffered output audio so the bot stops speaking
// promptly on a barge-in. The pending sub-chunk tail is discarded along with
// everything else: a barge-in cuts the bot off, tail included.
func (bo *BaseOutput) handleInterruption() {
	bo.bufMu.Lock()
	bo.buffer = nil
	bo.bufMu.Unlock()
	bo.flushFrame.Store(nil)
	for {
		select {
		case <-bo.audioOut:
		default:
			return
		}
	}
}

// flushTailChunk pads and emits the sub-chunk remainder left in the buffer at a
// turn's end, so the final few milliseconds of the bot's audio play out instead
// of being stranded until the next turn. The emitted chunk is marked so the
// audio loop writes it without re-arming bot-speaking. It is safe against a
// barge-in: an interruption clears the buffer before this can read it, or drains
// the emitted chunk back out of the queue.
func (bo *BaseOutput) flushTailChunk(ctx context.Context) {
	bo.bufMu.Lock()
	if len(bo.buffer) == 0 || bo.chunkSize == 0 {
		bo.bufMu.Unlock()
		return
	}
	tail := frames.NewOutputAudioRawFrame(padChunk(bo.buffer, bo.chunkSize), bo.sampleRate, bo.channels)
	bo.buffer = nil
	out := bo.audioOut
	bo.bufMu.Unlock()

	bo.flushFrame.Store(tail)
	sendAudio(ctx, out, frames.Frame(tail))
}

// markBotSpeaking emits BotStartedSpeakingFrame on the first audio chunk and
// arms a debounce timer that emits BotStoppedSpeakingFrame once audio drains.
// Both go upstream so the turn and idle controllers see them.
func (bo *BaseOutput) markBotSpeaking(ctx context.Context) {
	bo.botMu.Lock()
	if !bo.botSpeaking {
		bo.botSpeaking = true
		_ = bo.PushFrame(ctx, frames.NewBotStartedSpeakingFrame(), processor.Upstream)
	}
	if bo.botStopCancel != nil {
		bo.botStopCancel()
	}
	stopped := false
	timer := time.AfterFunc(botStopDebounce, func() {
		bo.botMu.Lock()
		if stopped {
			bo.botMu.Unlock()
			return
		}
		bo.botSpeaking = false
		bo.botStopCancel = nil
		bo.botMu.Unlock()
		// The turn has gone quiet: flush its trailing sub-chunk so the last of
		// the bot's audio plays before we report it stopped speaking.
		bo.flushTailChunk(ctx)
		_ = bo.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Upstream)
	})
	bo.botStopCancel = func() {
		stopped = true
		timer.Stop()
	}
	bo.botMu.Unlock()
}

// stopBotSpeaking ends bot-speaking immediately (on interruption or shutdown).
func (bo *BaseOutput) stopBotSpeaking(ctx context.Context) {
	bo.botMu.Lock()
	if bo.botStopCancel != nil {
		bo.botStopCancel()
		bo.botStopCancel = nil
	}
	was := bo.botSpeaking
	bo.botSpeaking = false
	bo.botMu.Unlock()
	if was {
		_ = bo.PushFrame(ctx, frames.NewBotStoppedSpeakingFrame(), processor.Upstream)
	}
}

// audioLoop sends buffered chunks over the transport and forwards them
// downstream.
func (bo *BaseOutput) audioLoop(ctx context.Context) {
	defer bo.audioWG.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case queued := <-bo.audioOut:
			if queued == drainMarker {
				bo.bufMu.Lock()
				w := bo.drainWait
				bo.drainWait = nil
				bo.bufMu.Unlock()
				if w != nil {
					close(w)
				}
				continue
			}
			// A word-aligned text frame carries no audio; it has waited behind the
			// audio it belongs to, so forward it downstream now that that audio has
			// played and move on.
			if wf, ok := queued.(*frames.TTSTextFrame); ok {
				_ = bo.PushFrame(ctx, wf, processor.Downstream)
				continue
			}
			chunk, ok := queued.(*frames.OutputAudioRawFrame)
			if !ok {
				continue
			}
			// A trailing flush chunk is the tail of a turn that already ended; it
			// plays but must not re-arm bot-speaking, or it would open a spurious
			// speaking cycle on a floor the bot has given up.
			trailing := bo.flushFrame.CompareAndSwap(chunk, nil)
			audio := chunk.Audio
			if bo.params.AudioOutMixer != nil {
				if mixed, err := bo.params.AudioOutMixer.Mix(ctx, audio); err == nil {
					audio = mixed
				}
			}
			if err := bo.self.WriteAudio(ctx, audio); err != nil {
				slog.Error("write audio to transport", "processor", bo.Name(), "err", err)
				continue
			}
			if !trailing {
				bo.markBotSpeaking(ctx)
			}
			_ = bo.PushFrame(ctx, chunk, processor.Downstream)
		}
	}
}
