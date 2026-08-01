package transport

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

const (
	// botVADStop is how long a speech stream has to stay silent before the
	// bot counts as having stopped speaking. It applies only to a speech stream,
	// which carries silence between utterances; TTS output is ended by its
	// TTSStoppedFrame instead.
	botVADStop = 350 * time.Millisecond

	// botVADStopFallback is the fallback, used when nothing reaches the
	// output at all: with no audio and no stop frame for this long, the bot
	// counts as having stopped speaking.
	botVADStopFallback = 3 * time.Second

	// botSpeakingFramePeriod is how often a BotSpeakingFrame is broadcast while
	// the bot holds the floor. It only has an effect while it stays longer than
	// one chunk's duration.
	botSpeakingFramePeriod = 200 * time.Millisecond
)

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
	// newChunk rebuilds a buffered chunk as the same frame type as the audio it
	// was buffered from, so TTS audio stays a TTSAudioRawFrame and a speech
	// stream stays a SpeechOutputAudioRawFrame. The bot-speaking bookkeeping
	// reads that type to tell which of the two it is pacing out.
	newChunk func(pcm []byte, sampleRate, numChannels int) frames.Frame

	audioOut    chan frames.Frame
	audioCtx    context.Context
	audioCancel context.CancelFunc
	audioWG     sync.WaitGroup
	// clockQ delivers frames carrying a presentation timestamp at the moment
	// that timestamp names. See clockqueue.go.
	clockQ *clockQueue
	// drainWait, set under bufMu before a drain marker is queued, is closed by
	// the audio loop once it has paced out everything ahead of the marker.
	drainWait chan struct{}

	// Bot-speaking detection: BotStartedSpeakingFrame is broadcast when the
	// bot's audio starts flowing and BotStoppedSpeakingFrame once it ends, so
	// the turn and idle controllers know when the bot holds the floor.
	botMu sync.Mutex
	// botSpeaking is whether the bot currently holds the floor.
	botSpeaking bool
	// ttsAudioReceived gates ending a turn on TTSStoppedFrame: a stop only
	// counts once TTS audio has actually arrived for the turn.
	ttsAudioReceived bool
	// botSpeakingFrameAt is when the last periodic BotSpeakingFrame went out.
	botSpeakingFrameAt time.Time
	// botSpeechLastAt is when the bot was last audibly speaking.
	botSpeechLastAt time.Time
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

// WriteTransportFrame is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) WriteTransportFrame(context.Context, frames.Frame) error { return nil }

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
		bo.botStoppedSpeaking(ctx)
		return nil
	case *frames.OutputTransportMessageFrame, *frames.OutputTransportMessageUrgentFrame:
		return bo.handleTransportMessage(ctx, f, dir)
	case frames.MixerControlFrame:
		return bo.handleMixerControl(ctx, fr, dir)
	case *frames.TTSAudioRawFrame, *frames.SpeechOutputAudioRawFrame, *frames.OutputAudioRawFrame:
		if dir == processor.Downstream {
			bo.handleAudioFrame(f)
			return nil
		}
		return bo.PushFrame(ctx, f, dir)
	case *frames.TTSStoppedFrame:
		if dir == processor.Downstream {
			return bo.handleTTSStopped(ctx, fr)
		}
		return bo.PushFrame(ctx, f, dir)
	default:
		return bo.handleTimedFrame(ctx, f, dir)
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

	if bo.clockQ == nil {
		bo.clockQ = newClockQueue(bo)
	}
	bo.clockQ.start(ctx)
}

func (bo *BaseOutput) stopStreaming(ctx context.Context) {
	bo.botStoppedSpeaking(ctx)
	if bo.params.AudioOutMixer != nil {
		_ = bo.params.AudioOutMixer.Stop(ctx)
	}
	cancel := bo.audioCancel
	bo.audioCancel = nil
	if cancel != nil {
		cancel()
		bo.audioWG.Wait()
	}
	if bo.clockQ != nil {
		bo.clockQ.stop()
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
	bo.bufMu.Unlock()
	if ac == nil {
		return
	}

	// Pad and flush the sub-chunk tail so the last of the audio plays too.
	bo.enqueueFlushedAudioBuffer()

	bo.bufMu.Lock()
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

// handleMixerControl hands a mixer control frame to the output mixer and
// forwards it on. The mixer reads the frame itself, so a control the mixer
// understands does not have to be translated on the way through, and a new one
// needs no change here.
func (bo *BaseOutput) handleMixerControl(
	ctx context.Context, f frames.MixerControlFrame, dir processor.Direction,
) error {
	if bo.params.AudioOutMixer != nil {
		_ = bo.params.AudioOutMixer.ProcessFrame(ctx, f)
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
		// Logged, not returned. A returned error becomes an ErrorFrame, and
		// anything that reports errors to the client turns that into another
		// message to send: if the connection is what failed, sending the report
		// fails too and the pipeline feeds itself errors until it runs out of
		// memory. A connection that cannot carry a message cannot carry the
		// complaint about it either.
		slog.Error("send transport message", "processor", bo.Name(), "err", err)
		return nil
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
// frame (a TTSAudioRawFrame, a SpeechOutputAudioRawFrame, or a plain
// OutputAudioRawFrame).
func outputAudio(f frames.Frame) (pcm []byte, sampleRate, channels int) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels
	case *frames.SpeechOutputAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels
	case *frames.OutputAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels
	}
	return nil, 0, 0
}

// chunkBuilder returns the constructor for f's own frame type, so a chunk sliced
// out of f is rebuilt as the same kind of frame rather than flattened to a plain
// OutputAudioRawFrame.
func chunkBuilder(f frames.Frame) func(pcm []byte, sampleRate, numChannels int) frames.Frame {
	switch f.(type) {
	case *frames.TTSAudioRawFrame:
		return func(pcm []byte, rate, ch int) frames.Frame {
			return frames.NewTTSAudioRawFrame(pcm, rate, ch)
		}
	case *frames.SpeechOutputAudioRawFrame:
		return func(pcm []byte, rate, ch int) frames.Frame {
			return frames.NewSpeechOutputAudioRawFrame(pcm, rate, ch)
		}
	default:
		return func(pcm []byte, rate, ch int) frames.Frame {
			return frames.NewOutputAudioRawFrame(pcm, rate, ch)
		}
	}
}

// handleAudioFrame resamples incoming audio to the output rate, buffers it, and
// emits fixed-size chunks, each rebuilt as the frame type it was buffered from.
func (bo *BaseOutput) handleAudioFrame(f frames.Frame) {
	if !bo.params.AudioOutEnabled || bo.chunkSize == 0 {
		return
	}
	pcm, sampleRate, channels := outputAudio(f)
	pcm = bo.resample(pcm, sampleRate, channels)

	bo.bufMu.Lock()
	bo.newChunk = chunkBuilder(f)
	build := bo.newChunk
	bo.buffer = append(bo.buffer, pcm...)
	var chunks []frames.Frame
	for len(bo.buffer) >= bo.chunkSize {
		chunk := make([]byte, bo.chunkSize)
		copy(chunk, bo.buffer[:bo.chunkSize])
		chunks = append(chunks, build(chunk, bo.sampleRate, channels))
		bo.buffer = bo.buffer[bo.chunkSize:]
	}
	ctx := bo.audioCtx
	out := bo.audioOut
	bo.bufMu.Unlock()

	for _, chunk := range chunks {
		sendAudio(ctx, out, chunk)
	}
}

// handleTTSStopped queues a TTSStoppedFrame behind the audio it ends, flushing
// the trailing partial chunk first. handleAudioFrame only queues whole chunks,
// so up to one chunk of a turn's audio can still be sitting in the buffer;
// queueing it now plays it before the stop frame is handled, instead of leaving
// it to be discarded when the buffer is cleared.
func (bo *BaseOutput) handleTTSStopped(ctx context.Context, f *frames.TTSStoppedFrame) error {
	bo.enqueueFlushedAudioBuffer()

	bo.bufMu.Lock()
	audioCtx, out := bo.audioCtx, bo.audioOut
	bo.bufMu.Unlock()
	if audioCtx == nil || out == nil {
		return bo.PushFrame(ctx, f, processor.Downstream)
	}
	// The audio loop forwards it downstream once it reaches it, so that the stop
	// lands after the audio it ends rather than ahead of it.
	sendAudio(audioCtx, out, frames.Frame(f))
	return nil
}

// enqueueFlushedAudioBuffer pads whatever is left in the buffer out to a full
// chunk with silence and queues it for playback, as the same frame type as the
// audio it was buffered from. It goes through the normal playback path (write,
// error handling, bot-speaking bookkeeping) like any other chunk, and keeps its
// order relative to whatever is queued after it.
func (bo *BaseOutput) enqueueFlushedAudioBuffer() {
	bo.bufMu.Lock()
	if len(bo.buffer) == 0 || bo.chunkSize == 0 {
		bo.bufMu.Unlock()
		return
	}
	build := bo.newChunk
	if build == nil {
		build = chunkBuilder(nil)
	}
	tail := build(padChunk(bo.buffer, bo.chunkSize), bo.sampleRate, bo.channels)
	bo.buffer = nil
	audioCtx, out := bo.audioCtx, bo.audioOut
	bo.bufMu.Unlock()

	if audioCtx == nil || out == nil {
		return
	}
	sendAudio(audioCtx, out, tail)
}

// enqueueWordFrame queues a downstream word-aligned TTSTextFrame on the audio
// channel so the audio loop forwards it downstream in playback order with the
// surrounding audio. A frame with no timestamp, or one going upstream, has
// nothing to wait for and is forwarded as it arrives.
func (bo *BaseOutput) handleTimedFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	pts, timed := f.Base().PTS()
	if !timed || dir != processor.Downstream || bo.clockQ == nil {
		return bo.PushFrame(ctx, f, dir)
	}
	bo.clockQ.push(pts, f)
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
	if bo.clockQ != nil {
		// The frames waiting on the clock belong to audio that will never play.
		bo.clockQ.drop()
	}
	for {
		select {
		case <-bo.audioOut:
		default:
			return
		}
	}
}

// handleBotSpeech updates the bot-speaking state from one chunk of outgoing
// audio. The two kinds of audio end a turn differently: TTS output is ended by
// its TTSStoppedFrame, while a speech stream carries its own silence and has to
// be measured.
func (bo *BaseOutput) handleBotSpeech(ctx context.Context, f frames.Frame) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		// A TTSStoppedFrame only ends the turn once TTS audio has arrived for it.
		bo.botMu.Lock()
		bo.ttsAudioReceived = true
		bo.botMu.Unlock()
		bo.botCurrentlySpeaking(ctx)
	case *frames.SpeechOutputAudioRawFrame:
		bo.maybeBotCurrentlySpeaking(ctx, fr)
	}
}

// maybeBotCurrentlySpeaking tracks a speech stream, which carries silence
// between utterances: audible audio holds the floor, and silence for longer than
// botVADStop gives it up.
func (bo *BaseOutput) maybeBotCurrentlySpeaking(ctx context.Context, f *frames.SpeechOutputAudioRawFrame) {
	if !audio.IsSilence(f.Audio) {
		bo.botCurrentlySpeaking(ctx)
		return
	}
	bo.botMu.Lock()
	last := bo.botSpeechLastAt
	bo.botMu.Unlock()
	if time.Since(last) > botVADStop {
		bo.botStoppedSpeaking(ctx)
	}
}

// botCurrentlySpeaking marks the bot as holding the floor and broadcasts a
// BotSpeakingFrame at most once per botSpeakingFramePeriod while it does.
func (bo *BaseOutput) botCurrentlySpeaking(ctx context.Context) {
	bo.botStartedSpeaking(ctx)

	now := time.Now()
	bo.botMu.Lock()
	due := now.Sub(bo.botSpeakingFrameAt) >= botSpeakingFramePeriod
	if due {
		bo.botSpeakingFrameAt = now
	}
	bo.botSpeechLastAt = now
	bo.botMu.Unlock()

	if due {
		_ = bo.Broadcast(ctx, func() frames.Frame { return frames.NewBotSpeakingFrame() })
	}
}

// botStartedSpeaking broadcasts that the bot took the floor, once per run of
// speech.
func (bo *BaseOutput) botStartedSpeaking(ctx context.Context) {
	bo.botMu.Lock()
	if bo.botSpeaking {
		bo.botMu.Unlock()
		return
	}
	bo.botSpeaking = true
	bo.botMu.Unlock()

	_ = bo.Broadcast(ctx, func() frames.Frame { return frames.NewBotStartedSpeakingFrame() })
}

// botStoppedSpeaking broadcasts that the bot gave up the floor. Whatever is left
// buffered is dropped rather than flushed: after an interruption, or once a turn
// has ended, that audio is no longer wanted.
func (bo *BaseOutput) botStoppedSpeaking(ctx context.Context) {
	bo.botMu.Lock()
	if !bo.botSpeaking {
		bo.botMu.Unlock()
		return
	}
	bo.botSpeaking = false
	bo.ttsAudioReceived = false
	bo.botMu.Unlock()

	bo.bufMu.Lock()
	bo.buffer = nil
	bo.bufMu.Unlock()

	_ = bo.Broadcast(ctx, func() frames.Frame { return frames.NewBotStoppedSpeakingFrame() })
}

// ttsStopped ends the bot's turn on a TTSStoppedFrame, but only when TTS audio
// actually arrived for that turn.
func (bo *BaseOutput) ttsStopped(ctx context.Context) {
	bo.botMu.Lock()
	received := bo.ttsAudioReceived
	bo.botMu.Unlock()
	if received {
		bo.botStoppedSpeaking(ctx)
	}
}

// audioLoop paces queued frames out to the transport. Receiving nothing at all
// for botVADStopFallback is the fallback that ends the bot's turn when no
// explicit stop reaches the output.
func (bo *BaseOutput) audioLoop(ctx context.Context) {
	defer bo.audioWG.Done()
	idle := time.NewTimer(botVADStopFallback)
	defer idle.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-idle.C:
			bo.botStoppedSpeaking(ctx)
			idle.Reset(botVADStopFallback)
		case queued := <-bo.audioOut:
			if !idle.Stop() {
				select {
				case <-idle.C:
				default:
				}
			}
			idle.Reset(botVADStopFallback)

			if queued == drainMarker {
				bo.signalDrained()
				continue
			}
			bo.handleQueuedFrame(ctx, queued)
		}
	}
}

// signalDrained releases a drainAudio waiter once the loop reaches its marker.
func (bo *BaseOutput) signalDrained() {
	bo.bufMu.Lock()
	w := bo.drainWait
	bo.drainWait = nil
	bo.bufMu.Unlock()
	if w != nil {
		close(w)
	}
}

// handleQueuedFrame plays one queued frame: it applies the frame's own effect,
// writes whatever audio it carries to the transport, and forwards it downstream.
// A frame whose audio could not be written is not forwarded, so nothing
// downstream treats it as having been sent.
func (bo *BaseOutput) handleQueuedFrame(ctx context.Context, f frames.Frame) {
	pushDownstream := true

	switch f.(type) {
	case *frames.TTSAudioRawFrame, *frames.SpeechOutputAudioRawFrame, *frames.OutputAudioRawFrame:
		bo.handleBotSpeech(ctx, f)

		pcm, _, _ := outputAudio(f)
		if bo.params.AudioOutMixer != nil {
			if mixed, err := bo.params.AudioOutMixer.Mix(ctx, pcm); err == nil {
				pcm = mixed
			}
		}
		if err := bo.self.WriteAudio(ctx, pcm); err != nil {
			slog.Error("write audio to transport", "processor", bo.Name(), "err", err)
			pushDownstream = false
		}
	case *frames.TTSStoppedFrame:
		bo.ttsStopped(ctx)
	default:
		// A frame that carries no audio has waited behind the audio it belongs
		// to (a word-aligned text frame, say). Give the concrete transport a
		// chance to act on it now that that audio has played.
		if err := bo.self.WriteTransportFrame(ctx, f); err != nil {
			slog.Error("write transport frame", "processor", bo.Name(), "frame", f.Name(), "err", err)
		}
	}

	if pushDownstream {
		_ = bo.PushFrame(ctx, f, processor.Downstream)
	}
}
