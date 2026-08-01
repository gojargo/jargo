package transport

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// mediaSender streams one outgoing destination. Everything it holds is per
// destination: the buffer audio is chunked out of, the resampler feeding it, the
// mixer blended into it, and whether the bot currently holds the floor on it. A
// transport carrying several outgoing streams has one sender each, so a turn on
// one stream neither shares a buffer with nor silences another.
type mediaSender struct {
	out         *BaseOutput
	destination string

	sampleRate int
	channels   int
	chunkSize  int
	mixer      audio.Mixer

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
	// bot's audio starts flowing on this destination and BotStoppedSpeakingFrame
	// once it ends, so the turn and idle controllers know when the bot holds the
	// floor.
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

// newMediaSender builds the sender for one destination, taking the audio shape
// the transport settled on when it started.
func newMediaSender(out *BaseOutput, destination string) *mediaSender {
	return &mediaSender{
		out:         out,
		destination: destination,
		sampleRate:  out.sampleRate,
		channels:    out.channels,
		chunkSize:   out.chunkSize,
		mixer:       out.mixerFor(destination),
	}
}

// start brings the sender's audio loop and clock up.
func (s *mediaSender) start(ctx context.Context) {
	s.bufMu.Lock()
	s.buffer = nil
	s.bufMu.Unlock()

	if s.mixer != nil {
		_ = s.mixer.Start(ctx, s.sampleRate)
	}

	s.audioCtx, s.audioCancel = context.WithCancel(ctx)
	s.audioOut = make(chan frames.Frame, audioFrameChanCap)
	s.audioWG.Add(1)
	go s.audioLoop(s.audioCtx)

	if s.clockQ == nil {
		s.clockQ = newClockQueue(s.out)
	}
	s.clockQ.start(ctx)
}

// stop tears the sender down.
func (s *mediaSender) stop(ctx context.Context) {
	s.botStoppedSpeaking(ctx)
	if s.mixer != nil {
		_ = s.mixer.Stop(ctx)
	}
	cancel := s.audioCancel
	s.audioCancel = nil
	if cancel != nil {
		cancel()
		s.audioWG.Wait()
	}
	if s.clockQ != nil {
		s.clockQ.stop()
	}
}

// closeResampler frees the native resampler handle, if one was created.
func (s *mediaSender) closeResampler() {
	if s.resampler != nil {
		s.resampler.Close()
		s.resampler = nil
	}
}

// drainAudio blocks until the audio loop has paced out everything it has queued,
// so a graceful EndFrame lets the bot finish speaking (a farewell, say) instead
// of cutting off. A CancelFrame or an interruption skips it and stops at once. It
// queues a marker behind the buffered audio and waits for the loop to reach it.
func (s *mediaSender) drainAudio(ctx context.Context) {
	s.bufMu.Lock()
	ac := s.audioCtx
	s.bufMu.Unlock()
	if ac == nil {
		return
	}

	// Pad and flush the sub-chunk tail so the last of the audio plays too.
	s.enqueueFlushedAudioBuffer()

	s.bufMu.Lock()
	done := make(chan struct{})
	s.drainWait = done
	s.bufMu.Unlock()

	sendAudio(ac, s.audioOut, frames.Frame(drainMarker))
	select {
	case <-done:
	case <-ac.Done():
	case <-ctx.Done():
	}
}

// handleAudioFrame resamples incoming audio to the output rate, buffers it, and
// emits fixed-size chunks, each rebuilt as the frame type it was buffered from
// and addressed to this sender's destination.
func (s *mediaSender) handleAudioFrame(f frames.Frame) {
	if !s.out.params.AudioOutEnabled || s.chunkSize == 0 {
		return
	}
	pcm, sampleRate, channels := outputAudio(f)
	pcm = s.resample(pcm, sampleRate, channels)

	s.bufMu.Lock()
	s.newChunk = chunkBuilder(f)
	build := s.newChunk
	s.buffer = append(s.buffer, pcm...)
	var chunks []frames.Frame
	for len(s.buffer) >= s.chunkSize {
		chunk := make([]byte, s.chunkSize)
		copy(chunk, s.buffer[:s.chunkSize])
		chunks = append(chunks, s.address(build(chunk, s.sampleRate, channels)))
		s.buffer = s.buffer[s.chunkSize:]
	}
	ctx := s.audioCtx
	out := s.audioOut
	s.bufMu.Unlock()

	for _, chunk := range chunks {
		sendAudio(ctx, out, chunk)
	}
}

// address stamps a frame this sender produced with its destination, so anything
// reading it downstream can tell which outgoing stream it belongs to.
func (s *mediaSender) address(f frames.Frame) frames.Frame {
	f.Base().SetTransportDestination(s.destination)
	return f
}

// handleTTSStopped queues a TTSStoppedFrame behind the audio it ends, flushing
// the trailing partial chunk first. handleAudioFrame only queues whole chunks,
// so up to one chunk of a turn's audio can still be sitting in the buffer;
// queueing it now plays it before the stop frame is handled, instead of leaving
// it to be discarded when the buffer is cleared.
func (s *mediaSender) handleTTSStopped(ctx context.Context, f *frames.TTSStoppedFrame) error {
	s.enqueueFlushedAudioBuffer()

	s.bufMu.Lock()
	audioCtx, out := s.audioCtx, s.audioOut
	s.bufMu.Unlock()
	if audioCtx == nil || out == nil {
		return s.out.PushFrame(ctx, f, processor.Downstream)
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
func (s *mediaSender) enqueueFlushedAudioBuffer() {
	s.bufMu.Lock()
	if len(s.buffer) == 0 || s.chunkSize == 0 {
		s.bufMu.Unlock()
		return
	}
	build := s.newChunk
	if build == nil {
		build = chunkBuilder(nil)
	}
	tail := s.address(build(padChunk(s.buffer, s.chunkSize), s.sampleRate, s.channels))
	s.buffer = nil
	audioCtx, out := s.audioCtx, s.audioOut
	s.bufMu.Unlock()

	if audioCtx == nil || out == nil {
		return
	}
	sendAudio(audioCtx, out, tail)
}

// resample converts audio at sampleRate to the transport output rate. The
// resampler is created lazily and reused across frames; it is only touched on
// the process goroutine, so it needs no lock.
func (s *mediaSender) resample(pcm []byte, sampleRate, channels int) []byte {
	if sampleRate == s.sampleRate {
		return pcm
	}
	if s.resampler == nil || s.resampleIn != sampleRate {
		s.closeResampler()
		r, err := resample.New(sampleRate, s.sampleRate, channels)
		if err != nil {
			slog.Error("transport: create resampler",
				"from", sampleRate, "to", s.sampleRate, "destination", s.destination, "err", err)
			return pcm
		}
		s.resampler = r
		s.resampleIn = sampleRate
	}
	return s.resampler.Process(pcm)
}

// handleInterruption drops buffered output audio so the bot stops speaking
// promptly on a barge-in. The pending sub-chunk tail is discarded along with
// everything else: a barge-in cuts the bot off, tail included.
func (s *mediaSender) handleInterruption() {
	s.bufMu.Lock()
	s.buffer = nil
	s.bufMu.Unlock()
	if s.clockQ != nil {
		// The frames waiting on the clock belong to audio that will never play.
		s.clockQ.drop()
	}
	for {
		select {
		case <-s.audioOut:
		default:
			return
		}
	}
}

// handleBotSpeech updates the bot-speaking state from one chunk of outgoing
// audio. The two kinds of audio end a turn differently: TTS output is ended by
// its TTSStoppedFrame, while a speech stream carries its own silence and has to
// be measured.
func (s *mediaSender) handleBotSpeech(ctx context.Context, f frames.Frame) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		// A TTSStoppedFrame only ends the turn once TTS audio has arrived for it.
		s.botMu.Lock()
		s.ttsAudioReceived = true
		s.botMu.Unlock()
		s.botCurrentlySpeaking(ctx)
	case *frames.SpeechOutputAudioRawFrame:
		s.maybeBotCurrentlySpeaking(ctx, fr)
	}
}

// maybeBotCurrentlySpeaking tracks a speech stream, which carries silence
// between utterances: audible audio holds the floor, and silence for longer than
// botVADStop gives it up.
func (s *mediaSender) maybeBotCurrentlySpeaking(ctx context.Context, f *frames.SpeechOutputAudioRawFrame) {
	if !audio.IsSilence(f.Audio) {
		s.botCurrentlySpeaking(ctx)
		return
	}
	s.botMu.Lock()
	last := s.botSpeechLastAt
	s.botMu.Unlock()
	if time.Since(last) > botVADStop {
		s.botStoppedSpeaking(ctx)
	}
}

// botCurrentlySpeaking marks the bot as holding the floor and broadcasts a
// BotSpeakingFrame at most once per botSpeakingFramePeriod while it does.
func (s *mediaSender) botCurrentlySpeaking(ctx context.Context) {
	s.botStartedSpeaking(ctx)

	now := time.Now()
	s.botMu.Lock()
	due := now.Sub(s.botSpeakingFrameAt) >= botSpeakingFramePeriod
	if due {
		s.botSpeakingFrameAt = now
	}
	s.botSpeechLastAt = now
	s.botMu.Unlock()

	if due {
		_ = s.out.Broadcast(ctx, func() frames.Frame {
			return s.address(frames.NewBotSpeakingFrame())
		})
	}
}

// botStartedSpeaking broadcasts that the bot took the floor on this destination,
// once per run of speech.
func (s *mediaSender) botStartedSpeaking(ctx context.Context) {
	s.botMu.Lock()
	if s.botSpeaking {
		s.botMu.Unlock()
		return
	}
	s.botSpeaking = true
	s.botMu.Unlock()

	_ = s.out.Broadcast(ctx, func() frames.Frame {
		return s.address(frames.NewBotStartedSpeakingFrame())
	})
}

// botStoppedSpeaking broadcasts that the bot gave up the floor. Whatever is left
// buffered is dropped rather than flushed: after an interruption, or once a turn
// has ended, that audio is no longer wanted.
func (s *mediaSender) botStoppedSpeaking(ctx context.Context) {
	s.botMu.Lock()
	if !s.botSpeaking {
		s.botMu.Unlock()
		return
	}
	s.botSpeaking = false
	s.ttsAudioReceived = false
	s.botMu.Unlock()

	s.bufMu.Lock()
	s.buffer = nil
	s.bufMu.Unlock()

	_ = s.out.Broadcast(ctx, func() frames.Frame {
		return s.address(frames.NewBotStoppedSpeakingFrame())
	})
}

// ttsStopped ends the bot's turn on a TTSStoppedFrame, but only when TTS audio
// actually arrived for that turn.
func (s *mediaSender) ttsStopped(ctx context.Context) {
	s.botMu.Lock()
	received := s.ttsAudioReceived
	s.botMu.Unlock()
	if received {
		s.botStoppedSpeaking(ctx)
	}
}

// audioLoop paces queued frames out to the transport. A mixer changes how the
// gaps between frames are handled, so the two cases are separate loops.
func (s *mediaSender) audioLoop(ctx context.Context) {
	defer s.audioWG.Done()
	if s.mixer != nil {
		s.audioLoopWithMixer(ctx)
		return
	}
	s.audioLoopWithoutMixer(ctx)
}

// audioLoopWithMixer blends the mixer into queued audio and fills the gaps
// between frames with the mixer's own audio. Without that filling, auxiliary
// audio would only be audible while the bot speaks and would cut out between
// turns. The generated audio is plain output audio, so it paces the loop through
// the transport without ever marking the bot as speaking.
func (s *mediaSender) audioLoopWithMixer(ctx context.Context) {
	silence := make([]byte, s.chunkSize)
	var lastAudioAt time.Time

	for {
		select {
		case <-ctx.Done():
			return
		case queued := <-s.audioOut:
			if queued == drainMarker {
				s.signalDrained()
				continue
			}
			if pcm, _, _ := outputAudio(queued); pcm != nil {
				if mixed, err := s.mixer.Mix(ctx, pcm); err == nil {
					setOutputAudio(queued, mixed)
				}
				lastAudioAt = time.Now()
			}
			s.handleQueuedFrame(ctx, queued)
		default:
			// Nothing is queued. The bot has stopped speaking once it has been
			// quiet for long enough, and the mixer plays on regardless.
			if time.Since(lastAudioAt) > botVADStopFallback {
				s.botStoppedSpeaking(ctx)
			}
			mixed, err := s.mixer.Mix(ctx, silence)
			if err != nil {
				mixed = silence
			}
			s.handleQueuedFrame(ctx, s.address(
				frames.NewOutputAudioRawFrame(mixed, s.sampleRate, s.channels)))
		}
	}
}

// audioLoopWithoutMixer paces queued frames out to the transport. Receiving
// nothing at all for botVADStopFallback is the fallback that ends the bot's turn
// when no explicit stop reaches the output.
func (s *mediaSender) audioLoopWithoutMixer(ctx context.Context) {
	idle := time.NewTimer(botVADStopFallback)
	defer idle.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-idle.C:
			s.botStoppedSpeaking(ctx)
			idle.Reset(botVADStopFallback)
		case queued := <-s.audioOut:
			if !idle.Stop() {
				select {
				case <-idle.C:
				default:
				}
			}
			idle.Reset(botVADStopFallback)

			if queued == drainMarker {
				s.signalDrained()
				continue
			}
			s.handleQueuedFrame(ctx, queued)
		}
	}
}

// signalDrained releases a drainAudio waiter once the loop reaches its marker.
func (s *mediaSender) signalDrained() {
	s.bufMu.Lock()
	w := s.drainWait
	s.drainWait = nil
	s.bufMu.Unlock()
	if w != nil {
		close(w)
	}
}

// handleQueuedFrame plays one queued frame: it applies the frame's own effect,
// writes whatever audio it carries to the transport, and forwards it downstream.
// A frame whose audio could not be written is not forwarded, so nothing
// downstream treats it as having been sent.
func (s *mediaSender) handleQueuedFrame(ctx context.Context, f frames.Frame) {
	pushDownstream := true

	switch af := f.(type) {
	case frames.OutputAudioFrame:
		s.handleBotSpeech(ctx, f)

		// The mixer has already been blended in by the loop that sourced this
		// frame, so what the frame carries is what goes out, destination and all.
		if err := s.out.self.WriteAudio(ctx, af); err != nil {
			slog.Error("write audio to transport",
				"processor", s.out.Name(), "destination", s.destination, "err", err)
			pushDownstream = false
		}
	case *frames.TTSStoppedFrame:
		s.ttsStopped(ctx)
	default:
		// A frame that carries no audio has waited behind the audio it belongs
		// to (a word-aligned text frame, say). Give the concrete transport a
		// chance to act on it now that that audio has played.
		if err := s.out.self.WriteTransportFrame(ctx, f); err != nil {
			slog.Error("write transport frame",
				"processor", s.out.Name(), "frame", f.Name(), "err", err)
		}
	}

	if pushDownstream {
		_ = s.out.PushFrame(ctx, f, processor.Downstream)
	}
}
