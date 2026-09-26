package transport

import (
	"context"
	"errors"
	"fmt"
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

	// resampleMu guards the resampler. Converting happens on the process
	// goroutine, but the audio loop ends a turn and resets it from its own, so
	// the two have to be kept apart.
	resampleMu sync.Mutex
	resampler  *resample.Resampler
	resampleIn int

	bufMu sync.Mutex
	// audioRuns is the incoming audio waiting to be cut into chunks, as runs of
	// audio that came from frames with the same interruptible flag, so a chunk
	// keeps the flag of the audio it holds.
	audioRuns []audioRun
	// newChunk rebuilds a buffered chunk as the same frame type as the audio it
	// was buffered from, so TTS audio stays a TTSAudioRawFrame and a speech
	// stream stays a SpeechOutputAudioRawFrame. The bot-speaking bookkeeping
	// reads that type to tell which of the two it is pacing out.
	newChunk func(pcm []byte, sampleRate, numChannels int) frames.Frame

	// lifeMu serializes starting, restarting and stopping the audio loop, so a
	// barge-in restarting it cannot add to the wait group while a teardown is
	// waiting on it.
	lifeMu sync.Mutex
	// parentCtx is what the audio context is derived from, kept so a restart
	// after an interruption outlives the frame that caused it.
	parentCtx   context.Context
	audioOut    *frameQueue
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
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()

	s.parentCtx = ctx
	s.bufMu.Lock()
	s.clearAudioBuffer()
	s.bufMu.Unlock()

	if s.mixer != nil {
		_ = s.mixer.Start(ctx, s.sampleRate)
	}

	s.audioCtx, s.audioCancel = context.WithCancel(ctx)
	s.audioOut = newFrameQueue()
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

	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()

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
	s.resampleMu.Lock()
	defer s.resampleMu.Unlock()
	if s.resampler != nil {
		s.resampler.Close()
		s.resampler = nil
	}
}

// flushResampler returns the audio the resampler is still holding in its filter
// and clears it, for a run of speech that has ended. See resample for why the
// resampler never clears itself.
func (s *mediaSender) flushResampler() []byte {
	s.resampleMu.Lock()
	defer s.resampleMu.Unlock()
	if s.resampler == nil {
		return nil
	}
	return s.resampler.Flush()
}

// resetResampler discards the audio the resampler is still holding, for a run of
// speech that was abandoned rather than finished.
func (s *mediaSender) resetResampler() {
	s.resampleMu.Lock()
	defer s.resampleMu.Unlock()
	if s.resampler != nil {
		s.resampler.Clear()
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

	s.audioOut.push(drainMarker)
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
	s.bufferAudio(pcm, !frames.Interruptible(f))
	var chunks []frames.Frame
	for s.bufferedAudioBytes() >= s.chunkSize {
		audio, uninterruptible := s.takeAudioChunk()
		chunk := s.address(build(audio, s.sampleRate, channels))
		chunk.Base().SetInterruptible(!uninterruptible)
		chunks = append(chunks, chunk)
	}
	out := s.audioOut
	s.bufMu.Unlock()

	for _, chunk := range chunks {
		out.push(chunk)
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
	out.push(f)
	return nil
}

// handleSyncFrame queues a frame that carries no audio behind the audio already
// queued for this stream, so it is forwarded in step with playback rather than
// as it arrives. A text frame that belongs with the words being spoken, say,
// would otherwise overtake the audio it describes by however much is buffered.
func (s *mediaSender) handleSyncFrame(ctx context.Context, f frames.Frame) {
	s.bufMu.Lock()
	audioCtx, out := s.audioCtx, s.audioOut
	s.bufMu.Unlock()
	if audioCtx == nil || out == nil {
		// Nothing is pacing anything yet, so there is nothing to wait behind.
		_ = s.out.PushFrame(ctx, f, processor.Downstream)
		return
	}
	out.push(f)
}

// enqueueFlushedAudioBuffer queues every bit of audio still held back at the end
// of a run of speech: what the resampler is holding in its filter, and what is
// left in the buffer, padded out to a full chunk with silence. Each goes out as
// the same frame type as the audio it was buffered from, through the normal
// playback path (write, error handling, bot-speaking bookkeeping) like any other
// chunk, keeping its order relative to whatever is queued after it.
func (s *mediaSender) enqueueFlushedAudioBuffer() {
	if s.chunkSize == 0 {
		return
	}
	// The resampler holds the tail of the audio it was fed, which belongs at the
	// end of this run of speech, with the last run's flag.
	tail := s.flushResampler()

	s.bufMu.Lock()
	lastUninterruptible := false
	if n := len(s.audioRuns); n > 0 {
		lastUninterruptible = s.audioRuns[n-1].uninterruptible
	}
	s.bufferAudio(tail, lastUninterruptible)
	if len(s.audioRuns) == 0 {
		s.bufMu.Unlock()
		return
	}
	build := s.newChunk
	if build == nil {
		build = chunkBuilder(nil)
	}
	// The flushed tail can be longer than a chunk, so queue whole chunks first
	// and pad only what is left over.
	var flushed []frames.Frame
	enqueue := func(audio []byte, uninterruptible bool) {
		chunk := s.address(build(audio, s.sampleRate, s.channels))
		chunk.Base().SetInterruptible(!uninterruptible)
		flushed = append(flushed, chunk)
	}
	for s.bufferedAudioBytes() >= s.chunkSize {
		enqueue(s.takeAudioChunk())
	}
	if len(s.audioRuns) > 0 {
		audio, uninterruptible := s.takeAudioChunk()
		enqueue(padChunk(audio, s.chunkSize), uninterruptible)
	}
	audioCtx, out := s.audioCtx, s.audioOut
	s.bufMu.Unlock()

	if audioCtx == nil || out == nil {
		return
	}
	for _, chunk := range flushed {
		out.push(chunk)
	}
}

// audioRun is audio waiting in the buffer that came from frames with the same
// interruptible flag.
type audioRun struct {
	audio           []byte
	uninterruptible bool
}

// bufferAudio adds output-rate audio to the buffer, noting whether it came from
// uninterruptible frames. The caller holds bufMu.
func (s *mediaSender) bufferAudio(audio []byte, uninterruptible bool) {
	if len(audio) == 0 {
		return
	}
	if n := len(s.audioRuns); n > 0 && s.audioRuns[n-1].uninterruptible == uninterruptible {
		s.audioRuns[n-1].audio = append(s.audioRuns[n-1].audio, audio...)
		return
	}
	s.audioRuns = append(s.audioRuns, audioRun{
		audio: append([]byte(nil), audio...), uninterruptible: uninterruptible,
	})
}

// bufferedAudioBytes is how much audio the buffer holds. The caller holds bufMu.
func (s *mediaSender) bufferedAudioBytes() int {
	n := 0
	for _, run := range s.audioRuns {
		n += len(run.audio)
	}
	return n
}

// takeAudioChunk cuts up to a chunk from the front of the buffer, and reports
// whether any of it is uninterruptible. The caller holds bufMu.
func (s *mediaSender) takeAudioChunk() ([]byte, bool) {
	chunk := make([]byte, 0, s.chunkSize)
	uninterruptible := false
	for len(chunk) < s.chunkSize && len(s.audioRuns) > 0 {
		run := &s.audioRuns[0]
		uninterruptible = uninterruptible || run.uninterruptible
		needed := s.chunkSize - len(chunk)
		if len(run.audio) <= needed {
			chunk = append(chunk, run.audio...)
			s.audioRuns = s.audioRuns[1:]
		} else {
			chunk = append(chunk, run.audio[:needed]...)
			run.audio = run.audio[needed:]
		}
	}
	return chunk, uninterruptible
}

// clearAudioBuffer drops the buffered audio. The caller holds bufMu.
func (s *mediaSender) clearAudioBuffer() { s.audioRuns = nil }

// resample converts audio at sampleRate to the transport output rate. The
// resampler is created lazily and reused across frames.
//
// It is built never to clear its own filter history. The boundaries of a run of
// speech are signaled explicitly, by flushResampler when one ends and
// resetResampler when one is abandoned, and the output runs ahead of playback,
// so a resampler left to decide for itself would read the pause between two TTS
// chunks as the end of the stream and throw away the tail of what it was given.
func (s *mediaSender) resample(pcm []byte, sampleRate, channels int) []byte {
	if sampleRate == s.sampleRate {
		return pcm
	}
	s.resampleMu.Lock()
	defer s.resampleMu.Unlock()
	if s.resampler == nil || s.resampleIn != sampleRate {
		if s.resampler != nil {
			s.resampler.Close()
			s.resampler = nil
		}
		r, err := resample.NewWithConfig(sampleRate, s.sampleRate, channels,
			resample.Config{ClearAfter: resample.NeverClear})
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
	s.clearAudioBuffer()
	s.bufMu.Unlock()
	if s.clockQ != nil {
		// The frames waiting on the clock belong to audio that will never play.
		s.clockQ.drop()
	}

	q := s.audioOut
	if q == nil {
		return
	}
	// Frames marked uninterruptible have to be delivered even through a
	// barge-in, and a mixer has to keep playing through one, so in either case
	// the loop keeps running and only the queue is cleared. Canceling it would
	// stop the mixer's own output too, leaving an audible gap in the background.
	if q.hasUninterruptible() || s.mixer != nil {
		q.reset()
		return
	}
	// Nothing has to survive, so restart the loop instead of draining it: that
	// cuts short a write already in flight rather than leaving the barge-in
	// waiting behind it.
	s.restartAudioLoop()
}

// restartAudioLoop stops the audio loop and starts a fresh one on an empty
// queue.
func (s *mediaSender) restartAudioLoop() {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()

	cancel := s.audioCancel
	if cancel == nil {
		return
	}
	cancel()
	s.audioWG.Wait()

	s.audioCtx, s.audioCancel = context.WithCancel(s.parentCtx)
	s.audioOut = newFrameQueue()
	s.audioWG.Add(1)
	go s.audioLoop(s.audioCtx)
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
// has ended, that audio is no longer wanted. The same goes for what the
// resampler is still holding, which would otherwise be prepended to whatever the
// bot says next.
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
	s.clearAudioBuffer()
	s.bufMu.Unlock()
	s.resetResampler()

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
	q := s.audioOut
	silence := make([]byte, s.chunkSize)
	var lastAudioAt time.Time

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if queued, ok := q.tryGet(); ok {
			if queued == drainMarker {
				// Everything queued has been paced out; the turn is over.
				s.sendEndSilence(ctx)
				s.signalDrained()
				continue
			}
			if s.blendMixer(ctx, queued) {
				lastAudioAt = time.Now()
			}
			if !s.handleQueuedFrame(ctx, queued) {
				return
			}
			continue
		}

		// Nothing is queued. The bot has stopped speaking once it has been quiet
		// for long enough, and the mixer plays on regardless.
		if time.Since(lastAudioAt) > botVADStopFallback {
			s.botStoppedSpeaking(ctx)
		}
		mixed, err := s.mixer.Mix(ctx, silence)
		if err != nil {
			mixed = silence
		}
		if !s.handleQueuedFrame(ctx, s.address(
			frames.NewOutputAudioRawFrame(mixed, s.sampleRate, s.channels))) {
			return
		}
	}
}

// blendMixer mixes the mixer's audio into a queued frame, in place. It reports
// whether the frame carried any audio to mix into, which is how the loop knows
// when audio last flowed. A mix that fails leaves the frame as it was, so the
// bot is still heard without the auxiliary audio behind it.
func (s *mediaSender) blendMixer(ctx context.Context, f frames.Frame) bool {
	pcm, _, _ := outputAudio(f)
	if pcm == nil {
		return false
	}
	if mixed, err := s.mixer.Mix(ctx, pcm); err == nil {
		setOutputAudio(f, mixed)
	}
	return true
}

// audioLoopWithoutMixer paces queued frames out to the transport. Receiving
// nothing at all for botVADStopFallback is the fallback that ends the bot's turn
// when no explicit stop reaches the output.
func (s *mediaSender) audioLoopWithoutMixer(ctx context.Context) {
	q := s.audioOut
	idle := time.NewTimer(botVADStopFallback)
	defer idle.Stop()
	for {
		// Check for a stop before taking anything else off the queue. A loop
		// that only looked when the queue ran dry would keep writing a cut-off
		// turn's audio all the way to the end of it.
		select {
		case <-ctx.Done():
			return
		default:
		}

		if queued, ok := q.tryGet(); ok {
			if !idle.Stop() {
				select {
				case <-idle.C:
				default:
				}
			}
			idle.Reset(botVADStopFallback)

			if queued == drainMarker {
				// Everything queued has been paced out; the turn is over.
				s.sendEndSilence(ctx)
				s.signalDrained()
				continue
			}
			if !s.handleQueuedFrame(ctx, queued) {
				return
			}
			continue
		}
		select {
		case <-ctx.Done():
			return
		case <-idle.C:
			s.botStoppedSpeaking(ctx)
			idle.Reset(botVADStopFallback)
		case <-q.wait():
		}
	}
}

// sendEndSilence writes a short run of silence after the last of the audio, so
// the closing words are not clipped by whatever closes on top of them. It is
// bounded by a timeout: a transport that cannot take the silence must not hold
// the shutdown up behind it.
func (s *mediaSender) sendEndSilence(ctx context.Context) {
	secs := s.out.params.AudioOutEndSilenceSecs
	if secs <= 0 {
		return
	}
	const sampleWidth = 2 // 16-bit samples
	silence := frames.NewOutputAudioRawFrame(
		make([]byte, s.sampleRate*sampleWidth*secs), s.sampleRate, 1)
	silence.SetTransportDestination(s.destination)

	s.writeAudio(ctx, silence)
}

// writeAudio hands one chunk to the transport, bounded, and reports whether the
// transport took it.
//
// A peer that stops reading blocks the write on socket buffers that never drain.
// Nothing about that is an error the transport can report, so the only signal
// available is that the write never returns, and by then the connection is no
// longer usable. Left unbounded the audio loop parks: the bot goes silent, the
// queue grows without limit, and the frame that ends the session is stranded
// inside the transport, since it is forwarded only once the stop returns.
//
// Reporting the timeout costs the transport its usability, and a write is
// skipped while it is unusable, so a peer is written off once however much audio
// is still queued behind it rather than once per chunk.
func (s *mediaSender) writeAudio(ctx context.Context, af frames.OutputAudioFrame) bool {
	if !s.out.Usable() {
		return false
	}
	timeout := s.out.params.AudioOutWrite()
	writeCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	sent, err := s.out.self.WriteAudio(writeCtx, af)
	if errors.Is(writeCtx.Err(), context.DeadlineExceeded) && !canceled(ctx) {
		s.out.PushError(ctx, fmt.Sprintf(
			"timed out after %s writing audio to the transport, the peer has stopped reading",
			timeout), writeCtx.Err(), false, processor.ForceTreatAsPermanent())
		return false
	}
	if err != nil && !canceled(ctx) {
		slog.Error("write audio to transport",
			"processor", s.out.Name(), "destination", s.destination, "err", err)
	}
	return sent && err == nil
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
// A frame whose audio the transport did not send is not forwarded, so nothing
// downstream treats it as having been heard.
//
// It reports whether the loop that called it should carry on. Cancellation ends
// it: the loop is being restarted by a barge-in or torn down by a stop, so the
// frame it was holding belongs to a turn that is over. That frame is abandoned
// where it stands, unreported and unforwarded, and the loop ends with it.
func (s *mediaSender) handleQueuedFrame(ctx context.Context, f frames.Frame) bool {
	pushDownstream := true

	switch af := f.(type) {
	case frames.OutputAudioFrame:
		s.handleBotSpeech(ctx, f)

		// The mixer has already been blended in by the loop that sourced this
		// frame, so what the frame carries is what goes out, destination and all.
		//
		// Audio the transport had nowhere to put is no more heard than audio a
		// failed write dropped, so neither is forwarded.
		pushDownstream = s.writeAudio(ctx, af)
	case *frames.OutputTransportMessageFrame:
		// The ordered message: it waited behind the audio around it, so it
		// reaches the client in step with what the client is hearing.
		s.out.sendTransportMessage(ctx, af)
	case *frames.TTSStoppedFrame:
		s.ttsStopped(ctx)
	case *frames.OutputDTMFFrame:
		// The keys waited behind the audio around them, so they land where the
		// caller meant them rather than over what was still being said.
		if err := s.out.WriteDTMF(ctx, af); err != nil && !canceled(ctx) {
			slog.Error("write dtmf to transport",
				"processor", s.out.Name(), "destination", s.destination, "err", err)
		}
	default:
		// A frame that carries no audio has waited behind the audio it belongs
		// to (a word-aligned text frame, say). Give the concrete transport a
		// chance to act on it now that that audio has played.
		if err := s.out.self.WriteTransportFrame(ctx, f); err != nil && !canceled(ctx) {
			slog.Error("write transport frame",
				"processor", s.out.Name(), "frame", f.Name(), "err", err)
		}
	}

	// Whichever of those calls the cancellation landed in, and whether or not it
	// reported anything back, the frame goes no further: one the transport never
	// finished sending is not forwarded as though it had been heard.
	if canceled(ctx) {
		return false
	}
	if pushDownstream {
		_ = s.out.PushFrame(ctx, f, processor.Downstream)
	}
	return true
}

// canceled reports whether ctx is done, which is how a barge-in restarting the
// audio loop, or a stop tearing it down, cuts short whatever send is in flight
// at the time. That is the interruption doing its job rather than the transport
// failing, so the send it caught is abandoned instead of being reported.
//
// It tests the caller's own context rather than the error the send returned, so
// that a context.Canceled surfacing from inside a transport (its connection
// going away mid-send, say) is still reported as the failure it is.
func canceled(ctx context.Context) bool { return ctx.Err() != nil }
