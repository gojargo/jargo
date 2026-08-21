package transport

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/dtmf"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// errNativeDTMFUnimplemented is returned by a transport that reports native DTMF
// support without providing it.
//
//nolint:gochecknoglobals // sentinel error
var errNativeDTMFUnimplemented = errors.New("transport: native DTMF reported as supported but not implemented")

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

// defaultDestination is the sender that serves frames carrying no destination of
// their own. A transport that exposes only one outgoing stream uses just this
// one.
const defaultDestination = ""

// BaseOutput is the tail of a pipeline: it routes each outgoing frame to the
// sender for the destination the frame names, and each sender buffers audio,
// slices it into fixed-size chunks, and hands each chunk to the concrete
// transport to send. Chunking into small, uniform pieces keeps output latency
// low and makes interruptions responsive. A concrete transport embeds it and
// implements OutputDriver to send the audio.
//
// The per-destination state lives on mediaSender rather than here, so a
// transport that carries several outgoing streams (several tracks, say) keeps
// one buffer, one mixer and one speaking state per stream instead of sharing
// one set between them.
type BaseOutput struct {
	*processor.Base
	params Params
	self   OutputDriver

	sampleRate int
	channels   int
	chunkSize  int

	sendersMu sync.RWMutex
	senders   map[string]*mediaSender
}

// NewBaseOutput builds a BaseOutput. self is the embedding transport, used to
// dispatch WriteAudio and to process frames.
func NewBaseOutput(name string, params Params, self OutputDriver) *BaseOutput {
	bo := &BaseOutput{params: params, self: self, senders: map[string]*mediaSender{}}
	bo.Base = processor.New(name, self)
	return bo
}

// SampleRate is the output sample rate in Hz, set when the transport starts.
func (bo *BaseOutput) SampleRate() int { return bo.sampleRate }

// ChunkSize is the size in bytes of the audio chunks the output writes, set when
// the transport starts.
func (bo *BaseOutput) ChunkSize() int { return bo.chunkSize }

// Params returns the transport parameters.
func (bo *BaseOutput) Params() Params { return bo.params }

// WriteAudio is the default no-op; a concrete transport overrides it. A base
// with no transport under it sends nothing, so it reports the audio as unsent.
func (bo *BaseOutput) WriteAudio(context.Context, frames.OutputAudioFrame) (bool, error) {
	return false, nil
}

// SendMessage is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) SendMessage(context.Context, []byte) error { return nil }

// SupportsNativeDTMF reports whether the transport signals a keypress itself.
// The default is false, so the keys are sounded as audio instead. A transport
// whose protocol carries keypresses overrides this and WriteDTMFNative, which
// keeps the keys out of the audio a recording would capture.
func (bo *BaseOutput) SupportsNativeDTMF() bool { return false }

// WriteDTMFNative signals the keys over the transport's own protocol. A
// transport that answers true to SupportsNativeDTMF overrides this; the default
// says it did not, which is what makes claiming support without implementing it
// an error rather than silence.
func (bo *BaseOutput) WriteDTMFNative(context.Context, frames.DTMFOutput) error {
	return errNativeDTMFUnimplemented
}

// WriteDTMF sends the keys the frame carries, natively when the transport can
// and as audio when it cannot.
func (bo *BaseOutput) WriteDTMF(ctx context.Context, f frames.DTMFOutput) error {
	if bo.self.SupportsNativeDTMF() {
		return bo.self.WriteDTMFNative(ctx, f)
	}
	return bo.writeDTMFAudio(ctx, f)
}

// writeDTMFAudio sounds each key in turn as a tone and writes it like any other
// audio, which is how a keypress reaches a call that has only an audio path.
func (bo *BaseOutput) writeDTMFAudio(ctx context.Context, f frames.DTMFOutput) error {
	for _, button := range f.Keys() {
		pcm, err := dtmf.Tone(button, bo.sampleRate)
		if err != nil {
			// A key with no tone is skipped rather than failing the rest: the
			// caller still means the keys around it.
			slog.Warn("skipping a key with no DTMF tone", "transport", bo.Name(), "err", err)
			continue
		}
		audio := frames.NewOutputAudioRawFrame(pcm, bo.sampleRate, 1)
		audio.SetTransportDestination(f.Base().TransportDestination())
		if _, err := bo.self.WriteAudio(ctx, audio); err != nil {
			return err
		}
	}
	return nil
}

// WriteTransportFrame is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) WriteTransportFrame(context.Context, frames.Frame) error { return nil }

// StartWriting is the default no-op; a concrete transport overrides it to open
// its outgoing media path.
func (bo *BaseOutput) StartWriting(context.Context) error { return nil }

// RegisterAudioDestination is the default no-op; a concrete transport overrides
// it to open the outgoing stream a destination names.
func (bo *BaseOutput) RegisterAudioDestination(context.Context, string) error { return nil }

// Setup resolves the rate the transport writes at. A rate configured on the
// transport wins; otherwise it takes the pipeline's output rate, which it knows
// from the moment it is set up rather than when the StartFrame arrives.
func (bo *BaseOutput) Setup(ctx context.Context, s processor.Setup) error {
	if err := bo.Base.Setup(ctx, s); err != nil {
		return err
	}
	bo.sampleRate = pick(bo.params.AudioOutSampleRate, s.AudioOutSampleRate)
	return nil
}

// ProcessFrame handles the transport lifecycle and routes frames to the sender
// for the destination they name.
func (bo *BaseOutput) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := bo.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		// Initialize before forwarding so the chunk size is set before any
		// audio frame can be processed. Nothing downstream of the output
		// transport needs the StartFrame ahead of this.
		bo.startStreaming(ctx)
		if err := bo.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		// Report ready only once the StartFrame has gone downstream, so the
		// pipeline is running by the time anything waiting on the transport is
		// released.
		return bo.PushFrame(ctx, frames.NewOutputTransportReadyFrame(), processor.Upstream)
	case *frames.EndFrame:
		bo.eachSender(func(s *mediaSender) { s.drainAudio(ctx) })
		bo.stopStreaming(ctx)
		return bo.PushFrame(ctx, f, dir)
	case *frames.CancelFrame:
		bo.stopStreaming(ctx)
		return bo.PushFrame(ctx, f, dir)
	case *frames.InterruptionFrame:
		if err := bo.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		// The barge-in cuts the stream the frame names. Another destination is
		// carrying something else (background audio, say), and cutting that too
		// would silence what the interruption was never about.
		if s := bo.senderFor(f); s != nil {
			s.handleInterruption()
			s.botStoppedSpeaking(ctx)
		}
		return nil
	case *frames.OutputDTMFUrgentFrame:
		// Urgent, so the keys go out ahead of the audio already queued, for a
		// keypress answering a prompt that is still playing.
		if err := bo.WriteDTMF(ctx, fr); err != nil {
			return err
		}
		return nil
	case *frames.OutputTransportMessageUrgentFrame:
		// Urgent, so it goes out at once, ahead of whatever is queued, and is
		// not forwarded on.
		bo.sendTransportMessage(ctx, fr.Message)
		return nil
	case frames.SystemFrame:
		// A system frame outranks the queue and is forwarded as it arrives.
		return bo.PushFrame(ctx, f, dir)
	default:
		if dir != processor.Downstream {
			return bo.PushFrame(ctx, f, dir)
		}
		return bo.handleFrame(ctx, f)
	}
}

// handleFrame routes one downstream frame to the sender for the destination it
// names. Audio is buffered and chunked, a mixer control reaches that stream's
// mixer, a frame carrying a presentation timestamp waits on the clock, and
// anything else is queued behind the audio already there so it is forwarded in
// step with playback rather than as it arrives.
func (bo *BaseOutput) handleFrame(ctx context.Context, f frames.Frame) error {
	s := bo.senderFor(f)
	if s == nil {
		return nil
	}
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame, *frames.SpeechOutputAudioRawFrame, *frames.OutputAudioRawFrame:
		s.handleAudioFrame(f)
	case frames.MixerControlFrame:
		if s.mixer != nil {
			_ = s.mixer.ProcessFrame(ctx, fr)
		}
	case *frames.TTSStoppedFrame:
		return s.handleTTSStopped(ctx, fr)
	default:
		if pts, timed := f.Base().PTS(); timed && s.clockQ != nil {
			s.clockQ.push(pts, f)
			return nil
		}
		s.handleSyncFrame(ctx, f)
	}
	return nil
}

// Cleanup stops the senders and the processor.
func (bo *BaseOutput) Cleanup(ctx context.Context) error {
	bo.stopStreaming(ctx)
	// Free the resampler handles after the base stops the process goroutine, so
	// no in-flight resample touches a freed native resampler.
	err := bo.Base.Cleanup(ctx)
	bo.eachSender(func(s *mediaSender) { s.closeResampler() })
	return err
}

// senderFor returns the sender serving the destination f names. A frame for a
// destination that was never registered is dropped with a warning rather than
// being sent somewhere it was not addressed to.
func (bo *BaseOutput) senderFor(f frames.Frame) *mediaSender {
	dest := f.Base().TransportDestination()

	bo.sendersMu.RLock()
	s, ok := bo.senders[dest]
	bo.sendersMu.RUnlock()

	if !ok {
		slog.Warn("transport: destination not registered",
			"processor", bo.Name(), "destination", dest, "frame", f.Name())
		return nil
	}
	return s
}

// eachSender runs fn over every sender.
func (bo *BaseOutput) eachSender(fn func(*mediaSender)) {
	bo.sendersMu.RLock()
	senders := make([]*mediaSender, 0, len(bo.senders))
	for _, s := range bo.senders {
		senders = append(senders, s)
	}
	bo.sendersMu.RUnlock()

	for _, s := range senders {
		fn(s)
	}
}

// mixerFor returns the mixer serving a destination. A per-destination mapping
// picks by name; a single mixer serves the default destination only, so a
// transport with several streams does not silently mix the same auxiliary audio
// into all of them.
func (bo *BaseOutput) mixerFor(destination string) audio.Mixer {
	if m, ok := bo.params.AudioOutMixers[destination]; ok {
		return m
	}
	if destination == defaultDestination {
		return bo.params.AudioOutMixer
	}
	return nil
}

func (bo *BaseOutput) startStreaming(ctx context.Context) {
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

	// Open the transport's own media path before anything can be queued for it.
	// An output that cannot open one can never send, so the failure is fatal
	// rather than something to stream into and drop.
	if err := bo.self.StartWriting(ctx); err != nil {
		bo.PushError(ctx, "transport: open the outgoing media path", err, true)
		return
	}
	bo.setTransportReady(ctx)
}

// setTransportReady registers the outgoing streams and starts a sender for each.
// It runs once the media path is open, so nothing is queued for a stream that
// cannot carry it yet. Reporting the transport ready is separate, because that
// has to wait for the StartFrame to have gone downstream first.
func (bo *BaseOutput) setTransportReady(ctx context.Context) {
	// The default sender always exists: it serves every frame that names no
	// destination of its own.
	destinations := []string{defaultDestination}
	for _, d := range bo.params.AudioOutDestinations {
		if d == defaultDestination {
			continue
		}
		if err := bo.self.RegisterAudioDestination(ctx, d); err != nil {
			bo.PushError(ctx, "register audio destination "+d, err, false)
			continue
		}
		destinations = append(destinations, d)
	}

	bo.sendersMu.Lock()
	bo.senders = make(map[string]*mediaSender, len(destinations))
	for _, d := range destinations {
		bo.senders[d] = newMediaSender(bo, d)
	}
	senders := make([]*mediaSender, 0, len(bo.senders))
	for _, s := range bo.senders {
		senders = append(senders, s)
	}
	bo.sendersMu.Unlock()

	for _, s := range senders {
		s.start(ctx)
	}
}

func (bo *BaseOutput) stopStreaming(ctx context.Context) {
	bo.eachSender(func(s *mediaSender) { s.stop(ctx) })
}

// sendTransportMessage serializes a transport message payload to JSON and hands
// it to the concrete transport. It serves both message frames: the ordered one,
// which reaches here in step with the surrounding audio, and the urgent one,
// which is a system frame and so arrives ahead of anything queued.
//
// A failure is logged, not returned. A returned error becomes an ErrorFrame, and
// anything that reports errors to the client turns that into another message to
// send: if the connection is what failed, sending the report fails too and the
// pipeline feeds itself errors until it runs out of memory. A connection that
// cannot carry a message cannot carry the complaint about it either.
func (bo *BaseOutput) sendTransportMessage(ctx context.Context, message any) {
	data, err := json.Marshal(message)
	if err == nil {
		err = bo.self.SendMessage(ctx, data)
	}
	if err != nil && !canceled(ctx) {
		slog.Error("send transport message", "processor", bo.Name(), "err", err)
	}
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

// setOutputAudio replaces the PCM a bot audio frame carries, leaving everything
// else about the frame alone.
func setOutputAudio(f frames.Frame, pcm []byte) {
	switch fr := f.(type) {
	case *frames.TTSAudioRawFrame:
		fr.Audio = pcm
	case *frames.SpeechOutputAudioRawFrame:
		fr.Audio = pcm
	case *frames.OutputAudioRawFrame:
		fr.Audio = pcm
	}
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

// drainMarker is a sentinel queued on a sender's audio channel to detect when its
// loop has paced out everything ahead of it. It is compared by identity and never
// played.
//
//nolint:gochecknoglobals // sentinel value
var drainMarker = &frames.OutputAudioRawFrame{}
