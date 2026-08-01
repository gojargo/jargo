package transport

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio"
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

// WriteAudio is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) WriteAudio(context.Context, frames.OutputAudioFrame) error { return nil }

// SendMessage is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) SendMessage(context.Context, []byte) error { return nil }

// WriteTransportFrame is the default no-op; a concrete transport overrides it.
func (bo *BaseOutput) WriteTransportFrame(context.Context, frames.Frame) error { return nil }

// RegisterAudioDestination is the default no-op; a concrete transport overrides
// it to open the outgoing stream a destination names.
func (bo *BaseOutput) RegisterAudioDestination(context.Context, string) error { return nil }

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
		bo.startStreaming(ctx, fr)
		return bo.PushFrame(ctx, f, dir)
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
		// A barge-in cuts every outgoing stream, not only the one the frame
		// happens to name.
		bo.eachSender(func(s *mediaSender) {
			s.handleInterruption()
			s.botStoppedSpeaking(ctx)
		})
		return nil
	case *frames.OutputTransportMessageFrame, *frames.OutputTransportMessageUrgentFrame:
		return bo.handleTransportMessage(ctx, f, dir)
	case frames.MixerControlFrame:
		return bo.handleMixerControl(ctx, fr, dir)
	case *frames.TTSAudioRawFrame, *frames.SpeechOutputAudioRawFrame, *frames.OutputAudioRawFrame:
		return bo.routeAudio(ctx, f, dir)
	case *frames.TTSStoppedFrame:
		return bo.routeTTSStopped(ctx, fr, dir)
	default:
		return bo.handleTimedFrame(ctx, f, dir)
	}
}

// routeAudio hands a bot audio frame to the sender for the destination it names.
func (bo *BaseOutput) routeAudio(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if dir != processor.Downstream {
		return bo.PushFrame(ctx, f, dir)
	}
	if s := bo.senderFor(f); s != nil {
		s.handleAudioFrame(f)
	}
	return nil
}

// routeTTSStopped hands a TTS stop to the sender for the destination it names, so
// it lands behind the audio it ends rather than ahead of it.
func (bo *BaseOutput) routeTTSStopped(
	ctx context.Context, f *frames.TTSStoppedFrame, dir processor.Direction,
) error {
	if dir != processor.Downstream {
		return bo.PushFrame(ctx, f, dir)
	}
	s := bo.senderFor(f)
	if s == nil {
		return bo.PushFrame(ctx, f, dir)
	}
	return s.handleTTSStopped(ctx, f)
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

// handleMixerControl hands a mixer control frame to the mixer of the destination
// it names and forwards it on. The mixer reads the frame itself, so a control the
// mixer understands does not have to be translated on the way through, and a new
// one needs no change here.
func (bo *BaseOutput) handleMixerControl(
	ctx context.Context, f frames.MixerControlFrame, dir processor.Direction,
) error {
	if s := bo.senderFor(f); s != nil && s.mixer != nil {
		_ = s.mixer.ProcessFrame(ctx, f)
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

// handleTimedFrame queues a downstream frame carrying a presentation timestamp
// (a word-aligned TTSTextFrame, say) on its destination's clock, so it is
// forwarded at the moment that timestamp names. A frame with no timestamp, or one
// going upstream, has nothing to wait for and is forwarded as it arrives.
func (bo *BaseOutput) handleTimedFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	pts, timed := f.Base().PTS()
	if !timed || dir != processor.Downstream {
		return bo.PushFrame(ctx, f, dir)
	}
	s := bo.senderFor(f)
	if s == nil || s.clockQ == nil {
		return bo.PushFrame(ctx, f, dir)
	}
	s.clockQ.push(pts, f)
	return nil
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
