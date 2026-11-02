package livekit

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/pion/webrtc/v4"
)

// readDeadline bounds a blocking RTP read so the read loop can notice
// cancellation between packets.
const readDeadline = 500 * time.Millisecond

// Transport is a LiveKit room transport. It provides the input and output
// processors for a pipeline.
type Transport struct {
	in  *inputTransport
	out *outputTransport
}

// options are the LiveKit-specific settings of a Transport.
type options struct {
	audioInUserTracks bool
}

// Option sets a LiveKit-specific setting of a Transport.
type Option func(*options)

// WithAudioInUserTracks says how several participants' audio enters the
// pipeline. Enabled, the default, each participant's audio is its own stream
// of UserAudioRawFrames tagged with their identity. All of these streams enter
// the pipeline through the one input transport, so a pipeline that runs a
// single speech-to-text and VAD should only receive one of them. Disabled, the
// participants' audio is mixed into a single stream of InputAudioRawFrames,
// which a pipeline with one speech-to-text and VAD can follow however many
// participants publish a microphone.
func WithAudioInUserTracks(enabled bool) Option {
	return func(o *options) { o.audioInUserTracks = enabled }
}

// NewTransport builds a transport over a joined LiveKit connection.
func NewTransport(conn *Connection, params transport.Params, opts ...Option) *Transport {
	o := options{audioInUserTracks: true}
	for _, opt := range opts {
		opt(&o)
	}
	return &Transport{
		in:  newInput(conn, params, o),
		out: newOutput(conn, params),
	}
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

func channels(n int) int {
	if n == 0 {
		return 1
	}
	return n
}

// The mix is made of 10 ms chunks, and waits for a participant with nothing
// buffered until another has 50 ms.
const (
	mixChunkMs   = 10
	mixMaxWaitMs = 50
)

// participantAudio is one decoded frame of a participant's audio at the
// codec's rate, or, with released set, word that their track is gone.
type participantAudio struct {
	participant string
	pcm         []byte
	released    bool
}

// inputTransport reads Opus RTP from every subscribed remote audio track,
// decodes it to PCM, and pushes it into the pipeline at the input sample rate:
// each participant's as UserAudioRawFrames, or all of them mixed into
// InputAudioRawFrames.
type inputTransport struct {
	*transport.BaseInput
	conn *Connection
	opts options

	// audio carries every participant's decoded audio to the one goroutine
	// that converts and pushes it, in the order it was decoded. mu guards it.
	audio chan participantAudio
	// resamplers convert each participant's audio from the 48 kHz the codec
	// decodes to into the rate the transport reads at, one each: a stream
	// resampler keeps filter state between calls, so sharing one would mix
	// the audio of participants who talk at once. Only the audio goroutine
	// touches them.
	resamplers map[string]*resample.Resampler
	// mixer and mixedResampler serve the mixed stream, with user tracks off.
	mixer          *participantAudioMixer
	mixedResampler *resample.Resampler

	readWG     sync.WaitGroup
	mu         sync.Mutex // guards readCancel and readers
	readCancel context.CancelFunc
	// readers stop each participant's track reader, by identity.
	readers map[string]context.CancelFunc
}

func newInput(conn *Connection, params transport.Params, opts options) *inputTransport {
	in := &inputTransport{conn: conn, opts: opts}
	in.BaseInput = transport.NewBaseInput("LiveKitInput", params, in)
	return in
}

// StartReading reads every participant's audio on goroutines of its own.
func (in *inputTransport) StartReading(ctx context.Context) error {
	readCtx, cancel := context.WithCancel(ctx)
	audio := make(chan participantAudio, 64)
	in.resamplers = map[string]*resample.Resampler{}
	in.mixer, in.mixedResampler = nil, nil
	in.mu.Lock()
	in.readCancel = cancel
	in.readers = map[string]context.CancelFunc{}
	in.audio = audio
	in.mu.Unlock()
	in.readWG.Go(func() { in.audioLoop(readCtx, audio) })

	in.conn.OnAudioTrack(func(track *webrtc.TrackRemote, participant string) {
		if readCtx.Err() != nil {
			return
		}
		in.readTrack(readCtx, audio, track, participant)
	})
	in.conn.OnAudioTrackUnsubscribed(func(participant string) {
		select {
		case audio <- participantAudio{participant: participant, released: true}:
		case <-readCtx.Done():
		}
	})

	in.conn.OnMessage(func(raw []byte, sender string) {
		if readCtx.Err() != nil {
			return
		}
		in.PushParticipantMessage(readCtx, raw, sender)
	})
	// A key a SIP caller presses becomes a frame, so the DTMF aggregator works
	// on a LiveKit call the same way it does on a telephony one.
	in.conn.OnDTMF(func(button frames.KeypadEntry) {
		if readCtx.Err() != nil {
			return
		}
		_ = in.PushFrame(readCtx, frames.NewInputDTMFFrame(button), processor.Downstream)
	})

	// The bot is a participant in the room like any other, and it joined when
	// the connection was made, which is before the pipeline was built. So the
	// join is reported here rather than waited for: by the time a pipeline is
	// running over this connection it has already happened.
	in.PushBotConnected(readCtx)

	in.conn.OnParticipantConnected(func(string) {
		if readCtx.Err() != nil {
			return
		}
		in.PushClientConnected(readCtx)
	})
	return nil
}

// StopReading stops the read goroutines.
func (in *inputTransport) StopReading(context.Context) error {
	in.mu.Lock()
	cancel := in.readCancel
	in.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	in.readWG.Wait()
	for id, r := range in.resamplers {
		r.Close()
		delete(in.resamplers, id)
	}
	if in.mixedResampler != nil {
		in.mixedResampler.Close()
		in.mixedResampler = nil
	}
	return nil
}

// sampleRate is the rate the transport reads at, falling back to the codec's
// own rate when none was settled.
func (in *inputTransport) sampleRate() int {
	if r := in.SampleRate(); r > 0 {
		return r
	}
	return opus.SampleRate
}

// readTrack starts reading a participant's audio track. A participant who
// publishes again (a mute and unmute cycle, say) has the reader of their
// previous track stopped first, so two readers never feed the audio goroutine
// for the same participant.
func (in *inputTransport) readTrack(
	ctx context.Context, audio chan<- participantAudio, track *webrtc.TrackRemote, participant string,
) {
	ch := channels(in.Params().AudioInChannels)
	dec, err := opus.NewDecoder(ch)
	if err != nil {
		slog.Error("livekit can't decode a participant's audio", "participant", participant, "err", err)
		return
	}
	trackCtx, stop := context.WithCancel(ctx)
	in.mu.Lock()
	if prev := in.readers[participant]; prev != nil {
		prev()
	}
	in.readers[participant] = stop
	in.mu.Unlock()

	in.readWG.Go(func() { in.readLoop(trackCtx, audio, track, dec, participant) })
}

// readLoop decodes one participant's track, handing each frame to the audio
// goroutine, until the track ends or ctx is done.
func (in *inputTransport) readLoop(
	ctx context.Context, audio chan<- participantAudio, track *webrtc.TrackRemote, dec *opus.Decoder,
	participant string,
) {
	for {
		if ctx.Err() != nil {
			return
		}
		_ = track.SetReadDeadline(time.Now().Add(readDeadline))
		pkt, _, err := track.ReadRTP()
		if err != nil {
			if errors.Is(err, os.ErrDeadlineExceeded) {
				continue
			}
			return
		}
		if len(pkt.Payload) == 0 {
			continue
		}
		pcm, err := dec.Decode(pkt.Payload)
		if err != nil {
			continue
		}
		select {
		case audio <- participantAudio{participant: participant, pcm: pcm}:
		case <-ctx.Done():
			return
		}
	}
}

// audioLoop converts the participants' audio and pushes it into the pipeline,
// in the order it was decoded.
func (in *inputTransport) audioLoop(ctx context.Context, audio <-chan participantAudio) {
	for {
		select {
		case <-ctx.Done():
			return
		case a := <-audio:
			in.handleAudio(ctx, a)
		}
	}
}

// handleAudio pushes one frame of a participant's audio, or drops their
// resampler once their track is gone.
func (in *inputTransport) handleAudio(ctx context.Context, a participantAudio) {
	if a.released {
		if r := in.resamplers[a.participant]; r != nil {
			r.Close()
			delete(in.resamplers, a.participant)
		}
		return
	}
	if !in.opts.audioInUserTracks {
		in.pushMixedAudio(ctx, a)
		return
	}
	r, err := in.resamplerFor(a.participant)
	if err != nil {
		return
	}
	// The codec always decodes at 48 kHz; the pipeline gets the rate it asked
	// for.
	pcm := r.Process(a.pcm)
	if len(pcm) == 0 {
		return
	}
	ch := channels(in.Params().AudioInChannels)
	in.PushAudioFrame(ctx, frames.NewUserAudioRawFrame(a.participant, pcm, in.sampleRate(), ch))
}

// resamplerFor returns a participant's resampler, building it on their first
// audio. Their audio is one continuous stream, so it keeps its filter history
// for as long as their track lasts.
func (in *inputTransport) resamplerFor(participant string) (*resample.Resampler, error) {
	if r := in.resamplers[participant]; r != nil {
		return r, nil
	}
	r, err := newStreamResampler(in.sampleRate(), channels(in.Params().AudioInChannels))
	if err != nil {
		return nil, err
	}
	in.resamplers[participant] = r
	return r, nil
}

// pushMixedAudio mixes a participant's audio into the room's stream and pushes
// what is ready.
func (in *inputTransport) pushMixedAudio(ctx context.Context, a participantAudio) {
	ch := channels(in.Params().AudioInChannels)
	if in.mixer == nil {
		in.mixer = newParticipantAudioMixer(opus.SampleRate, ch, mixChunkMs, mixMaxWaitMs)
	}
	if in.mixedResampler == nil {
		r, err := newStreamResampler(in.sampleRate(), ch)
		if err != nil {
			return
		}
		in.mixedResampler = r
	}
	for _, chunk := range in.mixer.add(a.participant, a.pcm) {
		pcm := in.mixedResampler.Process(chunk)
		if len(pcm) == 0 {
			continue
		}
		in.PushAudioFrame(ctx, frames.NewInputAudioRawFrame(pcm, in.sampleRate(), ch))
	}
}

// newStreamResampler builds a resampler from the codec's rate to rate for one
// continuous stream, which keeps its filter history for as long as it lasts.
func newStreamResampler(rate, ch int) (*resample.Resampler, error) {
	return resample.NewWithConfig(opus.SampleRate, rate, ch, resample.Config{ClearAfter: resample.NeverClear})
}

// outputTransport encodes outgoing PCM chunks into Opus and writes them to the
// published track.
type outputTransport struct {
	*transport.BaseOutput
	conn *Connection
	enc  *opus.Encoder
	tail []byte
	// frameBytes is the size of one 20 ms frame at the output sample rate. The
	// transport writes at that rate and the encoder brings it to the codec's
	// own.
	frameBytes int
	nextSend   time.Time
}

func newOutput(conn *Connection, params transport.Params) *outputTransport {
	out := &outputTransport{conn: conn}
	out.BaseOutput = transport.NewBaseOutput("LiveKitOutput", params, out)
	return out
}

// SendMessage publishes an application message to the room. The payload is
// encoded here rather than by the base, because what a wire carries is the
// transport's own business: a message that is already text is published as it
// is, and anything else is marshaled to JSON.
func (out *outputTransport) SendMessage(_ context.Context, f frames.OutputTransportMessage) error {
	data, err := encodeMessage(f.TransportMessage())
	if err != nil {
		return err
	}
	return out.conn.SendMessage(data)
}

// encodeMessage encodes an application message for the room.
func encodeMessage(message any) ([]byte, error) {
	if text, ok := message.(string); ok {
		return []byte(text), nil
	}
	return json.Marshal(message)
}

// WriteAudio encodes PCM at the output sample rate into 20 ms Opus frames and
// sends them, paced to
// wall-clock time. LiveKit's sample track packetizes and sends a frame the
// instant it is called, so writing a whole utterance back-to-back floods the
// client's jitter buffer; we pace explicitly, exactly as the Pion transport
// does. Audio that does not fill a whole frame is held until the next call.
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	pcm := f.AudioData().Audio
	ch := channels(out.Params().AudioOutChannels)
	if out.enc == nil {
		p := out.Params()
		rate := out.SampleRate()
		if rate <= 0 {
			rate = opus.SampleRate
		}
		enc, err := opus.NewEncoder(opus.EncoderConfig{
			Channels:   ch,
			SampleRate: rate,
			Bitrate:    p.AudioOutBitrate,
		})
		if err != nil {
			return false, err
		}
		out.enc = enc
		out.frameBytes = opus.FrameBytesAt(rate, ch)
	}

	frameBytes := out.frameBytes
	out.tail = append(out.tail, pcm...)
	for len(out.tail) >= frameBytes {
		packet, err := out.enc.Encode(out.tail[:frameBytes])
		if err != nil {
			return false, err
		}
		out.pace(ctx)
		if err := out.conn.WriteAudio(packet, opus.FrameDuration); err != nil {
			return false, err
		}
		out.tail = out.tail[frameBytes:]
	}
	return true, nil
}

// pace blocks until it is time to send the next 20 ms Opus frame, keeping output
// at real time. The clock resets after a gap longer than one frame so playback
// resumes immediately rather than bursting to catch up.
func (out *outputTransport) pace(ctx context.Context) {
	now := time.Now()
	if out.nextSend.IsZero() || now.Sub(out.nextSend) > opus.FrameDuration {
		out.nextSend = now
	}
	if d := time.Until(out.nextSend); d > 0 {
		timer := time.NewTimer(d)
		defer timer.Stop()
		select {
		case <-ctx.Done():
		case <-timer.C:
		}
	}
	out.nextSend = out.nextSend.Add(opus.FrameDuration)
}
