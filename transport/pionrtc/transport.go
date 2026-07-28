package pionrtc

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
)

// readDeadline bounds a blocking RTP read so the read loop can notice
// cancellation between packets.
const readDeadline = 500 * time.Millisecond

// Transport is a WebRTC transport backed by a Pion connection. It provides the
// input and output processors for a pipeline.
type Transport struct {
	in  *inputTransport
	out *outputTransport
}

// NewTransport builds a WebRTC transport over conn.
func NewTransport(conn *Connection, params transport.Params) *Transport {
	return &Transport{
		in:  newInput(conn, params),
		out: newOutput(conn, params),
	}
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

// inputTransport reads Opus RTP from the connection, decodes it to PCM, and
// pushes InputAudioRawFrames into the pipeline.
type inputTransport struct {
	*transport.BaseInput
	conn *Connection
	dec  *opus.Decoder

	readWG     sync.WaitGroup
	mu         sync.Mutex // guards readCancel
	readCancel context.CancelFunc
}

func newInput(conn *Connection, params transport.Params) *inputTransport {
	in := &inputTransport{conn: conn}
	in.BaseInput = transport.NewBaseInput("PionInput", params, in)
	return in
}

func channels(n int) int {
	if n == 0 {
		return 1
	}
	return n
}

// StartReading decodes incoming audio on its own goroutine.
func (in *inputTransport) StartReading(ctx context.Context) error {
	dec, err := opus.NewDecoder(channels(in.Params().AudioInChannels))
	if err != nil {
		return err
	}
	in.dec = dec

	readCtx, cancel := context.WithCancel(ctx)
	in.mu.Lock()
	in.readCancel = cancel
	in.mu.Unlock()
	in.readWG.Add(1)
	go in.readLoop(readCtx)

	// Surface data channel messages as frames in the pipeline.
	in.conn.OnMessage(func(raw []byte) {
		if readCtx.Err() != nil {
			return
		}
		in.PushTransportMessage(readCtx, raw)
	})
	return nil
}

// StopReading stops the read goroutine.
func (in *inputTransport) StopReading(context.Context) error {
	in.mu.Lock()
	cancel := in.readCancel
	in.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	in.readWG.Wait()
	return nil
}

func (in *inputTransport) readLoop(ctx context.Context) {
	defer in.readWG.Done()

	track, err := in.conn.RemoteAudioTrack(ctx)
	if err != nil {
		return
	}
	ch := channels(in.Params().AudioInChannels)

	for {
		if ctx.Err() != nil {
			return
		}
		_ = track.SetReadDeadline(time.Now().Add(readDeadline))
		pkt, _, err := track.ReadRTP()
		if err != nil {
			// A deadline lets us re-check ctx; any other error means the track
			// is gone.
			if errors.Is(err, os.ErrDeadlineExceeded) {
				continue
			}
			return
		}
		if len(pkt.Payload) == 0 {
			continue
		}
		pcm, err := in.dec.Decode(pkt.Payload)
		if err != nil {
			continue
		}
		in.PushAudioFrame(ctx, frames.NewInputAudioRawFrame(pcm, opus.SampleRate, ch))
	}
}

// queuedFrames bounds the frames waiting to be sent. The base output already
// buffers well ahead of this, so a shallow queue is enough to keep the sender fed
// without holding a second copy of the same audio.
const queuedFrames = 64

// pending is one 20 ms frame of PCM waiting its turn, and the channel closed
// once it has been sent.
type pending struct {
	pcm  []byte
	sent chan struct{}
}

// outputTransport encodes outgoing PCM into Opus and writes it to the
// connection's audio track at real time.
//
// The sender goroutine owns the clock and writes on every frame boundary for as
// long as the session lasts, falling back to silence whenever nothing is queued
// — the same thing the local audio transport does in its playback callback, and
// what a device's pull callback forces on you. That is what keeps RTP timestamps
// honest: they advance one frame per packet whatever the wall clock did, so a
// sender that goes quiet during a gap and resumes leaves the audio after it
// timestamped as though the gap never happened. A receiver schedules playout from
// those timestamps, reads that as delay, conceals by repeating the last frame,
// then compresses once packets bunch up again — stuttered and clipped words. With
// the sender writing every frame, elapsed frames and elapsed time cannot diverge.
type outputTransport struct {
	*transport.BaseOutput
	conn *Connection
	enc  *opus.Encoder
	tail []byte

	mu      sync.Mutex
	queue   chan *pending
	cancel  context.CancelFunc
	sendWG  sync.WaitGroup
	running bool
}

func newOutput(conn *Connection, params transport.Params) *outputTransport {
	out := &outputTransport{conn: conn}
	out.BaseOutput = transport.NewBaseOutput("PionOutput", params, out)
	return out
}

// ProcessFrame drives the sender's lifecycle around the base output: start it on
// the StartFrame so silence is already flowing before the first word, drop
// anything queued on a barge-in, and stop it when the session ends.
func (out *outputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := out.BaseOutput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir != processor.Downstream {
		return nil
	}
	switch f.(type) {
	case *frames.StartFrame:
		return out.startSending()
	case *frames.InterruptionFrame:
		out.discardQueued()
	case *frames.EndFrame, *frames.CancelFrame:
		out.stopSending()
	}
	return nil
}

// SendMessage sends an application message over the data channel.
func (out *outputTransport) SendMessage(_ context.Context, data []byte) error {
	return out.conn.SendMessage(data)
}

// startSending brings up the sender goroutine. It runs for the whole session, so
// the receiver is already being fed before the first word rather than having to
// build its buffer from a cold start mid-sentence.
func (out *outputTransport) startSending() error {
	out.mu.Lock()
	defer out.mu.Unlock()
	if out.running {
		return nil
	}
	ch := channels(out.Params().AudioOutChannels)
	p := out.Params()
	enc, err := opus.NewEncoder(opus.EncoderConfig{
		Channels:           ch,
		Bitrate:            p.AudioOutBitrate,
		InbandFEC:          p.AudioOutFEC,
		ExpectedPacketLoss: p.AudioOutExpectedPacketLoss,
	})
	if err != nil {
		return err
	}
	// The encoder is touched only by the sender goroutine from here on, so
	// packets cannot be sent in a different order than they were encoded.
	out.enc = enc
	out.queue = make(chan *pending, queuedFrames)
	ctx, cancel := context.WithCancel(context.Background())
	out.cancel = cancel
	out.running = true
	out.sendWG.Add(1)
	go out.sendLoop(ctx, opus.FrameBytes(ch))
	return nil
}

// stopSending shuts the sender down and waits for it to finish.
func (out *outputTransport) stopSending() {
	out.mu.Lock()
	cancel := out.cancel
	out.cancel = nil
	out.running = false
	out.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	out.sendWG.Wait()
}

// discardQueued drops audio that has not been sent yet. A barge-in has to take
// effect now: anything already queued belongs to the turn the user just cut off.
// Waiters are released rather than left blocked on audio that will never go out.
func (out *outputTransport) discardQueued() {
	out.mu.Lock()
	q := out.queue
	out.tail = nil
	out.mu.Unlock()
	if q == nil {
		return
	}
	for {
		select {
		case p := <-q:
			close(p.sent)
		default:
			return
		}
	}
}

// sendLoop writes one frame on every frame boundary until the session ends,
// sending queued audio when there is any and silence when there is not.
//
// The schedule comes from a fixed origin — frames sent times the frame duration —
// rather than from a ticker, which coalesces the ticks it misses and would let
// the stream fall permanently behind the wall clock while its timestamps claimed
// otherwise. Running late here instead sends back to back until the count catches
// up, so elapsed frames and elapsed time stay equal.
func (out *outputTransport) sendLoop(ctx context.Context, frameBytes int) {
	defer out.sendWG.Done()

	quiet := make([]byte, frameBytes)
	start := time.Now()
	var sent int64

	for {
		if d := time.Until(start.Add(time.Duration(sent) * opus.FrameDuration)); d > 0 {
			timer := time.NewTimer(d)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-out.conn.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-out.conn.Done():
			return
		default:
		}

		pcm, done := quiet, (chan struct{})(nil)
		select {
		case p := <-out.queue:
			pcm, done = p.pcm, p.sent
		default:
		}

		packet, err := out.enc.Encode(pcm)
		if err == nil {
			err = out.conn.WriteAudio(packet, opus.FrameDuration)
		}
		if err != nil {
			slog.Error("write audio", "processor", out.Name(), "err", err)
		}
		if done != nil {
			close(done)
		}
		sent++
	}
}

// WriteAudio hands PCM to the sender a frame at a time and returns once the last
// of it has gone out, so callers still see a write that takes as long as the
// audio does — the drain at the end of a turn depends on that. Audio that does
// not fill a whole frame is held until the next call.
func (out *outputTransport) WriteAudio(ctx context.Context, pcm []byte) error {
	out.mu.Lock()
	q, running := out.queue, out.running
	if !running {
		out.mu.Unlock()
		return nil
	}
	frameBytes := opus.FrameBytes(channels(out.Params().AudioOutChannels))
	out.tail = append(out.tail, pcm...)
	var last chan struct{}
	var batch []*pending
	for len(out.tail) >= frameBytes {
		frame := make([]byte, frameBytes)
		copy(frame, out.tail[:frameBytes])
		p := &pending{pcm: frame, sent: make(chan struct{})}
		batch = append(batch, p)
		last = p.sent
		out.tail = out.tail[frameBytes:]
	}
	out.mu.Unlock()

	for _, p := range batch {
		select {
		case q <- p:
		case <-ctx.Done():
			return ctx.Err()
		case <-out.conn.Done():
			return nil
		}
	}
	if last == nil {
		return nil
	}
	select {
	case <-last:
	case <-ctx.Done():
		return ctx.Err()
	case <-out.conn.Done():
	}
	return nil
}
