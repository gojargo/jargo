package pionrtc

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
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

// queuedFrames is how far the writer may run ahead of the sender. It has to be
// more than one: the sender takes whatever is queued at the instant it comes
// round, so a queue that only ever holds the frame being waited on leaves no
// slack, and any hesitation upstream — a goroutine not scheduled promptly, work
// done downstream between chunks — arrives too late and gets a frame of silence
// spliced into the middle of speech instead. The cushion is small enough that
// bot-speaking state stays close to what is actually playing.
const queuedFrames = 8

// starvationFrames is the longest run of silence that still counts as a gap in
// speech once real audio resumes after it. A longer run is taken as the end of
// an utterance.
const starvationFrames = 10

// gapTracker accounts for the silence the sender emits. Only silence that real
// audio resumes after is a fault: the writer fell behind and a word was chopped.
// Silence that nothing resumes is the talker having stopped, which is why a run
// stays pending until it is known which one it was — counting it on sight charges
// every utterance for its own ending.
type gapTracker struct {
	silence int64
	starved int64
	gaps    int64
	pending int64
	spoken  bool
	// sent is the frame the sender is on, so a gap can be placed in the session
	// rather than only counted at the end of it.
	sent int64
}

// real records a frame of real audio, settling any silence run before it.
func (g *gapTracker) real() {
	if g.pending > 0 && g.pending <= starvationFrames {
		g.starved += g.pending
		g.gaps++
		slog.Debug("audio gap spliced into speech",
			"gap", time.Duration(g.pending)*opus.FrameDuration,
			"at", time.Duration(g.sent)*opus.FrameDuration,
			"gaps", g.gaps)
	}
	g.pending = 0
	g.spoken = true
}

// quiet records a frame of silence sent because the queue was empty.
func (g *gapTracker) quiet() {
	g.silence++
	if g.spoken {
		g.pending++
	}
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
	queue   chan []byte
	cancel  context.CancelFunc
	sendWG  sync.WaitGroup
	running bool
	// queued counts frames handed over but not yet sent, so a graceful end can
	// wait for them rather than cutting the last words off.
	queued atomic.Int64
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
	case *frames.EndFrame:
		// The base output has handed over everything it had; let it play before
		// the sender goes away, or the farewell loses its last words.
		out.waitDrained(ctx)
		out.stopSending()
	case *frames.CancelFrame:
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
	out.queue = make(chan []byte, queuedFrames)
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
		case <-q:
			out.queued.Add(-1)
		default:
			return
		}
	}
}

// waitDrained blocks until everything handed to the sender has gone out, so a
// graceful end plays the last of the audio instead of clipping it. Bounded, so a
// sender that has already stopped cannot hang the shutdown.
func (out *outputTransport) waitDrained(ctx context.Context) {
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(opus.FrameDuration)
	defer tick.Stop()
	for out.queued.Load() > 0 {
		select {
		case <-tick.C:
		case <-ctx.Done():
			return
		case <-out.conn.Done():
			return
		case <-deadline.C:
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
	var gap gapTracker

	defer func() {
		slog.Info("pion sender stopped", "processor", out.Name(), "frames", sent,
			"silence", gap.silence, "starved", gap.starved, "gaps", gap.gaps)
	}()

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

		pcm := quiet
		gap.sent = sent
		select {
		case frame := <-out.queue:
			pcm = frame
			out.queued.Add(-1)
			gap.real()
		default:
			gap.quiet()
		}

		packet, err := out.enc.Encode(pcm)
		if err == nil {
			err = out.conn.WriteAudio(packet, opus.FrameDuration)
		}
		if err != nil {
			slog.Error("write audio", "processor", out.Name(), "err", err)
		}
		sent++
	}
}

// WriteAudio hands PCM to the sender a frame at a time. It blocks only when the
// sender is already as far ahead as the queue allows, which is what paces the
// pipeline to real time; waiting on each frame individually would instead keep
// the queue empty and starve the sender. Audio that does not fill a whole frame
// is held until the next call.
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) error {
	pcm := f.AudioData().Audio

	out.mu.Lock()
	q, running := out.queue, out.running
	if !running {
		out.mu.Unlock()
		return nil
	}
	frameBytes := opus.FrameBytes(channels(out.Params().AudioOutChannels))
	out.tail = append(out.tail, pcm...)
	var batch [][]byte
	for len(out.tail) >= frameBytes {
		frame := make([]byte, frameBytes)
		copy(frame, out.tail[:frameBytes])
		batch = append(batch, frame)
		out.tail = out.tail[frameBytes:]
	}
	out.mu.Unlock()

	for _, frame := range batch {
		out.queued.Add(1)
		select {
		case q <- frame:
		case <-ctx.Done():
			out.queued.Add(-1)
			return ctx.Err()
		case <-out.conn.Done():
			out.queued.Add(-1)
			return nil
		}
	}
	return nil
}
