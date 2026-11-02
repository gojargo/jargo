package rtc

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
)

// readDeadline bounds a blocking RTP read so the read loop can notice
// cancellation between packets.
const readDeadline = 500 * time.Millisecond

// Transport is a WebRTC transport backed by a Pion connection. It provides the
// input and output processors for a pipeline.
//
// It raises transport.EventClientConnected when the peer connection is
// established (again after a restart), transport.EventClientDisconnected when
// the client goes away, and transport.EventAppMessage for each message the
// client sends. The client passed to the first two is the *Connection.
type Transport struct {
	in   *inputTransport
	out  *outputTransport
	conn *Connection
	// closing is set once the bot disconnects, so the connection closing then
	// is not reported as the client leaving.
	closing atomic.Bool
}

// NewTransport builds a WebRTC transport over conn.
func NewTransport(conn *Connection, params transport.Params) *Transport {
	t := &Transport{conn: conn}
	t.in = newInput(conn, params)
	t.out = newOutput(conn, params, &t.closing)

	ev := t.in.Events()
	ev.Register(transport.EventAppMessage, false)
	ev.Register(transport.EventClientConnected, false)
	ev.Register(transport.EventClientDisconnected, false)

	conn.OnMessage(t.onAppMessage)
	conn.OnConnected(t.onClientConnected)
	conn.OnClosed(t.onClientClosed)
	return t
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

// Events returns the registry of the events the transport raises.
func (t *Transport) Events() *events.Registry { return t.in.Events() }

// onAppMessage surfaces a message from the client as a frame in the pipeline,
// then raises EventAppMessage with the message and the peer connection's id.
func (t *Transport) onAppMessage(raw []byte) {
	ctx := t.in.readContext()
	if ctx == nil {
		return
	}
	t.in.PushTransportMessage(ctx, raw)
	t.in.Events().Call(ctx, transport.EventAppMessage, t, json.RawMessage(raw), t.conn.PCID())
}

// onClientConnected raises EventClientConnected and reports the client to the
// pipeline. The connection raises it only once the pipeline is ready for it.
func (t *Transport) onClientConnected() {
	ctx := t.in.readContext()
	if ctx == nil {
		return
	}
	t.in.Events().Call(ctx, transport.EventClientConnected, t, t.conn)
	t.in.PushClientConnected(ctx)
}

// onClientClosed raises EventClientDisconnected when the client went away,
// rather than the bot disconnecting.
func (t *Transport) onClientClosed() {
	if t.closing.Load() {
		return
	}
	ctx := t.in.readContext()
	if ctx == nil {
		ctx = context.Background()
	}
	t.in.Events().Call(context.WithoutCancel(ctx), transport.EventClientDisconnected, t, t.conn)
}

// inputTransport reads Opus RTP from the connection, decodes it to PCM, and
// pushes InputAudioRawFrames into the pipeline at the input sample rate.
type inputTransport struct {
	*transport.BaseInput
	conn *Connection
	dec  *opus.Decoder
	// resampler converts the 48 kHz the codec decodes to into the rate the
	// transport reads at. It passes audio through when the two match.
	resampler *resample.Resampler

	readWG     sync.WaitGroup
	mu         sync.Mutex // guards readCancel and readCtx
	readCancel context.CancelFunc
	// readCtx is the context the input reads under, nil while it is not.
	readCtx context.Context //nolint:containedctx // the reading's lifetime, for the connection's callbacks
}

// readContext returns the context the input reads under, or nil when it is
// not reading.
func (in *inputTransport) readContext() context.Context {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.readCtx
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
	ch := channels(in.Params().AudioInChannels)
	dec, err := opus.NewDecoder(ch)
	if err != nil {
		return err
	}
	// The decoded audio is one continuous stream, so the resampler keeps its
	// filter history for as long as the session lasts.
	r, err := resample.NewWithConfig(opus.SampleRate, in.sampleRate(), ch,
		resample.Config{ClearAfter: resample.NeverClear})
	if err != nil {
		return err
	}
	in.dec = dec
	in.resampler = r

	readCtx, cancel := context.WithCancel(ctx)
	in.mu.Lock()
	in.readCancel = cancel
	in.readCtx = readCtx
	in.mu.Unlock()
	in.readWG.Add(1)
	go in.readLoop(readCtx)

	// The pipeline is ready for the connection now. The connected event is
	// raised now if the peer is already connected, or when it connects, and
	// the client's messages held until this point are released.
	in.conn.Connect()
	return nil
}

// StopReading stops the read goroutine.
func (in *inputTransport) StopReading(context.Context) error {
	in.mu.Lock()
	cancel := in.readCancel
	in.readCtx = nil
	in.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	in.readWG.Wait()
	if in.resampler != nil {
		in.resampler.Close()
		in.resampler = nil
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

func (in *inputTransport) readLoop(ctx context.Context) {
	defer in.readWG.Done()

	track, err := in.conn.RemoteAudioTrack(ctx)
	if err != nil {
		return
	}
	ch := channels(in.Params().AudioInChannels)
	rate := in.sampleRate()

	for {
		if ctx.Err() != nil {
			return
		}
		_ = track.SetReadDeadline(time.Now().Add(readDeadline))
		pkt, _, err := track.ReadRTP()
		if err != nil {
			// A deadline lets us re-check ctx.
			if errors.Is(err, os.ErrDeadlineExceeded) {
				continue
			}
			// Any other error means the track is gone. Reading it again would
			// fail the same way, so park until a restart brings a fresh one;
			// that is how the session survives the client restarting its peer
			// connection.
			slog.Warn("rtc: audio track stopped, waiting for a new one", "err", err)
			track, err = in.conn.nextRemoteAudioTrack(ctx, track)
			if err != nil {
				return
			}
			continue
		}
		if len(pkt.Payload) == 0 {
			continue
		}
		pcm, err := in.dec.Decode(pkt.Payload)
		if err != nil {
			continue
		}
		// The codec always decodes at 48 kHz; the pipeline gets the rate it
		// asked for.
		pcm = in.resampler.Process(pcm)
		if len(pcm) == 0 {
			continue
		}
		in.PushAudioFrame(ctx, frames.NewInputAudioRawFrame(pcm, rate, ch))
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
	// frameBytes is the size of one 20 ms frame at the output sample rate. The
	// sender runs at that rate and the encoder brings it to the codec's own.
	frameBytes int

	mu      sync.Mutex
	queue   chan []byte
	cancel  context.CancelFunc
	sendWG  sync.WaitGroup
	running bool
	// queued counts frames handed over but not yet sent, so a graceful end can
	// wait for them rather than cutting the last words off.
	queued atomic.Int64

	disconnectOnce sync.Once
	// closing is the transport's flag that the bot is disconnecting.
	closing *atomic.Bool
}

func newOutput(conn *Connection, params transport.Params, closing *atomic.Bool) *outputTransport {
	out := &outputTransport{conn: conn, closing: closing}
	out.BaseOutput = transport.NewBaseOutput("PionOutput", params, out)
	return out
}

// ProcessFrame drives the sender's lifecycle around the base output: start it on
// the StartFrame so silence is already flowing before the first word, drop
// anything queued on a barge-in, and stop it when the session ends.
// StartWriting brings the sender up, so the outgoing track is running before the
// base starts queueing audio for it.
func (out *outputTransport) StartWriting(context.Context) error {
	return out.startSending()
}

func (out *outputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := out.BaseOutput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir != processor.Downstream {
		return nil
	}
	switch f.(type) {
	case *frames.InterruptionFrame:
		out.discardQueued()
	case *frames.EndFrame:
		// The base output has handed over everything it had; let it play before
		// the sender goes away, or the farewell loses its last words.
		out.waitDrained(ctx)
		out.stopSending()
		out.disconnect()
	case *frames.CancelFrame:
		out.stopSending()
		out.disconnect()
	}
	return nil
}

// disconnect tells the client the bot has left and closes the connection, once.
// It happens here rather than on the input, which sees the EndFrame first:
// closing there would cut off the audio the output is still draining.
func (out *outputTransport) disconnect() {
	out.disconnectOnce.Do(func() {
		select {
		case <-out.conn.Done():
			// The client is already gone.
			return
		default:
		}
		slog.Info("rtc: disconnecting from the peer")
		out.closing.Store(true)
		_ = out.conn.Disconnect()
	})
}

// SendMessage sends an application message over the data channel. The payload is
// encoded here rather than by the base, because what a wire carries is the
// transport's own business: this one speaks JSON to a browser client, so a
// message that is already text goes as it is and anything else is marshaled.
func (out *outputTransport) SendMessage(_ context.Context, f frames.OutputTransportMessage) error {
	data, err := encodeMessage(f.TransportMessage())
	if err != nil {
		return err
	}
	return out.conn.SendMessage(data)
}

// encodeMessage encodes an application message for the data channel.
func encodeMessage(message any) ([]byte, error) {
	if text, ok := message.(string); ok {
		return []byte(text), nil
	}
	return json.Marshal(message)
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
		return err
	}
	// The encoder is touched only by the sender goroutine from here on, so
	// packets cannot be sent in a different order than they were encoded.
	out.enc = enc
	out.frameBytes = opus.FrameBytesAt(rate, ch)
	out.queue = make(chan []byte, queuedFrames)
	ctx, cancel := context.WithCancel(context.Background())
	out.cancel = cancel
	out.running = true
	out.sendWG.Add(1)
	go out.sendLoop(ctx, out.frameBytes)
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
	fillSilence := out.Params().AudioOutFillsSilence()
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

		gap.sent = sent
		pcm, ok := out.nextChunk(ctx, quiet, fillSilence, &gap)
		if !ok {
			return
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

// nextChunk takes the next frame of audio to send: whatever is queued, or
// silence when nothing is and the stream is to be kept flowing.
//
// With the filling turned off it waits for audio instead. The timestamps go on
// advancing one frame per packet either way, so a receiver still schedules
// playout from a clock that never diverged from the wall clock. It reports false
// when the session ended while it waited.
func (out *outputTransport) nextChunk(
	ctx context.Context, quiet []byte, fillSilence bool, gap *gapTracker,
) ([]byte, bool) {
	select {
	case frame := <-out.queue:
		out.queued.Add(-1)
		gap.real()
		return frame, true
	default:
	}
	if fillSilence {
		gap.quiet()
		return quiet, true
	}
	select {
	case <-ctx.Done():
		return nil, false
	case <-out.conn.Done():
		return nil, false
	case frame := <-out.queue:
		out.queued.Add(-1)
		gap.real()
		return frame, true
	}
}

// WriteAudio hands PCM to the sender a frame at a time. It blocks only when the
// sender is already as far ahead as the queue allows, which is what paces the
// pipeline to real time; waiting on each frame individually would instead keep
// the queue empty and starve the sender. Audio that does not fill a whole frame
// is held until the next call.
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	pcm := f.AudioData().Audio

	out.mu.Lock()
	q, running := out.queue, out.running
	if !running {
		out.mu.Unlock()
		// The outgoing track is not live, so there is nowhere to send it.
		return false, nil
	}
	frameBytes := out.frameBytes
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
			return false, ctx.Err()
		case <-out.conn.Done():
			out.queued.Add(-1)
			// The connection went away with the chunk part queued, so the rest
			// of it is never heard.
			return false, nil
		}
	}
	return true, nil
}
