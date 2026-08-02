// Package wsserver is a WebSocket media transport for telephony. It serves a
// WebSocket endpoint a phone provider (Twilio, Telnyx, Plivo) streams call audio
// over, and bridges that socket to a jargo pipeline: inbound messages become
// InputAudioRawFrames and outbound audio becomes provider media messages.
//
// The wire format is provider-specific, so it is supplied as a Serializer; the
// transport itself is provider-agnostic. Telephony audio is μ-law 8 kHz, so run
// the pipeline at 8 kHz (set the StartFrame and transport sample rates to 8000).
package wsserver

import (
	"context"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
)

// readLimit bounds a single inbound WebSocket message. Telephony media messages
// are small, but a generous limit keeps any provider's control messages safe.
const readLimit = 1 << 20

// Serializer converts between jargo frames and a provider's WebSocket wire
// format. One Serializer serves one session; it is not safe for concurrent use
// across sessions.
type Serializer interface {
	// Setup captures pipeline configuration from the StartFrame.
	Setup(f *frames.StartFrame) error
	// Serialize converts an outbound frame to a wire message, or (nil, nil) for
	// frames it does not send. Interruption, end and cancel frames are passed in
	// so the serializer can emit a "clear" message or hang up the call.
	Serialize(f frames.Frame) ([]byte, error)
	// Deserialize converts an inbound wire message to a frame, or (nil, nil) for
	// messages that carry no frame (handshake, marks, stop).
	Deserialize(data []byte) (frames.Frame, error)
}

// Transport bridges a WebSocket session to a pipeline.
type Transport struct {
	in   *inputTransport
	out  *outputTransport
	sess *Session
}

// Accept upgrades an HTTP request to a WebSocket and builds a Transport that
// uses ser for the wire format. Call it from an http.HandlerFunc; the returned
// Transport's Input and Output go at the head and tail of the pipeline, and
// Done reports when the call ends.
func Accept(w http.ResponseWriter, r *http.Request, ser Serializer, params transport.Params) (*Transport, error) {
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: []string{"*"}})
	if err != nil {
		return nil, err
	}
	c.SetReadLimit(readLimit)
	sess := &Session{conn: c, done: make(chan struct{})}
	return &Transport{
		sess: sess,
		in:   newInput(sess, ser, params),
		out:  newOutput(sess, ser, params),
	}, nil
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

// Done is closed when the call ends (the client closes the socket or the
// pipeline stops reading). Cancel the pipeline context on it.
func (t *Transport) Done() <-chan struct{} { return t.sess.done }

// Session owns one WebSocket connection and serializes writes, which
// coder/websocket requires.
type Session struct {
	conn      *websocket.Conn
	writeMu   sync.Mutex
	closeOnce sync.Once
	done      chan struct{}
}

func (s *Session) read(ctx context.Context) ([]byte, error) {
	_, data, err := s.conn.Read(ctx)
	return data, err
}

func (s *Session) write(ctx context.Context, msg []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(ctx, websocket.MessageText, msg)
}

// Close closes the socket and signals Done. It is idempotent.
//
// It is safe on the read path, where it is deferred by readLoop: a read that
// fails or is canceled has already closed the connection underneath, so Close
// finds it closed and returns at once. Called on a live connection it instead
// performs the close handshake and waits up to five seconds for the peer to
// reply, which must never happen on a frame-processing goroutine — see abort.
func (s *Session) Close() {
	s.closeOnce.Do(func() {
		close(s.done)
		_ = s.conn.Close(websocket.StatusNormalClosure, "")
	})
}

// abort tears the socket down without the close handshake and signals Done. It
// is for a session that never started: there is no conversation to end politely,
// and the caller is a frame-processing goroutine that must not block.
func (s *Session) abort() {
	s.closeOnce.Do(func() {
		close(s.done)
		_ = s.conn.CloseNow()
	})
}

func channels(n int) int {
	if n == 0 {
		return 1
	}
	return n
}

// inputTransport reads provider messages off the socket, deserializes them, and
// pushes the resulting frames into the pipeline.
type inputTransport struct {
	*transport.BaseInput
	sess *Session
	ser  Serializer

	readWG     sync.WaitGroup
	mu         sync.Mutex
	readCancel context.CancelFunc
	start      *frames.StartFrame
}

func newInput(sess *Session, ser Serializer, params transport.Params) *inputTransport {
	in := &inputTransport{sess: sess, ser: ser}
	in.BaseInput = transport.NewBaseInput("WSInput", params, in)
	return in
}

// ProcessFrame records the StartFrame for the serializer and defers to the base.
//
// The serializer is set up in StartReading rather than here, because the base
// pushes the StartFrame downstream before it calls StartReading. Configuring the
// serializer first would mean a failure swallowed the StartFrame, leaving every
// processor downstream uninitialized and the pipeline unable to finish.
func (in *inputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if sf, ok := f.(*frames.StartFrame); ok {
		in.mu.Lock()
		in.start = sf
		in.mu.Unlock()
	}
	return in.BaseInput.ProcessFrame(ctx, f, dir)
}

// StartReading configures the serializer and launches the socket read loop. It
// runs after the StartFrame has reached the rest of the pipeline and before any
// inbound message is deserialized, so the serializer always sees the pipeline's
// configuration before the first frame it must decode.
//
// A serializer that cannot configure itself leaves the session unable to speak
// the provider's wire format at all, so the failure is fatal: it ends the
// pipeline and closes the socket rather than holding a call open that can
// neither hear nor answer.
func (in *inputTransport) StartReading(ctx context.Context) error {
	in.mu.Lock()
	sf := in.start
	in.mu.Unlock()

	if sf != nil {
		if err := in.ser.Setup(sf); err != nil {
			in.PushError(ctx, "wsserver: serializer setup failed", err, true)
			in.sess.abort()
			return nil // reported as a fatal error frame; do not report it twice
		}
	}

	readCtx, cancel := context.WithCancel(ctx)
	in.mu.Lock()
	in.readCancel = cancel
	in.mu.Unlock()
	in.readWG.Add(1)
	go in.readLoop(readCtx)
	return nil
}

// StopReading stops the read loop.
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
	defer in.sess.Close()
	for {
		data, err := in.sess.read(ctx)
		if err != nil {
			return // socket closed or context canceled
		}
		f, err := in.ser.Deserialize(data)
		if err != nil {
			slog.Warn("wsserver: deserialize", "err", err)
			continue
		}
		if f == nil {
			continue
		}
		if af, ok := f.(*frames.InputAudioRawFrame); ok {
			in.PushAudioFrame(ctx, af)
			continue
		}
		_ = in.PushFrame(ctx, f, processor.Downstream)
	}
}

// outputTransport serializes outbound audio and control frames and writes them
// to the socket.
type outputTransport struct {
	*transport.BaseOutput
	sess *Session
	ser  Serializer

	// WriteAudio is called as soon as audio is produced, by the TTS say, and
	// this is only a network connection, so audio would otherwise go out far
	// faster than it plays. Blocking for as long as a chunk takes to play
	// emulates an audio device, keeping the provider's playout buffer shallow so
	// a barge-in cuts audio that has not been handed over yet.
	paceMu sync.Mutex
	// sendInterval is how long one chunk takes to play, set on the StartFrame.
	sendInterval time.Duration
	// nextSend is when the next chunk is due. The zero time means send now and
	// start the clock from there.
	nextSend time.Time
}

// chunkDuration is how long one chunk of 16-bit PCM takes to play.
func chunkDuration(chunkSize, sampleRate, numChannels int) time.Duration {
	bytesPerSec := sampleRate * numChannels * 2
	if chunkSize <= 0 || bytesPerSec <= 0 {
		return 0
	}
	return time.Duration(chunkSize) * time.Second / time.Duration(bytesPerSec)
}

func newOutput(sess *Session, ser Serializer, params transport.Params) *outputTransport {
	out := &outputTransport{sess: sess, ser: ser}
	out.BaseOutput = transport.NewBaseOutput("WSOutput", params, out)
	return out
}

// WriteAudio serializes a PCM chunk to a provider media message and sends it.
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) error {
	msg, err := out.ser.Serialize(f)
	if err != nil {
		return err
	}
	if msg == nil {
		return nil
	}
	if err := out.sess.write(ctx, msg); err != nil {
		return err
	}
	out.writeAudioSleep(ctx)
	return nil
}

// writeAudioSleep blocks until the next chunk is due, so audio leaves at the
// rate it plays rather than the rate it is produced.
func (out *outputTransport) writeAudioSleep(ctx context.Context) {
	out.paceMu.Lock()
	interval, next := out.sendInterval, out.nextSend
	out.paceMu.Unlock()
	if interval <= 0 {
		return
	}

	if wait := time.Until(next); wait > 0 {
		timer := time.NewTimer(wait)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return
		case <-timer.C:
		}

		out.paceMu.Lock()
		// An interruption may have reset the clock while this waited. Leave the
		// reset alone if so, or the next chunk would wait on a stale schedule
		// instead of going out at once.
		if out.nextSend.Equal(next) {
			out.nextSend = next.Add(interval)
		}
		out.paceMu.Unlock()
		return
	}

	// At or behind schedule: send now and time the next chunk from here.
	out.paceMu.Lock()
	out.nextSend = time.Now().Add(interval)
	out.paceMu.Unlock()
}

// SendMessage sends an already-encoded application message.
func (out *outputTransport) SendMessage(ctx context.Context, data []byte) error {
	return out.sess.write(ctx, data)
}

// ProcessFrame adds the control-frame handling the base output does not: an
// interruption becomes the provider's "clear" message (barge-in), and end or
// cancel triggers the serializer's hang-up.
// StartWriting sets the playout clock. The base has sized the chunks by the time
// it calls this, so the interval can be derived from them.
func (out *outputTransport) StartWriting(context.Context) error {
	out.paceMu.Lock()
	defer out.paceMu.Unlock()
	out.sendInterval = chunkDuration(out.ChunkSize(), out.SampleRate(), channels(out.Params().AudioOutChannels))
	out.nextSend = time.Time{}
	return nil
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
		out.sendControl(ctx, f)
		// Restart the playout clock on a barge-in so the next turn's audio goes
		// out at once rather than waiting on the cut-off turn's schedule.
		out.paceMu.Lock()
		out.nextSend = time.Time{}
		out.paceMu.Unlock()
	case *frames.EndFrame, *frames.CancelFrame:
		out.sendControl(ctx, f)
	}
	return nil
}

// sendControl serializes a control frame to the provider's own message, if the
// serializer has one for it, and sends it.
func (out *outputTransport) sendControl(ctx context.Context, f frames.Frame) {
	msg, err := out.ser.Serialize(f)
	if err == nil && msg != nil {
		_ = out.sess.write(ctx, msg)
	}
}
