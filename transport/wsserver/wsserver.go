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
func (s *Session) Close() {
	s.closeOnce.Do(func() {
		close(s.done)
		_ = s.conn.Close(websocket.StatusNormalClosure, "")
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
}

func newInput(sess *Session, ser Serializer, params transport.Params) *inputTransport {
	in := &inputTransport{sess: sess, ser: ser}
	in.BaseInput = transport.NewBaseInput("WSInput", params, in)
	return in
}

// ProcessFrame sets the serializer up from the StartFrame before the base starts
// reading, then defers to the base.
func (in *inputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if sf, ok := f.(*frames.StartFrame); ok {
		if err := in.ser.Setup(sf); err != nil {
			return err
		}
	}
	return in.BaseInput.ProcessFrame(ctx, f, dir)
}

// StartReading launches the socket read loop.
func (in *inputTransport) StartReading(ctx context.Context) error {
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
}

func newOutput(sess *Session, ser Serializer, params transport.Params) *outputTransport {
	out := &outputTransport{sess: sess, ser: ser}
	out.BaseOutput = transport.NewBaseOutput("WSOutput", params, out)
	return out
}

// WriteAudio serializes a PCM chunk to a provider media message and sends it.
func (out *outputTransport) WriteAudio(ctx context.Context, pcm []byte) error {
	f := frames.NewOutputAudioRawFrame(pcm, out.SampleRate(), channels(out.Params().AudioOutChannels))
	msg, err := out.ser.Serialize(f)
	if err != nil {
		return err
	}
	if msg == nil {
		return nil
	}
	return out.sess.write(ctx, msg)
}

// SendMessage sends an already-encoded application message.
func (out *outputTransport) SendMessage(ctx context.Context, data []byte) error {
	return out.sess.write(ctx, data)
}

// ProcessFrame adds the control-frame handling the base output does not: an
// interruption becomes the provider's "clear" message (barge-in), and end or
// cancel triggers the serializer's hang-up.
func (out *outputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := out.BaseOutput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir != processor.Downstream {
		return nil
	}
	switch f.(type) {
	case *frames.InterruptionFrame, *frames.EndFrame, *frames.CancelFrame:
		msg, err := out.ser.Serialize(f)
		if err == nil && msg != nil {
			_ = out.sess.write(ctx, msg)
		}
	}
	return nil
}
