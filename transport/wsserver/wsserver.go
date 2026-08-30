// Package wsserver is a WebSocket media transport for telephony. It serves a
// WebSocket endpoint a phone provider (Twilio, Telnyx, Plivo) streams call audio
// over, and bridges that socket to a jargo pipeline: inbound messages become
// InputAudioRawFrames and outbound audio becomes provider media messages.
//
// The wire format is provider-specific, so it is supplied as a Serializer; the
// transport itself is provider-agnostic.
//
// Telephony audio is companded 8 kHz on the wire, but the pipeline does not have
// to run at that rate. A Serializer built on Codec converts between the two at
// each edge, so the pipeline can run at whatever rate suits its services: a
// transcriber handed 8 kHz and a voice asked for 8 kHz are both worse than
// converting once on the way in and once on the way out. Set the pipeline rate
// as usual and the serializer follows it.
package wsserver

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/security"
)

// readLimit bounds a single inbound WebSocket message. Telephony media messages
// are small, but a generous limit keeps any provider's control messages safe.
const readLimit = 1 << 20

// Serializer converts between jargo frames and a provider's WebSocket wire
// format. One Serializer serves one session; it is not safe for concurrent use
// across sessions.
type Serializer interface {
	// Setup captures pipeline configuration from the StartFrame.
	Setup(s processor.Setup) error
	// Serialize converts an outbound frame to a wire message, or (nil, nil) for
	// frames it does not send. Interruption, end and cancel frames are passed in
	// so the serializer can emit a "clear" message or hang up the call.
	Serialize(f frames.Frame) ([]byte, error)
	// Deserialize converts an inbound wire message to a frame, or (nil, nil) for
	// messages that carry no frame (handshake, marks, stop).
	Deserialize(data []byte) (frames.Frame, error)
}

// RTVIMessageCarrier is an optional interface a Serializer implements when the
// wire it writes to is the one RTVI messages travel on, which is what a browser
// client connects over.
//
// It is off unless a serializer says otherwise, because the usual wire here is a
// telephony provider's media stream: RTVI means nothing to a provider expecting
// its own control messages, and a pipeline with an RTVI processor in it would
// otherwise write the whole protocol onto the call.
type RTVIMessageCarrier interface {
	// CarriesRTVIMessages reports whether RTVI messages belong on this wire.
	CarriesRTVIMessages() bool
}

// closableSerializer is a Serializer holding resources that have to be handed
// back when the session ends, a native resampler being the case here. It is
// optional, and separate from Serializer so that a serializer holding nothing
// does not have to say so.
type closableSerializer interface {
	Close()
}

// EventSessionTimeout fires when a session has run for Params.SessionTimeout
// without ending. It carries nothing: the session it concerns is this one.
//
//	events.On(tr.Events(), wsserver.EventSessionTimeout,
//	    func(ctx context.Context, _ struct{}) { … })
const EventSessionTimeout = "on_session_timeout"

// Transport bridges a WebSocket session to a pipeline.
type Transport struct {
	in   *inputTransport
	out  *outputTransport
	sess *Session
}

// Params configures a WebSocket server transport: the media parameters every
// transport takes, plus the ones only a server serving its own socket has.
type Params struct {
	transport.Params

	// AllowedOrigins are the origins a browser client may open the socket from.
	//
	// Empty, the default, allows every origin, which is what a telephony
	// provider needs: it is not a browser and sends no Origin header at all.
	// Naming origins allows only those, matched whole and without regard to
	// case, and turns away a request whose origin is missing.
	//
	// Set it for an endpoint a browser connects to. Without it, a page on any
	// other site can open this socket in a visitor's browser and hold a
	// conversation as them.
	AllowedOrigins []string

	// SessionTimeout bounds how long one session may run. When it elapses with
	// the socket still open, EventSessionTimeout fires; zero, the default,
	// leaves a session to run as long as it likes.
	//
	// Nothing is closed by it. What to do about a session that has gone on too
	// long is the endpoint's call, and it usually wants to say something before
	// ending the pipeline rather than cutting the caller off mid-sentence.
	SessionTimeout time.Duration
}

// DefaultParams returns the transport defaults with no origin restriction.
func DefaultParams() Params {
	return Params{Params: transport.DefaultParams()}
}

// ErrOriginNotAllowed is returned by Accept when the request's Origin header is
// missing from, or absent against, the origins Params.AllowedOrigins names.
var ErrOriginNotAllowed = errors.New("wsserver: origin not allowed")

// Accept upgrades an HTTP request to a WebSocket and builds a Transport that
// uses ser for the wire format. Call it from an http.HandlerFunc; the returned
// Transport's Input and Output go at the head and tail of the pipeline, and
// Done reports when the call ends.
//
// A request whose origin params.AllowedOrigins does not name is refused before
// the upgrade, with ErrOriginNotAllowed and no reply written. The caller
// answers it, so that an endpoint can choose what to tell a rejected client.
func Accept(w http.ResponseWriter, r *http.Request, ser Serializer, params Params) (*Transport, error) {
	if !security.IsOriginAllowed(r.Header.Get("Origin"), params.AllowedOrigins) {
		return nil, fmt.Errorf("%w: %q", ErrOriginNotAllowed, r.Header.Get("Origin"))
	}
	// The origins are checked above rather than handed to the library, whose own
	// rule is a different one: it compares the origin's host against the request
	// host, which turns away a client legitimately served from elsewhere.
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: []string{"*"}})
	if err != nil {
		return nil, err
	}
	c.SetReadLimit(readLimit)
	sess := &Session{conn: c, done: make(chan struct{})}
	return &Transport{
		sess: sess,
		in:   newInput(sess, ser, params),
		out:  newOutput(sess, ser, params.Params),
	}, nil
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

// Done is closed when the call ends (the client closes the socket or the
// pipeline stops reading). Cancel the pipeline context on it.
func (t *Transport) Done() <-chan struct{} { return t.sess.done }

// Events is the registry of events this transport raises. See
// EventSessionTimeout.
func (t *Transport) Events() *events.Registry { return t.in.Events() }

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
	// sessionTimeout bounds how long the session may run before the event fires.
	sessionTimeout time.Duration

	readWG     sync.WaitGroup
	mu         sync.Mutex
	readCancel context.CancelFunc
	setup      processor.Setup
}

func newInput(sess *Session, ser Serializer, params Params) *inputTransport {
	in := &inputTransport{sess: sess, ser: ser, sessionTimeout: params.SessionTimeout}
	in.BaseInput = transport.NewBaseInput("WSInput", params.Params, in)
	in.Events().Register(EventSessionTimeout, false)
	return in
}

// watchSessionLength raises EventSessionTimeout once the session has run for as
// long as it is allowed to, and does nothing for one that ends first. It is
// started with the read loop, so the clock runs from the point the session
// begins carrying conversation rather than from the socket being accepted.
func (in *inputTransport) watchSessionLength(ctx context.Context) {
	defer in.readWG.Done()
	timer := time.NewTimer(in.sessionTimeout)
	defer timer.Stop()
	select {
	case <-timer.C:
		select {
		case <-in.sess.done:
			// The session ended as the timer fired; there is nothing to report.
		default:
			in.Events().Call(ctx, EventSessionTimeout, in, struct{}{})
		}
	case <-in.sess.done:
	case <-ctx.Done():
	}
}

// Setup records the pipeline's configuration for the serializer and defers to
// the base.
//
// The serializer is set up in StartReading rather than here, because the base
// pushes the StartFrame downstream before it calls StartReading. Configuring the
// serializer first would mean a failure swallowed the StartFrame, leaving every
// processor downstream uninitialized and the pipeline unable to finish.
func (in *inputTransport) Setup(ctx context.Context, s processor.Setup) error {
	in.mu.Lock()
	in.setup = s
	in.mu.Unlock()
	return in.BaseInput.Setup(ctx, s)
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
	setup := in.setup
	in.mu.Unlock()

	if err := in.ser.Setup(setup); err != nil {
		in.PushError(ctx, "wsserver: serializer setup failed", err, true)
		in.sess.abort()
		return nil // reported as a fatal error frame; do not report it twice
	}

	// The socket was accepted before the pipeline was built, so the client is
	// already there. It is reported once the serializer can speak the provider's
	// wire format and before the first inbound message is read, so the
	// conversation starts with the connection already accounted for.
	in.PushClientConnected(ctx)

	readCtx, cancel := context.WithCancel(ctx)
	in.mu.Lock()
	in.readCancel = cancel
	in.mu.Unlock()
	in.readWG.Add(1)
	go in.readLoop(readCtx)
	if in.sessionTimeout > 0 {
		in.readWG.Add(1)
		go in.watchSessionLength(readCtx)
	}
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
		if mf, ok := f.(*frames.InputTransportMessageFrame); ok {
			// Broadcast, so whatever handles client messages hears them whether
			// it sits ahead of this transport or behind it.
			in.PushTransportMessage(ctx, mf.Message)
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
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	msg, err := out.ser.Serialize(f)
	if err != nil {
		return false, err
	}
	if msg == nil {
		// The serializer had nothing to send for this frame.
		return false, nil
	}
	if err := out.sess.write(ctx, msg); err != nil {
		return false, err
	}
	out.writeAudioSleep(ctx)
	return true, nil
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

// IgnoresMessage refuses an RTVI message unless the serializer says this wire
// carries them. See RTVIMessageCarrier.
func (out *outputTransport) IgnoresMessage(message any) bool {
	if c, ok := out.ser.(RTVIMessageCarrier); ok && c.CarriesRTVIMessages() {
		return false
	}
	return isRTVIMessage(message)
}

// isRTVIMessage reports whether the message is one of the RTVI protocol's,
// which every one of them says of itself in its label.
func isRTVIMessage(message any) bool {
	switch m := message.(type) {
	case rtvi.Message:
		return m.Label == rtvi.MessageLabel
	case *rtvi.Message:
		return m != nil && m.Label == rtvi.MessageLabel
	case map[string]any:
		// Built by hand rather than by the rtvi package, which a caller speaking
		// the protocol itself would produce.
		label, _ := m["label"].(string)
		return label == rtvi.MessageLabel
	default:
		return false
	}
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
		// The last use of the serializer in either direction. The frame reached
		// the input first, which stopped the read loop and waited for it, and
		// the base has drained the outgoing audio above, so nothing is still
		// converting by the time this releases what the serializer holds.
		if c, ok := out.ser.(closableSerializer); ok {
			c.Close()
		}
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
