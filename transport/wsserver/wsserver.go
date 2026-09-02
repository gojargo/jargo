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
	"io"
	"log/slog"
	"net"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
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
	// Serialize converts an outbound frame to a wire message, or an empty
	// Message for frames it does not send. Interruption, end and cancel frames
	// are passed in so the serializer can emit a "clear" message or hang up the
	// call, and so are application messages, which the serializer encodes for
	// its own wire after asking BaseSerializer.ShouldIgnoreFrame whether they
	// belong on it.
	Serialize(f frames.Frame) (Message, error)
	// Deserialize converts an inbound wire message to a frame, or (nil, nil) for
	// messages that carry no frame (handshake, marks, stop).
	Deserialize(data []byte) (frames.Frame, error)
}

// closableSerializer is a Serializer holding resources that have to be handed
// back when the session ends, a native resampler being the case here. It is
// optional, and separate from Serializer so that a serializer holding nothing
// does not have to say so.
type closableSerializer interface {
	Close()
}

// The events a WebSocket transport raises. Each carries nothing, because the
// session it concerns is this one:
//
//	events.On(tr.Events(), wsserver.EventClientDisconnected,
//	    func(ctx context.Context, _ struct{}) { … })
const (
	// EventClientConnected fires once the client is on the line and the
	// session can carry conversation. Register for it before running the
	// pipeline: the socket was accepted before the pipeline was built, so the
	// client is already there and this fires as the pipeline starts.
	EventClientConnected = "on_client_connected"
	// EventClientDisconnected fires when the client hangs up. It does not fire
	// for a session this side ended, which is a shutdown already under way
	// rather than news about the client.
	EventClientDisconnected = "on_client_disconnected"
	// EventSessionTimeout fires when a session has run for
	// Params.SessionTimeout without ending.
	EventSessionTimeout = "on_session_timeout"
)

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
	// DefaultParams fills it from security.AllowedOriginsEnv, so a deployment
	// can set the policy once for every endpoint it serves.
	//
	// Empty, which is what an unset variable leaves, allows every origin, and is
	// what a telephony provider needs: it is not a browser and sends no Origin
	// header at all.
	// Naming origins allows only those, matched whole and without regard to
	// case, and turns away a request whose origin is missing.
	//
	// Set it for an endpoint a browser connects to. Without it, a page on any
	// other site can open this socket in a visitor's browser and hold a
	// conversation as them.
	AllowedOrigins []string

	// AddWAVHeader wraps every outgoing chunk of audio in a WAV container
	// before the serializer encodes it, for a client that plays whole blobs
	// rather than a raw PCM stream. Off by default, which is what a provider
	// streaming a call expects.
	AddWAVHeader bool

	// FixedAudioPacketSize frames outgoing binary payloads at exactly this many
	// bytes, holding back whatever does not fill a packet until the next one
	// completes it. Zero, the default, sends each payload as the serializer
	// produced it.
	//
	// Set it for a media endpoint that requires strict framing: 640 bytes is
	// 20 ms of 16 kHz mono 16-bit PCM. It applies only to binary payloads, since
	// a wire whose messages are text frames them itself.
	FixedAudioPacketSize int

	// SessionTimeout bounds how long one session may run. When it elapses with
	// the socket still open, EventSessionTimeout fires; zero, the default,
	// leaves a session to run as long as it likes.
	//
	// Nothing is closed by it. What to do about a session that has gone on too
	// long is the endpoint's call, and it usually wants to say something before
	// ending the pipeline rather than cutting the caller off mid-sentence.
	SessionTimeout time.Duration
}

// DefaultParams returns the transport defaults, with the origins
// security.AllowedOriginsEnv names and no restriction when it is unset.
func DefaultParams() Params {
	return Params{
		Params:         transport.DefaultParams(),
		AllowedOrigins: security.DefaultAllowedOrigins(),
	}
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
		out:  newOutput(sess, ser, params),
	}, nil
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

// Done is closed when the call ends: the client hung up, or the pipeline
// finished with the session and the connection was closed. Cancel the pipeline
// context on it.
//
// A shutdown the pipeline started itself reports only once the output has sent
// the last of its audio, so canceling on this cannot cut off a goodbye spoken
// on the way out.
func (t *Transport) Done() <-chan struct{} { return t.sess.done }

// Events is the registry of events this transport raises. See
// EventClientConnected, EventClientDisconnected and EventSessionTimeout.
func (t *Transport) Events() *events.Registry { return t.in.Events() }

// Session owns one WebSocket connection and serializes writes, which
// coder/websocket requires.
//
// Both sides of the transport write to the same connection, and an EndFrame
// reaches the input before the output. Closing it as soon as the input was done
// would cut off whatever the output has still to send, which on a call that ends
// politely is the goodbye itself. So each side takes a hold while the pipeline
// is set up and gives it back when it stops, and the connection closes once
// neither side holds it.
type Session struct {
	conn      *websocket.Conn
	writeMu   sync.Mutex
	closeOnce sync.Once
	doneOnce  sync.Once
	done      chan struct{}

	mu sync.Mutex
	// holds counts the sides of the transport still using the connection.
	holds int
	// stopRead ends the read loop, registered by the input once it starts
	// reading. The read is ended here, as the connection closes, rather than
	// when the input stops: canceling a read closes the connection underneath,
	// and the output may still have audio to send.
	stopRead func()
}

func (s *Session) read(ctx context.Context) ([]byte, error) {
	_, data, err := s.conn.Read(ctx)
	return data, err
}

func (s *Session) write(ctx context.Context, m Message) error {
	typ := websocket.MessageText
	if m.Binary {
		typ = websocket.MessageBinary
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.Write(ctx, typ, m.Data)
}

// hold registers one side of the transport as a user of the connection. Both
// sides take theirs while the pipeline is being set up, before any frame flows,
// so neither can find the connection already closed by the time it starts.
func (s *Session) hold() {
	s.mu.Lock()
	s.holds++
	s.mu.Unlock()
}

// release gives back one hold and closes the connection once none remain.
// Whichever side finishes last is the one that closes, which is what lets the
// output send the last of its audio after the input has stopped reading. A
// pipeline built without the output in it leaves the input holding it alone, and
// its release closes the connection as before.
func (s *Session) release() {
	s.mu.Lock()
	s.holds--
	last := s.holds <= 0
	s.mu.Unlock()
	if last {
		s.Close()
	}
}

// setStopRead registers how the read loop is ended, which only the input knows.
func (s *Session) setStopRead(stop func()) {
	s.mu.Lock()
	s.stopRead = stop
	s.mu.Unlock()
}

// endRead ends the read loop and waits for it, so nothing is still reading the
// connection by the time it is closed. It does nothing for a session that never
// started reading.
func (s *Session) endRead() {
	s.mu.Lock()
	stop := s.stopRead
	s.mu.Unlock()
	if stop != nil {
		stop()
	}
}

// closed reports that the call is over, whether the peer hung up or this side
// finished with the session. Nothing more can be written to it.
func (s *Session) closed() bool {
	select {
	case <-s.done:
		return true
	default:
		return false
	}
}

// gone reports an error from a write that failed because the client is no
// longer there. It is the ordinary race on a disconnect, not a fault: the peer
// can go at any point, including between the check that it is still there and
// the write itself.
func gone(err error) bool {
	if errors.Is(err, net.ErrClosed) || errors.Is(err, io.EOF) {
		return true
	}
	var closeErr websocket.CloseError
	if errors.As(err, &closeErr) {
		return true
	}
	// A failed network write on a live session means the connection under it
	// has gone, which is a reset or a broken pipe.
	var opErr *net.OpError
	return errors.As(err, &opErr)
}

// signalDone reports the call as over to whoever waits on Done. It is
// idempotent, and separate from Close because the peer hanging up ends the call
// before the connection is closed: the pipeline is still draining, and waiting
// for the close would mean waiting on the very shutdown this report is meant to
// start.
func (s *Session) signalDone() {
	s.doneOnce.Do(func() { close(s.done) })
}

// Close ends the read loop, closes the socket and signals Done. It is
// idempotent.
//
// Ending the read is what actually takes the connection down: canceling a read
// closes it underneath, so the close handshake that follows finds it closed and
// returns at once rather than waiting five seconds for a peer that may be gone.
// That is what keeps teardown prompt on a frame-processing goroutine.
func (s *Session) Close() {
	s.closeOnce.Do(func() {
		s.endRead()
		s.signalDone()
		_ = s.conn.Close(websocket.StatusNormalClosure, "")
	})
}

// abort tears the socket down without the close handshake and signals Done. It
// is for a session that never started: there is no conversation to end politely,
// and the caller is a frame-processing goroutine that must not block.
func (s *Session) abort() {
	s.closeOnce.Do(func() {
		s.endRead()
		s.signalDone()
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

	readWG sync.WaitGroup
	mu     sync.Mutex
	setup  processor.Setup
	// stopping records that the pipeline has stopped, so a message that arrives
	// after it is not delivered and the read loop ending is not mistaken for the
	// peer hanging up.
	stopping bool

	holdOnce    sync.Once
	releaseOnce sync.Once
}

func newInput(sess *Session, ser Serializer, params Params) *inputTransport {
	in := &inputTransport{sess: sess, ser: ser, sessionTimeout: params.SessionTimeout}
	in.BaseInput = transport.NewBaseInput("WSInput", params.Params, in)
	in.Events().Register(EventClientConnected, false)
	in.Events().Register(EventClientDisconnected, false)
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
	// Taken here rather than when the read loop starts, so the output cannot
	// find the connection closed before its own StartFrame has reached it.
	in.holdOnce.Do(in.sess.hold)

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
	in.Events().Call(ctx, EventClientConnected, in, struct{}{})

	// The read deliberately outlives the context the base hands over. That one
	// is canceled the moment the pipeline stops streaming, and canceling a read
	// closes the connection underneath, which would cut off audio the output has
	// still to send: on a call that ends politely, the goodbye. The read is ended
	// by the session instead, when the connection is closed.
	readCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	in.sess.setStopRead(func() {
		cancel()
		in.readWG.Wait()
	})

	in.readWG.Add(1)
	go in.readLoop(readCtx)
	if in.sessionTimeout > 0 {
		in.readWG.Add(1)
		go in.watchSessionLength(readCtx)
	}
	return nil
}

// StopReading stops delivering inbound frames and gives back the input's hold on
// the connection.
//
// It neither ends the read nor waits for it. The output may still have audio to
// send, and ending the read would close the connection out from under it, so
// that is left to whichever side releases its hold last. On a pipeline built
// without the output in it that is this call, and the connection closes here as
// it always did.
func (in *inputTransport) StopReading(context.Context) error {
	in.mu.Lock()
	in.stopping = true
	in.mu.Unlock()
	in.releaseOnce.Do(in.sess.release)
	return nil
}

// stopped reports that the pipeline has stopped and the read loop is no longer
// to deliver what it reads.
func (in *inputTransport) stopped() bool {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.stopping
}

func (in *inputTransport) readLoop(ctx context.Context) {
	defer in.readWG.Done()
	defer in.reportPeerGone(ctx)
	for {
		data, err := in.sess.read(ctx)
		if err != nil {
			return // socket closed or context canceled
		}
		if in.stopped() {
			// The pipeline has stopped and is draining what it already has.
			// Whatever arrives now belongs to a call that is already over.
			return
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

// reportPeerGone ends the call when the read stopped because the peer did. A
// read that ends because this side stopped it is a shutdown already under way,
// and reporting that would have the endpoint cancel the pipeline in the middle
// of its own drain, taking the goodbye with it.
func (in *inputTransport) reportPeerGone(ctx context.Context) {
	if in.stopped() {
		return
	}
	in.sess.signalDone()
	in.Events().Call(ctx, EventClientDisconnected, in, struct{}{})
}

// outputTransport serializes outbound audio and control frames and writes them
// to the socket.
type outputTransport struct {
	*transport.BaseOutput
	sess   *Session
	ser    Serializer
	params Params

	// packetMu guards the buffer that holds back audio which does not fill a
	// whole packet. See Params.FixedAudioPacketSize.
	packetMu     sync.Mutex
	packetBuffer []byte

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

	holdOnce    sync.Once
	releaseOnce sync.Once
}

// chunkDuration is how long one chunk of 16-bit PCM takes to play.
func chunkDuration(chunkSize, sampleRate, numChannels int) time.Duration {
	bytesPerSec := sampleRate * numChannels * 2
	if chunkSize <= 0 || bytesPerSec <= 0 {
		return 0
	}
	return time.Duration(chunkSize) * time.Second / time.Duration(bytesPerSec)
}

func newOutput(sess *Session, ser Serializer, params Params) *outputTransport {
	out := &outputTransport{sess: sess, ser: ser, params: params}
	out.BaseOutput = transport.NewBaseOutput("WSOutput", params.Params, out)
	return out
}

// WriteAudio serializes a PCM chunk to a provider media message and sends it.
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	if out.sess.closed() {
		// The client is gone. The audio is not heard, but nothing failed, so it
		// is reported as unsent rather than as an error the pipeline reports on.
		return false, nil
	}
	if out.params.AddWAVHeader {
		a := f.AudioData()
		f = frames.NewOutputAudioRawFrame(
			audio.PCMToWAV(a.Audio, a.SampleRate, a.NumChannels), a.SampleRate, a.NumChannels)
	}

	msg, err := out.ser.Serialize(f)
	if err != nil {
		return false, err
	}
	if msg.Empty() {
		// The serializer had nothing to send for this frame.
		return false, nil
	}
	if err := out.write(ctx, msg); err != nil {
		if gone(err) {
			slog.Debug("wsserver: the client went away while audio was being sent", "err", err)
			return false, nil
		}
		return false, err
	}
	out.writeAudioSleep(ctx)
	return true, nil
}

// write sends one wire message, framing binary payloads at a fixed size when the
// endpoint asked for that. Whatever does not fill a packet is held back until
// the next payload completes it, so the endpoint never sees a short frame.
func (out *outputTransport) write(ctx context.Context, msg Message) error {
	size := out.params.FixedAudioPacketSize
	if size <= 0 || !msg.Binary {
		return out.sess.write(ctx, msg)
	}

	out.packetMu.Lock()
	out.packetBuffer = append(out.packetBuffer, msg.Data...)
	var packets [][]byte
	for len(out.packetBuffer) >= size {
		p := make([]byte, size)
		copy(p, out.packetBuffer[:size])
		packets = append(packets, p)
		out.packetBuffer = out.packetBuffer[size:]
	}
	out.packetMu.Unlock()

	for _, p := range packets {
		if err := out.sess.write(ctx, Message{Data: p, Binary: true}); err != nil {
			return err
		}
	}
	return nil
}

// dropBufferedPacket throws away audio that has not filled a whole packet. A
// barge-in cuts what the bot was saying, and holding the tail of it would splice
// it onto the front of whatever is said next.
func (out *outputTransport) dropBufferedPacket() {
	out.packetMu.Lock()
	out.packetBuffer = nil
	out.packetMu.Unlock()
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

// SendMessage hands the message frame to the serializer and writes what it
// produces. The serializer decides both whether the message belongs on this wire
// and how it is encoded, which is the same path an audio frame takes.
func (out *outputTransport) SendMessage(ctx context.Context, f frames.OutputTransportMessage) error {
	if out.sess.closed() {
		return nil
	}
	msg, err := out.ser.Serialize(f)
	if err != nil {
		return err
	}
	if msg.Empty() {
		// The serializer had nothing to send for this frame.
		return nil
	}
	if err := out.write(ctx, msg); err != nil && !gone(err) {
		return err
	}
	return nil
}

// Setup takes the output's hold on the connection, so that it is not closed
// under the output while there is still audio to send. See Session.
func (out *outputTransport) Setup(ctx context.Context, s processor.Setup) error {
	out.holdOnce.Do(out.sess.hold)
	return out.BaseOutput.Setup(ctx, s)
}

// Cleanup gives back the output's hold for a teardown that never delivered an
// end or cancel frame. It is the backstop: the ordinary path releases as that
// frame arrives, so the connection closes once the last of the audio is out
// rather than waiting for the pipeline to be torn down.
func (out *outputTransport) Cleanup(ctx context.Context) error {
	err := out.BaseOutput.Cleanup(ctx)
	out.releaseOnce.Do(out.sess.release)
	return err
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
		out.dropBufferedPacket()
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
		// The base drained the outgoing audio before this ran, so everything the
		// call had to say has been written. Give the hold back, which closes the
		// connection once the input has given up its own.
		out.releaseOnce.Do(out.sess.release)
	}
	return nil
}

// sendControl serializes a control frame to the provider's own message, if the
// serializer has one for it, and sends it.
func (out *outputTransport) sendControl(ctx context.Context, f frames.Frame) {
	if out.sess.closed() {
		return
	}
	msg, err := out.ser.Serialize(f)
	if err == nil && !msg.Empty() {
		_ = out.write(ctx, msg)
	}
}
