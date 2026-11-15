// Package wsclient is a WebSocket client transport: it dials a WebSocket server
// and bridges the socket to a jargo pipeline, so a pipeline can talk to a remote
// peer in both directions.
//
// Inbound messages are deserialized into frames and pushed downstream; outgoing
// audio and messages are serialized and sent, with the audio paced to the rate
// it plays. The wire format is supplied as a wsserver.Serializer, the same one a
// server transport takes, and defaults to protobuf.
//
// RTVITransport speaks RTVI as a client to a bot over this transport.
package wsclient

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"sync"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/protobuf"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/shared"
)

// The events a WebSocket client transport raises. Each carries the
// *websocket.Conn the event is about:
//
//	events.On(tr.Events(), wsclient.EventConnected,
//	    func(ctx context.Context, conn *websocket.Conn) { … })
const (
	// EventConnected fires once the connection to the server is established.
	EventConnected = "on_connected"
	// EventDisconnected fires when the connection to the server is closed by
	// the server or lost. It does not fire for a disconnect this side made.
	EventDisconnected = "on_disconnected"
)

const (
	// openTimeout bounds dialing the server.
	openTimeout = 10 * time.Second
	// readLimit bounds a single inbound WebSocket message.
	readLimit = 1 << 20
)

// Params configures a WebSocket client transport: the media parameters every
// transport takes, plus the ones a client dialing its own socket has.
type Params struct {
	transport.Params

	// AddWAVHeader wraps every outgoing chunk of audio in a WAV container
	// before the serializer encodes it. Nil adds it.
	AddWAVHeader *bool
	// AdditionalHeaders are sent with the request that opens the connection.
	AdditionalHeaders http.Header
	// Serializer encodes and decodes the wire format. Nil uses protobuf.
	Serializer wsserver.Serializer
}

// AddsWAVHeader reports whether outgoing audio is wrapped in a WAV container,
// which it is unless the parameter says otherwise.
func (p Params) AddsWAVHeader() bool {
	return p.AddWAVHeader == nil || *p.AddWAVHeader
}

// Callbacks are the functions a Session calls about its connection.
type Callbacks struct {
	// OnConnected is called when the connection is established.
	OnConnected func(ctx context.Context, conn *websocket.Conn)
	// OnDisconnected is called when the connection is closed by the server or
	// lost.
	OnDisconnected func(ctx context.Context, conn *websocket.Conn)
	// OnMessage is called for each message received on the connection.
	OnMessage func(ctx context.Context, conn *websocket.Conn, message []byte)
}

// connState is where a Session's connection is in its life.
type connState int

const (
	stateNone connState = iota
	stateOpen
	stateClosing
	stateClosed
)

// Session manages one WebSocket client connection: its lifecycle, sending and
// receiving messages, and the callbacks reporting connection events.
//
// An input and an output transport share a Session and both connect it while
// they are set up, so only the first of them dials; and both disconnect it, so
// only the last of them closes it.
type Session struct {
	uri           string
	params        Params
	callbacks     Callbacks
	transportName string

	connection shared.Resource

	writeMu sync.Mutex

	mu    sync.Mutex
	conn  *websocket.Conn
	state connState
	// stopping records that this side is disconnecting, so the read loop ending
	// is not reported as the server going away.
	stopping   bool
	readCancel context.CancelFunc
	readDone   chan struct{}
}

// NewSession builds a Session that connects to uri. transportName names the
// transport that owns it, for logging.
func NewSession(uri string, params Params, callbacks Callbacks, transportName string) *Session {
	return &Session{uri: uri, params: params, callbacks: callbacks, transportName: transportName}
}

// String implements fmt.Stringer.
func (s *Session) String() string { return s.transportName + "::WebsocketClientSession" }

// Connect connects to the WebSocket server.
//
// An input and an output transport share this session and both connect it
// while they are set up, concurrently, so only the first of them dials.
func (s *Session) Connect(ctx context.Context) error {
	return s.connection.Acquire(func() error { return s.connect(ctx) })
}

func (s *Session) connect(ctx context.Context) error {
	dialCtx, cancel := context.WithTimeout(ctx, openTimeout)
	defer cancel()
	conn, resp, err := websocket.Dial(dialCtx, s.uri, &websocket.DialOptions{HTTPHeader: s.params.AdditionalHeaders})
	if resp != nil && resp.Body != nil {
		_ = resp.Body.Close()
	}
	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) && ctx.Err() == nil {
			slog.Error("timeout connecting", "session", s.String(), "uri", s.uri)
			return nil
		}
		return err
	}
	conn.SetReadLimit(readLimit)

	// The read outlives the context it was connected under, which belongs to
	// setting the pipeline up. It is ended by Disconnect.
	readCtx, readCancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	s.mu.Lock()
	s.conn, s.state, s.stopping = conn, stateOpen, false
	s.readCancel, s.readDone = readCancel, done
	s.mu.Unlock()

	go s.clientTaskHandler(readCtx, conn, done)
	if s.callbacks.OnConnected != nil {
		s.callbacks.OnConnected(ctx, conn)
	}
	return nil
}

// Disconnect disconnects from the WebSocket server.
//
// It runs once the last of the transports sharing this session has let go, so
// neither is left sending over a closed socket.
func (s *Session) Disconnect(context.Context) error {
	return s.connection.Release(func() error {
		s.disconnect()
		return nil
	})
}

func (s *Session) disconnect() {
	s.mu.Lock()
	conn := s.conn
	if conn == nil {
		s.mu.Unlock()
		return
	}
	s.state, s.stopping = stateClosing, true
	cancel, done := s.readCancel, s.readDone
	s.mu.Unlock()

	// The close frame goes out first so the server sees the session end
	// normally. Canceling the read first would drop the connection under it.
	_ = conn.Close(websocket.StatusNormalClosure, "")
	cancel()
	<-done
	_ = conn.CloseNow()

	s.mu.Lock()
	s.conn, s.state = nil, stateNone
	s.mu.Unlock()
}

// Send sends a message on the connection, reporting whether it went out.
func (s *Session) Send(ctx context.Context, message wsserver.Message) bool {
	s.mu.Lock()
	conn := s.conn
	s.mu.Unlock()
	if conn == nil {
		return false
	}
	typ := websocket.MessageText
	if message.Binary {
		typ = websocket.MessageBinary
	}
	s.writeMu.Lock()
	err := conn.Write(ctx, typ, message.Data)
	s.writeMu.Unlock()
	if err != nil {
		slog.Error("exception sending data", "session", s.String(), "err", err)
		return false
	}
	return true
}

// IsConnected reports whether the connection is open.
func (s *Session) IsConnected() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == stateOpen
}

// IsClosing reports whether the connection is being closed.
func (s *Session) IsClosing() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == stateClosing
}

// clientTaskHandler reads the connection's messages until it closes, then
// reports the connection gone unless this side closed it.
func (s *Session) clientTaskHandler(ctx context.Context, conn *websocket.Conn, done chan struct{}) {
	defer close(done)
	for {
		_, message, err := conn.Read(ctx)
		if err != nil {
			s.mu.Lock()
			stopping := s.stopping
			if !stopping {
				s.state = stateClosed
			}
			s.mu.Unlock()
			if stopping {
				// Disconnect ended the read; nothing went away.
				return
			}
			if status := websocket.CloseStatus(err); status != websocket.StatusNormalClosure &&
				status != websocket.StatusGoingAway {
				slog.Error("exception receiving data", "session", s.String(), "err", err)
			}
			break
		}
		if s.callbacks.OnMessage != nil {
			s.callbacks.OnMessage(ctx, conn, message)
		}
	}
	if s.callbacks.OnDisconnected != nil {
		s.callbacks.OnDisconnected(ctx, conn)
	}
}

// InputTransport receives frames over a WebSocket client connection: it
// deserializes incoming messages and pushes the frames downstream.
type InputTransport struct {
	*transport.BaseInput
	transport *Transport
	session   *Session
	params    Params
}

func newInput(t *Transport, session *Session, params Params) *InputTransport {
	in := &InputTransport{transport: t, session: session, params: params}
	in.BaseInput = transport.NewBaseInput(t.name+"::Input", params.Params, in)
	return in
}

// Setup sets the serializer up and connects the session.
func (in *InputTransport) Setup(ctx context.Context, s processor.Setup) error {
	if err := in.BaseInput.Setup(ctx, s); err != nil {
		return err
	}
	if in.params.Serializer != nil {
		if err := in.params.Serializer.Setup(s); err != nil {
			return err
		}
	}
	return in.session.Connect(ctx)
}

// Cleanup disconnects the session and cleans the transport up.
func (in *InputTransport) Cleanup(ctx context.Context) error {
	err := in.BaseInput.Cleanup(ctx)
	_ = in.session.Disconnect(ctx)
	in.transport.cleanup(ctx)
	return err
}

// ProcessFrame disconnects the session once an end or cancel frame has stopped
// the input.
func (in *InputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := in.BaseInput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch f.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		return in.session.Disconnect(ctx)
	}
	return nil
}

// onMessage deserializes a message received on the connection and pushes its
// frame.
func (in *InputTransport) onMessage(ctx context.Context, _ *websocket.Conn, message []byte) {
	if in.params.Serializer == nil {
		return
	}
	f, err := in.params.Serializer.Deserialize(message)
	if err != nil {
		slog.Warn("wsclient: deserialize", "err", err)
		return
	}
	if f == nil {
		return
	}
	if af, ok := f.(frames.InputAudioFrame); ok && in.params.AudioInEnabled {
		in.PushAudioFrame(ctx, af)
		return
	}
	if mf, ok := f.(*frames.InputTransportMessageFrame); ok {
		in.PushTransportMessage(ctx, mf.Message)
		return
	}
	_ = in.PushFrame(ctx, f, processor.Downstream)
}

// OutputTransport sends frames over a WebSocket client connection: it
// serializes outgoing frames and paces audio to the rate it plays.
type OutputTransport struct {
	*transport.BaseOutput
	transport *Transport
	session   *Session
	params    Params

	// WriteAudio is called as soon as there is audio (from the TTS, say), and
	// this is only a network connection, so the audio would otherwise go out
	// far faster than it plays. Blocking for as long as a chunk takes to play
	// emulates an audio device; sendInterval is that time, computed during
	// setup.
	paceMu       sync.Mutex
	sendInterval time.Duration
	nextSendTime time.Time
}

func newOutput(t *Transport, session *Session, params Params) *OutputTransport {
	out := &OutputTransport{transport: t, session: session, params: params}
	out.BaseOutput = transport.NewBaseOutput(t.name+"::Output", params.Params, out)
	return out
}

// channels is how many channels the outgoing audio carries.
func (out *OutputTransport) channels() int {
	if out.params.AudioOutChannels == 0 {
		return 1
	}
	return out.params.AudioOutChannels
}

// Setup computes the playout interval, sets the serializer up and connects the
// session.
func (out *OutputTransport) Setup(ctx context.Context, s processor.Setup) error {
	if err := out.BaseOutput.Setup(ctx, s); err != nil {
		return err
	}

	bytesPerSecond := out.SampleRate() * out.channels() * 2
	out.paceMu.Lock()
	out.sendInterval = 0
	if bytesPerSecond > 0 {
		out.sendInterval = time.Duration(out.ChunkSize()) * time.Second / time.Duration(bytesPerSecond)
	}
	out.paceMu.Unlock()

	if out.params.Serializer != nil {
		if err := out.params.Serializer.Setup(s); err != nil {
			return err
		}
	}
	return out.session.Connect(ctx)
}

// ProcessFrame disconnects the session once an end or cancel frame has stopped
// the output.
func (out *OutputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := out.BaseOutput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch f.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		return out.session.Disconnect(ctx)
	}
	return nil
}

// Cleanup disconnects the session and cleans the transport up.
func (out *OutputTransport) Cleanup(ctx context.Context) error {
	err := out.BaseOutput.Cleanup(ctx)
	_ = out.session.Disconnect(ctx)
	out.transport.cleanup(ctx)
	return err
}

// SendMessage sends a transport message over the connection.
func (out *OutputTransport) SendMessage(ctx context.Context, f frames.OutputTransportMessage) error {
	_, err := out.writeFrame(ctx, f)
	return err
}

// WriteAudio writes an audio frame to the connection, with a WAV header when
// the transport adds one, and waits for it to play.
func (out *OutputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	if out.session.IsClosing() || !out.session.IsConnected() {
		return false, nil
	}

	pcm := f.AudioData().Audio
	frame := frames.NewOutputAudioRawFrame(pcm, out.SampleRate(), out.channels())
	if out.params.AddsWAVHeader() {
		frame = frames.NewOutputAudioRawFrame(
			audio.PCMToWAV(frame.Audio, frame.SampleRate, frame.NumChannels), frame.SampleRate, frame.NumChannels)
	}

	if ok, err := out.writeFrame(ctx, frame); !ok || err != nil {
		return false, err
	}

	// Simulate audio playback with a sleep.
	out.writeAudioSleep(ctx)
	return true, nil
}

// writeFrame serializes a frame and sends it. It reports whether the transport
// took the frame: a serializer that emits no payload has still taken it, see
// wsserver.Serializer.
func (out *OutputTransport) writeFrame(ctx context.Context, f frames.Frame) (bool, error) {
	if out.session.IsClosing() || !out.session.IsConnected() {
		return false, nil
	}
	if out.params.Serializer == nil {
		return false, nil
	}
	payload, err := out.params.Serializer.Serialize(f)
	if err != nil {
		return false, err
	}
	if !payload.Empty() {
		out.session.Send(ctx, payload)
	}
	return true, nil
}

// writeAudioSleep simulates audio playback timing, blocking until the next
// chunk is due.
func (out *OutputTransport) writeAudioSleep(ctx context.Context) {
	out.paceMu.Lock()
	interval, next := out.sendInterval, out.nextSendTime
	out.paceMu.Unlock()

	sleep := max(0, time.Until(next))
	if sleep > 0 {
		timer := time.NewTimer(sleep)
		select {
		case <-ctx.Done():
		case <-timer.C:
		}
		timer.Stop()
	}

	out.paceMu.Lock()
	if sleep == 0 {
		out.nextSendTime = time.Now().Add(interval)
	} else {
		out.nextSendTime = out.nextSendTime.Add(interval)
	}
	out.paceMu.Unlock()
}

// connectionHandler is what a transport does about its connection. Transport
// implements it, and a transport built on Transport overrides what it adds to.
type connectionHandler interface {
	onConnected(ctx context.Context, conn *websocket.Conn)
	onDisconnected(ctx context.Context, conn *websocket.Conn)
	onMessage(ctx context.Context, conn *websocket.Conn, message []byte)
}

// Transport is a WebSocket client transport for bidirectional communication:
// input and output processors over one connection, and the events reporting
// it. See EventConnected and EventDisconnected.
type Transport struct {
	name    string
	params  Params
	session *Session
	events  events.Registry
	// self is the transport handling connection events: this one, or the
	// transport built on it.
	self connectionHandler

	mu     sync.Mutex
	input  *InputTransport
	output *OutputTransport
}

// NewTransport builds a transport that connects to uri. The connection is made
// as the pipeline is set up.
func NewTransport(uri string, params Params) *Transport {
	return newTransport("WebsocketClientTransport", uri, params)
}

func newTransport(name, uri string, params Params) *Transport {
	if params.Serializer == nil {
		params.Serializer = protobuf.New()
	}
	t := &Transport{name: name, params: params}
	t.self = t

	callbacks := Callbacks{
		OnConnected:    func(ctx context.Context, conn *websocket.Conn) { t.self.onConnected(ctx, conn) },
		OnDisconnected: func(ctx context.Context, conn *websocket.Conn) { t.self.onDisconnected(ctx, conn) },
		OnMessage: func(ctx context.Context, conn *websocket.Conn, message []byte) {
			t.self.onMessage(ctx, conn, message)
		},
	}
	t.session = NewSession(uri, params, callbacks, name)

	// Register supported handlers. These are the only events a handler can be
	// attached to.
	t.events.Register(EventConnected, false)
	t.events.Register(EventDisconnected, false)
	return t
}

// Input returns the input processor, which receives frames.
func (t *Transport) Input() processor.Processor { return t.inputTransport() }

// Output returns the output processor, which sends frames.
func (t *Transport) Output() processor.Processor { return t.outputTransport() }

// Events is the registry of the events this transport raises. See
// EventConnected and EventDisconnected.
func (t *Transport) Events() *events.Registry { return &t.events }

func (t *Transport) inputTransport() *InputTransport {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.input == nil {
		t.input = newInput(t, t.session, t.params)
	}
	return t.input
}

func (t *Transport) outputTransport() *OutputTransport {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.output == nil {
		t.output = newOutput(t, t.session, t.params)
	}
	return t.output
}

// cleanup waits for the event handlers still running.
func (t *Transport) cleanup(ctx context.Context) { t.events.Cleanup(ctx) }

func (t *Transport) onConnected(ctx context.Context, conn *websocket.Conn) {
	t.events.Call(ctx, EventConnected, t.self, conn)
}

func (t *Transport) onDisconnected(ctx context.Context, conn *websocket.Conn) {
	t.events.Call(ctx, EventDisconnected, t.self, conn)
}

func (t *Transport) onMessage(ctx context.Context, conn *websocket.Conn, message []byte) {
	t.mu.Lock()
	in := t.input
	t.mu.Unlock()
	if in != nil {
		in.onMessage(ctx, conn, message)
	}
}
