// Package rtc implements a WebRTC transport for jargo. It terminates a peer
// connection, decodes received Opus audio into PCM frames, and encodes outgoing
// PCM frames into Opus sent over an RTP track.
//
// Video goes both ways too. The client's camera and screen share are decoded
// (VP8 through libvpx, H.264 through OpenH264) into RGB images, and the images
// the bot shows are encoded to VP8 on its own video track. The codec libraries
// are loaded at run time, the first time video needs them; VideoCodecConfig
// locates them.
//
// Pion is the WebRTC implementation it is built on, which is why it imports
// github.com/pion/webrtc. That is an implementation detail rather than part of
// the interface this package offers.
//
// Signaling runs over HTTP and is driven by the application. The simplest form
// is a single SDP offer/answer exchange through Connection.Answer. RequestHandler
// adds the rest of the protocol a browser client speaks: peer connections keyed
// by pc_id, renegotiation and restarts, and remote ICE candidates trickled in
// after the offer.
package rtc

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/interceptor/pkg/nack"
	"github.com/pion/rtcp"
	"github.com/pion/sdp/v3"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// The positions of the client's transceivers, in the order the client offers
// them: its microphone, its camera, then its screen share.
const (
	AudioTransceiverIndex       = 0
	VideoTransceiverIndex       = 1
	ScreenVideoTransceiverIndex = 2
)

// screenVideoJitterBufferCapacity is how many packets the screen share
// receiver holds while it assembles a frame, which bounds the size of a frame
// it can assemble. The camera's default, defaultJitterBufferCapacity, is
// smaller than a keyframe of a detailed, high-resolution screen share, so such
// a keyframe would be dropped, the next keyframe asked for would be as large,
// and no frame would ever decode. 512 packets fit frames of about 600 KB. The
// capacity does not delay frames, but it is how many packets arrive before a
// lost one is given up on, so the camera keeps the default.
const screenVideoJitterBufferCapacity = 512

// defaultJitterBufferCapacity is how many packets a camera receiver holds
// while it assembles a frame.
const defaultJitterBufferCapacity = 128

// errConnectionClosed is returned when the peer connection closes while a
// caller is waiting on it.
var errConnectionClosed = errors.New("rtc: connection closed")

// ErrNoAnswer is returned when a connection has no SDP answer to give.
var ErrNoAnswer = errors.New("rtc: connection produced no SDP answer")

const (
	// signalingType is the message type that carries signaling over the data
	// channel rather than an application message. The wire spelling is the
	// protocol's.
	signalingType = "signalling" //nolint:misspell // wire value

	// maxMessageQueueSize is how many outgoing messages are held while the data
	// channel is not open yet.
	maxMessageQueueSize = 50

	// dataChannelTimeout is how long after the peer connection is established
	// the data channel has to open before outgoing messages stop being queued.
	dataChannelTimeout = 10 * time.Second

	// defaultConnectionTimeout is how long the peer connection may stay in the
	// connecting state before it is closed.
	defaultConnectionTimeout = 60 * time.Second

	// renegotiationSettle is how long a renegotiation is considered in progress,
	// during which a further request to renegotiate is not sent.
	renegotiationSettle = 2 * time.Second

	// pingWindow is how recent the client's last keep-alive ping must be for the
	// connection to count as connected, once the client has sent one.
	pingWindow = 3 * time.Second
)

// connectionCount numbers connections, so each pc_id names the connection it
// belongs to.
var connectionCount atomic.Int64 //nolint:gochecknoglobals // a process-wide counter

// Answer is the SDP answer to an offer, with the identifier of the peer
// connection that produced it. A client sends the identifier back with a
// renegotiation offer and with the ICE candidates it trickles in.
type Answer struct {
	SDP  string `json:"sdp"`
	Type string `json:"type"`
	PCID string `json:"pc_id"`
}

// signalingMessage is a signaling message sent over the data channel.
type signalingMessage struct {
	Type    string      `json:"type"`
	Message typedSignal `json:"message"`
}

// typedSignal is a signaling message that carries nothing but its type.
type typedSignal struct {
	Type string `json:"type"`
}

// trackStatusMessage turns one of the client's tracks on or off.
type trackStatusMessage struct {
	Type          string `json:"type"`
	ReceiverIndex int    `json:"receiver_index"`
	Enabled       bool   `json:"enabled"`
}

// VideoCodecConfig locates the native libraries video is decoded and encoded
// with. They are loaded the first time video needs them, so a session without
// video never loads them.
type VideoCodecConfig struct {
	// VPXLibPath is libvpx, which decodes and encodes VP8. Empty looks for its
	// conventional names (libvpx.so.12, libvpx.so.11, and so on) on the
	// loader's default search path.
	VPXLibPath string
	// OpenH264LibPath is libopenh264, which decodes H.264. Empty looks for its
	// conventional names (libopenh264.so.8, libopenh264.so.7, and so on).
	OpenH264LibPath string
}

// Connection wraps a Pion peer connection for one client session. It owns the
// outgoing audio track and exposes the incoming one, and the client's camera
// and screen share as Tracks. It is safe for concurrent use.
//
// The peer connection behind it can be replaced: a renegotiation that asks for a
// restart closes the current one and answers on a new one, which also takes a
// new pc_id. The Connection itself, and the session built on it, carry on.
type Connection struct {
	iceServers        []webrtc.ICEServer
	api               *webrtc.API
	localAudio        *webrtc.TrackLocalStaticSample
	connectionTimeout time.Duration
	name              string
	videoCodecs       VideoCodecConfig

	// sigMu serializes signaling: an offer, a renegotiation and the candidates
	// trickled in for them arrive on separate requests.
	sigMu sync.Mutex

	mu sync.Mutex
	// gen counts peer connections. A callback from a peer connection that has
	// since been replaced carries an older generation and is ignored.
	gen    int
	pc     *webrtc.PeerConnection
	pcID   string
	answer *webrtc.SessionDescription

	remoteAudio *webrtc.TrackRemote
	// trackReady is closed when remoteAudio is set, and replaced when the peer
	// connection is.
	trackReady chan struct{}
	// remoteVideo holds the incoming video tracks by the receiver they arrived
	// on, and remoteVideoReady is closed, then replaced, as each arrives.
	remoteVideo      map[*webrtc.RTPReceiver]*webrtc.TrackRemote
	remoteVideoReady chan struct{}
	// trackMap holds the input Tracks handed out, by transceiver index.
	trackMap map[int]*Track
	// replaced is closed when the current peer connection is replaced by a
	// restart.
	replaced chan struct{}

	// connectInvoked is set once the pipeline is ready for the connection. The
	// connected event and application messages wait for it.
	connectInvoked bool
	// lastReceived is when the client's last keep-alive ping arrived.
	lastReceived  time.Time
	renegotiating bool

	renegotiationTimer *time.Timer
	connectingTimer    *time.Timer
	dataChannelTimer   *time.Timer

	dc                 *webrtc.DataChannel
	dcOpen             bool
	dataChannelEnabled bool
	outgoing           [][]byte
	pendingApp         [][]byte

	onAppMessage   []func([]byte)
	onConnected    []func()
	onDisconnected []func()
	onClosed       []func()
	onFailed       []func()
	onTrackStatus  []func(receiverIndex int, enabled bool)

	connectedOnce sync.Once
	connected     chan struct{}
	closeOnce     sync.Once
	closed        chan struct{}

	// events runs the event handlers, one at a time and in the order the
	// events happened, off the goroutine that raised them: most are raised from
	// inside Pion's callbacks, which a handler that blocks would stall.
	events serialExecutor
}

// serialExecutor runs functions one at a time, in the order they were handed
// over, on a goroutine of its own that lasts while there is work.
type serialExecutor struct {
	mu      sync.Mutex
	queue   []func()
	running bool
}

// run queues fn behind everything handed over before it.
func (e *serialExecutor) run(fn func()) {
	e.mu.Lock()
	e.queue = append(e.queue, fn)
	if e.running {
		e.mu.Unlock()
		return
	}
	e.running = true
	e.mu.Unlock()
	go func() {
		for {
			e.mu.Lock()
			if len(e.queue) == 0 {
				e.running = false
				e.mu.Unlock()
				return
			}
			next := e.queue[0]
			e.queue = e.queue[1:]
			e.mu.Unlock()
			next()
		}
	}()
}

// Option configures a Connection.
type Option func(*Connection)

// WithICEServers sets the ICE servers used for connectivity. Passing none (or
// an empty list) disables STUN, leaving only host candidates (useful for
// same-host connections and tests).
func WithICEServers(servers ...webrtc.ICEServer) Option {
	return func(c *Connection) { c.iceServers = servers }
}

// WithVideoCodecs locates the native libraries video is decoded and encoded
// with. Without it they are looked for by their conventional names.
func WithVideoCodecs(cfg VideoCodecConfig) Option {
	return func(c *Connection) { c.videoCodecs = cfg }
}

// WithConnectionTimeout sets how long the peer connection may stay in the
// connecting state before it is closed. The default is 60 seconds.
func WithConnectionTimeout(d time.Duration) Option {
	return func(c *Connection) { c.connectionTimeout = d }
}

// NewConnection creates a peer connection with an outgoing Opus audio track. By
// default a public STUN server is configured so it works across NATs; override
// with WithICEServers.
func NewConnection(opts ...Option) (*Connection, error) {
	c := &Connection{
		iceServers:        []webrtc.ICEServer{{URLs: []string{"stun:stun.l.google.com:19302"}}},
		connectionTimeout: defaultConnectionTimeout,
		name:              fmt.Sprintf("Connection#%d", connectionCount.Add(1)-1),
		connected:         make(chan struct{}),
		closed:            make(chan struct{}),
	}
	for _, opt := range opts {
		opt(c)
	}

	api, err := newAPI()
	if err != nil {
		return nil, err
	}
	c.api = api

	local, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2},
		"audio", "jargo",
	)
	if err != nil {
		return nil, fmt.Errorf("new local audio track: %w", err)
	}
	c.localAudio = local

	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.initializeLocked(); err != nil {
		return nil, err
	}
	return c, nil
}

// The video codecs offered, in order of preference, each with its own payload
// type and the RTX payload type after it.
//
//nolint:gochecknoglobals // a fixed table
var videoCodecs = []struct {
	mimeType string
	fmtp     string
}{
	{webrtc.MimeTypeVP8, ""},
	{webrtc.MimeTypeH264, "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f"},
	{webrtc.MimeTypeH264, "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f"},
}

// firstVideoPayloadType is the payload type of the first video codec.
const firstVideoPayloadType = 97

// newAPI builds the Pion API the peer connections are made with: Opus and the
// G.711 and G.722 codecs for audio, VP8 and H.264 for video, the media id
// header extension that routes packets to their transceiver, and NACKs and
// RTCP reports.
func newAPI() (*webrtc.API, error) {
	m := &webrtc.MediaEngine{}
	for _, codec := range []webrtc.RTPCodecParameters{
		{
			MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2, SDPFmtpLine: "minptime=10;useinbandfec=1",
			PayloadType: 111,
		},
		{MimeType: webrtc.MimeTypeG722, ClockRate: 8000, PayloadType: 9},
		{MimeType: webrtc.MimeTypePCMU, ClockRate: 8000, PayloadType: 0},
		{MimeType: webrtc.MimeTypePCMA, ClockRate: 8000, PayloadType: 8},
	} {
		if err := m.RegisterCodec(codec, webrtc.RTPCodecTypeAudio); err != nil {
			return nil, fmt.Errorf("register codecs: %w", err)
		}
	}
	feedback := []webrtc.RTCPFeedback{{Type: "nack"}, {Type: "nack", Parameter: "pli"}, {Type: "goog-remb"}}
	pt := webrtc.PayloadType(firstVideoPayloadType)
	for _, vc := range videoCodecs {
		for _, codec := range []webrtc.RTPCodecParameters{
			{
				MimeType: vc.mimeType, ClockRate: 90000, SDPFmtpLine: vc.fmtp, RTCPFeedback: feedback,
				PayloadType: pt,
			},
			{
				MimeType: webrtc.MimeTypeRTX, ClockRate: 90000, SDPFmtpLine: fmt.Sprintf("apt=%d", pt),
				PayloadType: pt + 1,
			},
		} {
			if err := m.RegisterCodec(codec, webrtc.RTPCodecTypeVideo); err != nil {
				return nil, fmt.Errorf("register codecs: %w", err)
			}
		}
		pt += 2
	}
	for _, kind := range []webrtc.RTPCodecType{webrtc.RTPCodecTypeAudio, webrtc.RTPCodecTypeVideo} {
		if err := m.RegisterHeaderExtension(webrtc.RTPHeaderExtensionCapability{URI: sdp.SDESMidURI}, kind); err != nil {
			return nil, fmt.Errorf("register header extension: %w", err)
		}
	}

	registry := &interceptor.Registry{}
	generator, err := nack.NewGeneratorInterceptor()
	if err != nil {
		return nil, fmt.Errorf("nack generator: %w", err)
	}
	responder, err := nack.NewResponderInterceptor()
	if err != nil {
		return nil, fmt.Errorf("nack responder: %w", err)
	}
	registry.Add(responder)
	registry.Add(generator)
	if err := webrtc.ConfigureRTCPReports(registry); err != nil {
		return nil, fmt.Errorf("rtcp reports: %w", err)
	}
	return webrtc.NewAPI(webrtc.WithMediaEngine(m), webrtc.WithInterceptorRegistry(registry)), nil
}

// initializeLocked builds a new peer connection and the state that goes with
// it. It runs once on creation and again on every restart. c.mu must be held.
func (c *Connection) initializeLocked() error {
	// One transport for every media line, so a line the client adds later is
	// carried on the transport already established.
	pc, err := c.api.NewPeerConnection(webrtc.Configuration{
		ICEServers:   c.iceServers,
		BundlePolicy: webrtc.BundlePolicyMaxBundle,
	})
	if err != nil {
		return fmt.Errorf("new peer connection: %w", err)
	}
	if _, err := pc.AddTrack(c.localAudio); err != nil {
		_ = pc.Close()
		return fmt.Errorf("add local audio track: %w", err)
	}

	if c.replaced != nil {
		close(c.replaced)
	}
	c.stopTracksLocked()
	c.remoteVideo = map[*webrtc.RTPReceiver]*webrtc.TrackRemote{}
	c.remoteVideoReady = make(chan struct{})
	c.replaced = make(chan struct{})
	c.gen++
	c.pc = pc
	c.pcID = c.name + "-" + randomHex(16)
	c.answer = nil
	c.remoteAudio = nil
	c.trackReady = make(chan struct{})
	c.dc = nil
	c.dcOpen = false
	c.renegotiating = false
	c.lastReceived = time.Time{}
	c.outgoing = nil
	c.dataChannelEnabled = true
	c.pendingApp = nil
	c.connectingTimer = nil
	c.dataChannelTimer = nil
	c.setupListeners(pc, c.gen)
	return nil
}

// randomHex returns n random bytes as hex.
func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// setupListeners wires the peer connection's callbacks. Each one checks the
// generation it was registered for, so a peer connection replaced by a restart
// can no longer act on the Connection.
func (c *Connection) setupListeners(pc *webrtc.PeerConnection, gen int) {
	pc.OnTrack(func(tr *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) {
		slog.Debug("rtc: track received", "kind", tr.Kind().String())
		c.mu.Lock()
		defer c.mu.Unlock()
		if gen != c.gen {
			return
		}
		if tr.Kind() == webrtc.RTPCodecTypeVideo {
			if _, ok := c.remoteVideo[receiver]; !ok {
				c.remoteVideo[receiver] = tr
				close(c.remoteVideoReady)
				c.remoteVideoReady = make(chan struct{})
			}
			return
		}
		if tr.Kind() != webrtc.RTPCodecTypeAudio || c.remoteAudio != nil {
			return
		}
		c.remoteAudio = tr
		close(c.trackReady)
	})

	pc.OnDataChannel(func(dc *webrtc.DataChannel) {
		c.mu.Lock()
		if gen != c.gen {
			c.mu.Unlock()
			return
		}
		c.dc = dc
		c.mu.Unlock()

		dc.OnOpen(func() {
			slog.Debug("rtc: data channel is open")
			c.flushMessageQueue(gen)
		})
		dc.OnMessage(func(msg webrtc.DataChannelMessage) {
			c.handleDataChannelMessage(gen, msg)
		})
		// A channel the remote peer created may already be open, in which case
		// no open event follows.
		if dc.ReadyState() == webrtc.DataChannelStateOpen {
			c.flushMessageQueue(gen)
		}
	})

	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		c.handleConnectionState(gen, s)
	})

	pc.OnICEConnectionStateChange(func(s webrtc.ICEConnectionState) {
		slog.Debug("rtc: ICE connection state", "state", s.String())
	})
	pc.OnICEGatheringStateChange(func(s webrtc.ICEGatheringState) {
		slog.Debug("rtc: ICE gathering state", "state", s.String())
	})
}

// PCID identifies the current peer connection. It changes when the connection
// is restarted.
func (c *Connection) PCID() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.pcID
}

// Initialize applies the client's offer and prepares the answer, which
// GetAnswer then returns.
func (c *Connection) Initialize(offer webrtc.SessionDescription) error {
	c.sigMu.Lock()
	defer c.sigMu.Unlock()
	return c.createAnswer(offer)
}

// Answer completes the signaling handshake in one call: it applies the remote
// offer and returns an answer carrying every local candidate.
func (c *Connection) Answer(offer webrtc.SessionDescription) (webrtc.SessionDescription, error) {
	if err := c.Initialize(offer); err != nil {
		return webrtc.SessionDescription{}, err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.answer == nil {
		return webrtc.SessionDescription{}, ErrNoAnswer
	}
	return *c.answer, nil
}

// createAnswer applies the offer to the current peer connection and builds the
// answer. It waits for ICE gathering to finish, so the answer carries all the
// local candidates. c.sigMu must be held.
func (c *Connection) createAnswer(offer webrtc.SessionDescription) error {
	c.mu.Lock()
	pc := c.pc
	c.mu.Unlock()

	if err := pc.SetRemoteDescription(offer); err != nil {
		return fmt.Errorf("set remote description: %w", err)
	}
	c.forceTransceiversToSendRecv(pc)
	slog.Debug("rtc: creating answer")
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		return fmt.Errorf("create answer: %w", err)
	}
	gatherComplete := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(answer); err != nil {
		return fmt.Errorf("set local description: %w", err)
	}
	<-gatherComplete

	c.mu.Lock()
	defer c.mu.Unlock()
	if pc == c.pc {
		c.answer = pc.LocalDescription()
	}
	return nil
}

// mediaTransceivers returns the peer connection's transceivers that carry a
// media line of the offer, in the order of their lines.
func mediaTransceivers(pc *webrtc.PeerConnection) []*webrtc.RTPTransceiver {
	var out []*webrtc.RTPTransceiver
	for _, t := range pc.GetTransceivers() {
		if t.Mid() != "" {
			out = append(out, t)
		}
	}
	return out
}

// forceTransceiversToSendRecv makes the first two transceivers, the
// microphone's and the camera's, send as well as receive, so the bot can send
// its audio and video on them. The others, a screen share, only receive.
//
// The microphone's transceiver already sends the outgoing audio. The camera's
// gets a sender here, with a track sending nothing until the transport
// replaces it with its own (see ReplaceVideoTrack). The bot's video is
// encoded to VP8, so a camera line the client offers without VP8 is left
// receiving only.
func (c *Connection) forceTransceiversToSendRecv(pc *webrtc.PeerConnection) {
	for i, t := range mediaTransceivers(pc) {
		if i >= 2 || t.Sender() != nil || t.Kind() != webrtc.RTPCodecTypeVideo {
			continue
		}
		receiver := t.Receiver()
		if receiver == nil || !hasCodec(receiver.GetParameters().Codecs, webrtc.MimeTypeVP8) {
			slog.Warn("rtc: the client's camera does not take VP8, the bot cannot send it video")
			continue
		}
		placeholder, err := newLocalVideoTrack()
		if err != nil {
			slog.Error("rtc: new local video track", "err", err)
			continue
		}
		sender, err := c.api.NewRTPSender(placeholder, receiver.Transport())
		if err != nil {
			slog.Error("rtc: new video sender", "err", err)
			continue
		}
		if err := t.SetSender(sender, placeholder); err != nil {
			slog.Error("rtc: set video sender", "err", err)
			_ = sender.Stop()
			continue
		}
		go readSenderRTCP(sender)
	}
}

// newLocalVideoTrack builds an outgoing VP8 video track.
func newLocalVideoTrack() (*webrtc.TrackLocalStaticSample, error) {
	return webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8, ClockRate: 90000},
		"video", "jargo",
	)
}

// hasCodec reports whether codecs include one of the given MIME type.
func hasCodec(codecs []webrtc.RTPCodecParameters, mimeType string) bool {
	for _, codec := range codecs {
		if strings.EqualFold(codec.MimeType, mimeType) {
			return true
		}
	}
	return false
}

// keyframeRequester is an outgoing video track that can be told the receiver
// wants a keyframe.
type keyframeRequester interface {
	requestKeyframe()
}

// bitrateTarget is an outgoing video track whose bitrate the receiver's
// estimate of the available bandwidth sets.
type bitrateTarget interface {
	setTargetBitrate(bitrate int)
}

// readSenderRTCP reads the RTCP the client sends about the bot's video until
// the sender stops. A picture loss or a full intra request makes the next
// frame a keyframe, and a bandwidth estimate sets the encoder's bitrate. The
// reading also runs the NACK responder that retransmits lost packets.
func readSenderRTCP(sender *webrtc.RTPSender) {
	for {
		packets, _, err := sender.ReadRTCP()
		if err != nil {
			return
		}
		track := sender.Track()
		for _, packet := range packets {
			switch p := packet.(type) {
			case *rtcp.PictureLossIndication, *rtcp.FullIntraRequest:
				if kr, ok := track.(keyframeRequester); ok {
					kr.requestKeyframe()
				}
			case *rtcp.ReceiverEstimatedMaximumBitrate:
				if bt, ok := track.(bitrateTarget); ok {
					bt.setTargetBitrate(int(p.Bitrate))
				}
			}
		}
	}
}

// replaceTrack sends track on the transceiver at index, or logs that there is
// no transceiver there sending.
func (c *Connection) replaceTrack(index int, track webrtc.TrackLocal, what string) {
	slog.Debug("rtc: replacing track", "track", what)
	c.mu.Lock()
	pc := c.pc
	c.mu.Unlock()
	transceivers := mediaTransceivers(pc)
	if len(transceivers) <= index || transceivers[index].Sender() == nil {
		slog.Warn("rtc: transceiver not found, cannot replace track", "track", what)
		return
	}
	if err := transceivers[index].Sender().ReplaceTrack(track); err != nil {
		slog.Warn("rtc: cannot replace track", "track", what, "err", err)
	}
}

// ReplaceVideoTrack sends track on the camera's transceiver, the second.
func (c *Connection) ReplaceVideoTrack(track webrtc.TrackLocal) {
	c.replaceTrack(VideoTransceiverIndex, track, "video")
}

// ReplaceScreenVideoTrack sends track on the screen share's transceiver, the
// third.
func (c *Connection) ReplaceScreenVideoTrack(track webrtc.TrackLocal) {
	c.replaceTrack(ScreenVideoTransceiverIndex, track, "screen video")
}

// VideoInputTrack returns the client's camera, the track of the second
// transceiver, or nil when the client offered none.
func (c *Connection) VideoInputTrack() *Track {
	return c.inputTrack(VideoTransceiverIndex, defaultJitterBufferCapacity, "video")
}

// ScreenVideoInputTrack returns the client's screen share, the track of the
// third transceiver, or nil when the client offered none.
func (c *Connection) ScreenVideoInputTrack() *Track {
	return c.inputTrack(ScreenVideoTransceiverIndex, screenVideoJitterBufferCapacity, "screen video")
}

// inputTrack returns the Track of the transceiver at index, creating it the
// first time.
func (c *Connection) inputTrack(index int, capacity uint16, what string) *Track {
	c.mu.Lock()
	defer c.mu.Unlock()
	if t := c.trackMap[index]; t != nil {
		return t
	}
	transceivers := mediaTransceivers(c.pc)
	if len(transceivers) <= index || transceivers[index].Receiver() == nil {
		slog.Warn("rtc: no transceiver is available", "track", what)
		return nil
	}
	t := newTrack(c, c.gen, transceivers[index].Receiver(), transceivers[index].Kind(), capacity)
	if c.trackMap == nil {
		c.trackMap = map[int]*Track{}
	}
	c.trackMap[index] = t
	return t
}

// stopTracksLocked stops the input Tracks handed out and forgets them. c.mu
// must be held.
func (c *Connection) stopTracksLocked() {
	for _, t := range c.trackMap {
		t.Stop()
	}
	c.trackMap = nil
}

// remoteVideoTrack waits for the incoming video track of receiver, from the
// peer connection of generation gen. It returns nil once that peer connection
// is replaced or closed, or stop is closed.
func (c *Connection) remoteVideoTrack(gen int, receiver *webrtc.RTPReceiver, stop <-chan struct{}) *webrtc.TrackRemote {
	for {
		c.mu.Lock()
		if gen != c.gen {
			c.mu.Unlock()
			return nil
		}
		tr, ready, replaced := c.remoteVideo[receiver], c.remoteVideoReady, c.replaced
		c.mu.Unlock()
		if tr != nil {
			return tr
		}
		select {
		case <-ready:
		case <-replaced:
			return nil
		case <-c.closed:
			return nil
		case <-stop:
			return nil
		}
	}
}

// sendPictureLossIndication asks the client for a keyframe of the stream ssrc,
// on the peer connection of generation gen.
func (c *Connection) sendPictureLossIndication(gen int, ssrc webrtc.SSRC) {
	c.mu.Lock()
	pc := c.pc
	current := gen == c.gen
	c.mu.Unlock()
	if !current {
		return
	}
	if err := pc.WriteRTCP([]rtcp.Packet{&rtcp.PictureLossIndication{MediaSSRC: uint32(ssrc)}}); err != nil {
		slog.Debug("rtc: send picture loss indication", "err", err)
	}
}

// GetAnswer returns the answer to the last offer applied, or nil when there is
// none yet.
func (c *Connection) GetAnswer() *Answer {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.answer == nil {
		return nil
	}
	return &Answer{SDP: c.answer.SDP, Type: c.answer.Type.String(), PCID: c.pcID}
}

// Renegotiate answers a new offer from the client on the same session. With
// restartPC the current peer connection is closed and the offer is answered on
// a new one; without it the offer is applied to the current one.
func (c *Connection) Renegotiate(offer webrtc.SessionDescription, restartPC bool) error {
	c.sigMu.Lock()
	defer c.sigMu.Unlock()
	slog.Debug("rtc: renegotiating", "pc_id", c.PCID())

	if restartPC {
		c.emit(c.disconnectedHandlers())
		slog.Debug("rtc: closing old peer connection")
		c.mu.Lock()
		old := c.pc
		c.stopTimersLocked()
		// The new peer connection's generation leaves the old one's callbacks
		// with nothing to act on, so closing it does not close the session.
		if err := c.initializeLocked(); err != nil {
			c.mu.Unlock()
			_ = old.Close()
			return err
		}
		c.mu.Unlock()
		_ = old.Close()
	}

	if err := c.createAnswer(offer); err != nil {
		return err
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.renegotiationTimer != nil {
		c.renegotiationTimer.Stop()
	}
	var timer *time.Timer
	timer = time.AfterFunc(renegotiationSettle, func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		c.renegotiating = false
		// A renegotiation arriving inside the settle time replaces this timer.
		if c.renegotiationTimer == timer {
			c.renegotiationTimer = nil
		}
	})
	c.renegotiationTimer = timer
	return nil
}

// AddICECandidate adds a remote ICE candidate the client trickled in after its
// offer. A candidate with an empty candidate string is the end-of-candidates
// marker for its media line.
func (c *Connection) AddICECandidate(candidate webrtc.ICECandidateInit) error {
	c.sigMu.Lock()
	defer c.sigMu.Unlock()
	slog.Debug("rtc: adding remote candidate", "candidate", candidate.Candidate)
	c.mu.Lock()
	pc := c.pc
	c.mu.Unlock()
	return pc.AddICECandidate(candidate)
}

// Connect tells the connection the pipeline is ready for it. The connected
// event waits for this, and application messages received before it are held
// and delivered now.
func (c *Connection) Connect() {
	c.mu.Lock()
	c.connectInvoked = true
	connected := c.isConnectedLocked()
	var pending [][]byte
	if connected {
		pending = c.pendingApp
		c.pendingApp = nil
	}
	c.mu.Unlock()

	// When the peer connected first, the event was held back until now.
	if !connected {
		return
	}
	c.emit(c.connectedHandlers())
	slog.Debug("rtc: flushing pending app messages", "count", len(pending))
	for _, m := range pending {
		c.emitAppMessage(m)
	}
	// The first frames of video were likely lost while the pipeline was not
	// ready, and a decoder can show nothing until the next keyframe, which can
	// be seconds away. A renegotiation makes the client send one now.
	videoInputTrack := c.VideoInputTrack()
	if videoInputTrack != nil {
		videoInputTrack.DiscardOldFrames()
	}
	screenVideoInputTrack := c.ScreenVideoInputTrack()
	if screenVideoInputTrack != nil {
		screenVideoInputTrack.DiscardOldFrames()
	}
	if videoInputTrack != nil || screenVideoInputTrack != nil {
		c.AskToRenegotiate()
	}
}

// IsConnected reports whether the connection is up and the pipeline has been
// connected to it. Once the client has sent a keep-alive ping, it also requires
// the last one to be recent.
func (c *Connection) IsConnected() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.isConnectedLocked()
}

func (c *Connection) isConnectedLocked() bool {
	if !c.connectInvoked {
		return false
	}
	if c.lastReceived.IsZero() {
		// A client that never pings may not have created a data channel at all,
		// so the peer connection's own state is all there is to go on.
		return c.pc.ConnectionState() == webrtc.PeerConnectionStateConnected
	}
	return time.Since(c.lastReceived) < pingWindow
}

// handleConnectionState reacts to a change in the peer connection's state.
func (c *Connection) handleConnectionState(gen int, s webrtc.PeerConnectionState) {
	c.mu.Lock()
	if gen != c.gen {
		c.mu.Unlock()
		return
	}
	if s == webrtc.PeerConnectionStateConnecting {
		c.monitorConnectingStateLocked(gen)
	} else {
		c.cancelConnectingTimerLocked()
	}
	if s == webrtc.PeerConnectionStateConnected && c.dataChannelTimer == nil {
		c.startDataChannelTimeoutLocked(gen)
	}
	invoked := c.connectInvoked
	c.mu.Unlock()

	if s == webrtc.PeerConnectionStateConnected {
		c.connectedOnce.Do(func() { close(c.connected) })
		// The event waits until the pipeline is ready; Connect raises it then.
		if !invoked {
			return
		}
	}
	slog.Debug("rtc: connection state changed", "state", s.String())
	switch s {
	case webrtc.PeerConnectionStateConnected:
		c.emit(c.connectedHandlers())
	case webrtc.PeerConnectionStateDisconnected:
		c.emit(c.disconnectedHandlers())
	case webrtc.PeerConnectionStateFailed:
		c.emit(c.failedHandlers())
		slog.Warn("rtc: connection failed, closing peer connection")
		// Closing from inside a Pion callback can block on that callback.
		go c.closePeer(gen)
	case webrtc.PeerConnectionStateClosed:
		c.markClosed()
	default:
		// New and Connecting raise no event.
	}
}

// monitorConnectingStateLocked closes the peer connection if it stays in the
// connecting state for longer than the connection timeout. c.mu must be held.
func (c *Connection) monitorConnectingStateLocked(gen int) {
	slog.Debug("rtc: monitoring connecting state")
	c.cancelConnectingTimerLocked()
	c.connectingTimer = time.AfterFunc(c.connectionTimeout, func() {
		slog.Warn("rtc: timeout establishing the connection to the remote peer, closing")
		c.closePeer(gen)
	})
}

func (c *Connection) cancelConnectingTimerLocked() {
	if c.connectingTimer != nil {
		c.connectingTimer.Stop()
		c.connectingTimer = nil
	}
}

// startDataChannelTimeoutLocked gives the data channel a bounded time to open
// once the peer connection is up. A client may never create one; past the
// timeout the queue is dropped and further messages are discarded rather than
// queued. c.mu must be held.
func (c *Connection) startDataChannelTimeoutLocked(gen int) {
	c.dataChannelTimer = time.AfterFunc(dataChannelTimeout, func() {
		c.mu.Lock()
		defer c.mu.Unlock()
		if gen != c.gen || c.dcOpen {
			return
		}
		slog.Warn("rtc: data channel not established after connection, " +
			"clearing message queue and disabling future queueing")
		c.outgoing = nil
		c.dataChannelEnabled = false
	})
}

func (c *Connection) cancelDataChannelTimerLocked() {
	if c.dataChannelTimer != nil {
		c.dataChannelTimer.Stop()
		c.dataChannelTimer = nil
	}
}

func (c *Connection) stopTimersLocked() {
	c.cancelConnectingTimerLocked()
	c.cancelDataChannelTimerLocked()
}

// handleDataChannelMessage routes one message from the data channel: a
// keep-alive ping, a signaling message, or an application message.
func (c *Connection) handleDataChannelMessage(gen int, msg webrtc.DataChannelMessage) {
	c.mu.Lock()
	if gen != c.gen {
		c.mu.Unlock()
		return
	}
	// The client pings to say it is still there.
	if msg.IsString && strings.HasPrefix(string(msg.Data), "ping") {
		c.lastReceived = time.Now()
		c.mu.Unlock()
		return
	}
	c.mu.Unlock()

	var envelope struct {
		Type    string          `json:"type"`
		Message json.RawMessage `json:"message"`
	}
	if err := json.Unmarshal(msg.Data, &envelope); err != nil {
		slog.Error("rtc: error parsing JSON message", "message", string(msg.Data), "err", err)
		return
	}
	if envelope.Type == signalingType && len(envelope.Message) > 0 && string(envelope.Message) != "null" {
		c.handleSignalingMessage(envelope.Message)
		return
	}

	c.mu.Lock()
	if !c.isConnectedLocked() {
		slog.Debug("rtc: client not connected, queuing app message")
		c.pendingApp = append(c.pendingApp, msg.Data)
		c.mu.Unlock()
		return
	}
	c.mu.Unlock()
	c.emitAppMessage(msg.Data)
}

// handleSignalingMessage acts on a signaling message from the client.
func (c *Connection) handleSignalingMessage(raw json.RawMessage) {
	slog.Debug("rtc: signaling message received", "message", string(raw))
	var m trackStatusMessage
	if err := json.Unmarshal(raw, &m); err != nil || m.Type != "trackStatus" {
		slog.Warn("rtc: unsupported signaling message", "message", string(raw))
		return
	}
	var track *Track
	switch m.ReceiverIndex {
	case VideoTransceiverIndex:
		track = c.VideoInputTrack()
	case ScreenVideoTransceiverIndex:
		track = c.ScreenVideoInputTrack()
	}
	if track != nil {
		track.SetEnabled(m.Enabled)
	}
	c.mu.Lock()
	hs := append([]func(int, bool){}, c.onTrackStatus...)
	c.mu.Unlock()
	c.events.run(func() {
		for _, h := range hs {
			h(m.ReceiverIndex, m.Enabled)
		}
	})
}

// AskToRenegotiate asks the client to send a new offer. A request made while a
// renegotiation is still settling is dropped.
func (c *Connection) AskToRenegotiate() {
	c.mu.Lock()
	if c.renegotiating {
		c.mu.Unlock()
		return
	}
	c.renegotiating = true
	c.mu.Unlock()
	_ = c.SendAppMessage(signalingMessage{Type: signalingType, Message: typedSignal{Type: "renegotiate"}})
}

// WaitConnected blocks until the peer connection is established, ctx is done, or
// the connection closes.
func (c *Connection) WaitConnected(ctx context.Context) error {
	select {
	case <-c.connected:
		return nil
	case <-c.closed:
		return errConnectionClosed
	case <-ctx.Done():
		return ctx.Err()
	}
}

// RemoteAudioTrack returns the incoming audio track, blocking until it arrives,
// ctx is done, or the connection closes.
func (c *Connection) RemoteAudioTrack(ctx context.Context) (*webrtc.TrackRemote, error) {
	return c.nextRemoteAudioTrack(ctx, nil)
}

// nextRemoteAudioTrack returns an incoming audio track other than prev,
// blocking until one arrives. A track that stopped delivering is replaced only
// when the peer connection is, so waiting for a different one parks the reader
// until a restart brings a fresh track instead of reading the dead one again.
func (c *Connection) nextRemoteAudioTrack(ctx context.Context, prev *webrtc.TrackRemote) (*webrtc.TrackRemote, error) {
	for {
		c.mu.Lock()
		tr, ready, replaced := c.remoteAudio, c.trackReady, c.replaced
		c.mu.Unlock()
		if tr != nil && tr != prev {
			return tr, nil
		}
		if tr != nil {
			// The current track is the one that stopped: wait for the peer
			// connection to be replaced.
			ready = replaced
		}
		select {
		case <-ready:
		case <-c.closed:
			return nil, errConnectionClosed
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// WriteAudio sends one Opus packet on the outgoing track. dur is the packet's
// playback duration, used to advance RTP timestamps.
func (c *Connection) WriteAudio(packet []byte, dur time.Duration) error {
	return c.localAudio.WriteSample(media.Sample{Data: packet, Duration: dur})
}

// OnMessage registers a handler for application messages received on the data
// channel. Messages that arrive before the pipeline has connected are held, and
// delivered in order by Connect.
func (c *Connection) OnMessage(h func([]byte)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onAppMessage = append(c.onAppMessage, h)
}

// OnConnected registers a handler for the peer connection being established,
// raised once the pipeline has connected (see Connect).
func (c *Connection) OnConnected(h func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onConnected = append(c.onConnected, h)
}

// OnDisconnected registers a handler for the peer connection being lost. The
// session may still recover, through ICE or a restart.
func (c *Connection) OnDisconnected(h func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onDisconnected = append(c.onDisconnected, h)
}

// OnClosed registers a handler for the connection closing for good.
func (c *Connection) OnClosed(h func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onClosed = append(c.onClosed, h)
}

// OnFailed registers a handler for the peer connection failing. The connection
// closes right after.
func (c *Connection) OnFailed(h func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onFailed = append(c.onFailed, h)
}

// OnTrackStatus registers a handler for the client turning one of its tracks on
// or off. receiverIndex is the track's position: 0 for audio, 1 for the camera,
// 2 for a screen share.
func (c *Connection) OnTrackStatus(h func(receiverIndex int, enabled bool)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onTrackStatus = append(c.onTrackStatus, h)
}

func (c *Connection) connectedHandlers() []func() {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]func(){}, c.onConnected...)
}

func (c *Connection) disconnectedHandlers() []func() {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]func(){}, c.onDisconnected...)
}

func (c *Connection) failedHandlers() []func() {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]func(){}, c.onFailed...)
}

// emit raises an event: its handlers run after those of every event raised
// before it.
func (c *Connection) emit(hs []func()) {
	c.events.run(func() {
		for _, h := range hs {
			h()
		}
	})
}

// emitAppMessage raises an application message, in order with the events.
func (c *Connection) emitAppMessage(m []byte) {
	c.mu.Lock()
	hs := append([]func([]byte){}, c.onAppMessage...)
	c.mu.Unlock()
	c.events.run(func() {
		for _, h := range hs {
			h(m)
		}
	})
}

// SendMessage sends an application message to the client over the data channel.
// While the channel is not open yet the message is queued, up to a limit, and
// flushed in order once it opens. If the channel never opens, queueing stops and
// messages are discarded.
func (c *Connection) SendMessage(data []byte) error {
	c.mu.Lock()
	if c.dc != nil && c.dcOpen {
		dc := c.dc
		c.mu.Unlock()
		return dc.SendText(string(data))
	}
	defer c.mu.Unlock()
	switch {
	case !c.dataChannelEnabled:
		// The client may never create a data channel.
		slog.Debug("rtc: data channel unavailable and queueing disabled, discarding message")
	case len(c.outgoing) < maxMessageQueueSize:
		slog.Debug("rtc: data channel not ready, queuing message")
		c.outgoing = append(c.outgoing, data)
	default:
		slog.Warn("rtc: message queue is full, discarding message", "size", maxMessageQueueSize)
	}
	return nil
}

// SendAppMessage encodes message as JSON and sends it like SendMessage.
func (c *Connection) SendAppMessage(message any) error {
	data, err := json.Marshal(message)
	if err != nil {
		return err
	}
	return c.SendMessage(data)
}

// flushMessageQueue sends everything queued while the data channel was not
// open, now that it is.
func (c *Connection) flushMessageQueue(gen int) {
	c.mu.Lock()
	if gen != c.gen || c.dc == nil {
		c.mu.Unlock()
		return
	}
	c.cancelDataChannelTimerLocked()
	c.dcOpen = true
	dc := c.dc
	out := c.outgoing
	c.outgoing = nil
	c.mu.Unlock()
	if len(out) > 0 {
		slog.Debug("rtc: data channel is open, flushing queued messages", "count", len(out))
	}
	for _, m := range out {
		_ = dc.SendText(string(m))
	}
}

// Done returns a channel closed when the connection closes for good: it was
// closed, it failed, or it never finished connecting. A peer connection that is
// only disconnected may still recover, so it does not close it.
func (c *Connection) Done() <-chan struct{} { return c.closed }

// markClosed records the connection as closed and raises the closed event, once.
func (c *Connection) markClosed() {
	c.closeOnce.Do(func() {
		close(c.closed)
		c.mu.Lock()
		hs := append([]func(){}, c.onClosed...)
		c.mu.Unlock()
		c.emit(hs)
	})
}

// closePeer closes the peer connection of generation gen, unless it has since
// been replaced.
func (c *Connection) closePeer(gen int) {
	c.mu.Lock()
	if gen != c.gen {
		c.mu.Unlock()
		return
	}
	pc := c.pc
	c.stopTimersLocked()
	c.stopTracksLocked()
	c.outgoing = nil
	c.pendingApp = nil
	c.dataChannelEnabled = true
	c.mu.Unlock()
	_ = pc.Close()
	c.markClosed()
}

// Disconnect tells the client the bot is leaving, then closes the connection.
func (c *Connection) Disconnect() error {
	_ = c.SendAppMessage(signalingMessage{Type: signalingType, Message: typedSignal{Type: "peerLeft"}})
	return c.Close()
}

// Close tears down the peer connection.
func (c *Connection) Close() error {
	c.mu.Lock()
	pc := c.pc
	c.stopTimersLocked()
	if c.renegotiationTimer != nil {
		c.renegotiationTimer.Stop()
		c.renegotiationTimer = nil
	}
	c.stopTracksLocked()
	c.outgoing = nil
	c.pendingApp = nil
	c.mu.Unlock()
	err := pc.Close()
	c.markClosed()
	return err
}
