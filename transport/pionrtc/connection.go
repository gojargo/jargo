// Package pionrtc implements a WebRTC transport for jargo using Pion. It
// terminates a peer connection, decodes received Opus audio into PCM frames,
// and encodes outgoing PCM frames into Opus sent over an RTP track.
//
// Signaling is a single SDP offer/answer exchange (non-trickle ICE) that the
// application drives over HTTP.
package pionrtc

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// errConnectionClosed is returned when the peer connection closes while a
// caller is waiting on it.
var errConnectionClosed = errors.New("pionrtc: connection closed")

// Connection wraps a Pion peer connection for one client session. It owns the
// outgoing audio track and exposes the incoming one. It is safe for concurrent
// use.
type Connection struct {
	pc         *webrtc.PeerConnection
	localAudio *webrtc.TrackLocalStaticSample

	mu          sync.Mutex
	remoteAudio *webrtc.TrackRemote
	remoteCh    chan *webrtc.TrackRemote

	connectedOnce sync.Once
	connected     chan struct{}
	closeOnce     sync.Once
	closed        chan struct{}

	// Data channel state. The client creates the channel; messages are JSON.
	dcMu       sync.Mutex
	dc         *webrtc.DataChannel
	dcOpen     bool
	msgHandler func([]byte)
	pendingIn  [][]byte
	pendingOut [][]byte
}

// Option configures a Connection.
type Option func(*webrtc.Configuration)

// WithICEServers sets the ICE servers used for connectivity. Passing none (or
// an empty list) disables STUN, leaving only host candidates — useful for
// same-host connections and tests.
func WithICEServers(servers ...webrtc.ICEServer) Option {
	return func(c *webrtc.Configuration) { c.ICEServers = servers }
}

// NewConnection creates a peer connection with an outgoing Opus audio track. By
// default a public STUN server is configured so it works across NATs; override
// with WithICEServers.
func NewConnection(opts ...Option) (*Connection, error) {
	cfg := webrtc.Configuration{
		ICEServers: []webrtc.ICEServer{{URLs: []string{"stun:stun.l.google.com:19302"}}},
	}
	for _, opt := range opts {
		opt(&cfg)
	}

	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		return nil, fmt.Errorf("register codecs: %w", err)
	}
	api := webrtc.NewAPI(webrtc.WithMediaEngine(m))

	pc, err := api.NewPeerConnection(cfg)
	if err != nil {
		return nil, fmt.Errorf("new peer connection: %w", err)
	}

	local, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2},
		"audio", "jargo",
	)
	if err != nil {
		_ = pc.Close()
		return nil, fmt.Errorf("new local audio track: %w", err)
	}
	if _, err := pc.AddTrack(local); err != nil {
		_ = pc.Close()
		return nil, fmt.Errorf("add local audio track: %w", err)
	}

	c := &Connection{
		pc:         pc,
		localAudio: local,
		remoteCh:   make(chan *webrtc.TrackRemote, 1),
		connected:  make(chan struct{}),
		closed:     make(chan struct{}),
	}

	pc.OnTrack(func(tr *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if tr.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		c.mu.Lock()
		c.remoteAudio = tr
		c.mu.Unlock()
		select {
		case c.remoteCh <- tr:
		default:
		}
	})

	pc.OnDataChannel(func(dc *webrtc.DataChannel) {
		c.dcMu.Lock()
		c.dc = dc
		c.dcMu.Unlock()

		dc.OnOpen(func() {
			c.dcMu.Lock()
			c.dcOpen = true
			out := c.pendingOut
			c.pendingOut = nil
			c.dcMu.Unlock()
			for _, m := range out {
				_ = dc.SendText(string(m))
			}
		})

		dc.OnMessage(func(msg webrtc.DataChannelMessage) {
			c.dcMu.Lock()
			h := c.msgHandler
			if h == nil {
				c.pendingIn = append(c.pendingIn, msg.Data)
				c.dcMu.Unlock()
				return
			}
			c.dcMu.Unlock()
			h(msg.Data)
		})
	})

	pc.OnConnectionStateChange(func(s webrtc.PeerConnectionState) {
		switch s {
		case webrtc.PeerConnectionStateConnected:
			c.connectedOnce.Do(func() { close(c.connected) })
		case webrtc.PeerConnectionStateFailed,
			webrtc.PeerConnectionStateDisconnected,
			webrtc.PeerConnectionStateClosed:
			c.closeOnce.Do(func() { close(c.closed) })
		default:
			// New / Connecting / Unknown need no action.
		}
	})

	return c, nil
}

// Answer completes the signaling handshake: it applies the remote offer, builds
// an answer, and waits for ICE gathering to finish so the returned answer
// carries all candidates.
func (c *Connection) Answer(offer webrtc.SessionDescription) (webrtc.SessionDescription, error) {
	if err := c.pc.SetRemoteDescription(offer); err != nil {
		return webrtc.SessionDescription{}, fmt.Errorf("set remote description: %w", err)
	}
	answer, err := c.pc.CreateAnswer(nil)
	if err != nil {
		return webrtc.SessionDescription{}, fmt.Errorf("create answer: %w", err)
	}
	gatherComplete := webrtc.GatheringCompletePromise(c.pc)
	if err := c.pc.SetLocalDescription(answer); err != nil {
		return webrtc.SessionDescription{}, fmt.Errorf("set local description: %w", err)
	}
	<-gatherComplete
	return *c.pc.LocalDescription(), nil
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
	c.mu.Lock()
	tr := c.remoteAudio
	c.mu.Unlock()
	if tr != nil {
		return tr, nil
	}
	select {
	case tr := <-c.remoteCh:
		return tr, nil
	case <-c.closed:
		return nil, errConnectionClosed
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// WriteAudio sends one Opus packet on the outgoing track. dur is the packet's
// playback duration, used to advance RTP timestamps.
func (c *Connection) WriteAudio(packet []byte, dur time.Duration) error {
	return c.localAudio.WriteSample(media.Sample{Data: packet, Duration: dur})
}

// OnMessage registers the handler for application messages received on the data
// channel. Any messages that arrived before a handler was set are delivered
// immediately, in order.
func (c *Connection) OnMessage(h func([]byte)) {
	c.dcMu.Lock()
	c.msgHandler = h
	in := c.pendingIn
	c.pendingIn = nil
	c.dcMu.Unlock()
	for _, m := range in {
		h(m)
	}
}

// SendMessage sends an application message to the client over the data channel.
// If the channel is not open yet the message is queued and flushed on open.
func (c *Connection) SendMessage(data []byte) error {
	c.dcMu.Lock()
	if c.dc == nil || !c.dcOpen {
		c.pendingOut = append(c.pendingOut, data)
		c.dcMu.Unlock()
		return nil
	}
	dc := c.dc
	c.dcMu.Unlock()
	return dc.SendText(string(data))
}

// Done returns a channel closed when the connection fails, disconnects, or is
// closed.
func (c *Connection) Done() <-chan struct{} { return c.closed }

// Close tears down the peer connection.
func (c *Connection) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	return c.pc.Close()
}
