// Package livekit implements a jargo transport that joins a LiveKit room as a
// participant. It subscribes to a remote participant's Opus audio, decodes it to
// PCM frames for the pipeline, and publishes the pipeline's audio back as an
// Opus track. Room signaling and the access token are handled by the LiveKit
// server SDK; the media path mirrors the Pion transport (transport/rtc).
package livekit

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/gojargo/jargo/internal/validate"
	lksdk "github.com/livekit/server-sdk-go/v2"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// errConnectionClosed is returned when the room disconnects while a caller waits.
var errConnectionClosed = errors.New("livekit: connection closed")

// Config configures a LiveKit room connection. The SDK mints the participant
// access token from the API key and secret.
type Config struct {
	// URL is the LiveKit server URL (ws(s)://...). Required.
	URL string `validate:"required"`
	// APIKey is the LiveKit API key. Required.
	APIKey string `validate:"required"`
	// APISecret is the LiveKit API secret. Required.
	APISecret string `validate:"required"`
	// RoomName is the room to join. Required.
	RoomName string `validate:"required"`
	// Identity is this participant's identity. Required.
	Identity string `validate:"required"`
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// Connection wraps a joined LiveKit room for one session. It owns the published
// audio track and exposes the first subscribed remote audio track. It is safe
// for concurrent use.
type Connection struct {
	room       *lksdk.Room
	localTrack *lksdk.LocalTrack

	mu          sync.Mutex
	remoteAudio *webrtc.TrackRemote
	remoteCh    chan *webrtc.TrackRemote

	closeOnce sync.Once
	closed    chan struct{}

	msgMu      sync.Mutex
	msgHandler func([]byte)
	pendingIn  [][]byte
}

// Connect joins the configured room, publishes an Opus audio track, and returns
// a Connection ready to be wrapped with NewTransport.
func Connect(cfg Config) (*Connection, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	c := &Connection{
		remoteCh: make(chan *webrtc.TrackRemote, 1),
		closed:   make(chan struct{}),
	}

	cb := lksdk.NewRoomCallback()
	cb.OnTrackSubscribed = func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, _ *lksdk.RemoteParticipant) {
		if track.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		c.mu.Lock()
		c.remoteAudio = track
		c.mu.Unlock()
		select {
		case c.remoteCh <- track:
		default:
		}
	}
	cb.OnDataPacket = func(data lksdk.DataPacket, _ lksdk.DataReceiveParams) {
		if u, ok := data.(*lksdk.UserDataPacket); ok {
			c.deliver(u.Payload)
		}
	}
	cb.OnDisconnected = func() { c.closeOnce.Do(func() { close(c.closed) }) }

	room, err := lksdk.ConnectToRoom(cfg.URL, lksdk.ConnectInfo{
		APIKey:              cfg.APIKey,
		APISecret:           cfg.APISecret,
		RoomName:            cfg.RoomName,
		ParticipantIdentity: cfg.Identity,
	}, cb)
	if err != nil {
		return nil, err
	}
	c.room = room

	track, err := lksdk.NewLocalSampleTrack(webrtc.RTPCodecCapability{
		MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2,
	})
	if err != nil {
		room.Disconnect()
		return nil, err
	}
	opts := &lksdk.TrackPublicationOptions{Name: "audio", Stereo: true}
	if _, err := room.LocalParticipant.PublishTrack(track, opts); err != nil {
		room.Disconnect()
		return nil, err
	}
	c.localTrack = track
	return c, nil
}

// RemoteAudioTrack returns the first subscribed remote audio track, blocking
// until it arrives, ctx is done, or the connection closes.
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

// WriteAudio sends one Opus packet on the published track. dur is the packet's
// playback duration, used to advance RTP timestamps.
func (c *Connection) WriteAudio(packet []byte, dur time.Duration) error {
	return c.localTrack.WriteSample(media.Sample{Data: packet, Duration: dur}, nil)
}

// OnMessage registers the handler for data messages received in the room. Any
// messages that arrived before a handler was set are delivered immediately.
func (c *Connection) OnMessage(h func([]byte)) {
	c.msgMu.Lock()
	c.msgHandler = h
	in := c.pendingIn
	c.pendingIn = nil
	c.msgMu.Unlock()
	for _, m := range in {
		h(m)
	}
}

func (c *Connection) deliver(data []byte) {
	c.msgMu.Lock()
	h := c.msgHandler
	if h == nil {
		c.pendingIn = append(c.pendingIn, data)
		c.msgMu.Unlock()
		return
	}
	c.msgMu.Unlock()
	h(data)
}

// SendMessage publishes an application message to the room.
func (c *Connection) SendMessage(data []byte) error {
	return c.room.LocalParticipant.PublishDataPacket(&lksdk.UserDataPacket{Payload: data})
}

// Done returns a channel closed when the room disconnects or Close is called.
func (c *Connection) Done() <-chan struct{} { return c.closed }

// Close leaves the room.
func (c *Connection) Close() error {
	c.closeOnce.Do(func() { close(c.closed) })
	c.room.Disconnect()
	return nil
}
