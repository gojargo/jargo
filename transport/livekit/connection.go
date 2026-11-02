// Package livekit implements a jargo transport that joins a LiveKit room as a
// participant. It subscribes to a remote participant's Opus audio, decodes it to
// PCM frames for the pipeline, and publishes the pipeline's audio back as an
// Opus track. Room signaling and the access token are handled by the LiveKit
// server SDK; the media path mirrors the Pion transport (transport/rtc).
package livekit

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/livekit/protocol/livekit"
	lksdk "github.com/livekit/server-sdk-go/v2"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// errConnectionClosed is returned when the room disconnects while a caller waits.
var errConnectionClosed = errors.New("livekit: connection closed")

// Config configures a LiveKit room connection. Either Token is an access token
// already minted for the room, or the SDK mints one from the API key and secret
// for RoomName and Identity.
type Config struct {
	// URL is the LiveKit server URL (ws(s)://...). Required.
	URL string `validate:"required"`
	// Token is an access token for the room, which names the room and the
	// participant itself. When set, the fields below are not needed.
	Token string
	// APIKey is the LiveKit API key. Required without Token.
	APIKey string `validate:"required_without=Token"`
	// APISecret is the LiveKit API secret. Required without Token.
	APISecret string `validate:"required_without=Token"`
	// RoomName is the room to join. Required without Token.
	RoomName string `validate:"required_without=Token"`
	// Identity is this participant's identity. Required without Token.
	Identity string `validate:"required_without=Token"`
}

// Validate reports whether the configuration is usable.
func (c Config) Validate() error { return validate.Struct(c) }

// Connection wraps a joined LiveKit room for one session. It owns the published
// audio track and hands on every subscribed remote audio track, with the
// participant publishing it. It is safe for concurrent use.
type Connection struct {
	room       *lksdk.Room
	localTrack *lksdk.LocalTrack

	mu          sync.Mutex
	remoteAudio *webrtc.TrackRemote
	remoteCh    chan *webrtc.TrackRemote

	closeOnce sync.Once
	closed    chan struct{}

	msgMu      sync.Mutex
	msgHandler func(data []byte, sender string)
	pendingIn  []dataMessage

	dtmfMu      sync.Mutex
	dtmfHandler func(frames.KeypadEntry)
	pendingDTMF []frames.KeypadEntry

	partMu      sync.Mutex
	partHandler func(identity string)
	pendingPart []string

	audioMu      sync.Mutex
	audioHandler func(track *webrtc.TrackRemote, participant string)
	pendingAudio []remoteAudio
	unsubHandler func(participant string)
	pendingUnsub []string

	speakerMu      sync.Mutex
	speakerHandler func(identity string)
	// activeSpeaker is the loudest speaker last reported, "" before any.
	activeSpeaker string
	// speakerPending reports that activeSpeaker changed before a handler was
	// registered, so the handler is told it when it is.
	speakerPending bool
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
	cb.OnTrackSubscribed = func(track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, rp *lksdk.RemoteParticipant) {
		if track.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		c.subscribedAudio(track, rp.Identity())
	}
	cb.OnTrackUnsubscribed = func(
		track *webrtc.TrackRemote, _ *lksdk.RemoteTrackPublication, rp *lksdk.RemoteParticipant,
	) {
		if track.Kind() != webrtc.RTPCodecTypeAudio {
			return
		}
		c.deliverAudioUnsubscribed(rp.Identity())
	}
	cb.OnDataPacket = func(data lksdk.DataPacket, params lksdk.DataReceiveParams) {
		switch p := data.(type) {
		case *lksdk.UserDataPacket:
			// A packet sent by a server SDK comes with no sender identity.
			c.deliver(p.Payload, params.SenderIdentity)
		case *livekit.SipDTMF:
			// A key pressed by a caller on a SIP leg of the room.
			c.deliverDTMF(p.Digit)
		}
	}
	cb.OnParticipantConnected = func(p *lksdk.RemoteParticipant) { c.deliverParticipant(p.Identity()) }
	cb.OnActiveSpeakersChanged = func(speakers []lksdk.Participant) {
		identities := make([]string, len(speakers))
		for i, p := range speakers {
			identities[i] = p.Identity()
		}
		c.activeSpeakersChanged(identities)
	}
	cb.OnDisconnected = func() { c.closeOnce.Do(func() { close(c.closed) }) }

	var room *lksdk.Room
	var err error
	if cfg.Token != "" {
		room, err = lksdk.ConnectToRoomWithToken(cfg.URL, cfg.Token, cb)
	} else {
		room, err = lksdk.ConnectToRoom(cfg.URL, lksdk.ConnectInfo{
			APIKey:              cfg.APIKey,
			APISecret:           cfg.APISecret,
			RoomName:            cfg.RoomName,
			ParticipantIdentity: cfg.Identity,
		}, cb)
	}
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

	// Participants already in the room raise the same events as those who join
	// later. The room reports only who joins after the bot does.
	var present []string
	for _, p := range room.GetRemoteParticipants() {
		present = append(present, p.Identity())
	}
	c.reportPresent(present)
	return c, nil
}

// reportPresent reports the participants who were in the room when the bot
// joined, as though each had just joined.
func (c *Connection) reportPresent(identities []string) {
	for _, identity := range identities {
		c.deliverParticipant(identity)
	}
}

// remoteAudio is a subscribed remote audio track and the participant
// publishing it.
type remoteAudio struct {
	track       *webrtc.TrackRemote
	participant string
}

// subscribedAudio records a remote audio track the room subscribed to and
// hands it on.
func (c *Connection) subscribedAudio(track *webrtc.TrackRemote, participant string) {
	c.mu.Lock()
	c.remoteAudio = track
	c.mu.Unlock()
	select {
	case c.remoteCh <- track:
	default:
	}
	c.deliverAudioTrack(track, participant)
}

// OnAudioTrack registers the handler for a remote audio track the room
// subscribed to, called with the track and the identity of the participant
// publishing it. A participant who publishes again (a mute and unmute cycle,
// say) is handed the new track. Tracks subscribed before a handler was set
// are handed to it at once, in order.
func (c *Connection) OnAudioTrack(h func(track *webrtc.TrackRemote, participant string)) {
	c.audioMu.Lock()
	c.audioHandler = h
	pending := c.pendingAudio
	c.pendingAudio = nil
	c.audioMu.Unlock()
	for _, a := range pending {
		h(a.track, a.participant)
	}
}

// deliverAudioTrack hands one track to the handler, holding it when none is
// registered yet.
func (c *Connection) deliverAudioTrack(track *webrtc.TrackRemote, participant string) {
	c.audioMu.Lock()
	h := c.audioHandler
	if h == nil {
		c.pendingAudio = append(c.pendingAudio, remoteAudio{track: track, participant: participant})
		c.audioMu.Unlock()
		return
	}
	c.audioMu.Unlock()
	h(track, participant)
}

// OnAudioTrackUnsubscribed registers the handler for a participant's audio
// track being unsubscribed, called with their identity. Unsubscriptions before
// a handler was set are reported to it at once, in order.
func (c *Connection) OnAudioTrackUnsubscribed(h func(participant string)) {
	c.audioMu.Lock()
	c.unsubHandler = h
	pending := c.pendingUnsub
	c.pendingUnsub = nil
	c.audioMu.Unlock()
	for _, p := range pending {
		h(p)
	}
}

// deliverAudioUnsubscribed reports one unsubscription to the handler, holding
// it when none is registered yet.
func (c *Connection) deliverAudioUnsubscribed(participant string) {
	c.audioMu.Lock()
	h := c.unsubHandler
	if h == nil {
		c.pendingUnsub = append(c.pendingUnsub, participant)
		c.audioMu.Unlock()
		return
	}
	c.audioMu.Unlock()
	h(participant)
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

// OnMessage registers the handler for data messages received in the room. It is
// given each message with the identity of the participant who sent it, empty for
// a message a server SDK sent. Any messages that arrived before a handler was
// set are delivered immediately.
func (c *Connection) OnMessage(h func(data []byte, sender string)) {
	c.msgMu.Lock()
	c.msgHandler = h
	in := c.pendingIn
	c.pendingIn = nil
	c.msgMu.Unlock()
	for _, m := range in {
		h(m.data, m.sender)
	}
}

// dataMessage is a data message held until a handler is set, with its sender.
type dataMessage struct {
	data   []byte
	sender string
}

func (c *Connection) deliver(data []byte, sender string) {
	c.msgMu.Lock()
	h := c.msgHandler
	if h == nil {
		c.pendingIn = append(c.pendingIn, dataMessage{data: data, sender: sender})
		c.msgMu.Unlock()
		return
	}
	c.msgMu.Unlock()
	h(data, sender)
}

// OnDTMF registers the handler for the keys a SIP caller presses. Keys that
// arrived before a handler was set are delivered immediately.
func (c *Connection) OnDTMF(h func(frames.KeypadEntry)) {
	c.dtmfMu.Lock()
	c.dtmfHandler = h
	pending := c.pendingDTMF
	c.pendingDTMF = nil
	c.dtmfMu.Unlock()
	for _, e := range pending {
		h(e)
	}
}

// deliverDTMF hands one keypress to the handler, holding it when none is
// registered yet. A digit that is not a keypad key is dropped: SIP carries the
// key as text and there is nothing to press that is not one of them.
func (c *Connection) deliverDTMF(digit string) {
	entry := frames.KeypadEntry(digit)
	if !entry.Valid() {
		slog.Warn("livekit ignoring an unsupported DTMF key", "digit", digit)
		return
	}
	c.dtmfMu.Lock()
	h := c.dtmfHandler
	if h == nil {
		c.pendingDTMF = append(c.pendingDTMF, entry)
		c.dtmfMu.Unlock()
		return
	}
	c.dtmfMu.Unlock()
	h(entry)
}

// OnParticipantConnected registers the handler for a remote participant joining
// the room. Participants that joined before a handler was set are reported
// immediately, in order.
//
// Holding them matters here because the room is joined before the transport is
// built: someone arriving in that window would otherwise go unreported. Someone
// already in the room when the bot joined is reported the same way, as though
// they had just joined.
func (c *Connection) OnParticipantConnected(h func(identity string)) {
	c.partMu.Lock()
	c.partHandler = h
	pending := c.pendingPart
	c.pendingPart = nil
	c.partMu.Unlock()
	for _, identity := range pending {
		h(identity)
	}
}

// deliverParticipant hands one arrival to the handler, holding it when none is
// registered yet.
func (c *Connection) deliverParticipant(identity string) {
	c.partMu.Lock()
	h := c.partHandler
	if h == nil {
		c.pendingPart = append(c.pendingPart, identity)
		c.partMu.Unlock()
		return
	}
	c.partMu.Unlock()
	h(identity)
}

// OnActiveSpeakerChanged registers the handler for a change of the room's
// loudest speaker, called with their identity. The bot can be the speaker. A
// change that came before a handler was set is reported to it at once, the
// latest only.
func (c *Connection) OnActiveSpeakerChanged(h func(identity string)) {
	c.speakerMu.Lock()
	c.speakerHandler = h
	pending, speaker := c.speakerPending, c.activeSpeaker
	c.speakerPending = false
	c.speakerMu.Unlock()
	if pending {
		h(speaker)
	}
}

// activeSpeakersChanged takes the room's active speakers, loudest first, and
// reports the loudest when it changes. A room that falls quiet keeps the
// current speaker: only a change of speaker is reported.
func (c *Connection) activeSpeakersChanged(identities []string) {
	if len(identities) == 0 {
		return
	}
	speaker := identities[0]
	c.speakerMu.Lock()
	if speaker == c.activeSpeaker {
		c.speakerMu.Unlock()
		return
	}
	c.activeSpeaker = speaker
	h := c.speakerHandler
	if h == nil {
		c.speakerPending = true
	}
	c.speakerMu.Unlock()
	if h != nil {
		h(speaker)
	}
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
