package rtc

import (
	"context"
	"errors"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/internal/videocodec"
	"github.com/pion/rtp"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media/samplebuilder"
)

// ErrTrackEnded is returned by Track.Recv once the track has ended: the client
// stopped sending it, or the connection stopped it.
var ErrTrackEnded = errors.New("rtc: track ended")

// trackIdleTimeout is how long a Track goes without Recv being called before
// it stops being consumed.
const trackIdleTimeout = 2 * time.Second

// Pixel formats of a VideoFrame.
const (
	// FormatYUV420P is planar YUV 4:2:0: a luma plane, then two chroma planes
	// at half the width and height, rounded up.
	FormatYUV420P = "yuv420p"
)

// VideoFrame is a picture decoded from one of the client's video tracks.
type VideoFrame struct {
	// Width and Height are the picture's size, in pixels.
	Width, Height int
	// Format is the pixel format of Planes, such as FormatYUV420P.
	Format string
	// Planes hold the picture, each plane's rows packed with no padding.
	Planes [][]byte
	// PTS is the picture's RTP timestamp, on the 90 kHz video clock.
	PTS int64
}

// videoDecoder decodes the frames of one codec.
type videoDecoder interface {
	Decode(data []byte) ([]*videocodec.I420Frame, error)
	Close()
}

// Track is one of the client's incoming video tracks. Its packets are
// assembled into frames, decoded, and queued for Recv.
//
// It is not consumed until Recv is first called: until then its packets are
// dropped, so pictures nobody reads do not pile up. It stops being consumed
// again, its queued pictures dropped, when Recv has not been called for two
// seconds.
//
// The client can turn the track off, which SetEnabled records; Recv then
// returns no picture.
type Track struct {
	conn     *Connection
	gen      int
	receiver *webrtc.RTPReceiver
	kind     webrtc.RTPCodecType
	// capacity is how many packets the receiver holds while assembling a
	// frame.
	capacity    uint16
	idleTimeout time.Duration

	stop     chan struct{}
	stopOnce sync.Once

	mu sync.Mutex
	// enabled is whether the client is sending the track, as it last said.
	enabled bool
	// receiving is whether the track is being consumed.
	receiving   bool
	lastRecv    time.Time
	idleRunning bool
	queue       []*VideoFrame
	// ready is closed, then replaced, when a picture is queued or the track
	// ends.
	ready chan struct{}
	ended bool
}

// newTrack builds the Track of receiver, on the peer connection of generation
// gen, and starts reading it.
func newTrack(
	conn *Connection, gen int, receiver *webrtc.RTPReceiver, kind webrtc.RTPCodecType, capacity uint16,
) *Track {
	t := &Track{
		conn:        conn,
		gen:         gen,
		receiver:    receiver,
		kind:        kind,
		capacity:    capacity,
		idleTimeout: trackIdleTimeout,
		stop:        make(chan struct{}),
		enabled:     true,
		ready:       make(chan struct{}),
	}
	go t.read()
	return t
}

// Kind is the track's kind, audio or video.
func (t *Track) Kind() webrtc.RTPCodecType { return t.kind }

// SetEnabled records whether the client is sending the track.
func (t *Track) SetEnabled(enabled bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.enabled = enabled
}

// IsEnabled reports whether the client is sending the track, as it last said.
func (t *Track) IsEnabled() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.enabled
}

// DiscardOldFrames drops the pictures queued and not read yet, so the next
// Recv returns a fresh one.
func (t *Track) DiscardOldFrames() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.queue) > 0 {
		slog.Debug("rtc: discarding old frames", "frames", len(t.queue))
	}
	t.queue = nil
}

// Recv returns the next picture, waiting for one until ctx is done. It starts
// the track being consumed. A video track the client turned off returns no
// picture, and no error, at once. Once the track has ended it returns
// ErrTrackEnded.
func (t *Track) Recv(ctx context.Context) (*VideoFrame, error) {
	t.mu.Lock()
	t.receiving = true
	t.lastRecv = time.Now()
	if !t.idleRunning {
		t.idleRunning = true
		go t.idleWatcher()
	}
	if !t.enabled && t.kind == webrtc.RTPCodecTypeVideo {
		t.mu.Unlock()
		return nil, nil //nolint:nilnil // a track turned off has no picture, and that is no error
	}
	for {
		if len(t.queue) > 0 {
			frame := t.queue[0]
			t.queue = t.queue[1:]
			t.mu.Unlock()
			return frame, nil
		}
		if t.ended {
			t.mu.Unlock()
			return nil, ErrTrackEnded
		}
		ready := t.ready
		t.mu.Unlock()
		select {
		case <-ready:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		t.mu.Lock()
	}
}

// idleWatcher stops consuming the track once Recv has not been called for the
// idle timeout, dropping the pictures queued so they do not pile up.
func (t *Track) idleWatcher() {
	for {
		timer := time.NewTimer(t.idleTimeout)
		select {
		case <-t.stop:
			timer.Stop()
			return
		case <-timer.C:
		}
		t.mu.Lock()
		if !t.receiving {
			t.idleRunning = false
			t.mu.Unlock()
			return
		}
		if idle := time.Since(t.lastRecv); idle >= t.idleTimeout {
			slog.Debug("rtc: disabling receiver for idle track", "kind", t.kind.String(), "idle", idle)
			t.queue = nil
			t.receiving = false
			t.idleRunning = false
			t.mu.Unlock()
			return
		}
		t.mu.Unlock()
	}
}

// Stop stops the track: it is no longer read, and Recv returns ErrTrackEnded.
func (t *Track) Stop() {
	t.stopOnce.Do(func() { close(t.stop) })
	t.end()
}

// end marks the track ended and wakes Recv.
func (t *Track) end() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.ended {
		return
	}
	t.ended = true
	t.receiving = false
	close(t.ready)
}

// push queues a picture for Recv.
func (t *Track) push(frame *VideoFrame) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.ended || !t.receiving {
		return
	}
	t.queue = append(t.queue, frame)
	close(t.ready)
	t.ready = make(chan struct{})
}

// isReceiving reports whether the track is being consumed.
func (t *Track) isReceiving() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.receiving
}

// read reads the track's packets until it ends: it waits for the client's
// track to arrive, then assembles the packets into frames, decodes them and
// queues the pictures. Packets that arrive while the track is not being
// consumed are dropped.
func (t *Track) read() {
	defer t.end()
	if t.kind != webrtc.RTPCodecTypeVideo {
		return
	}
	remote := t.conn.remoteVideoTrack(t.gen, t.receiver, t.stop)
	if remote == nil {
		return
	}
	r := frameReader{track: t, ssrc: remote.SSRC()}
	defer r.close()
	for {
		select {
		case <-t.stop:
			return
		default:
		}
		_ = remote.SetReadDeadline(time.Now().Add(readDeadline))
		pkt, _, err := remote.ReadRTP()
		if err != nil {
			if errors.Is(err, os.ErrDeadlineExceeded) {
				continue
			}
			slog.Debug("rtc: video track ended", "err", err)
			return
		}
		if !t.isReceiving() {
			// Not consumed: the gap this leaves is a reason to ask for a
			// keyframe once the track is consumed again.
			continue
		}
		r.handle(pkt, remote.Codec().MimeType)
	}
}

// frameReader assembles one track's packets into frames and decodes them.
type frameReader struct {
	track *Track
	ssrc  webrtc.SSRC

	mimeType string
	builder  *samplebuilder.SampleBuilder
	decoder  videoDecoder
	// failed is set once the decoder could not be created, so the failure is
	// reported once.
	failed bool

	lastSeq uint16
	hasLast bool
}

// close frees the decoder.
func (r *frameReader) close() {
	if r.decoder != nil {
		r.decoder.Close()
		r.decoder = nil
	}
}

// newBuilder starts assembling frames afresh.
func (r *frameReader) newBuilder() {
	var depacketizer rtp.Depacketizer
	if strings.EqualFold(r.mimeType, webrtc.MimeTypeH264) {
		depacketizer = &codecs.H264Packet{}
	} else {
		depacketizer = &codecs.VP8Packet{}
	}
	r.builder = samplebuilder.New(r.track.capacity, depacketizer, videoClockRate)
}

// setCodec switches to the codec the track now sends.
func (r *frameReader) setCodec(mimeType string) {
	r.close()
	r.mimeType = mimeType
	r.hasLast = false
	r.failed = false
	r.newBuilder()
	codecs := r.track.conn.videoCodecs
	var err error
	switch {
	case strings.EqualFold(mimeType, webrtc.MimeTypeVP8):
		r.decoder, err = videocodec.NewVP8Decoder(codecs.VPXLibPath)
	case strings.EqualFold(mimeType, webrtc.MimeTypeH264):
		r.decoder, err = videocodec.NewH264Decoder(codecs.OpenH264LibPath)
	default:
		slog.Warn("rtc: no decoder for the video codec", "codec", mimeType)
		r.failed = true
		return
	}
	if err != nil {
		slog.Error("rtc: cannot decode the client's video", "codec", mimeType, "err", err)
		r.decoder = nil
		r.failed = true
	}
}

// handle takes one packet: it is added to the frame being assembled, and the
// frames it completes are decoded and their pictures queued.
func (r *frameReader) handle(pkt *rtp.Packet, mimeType string) {
	if r.builder == nil || !strings.EqualFold(mimeType, r.mimeType) {
		r.setCodec(mimeType)
	}
	if r.failed {
		return
	}
	// A run of packets longer than the receiver holds, lost or dropped while
	// the track was not consumed, empties it, and asks the client for a
	// keyframe to decode from.
	if r.hasLast {
		if ahead := pkt.SequenceNumber - r.lastSeq; ahead >= r.track.capacity && ahead < 1<<15 {
			r.newBuilder()
			r.track.conn.sendPictureLossIndication(r.track.gen, r.ssrc)
		}
	}
	r.lastSeq, r.hasLast = pkt.SequenceNumber, true

	r.builder.Push(pkt)
	for sample := r.builder.Pop(); sample != nil; sample = r.builder.Pop() {
		if sample.PrevDroppedPackets > 0 {
			// A frame was given up on: the next ones cannot be decoded
			// without a keyframe.
			r.track.conn.sendPictureLossIndication(r.track.gen, r.ssrc)
		}
		pictures, err := r.decoder.Decode(sample.Data)
		if err != nil {
			slog.Debug("rtc: video frame not decoded", "codec", r.mimeType, "err", err)
			continue
		}
		for _, p := range pictures {
			r.track.push(&VideoFrame{
				Width:  p.Width,
				Height: p.Height,
				Format: FormatYUV420P,
				Planes: [][]byte{p.Y, p.U, p.V},
				PTS:    int64(sample.PacketTimestamp),
			})
		}
	}
}
