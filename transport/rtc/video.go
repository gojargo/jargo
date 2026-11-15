package rtc

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log/slog"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/videocodec"
	"github.com/gojargo/jargo/transport"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// The media sources a client sends.
const (
	camVideoSource    = transport.CameraVideoSource
	screenVideoSource = transport.ScreenVideoSource
	micAudioSource    = "microphone"
)

// peerID is the participant the video samplers are kept under: the transport
// has one peer.
const peerID = "peer"

// sourcesByReceiver is the source each of the connection's receivers carries.
//
//nolint:gochecknoglobals // a fixed table
var sourcesByReceiver = map[int]string{
	AudioTransceiverIndex:       micAudioSource,
	VideoTransceiverIndex:       camVideoSource,
	ScreenVideoTransceiverIndex: screenVideoSource,
}

// The video clock: RTP timestamps of video run at 90 kHz, and the outgoing
// video is timestamped at 30 frames per second, 3000 ticks a frame.
const (
	videoClockRate     = 90000
	videoFrameRate     = 30
	videoPTime         = time.Second / videoFrameRate
	videoTicksPerFrame = videoClockRate / videoFrameRate
)

// The read loop's pauses: how long Recv waits for a picture, and how long the
// loop sleeps when there is no track or no picture.
const (
	videoRecvTimeout = 2 * time.Second
	videoRetryPause  = 10 * time.Millisecond
)

// errUnsupportedFormat reports a picture in a pixel format with no conversion
// to RGB.
var errUnsupportedFormat = errors.New("rtc: unsupported video format")

// rawVideoTrack is the bot's outgoing video. The transport queues RGB images
// on it; it sends each in turn, on the video clock at 30 frames per second,
// converted to I420 and encoded to VP8.
type rawVideoTrack struct {
	*webrtc.TrackLocalStaticSample
	width, height int
	codecs        VideoCodecConfig

	mu    sync.Mutex
	queue []*frames.OutputImageRawFrame
	ready chan struct{}

	forceKeyframe atomic.Bool
	bitrate       atomic.Int64

	stop     chan struct{}
	stopOnce sync.Once
	done     chan struct{}
}

// newRawVideoTrack builds the outgoing video track for images of the given size
// and starts sending what is queued on it.
func newRawVideoTrack(width, height int, codecs VideoCodecConfig) (*rawVideoTrack, error) {
	local, err := newLocalVideoTrack()
	if err != nil {
		return nil, err
	}
	r := &rawVideoTrack{
		TrackLocalStaticSample: local,
		width:                  width,
		height:                 height,
		codecs:                 codecs,
		ready:                  make(chan struct{}, 1),
		stop:                   make(chan struct{}),
		done:                   make(chan struct{}),
	}
	go r.run()
	return r, nil
}

// addVideoFrame queues an image to be sent.
func (r *rawVideoTrack) addVideoFrame(frame *frames.OutputImageRawFrame) {
	r.mu.Lock()
	r.queue = append(r.queue, frame)
	r.mu.Unlock()
	select {
	case r.ready <- struct{}{}:
	default:
	}
}

// requestKeyframe makes the next frame sent a keyframe.
func (r *rawVideoTrack) requestKeyframe() { r.forceKeyframe.Store(true) }

// setTargetBitrate sets the bitrate the encoder aims for.
func (r *rawVideoTrack) setTargetBitrate(bitrate int) { r.bitrate.Store(int64(bitrate)) }

// stopTrack stops sending and waits for the sender to finish.
func (r *rawVideoTrack) stopTrack() {
	r.stopOnce.Do(func() { close(r.stop) })
	<-r.done
}

// next waits for the next queued image, or returns nil once the track stops.
func (r *rawVideoTrack) next() *frames.OutputImageRawFrame {
	for {
		r.mu.Lock()
		if len(r.queue) > 0 {
			frame := r.queue[0]
			r.queue = r.queue[1:]
			r.mu.Unlock()
			return frame
		}
		r.mu.Unlock()
		select {
		case <-r.stop:
			return nil
		case <-r.ready:
		}
	}
}

// videoClock times the outgoing video: each frame is timestamped one frame
// period after the last, and held until that time comes.
type videoClock struct {
	start     time.Time
	timestamp int64
	started   bool
}

// nextTimestamp returns the next frame's timestamp on the 90 kHz clock, once
// it is due, or false when stop closes first.
func (c *videoClock) nextTimestamp(stop <-chan struct{}) (int64, bool) {
	if !c.started {
		c.started = true
		c.start = time.Now()
		c.timestamp = 0
		return 0, true
	}
	c.timestamp += videoTicksPerFrame
	wait := time.Until(c.start.Add(time.Duration(c.timestamp) * time.Second / videoClockRate))
	if wait > 0 {
		timer := time.NewTimer(wait)
		defer timer.Stop()
		select {
		case <-stop:
			return 0, false
		case <-timer.C:
		}
	}
	return c.timestamp, true
}

// run sends the queued images until the track stops.
func (r *rawVideoTrack) run() {
	defer close(r.done)
	var clock videoClock
	var encoder *videocodec.VP8Encoder
	var failed bool
	defer func() {
		if encoder != nil {
			encoder.Close()
		}
	}()
	for {
		frame := r.next()
		if frame == nil {
			return
		}
		pts, ok := clock.nextTimestamp(r.stop)
		if !ok {
			return
		}
		if failed {
			continue
		}
		if encoder == nil {
			var err error
			if encoder, err = videocodec.NewVP8Encoder(r.codecs.VPXLibPath); err != nil {
				slog.Error("rtc: cannot encode the bot's video", "err", err)
				failed = true
				continue
			}
		}
		if err := r.send(encoder, frame, pts); err != nil {
			slog.Warn("rtc: video frame not sent", "err", err)
		}
	}
}

// send encodes one image and writes it on the track.
func (r *rawVideoTrack) send(encoder *videocodec.VP8Encoder, frame *frames.OutputImageRawFrame, pts int64) error {
	picture, err := videocodec.RGBToI420(frame.Image, r.width, r.height)
	if err != nil {
		return err
	}
	if bitrate := r.bitrate.Load(); bitrate > 0 {
		encoder.SetTargetBitrate(int(bitrate))
	}
	data, err := encoder.Encode(picture, pts, r.forceKeyframe.Swap(false))
	if err != nil {
		return err
	}
	if len(data) == 0 {
		return nil
	}
	return r.WriteSample(media.Sample{Data: data, Duration: videoPTime})
}

// videoInputTrack is a client's video track as the transport reads it: a
// *Track.
type videoInputTrack interface {
	Recv(ctx context.Context) (*VideoFrame, error)
	IsEnabled() bool
}

// inputTrack is t as a videoInputTrack, nil when t is.
func inputTrack(t *Track) videoInputTrack {
	if t == nil {
		return nil
	}
	return t
}

// client is the transport's view of the connection's video: the client's
// camera and screen share tracks while it is connected, and the bot's outgoing
// video track.
type client struct {
	conn   *Connection
	params transport.Params
	// closing is the transport's flag that the bot is disconnecting.
	closing *atomic.Bool

	mu               sync.Mutex
	videoInputTrack  videoInputTrack
	screenVideoTrack videoInputTrack
	videoOutputTrack *rawVideoTrack
}

// track is the input track of a video source.
func (c *client) track(videoSource string) videoInputTrack {
	c.mu.Lock()
	defer c.mu.Unlock()
	if videoSource == camVideoSource {
		return c.videoInputTrack
	}
	return c.screenVideoTrack
}

// clearTrack forgets the input track of a video source, if it is still track.
func (c *client) clearTrack(videoSource string, track videoInputTrack) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if videoSource == camVideoSource && c.videoInputTrack == track {
		c.videoInputTrack = nil
	} else if videoSource != camVideoSource && c.screenVideoTrack == track {
		c.screenVideoTrack = nil
	}
}

// videoSourceEnabled reports whether the peer is sending a video source: its
// track is present and the peer has not turned it off.
func (c *client) videoSourceEnabled(videoSource string) bool {
	track := c.track(videoSource)
	return track != nil && track.IsEnabled()
}

// convertFrame converts a picture to packed RGB.
func convertFrame(frame *VideoFrame) ([]byte, error) {
	switch frame.Format {
	case "rgb24":
		return frame.Planes[0], nil
	case FormatYUV420P, "yuvj420p":
		if len(frame.Planes) != 3 {
			return nil, fmt.Errorf("%w: %s with %d planes", errUnsupportedFormat, frame.Format, len(frame.Planes))
		}
		return videocodec.I420ToRGB(&videocodec.I420Frame{
			Width: frame.Width, Height: frame.Height,
			Y: frame.Planes[0], U: frame.Planes[1], V: frame.Planes[2],
		}), nil
	default:
		return nil, fmt.Errorf("%w: %s", errUnsupportedFormat, frame.Format)
	}
}

// sleepFor waits for d, and reports false if ctx was done first.
func sleepFor(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}

// readVideoFrames yields the pictures of a video source as RGB
// UserImageRawFrames, until ctx is done.
//
// It waits while the source has no track. A track that ended is forgotten, so
// the loop waits for the track a reconnection brings rather than reading the
// dead one again.
func (c *client) readVideoFrames(ctx context.Context, videoSource string) iter.Seq[*frames.UserImageRawFrame] {
	return func(yield func(*frames.UserImageRawFrame) bool) {
		for ctx.Err() == nil {
			track := c.track(videoSource)
			if track == nil {
				if !sleepFor(ctx, videoRetryPause) {
					return
				}
				continue
			}

			frame, err := c.recvVideoFrame(ctx, videoSource, track)
			if err != nil {
				return
			}
			if frame == nil {
				if !sleepFor(ctx, videoRetryPause) {
					return
				}
				continue
			}

			rgb, err := convertFrame(frame)
			if err != nil {
				slog.Error("rtc: cannot convert the video frame", "source", videoSource, "err", err)
				continue
			}
			image := frames.NewUserImageRawFrame(c.conn.PCID(), rgb, frame.Width, frame.Height, "RGB")
			image.SetTransportSource(videoSource)
			image.SetPTS(frame.PTS)
			if !yield(image) {
				return
			}
		}
	}
}

// recvVideoFrame waits for the next picture of a source's track, for at most
// videoRecvTimeout. It returns no picture when none came in that time, or the
// track ended, which forgets the track. It returns an error only once ctx is
// done.
func (c *client) recvVideoFrame(ctx context.Context, videoSource string, track videoInputTrack) (*VideoFrame, error) {
	recvCtx, cancel := context.WithTimeout(ctx, videoRecvTimeout)
	defer cancel()
	frame, err := track.Recv(recvCtx)
	switch {
	case err == nil:
		return frame, nil
	case ctx.Err() != nil:
		return nil, ctx.Err()
	case errors.Is(err, ErrTrackEnded):
		// Every further Recv would fail the same way: the loop waits for the
		// track a reconnection brings instead.
		slog.Warn("rtc: media stream error while reading the video, clearing track", "source", videoSource)
		c.clearTrack(videoSource, track)
	case errors.Is(err, context.DeadlineExceeded):
		if c.conn.IsConnected() && track.IsEnabled() {
			slog.Warn("rtc: timeout, no video frame received within the specified time", "source", videoSource)
		}
	default:
		slog.Warn("rtc: error reading the video", "source", videoSource, "err", err)
	}
	return nil, nil //nolint:nilnil // no picture this time, and the loop goes on
}

// writeVideoFrame queues an image on the outgoing video track. It reports
// whether the image was taken: not while the client is not connected, the bot
// is leaving, or the bot sends no video.
func (c *client) writeVideoFrame(frame *frames.OutputImageRawFrame) bool {
	c.mu.Lock()
	track := c.videoOutputTrack
	c.mu.Unlock()
	if c.canSend() && track != nil {
		track.addVideoFrame(frame)
		return true
	}
	return false
}

// canSend reports whether media can be sent to the client.
func (c *client) canSend() bool {
	return c.conn.IsConnected() && !c.closing.Load()
}

// handleClientConnected picks up the client's video tracks, and starts the
// bot's own video when the transport sends video.
func (c *client) handleClientConnected() {
	videoInputTrack := inputTrack(c.conn.VideoInputTrack())
	screenVideoTrack := inputTrack(c.conn.ScreenVideoInputTrack())
	var output *rawVideoTrack
	if c.params.VideoOutEnabled {
		width, height := c.params.VideoOutSize()
		track, err := newRawVideoTrack(width, height, c.conn.videoCodecs)
		if err != nil {
			slog.Error("rtc: new video output track", "err", err)
		} else {
			output = track
			c.conn.ReplaceVideoTrack(track)
		}
	}
	c.mu.Lock()
	previous := c.videoOutputTrack
	c.videoInputTrack = videoInputTrack
	c.screenVideoTrack = screenVideoTrack
	c.videoOutputTrack = output
	c.mu.Unlock()
	if previous != nil {
		previous.stopTrack()
	}
}

// handlePeerDisconnected forgets the client's tracks and stops the bot's
// video.
func (c *client) handlePeerDisconnected() {
	c.mu.Lock()
	output := c.videoOutputTrack
	c.videoInputTrack = nil
	c.screenVideoTrack = nil
	c.videoOutputTrack = nil
	c.mu.Unlock()
	if output != nil {
		output.stopTrack()
	}
}

// configuredVideoSources is the video sources Params.VideoInSources lists, in
// order.
func configuredVideoSources(params transport.Params) []string {
	sources := make([]string, 0, len(params.VideoInSources))
	for source := range params.VideoInSources {
		sources = append(sources, source)
	}
	sort.Strings(sources)
	return sources
}
