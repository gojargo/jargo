package transport

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/imageenc"
)

// videoResetFrames is how many frames of drift live video tolerates before its
// pacing starts over from the image in hand.
const videoResetFrames = 5

// videoOut is the video half of a sender: the images it repeats, or the live
// images it paces out, and the loop that sends them at the video output
// framerate.
type videoOut struct {
	mu sync.Mutex
	// images are the images repeated while no others arrive: one image, or the
	// frames of a sprite, cycled in turn.
	images []*frames.OutputImageRawFrame
	next   int
	// queue holds the live images still to be sent, and ready signals that one
	// was added.
	queue []*frames.OutputImageRawFrame
	ready chan struct{}

	// lifeMu guards the loop's lifecycle.
	lifeMu sync.Mutex
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// handleImageFrame takes an image or a sprite bound for the outgoing video.
//
// An image synced with the audio waits in the audio queue, so it is only shown
// once the audio ahead of it has been sent. Otherwise live video queues the
// image to be sent as its next frame, and video that is not live shows it (or
// cycles the sprite's images) from now on, until another arrives.
func (s *mediaSender) handleImageFrame(f frames.Frame) {
	if !s.out.params.VideoOutEnabled {
		return
	}
	image, isImage := f.(frames.OutputImageFrame)
	switch {
	case isImage && image.OutputImage().SyncWithAudio:
		s.bufMu.Lock()
		out := s.audioOut
		s.bufMu.Unlock()
		if out != nil {
			out.push(f)
			return
		}
		s.setVideoImage(image.OutputImage())
	case isImage && s.out.params.VideoOutIsLive:
		s.queueLiveImage(image.OutputImage())
	case isImage:
		s.setVideoImage(image.OutputImage())
	default:
		if sprite, ok := f.(*frames.SpriteFrame); ok {
			s.setVideoImages(sprite.Images)
		}
	}
}

// showQueuedImage shows an image or a sprite taken from the audio queue, from
// now on.
func (s *mediaSender) showQueuedImage(f frames.Frame) {
	switch fr := f.(type) {
	case frames.OutputImageFrame:
		s.setVideoImage(fr.OutputImage())
	case *frames.SpriteFrame:
		s.setVideoImages(fr.Images)
	}
}

// setVideoImage repeats one image as the outgoing video.
func (s *mediaSender) setVideoImage(image *frames.OutputImageRawFrame) {
	s.setVideoImages([]*frames.OutputImageRawFrame{image})
}

// setVideoImages cycles a run of images as the outgoing video.
func (s *mediaSender) setVideoImages(images []*frames.OutputImageRawFrame) {
	s.video.mu.Lock()
	s.video.images = images
	s.video.next = 0
	s.video.mu.Unlock()
}

// nextVideoImage is the next image of the cycle, or nil when there is none.
func (s *mediaSender) nextVideoImage() *frames.OutputImageRawFrame {
	s.video.mu.Lock()
	defer s.video.mu.Unlock()
	if len(s.video.images) == 0 {
		return nil
	}
	image := s.video.images[s.video.next%len(s.video.images)]
	s.video.next = (s.video.next + 1) % len(s.video.images)
	return image
}

// queueLiveImage queues an image to be sent as a frame of live video.
func (s *mediaSender) queueLiveImage(image *frames.OutputImageRawFrame) {
	s.video.mu.Lock()
	s.video.queue = append(s.video.queue, image)
	ready := s.video.ready
	s.video.mu.Unlock()
	if ready != nil {
		select {
		case ready <- struct{}{}:
		default:
		}
	}
}

// takeLiveImage waits for the next live image, or returns nil once ctx is done.
func (s *mediaSender) takeLiveImage(ctx context.Context, ready chan struct{}) *frames.OutputImageRawFrame {
	for {
		s.video.mu.Lock()
		if len(s.video.queue) > 0 {
			image := s.video.queue[0]
			s.video.queue = s.video.queue[1:]
			s.video.mu.Unlock()
			return image
		}
		s.video.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil
		case <-ready:
		}
	}
}

// createVideoTask starts the video loop, when the transport sends video and it
// is not running already. Live images queued before it are dropped: the loop
// starts with an empty queue.
func (s *mediaSender) createVideoTask(ctx context.Context) {
	if !s.out.params.VideoOutEnabled {
		return
	}
	s.video.lifeMu.Lock()
	defer s.video.lifeMu.Unlock()
	if s.video.cancel != nil {
		return
	}
	ready := make(chan struct{}, 1)
	s.video.mu.Lock()
	s.video.queue = nil
	s.video.ready = ready
	s.video.mu.Unlock()

	videoCtx, cancel := context.WithCancel(ctx)
	s.video.cancel = cancel
	s.video.wg.Add(1)
	go s.videoLoop(videoCtx, ready)
}

// cancelVideoTask stops the video loop and waits for it.
func (s *mediaSender) cancelVideoTask() {
	s.video.lifeMu.Lock()
	defer s.video.lifeMu.Unlock()
	if s.video.cancel == nil {
		return
	}
	s.video.cancel()
	s.video.cancel = nil
	s.video.wg.Wait()
}

// videoLoop sends the outgoing video at the video output framerate: each live
// image in turn, or the images being cycled, one per frame.
func (s *mediaSender) videoLoop(ctx context.Context, ready chan struct{}) {
	defer s.video.wg.Done()
	duration := time.Second / time.Duration(s.out.params.VideoOutRate())
	live := videoPacing{duration: duration, reset: duration * videoResetFrames}
	for ctx.Err() == nil {
		if s.out.params.VideoOutIsLive {
			image := s.takeLiveImage(ctx, ready)
			if image == nil {
				return
			}
			if !live.wait(ctx) {
				return
			}
			s.drawImage(ctx, image)
			continue
		}
		if image := s.nextVideoImage(); image != nil {
			s.drawImage(ctx, image)
		}
		if !sleepCtx(ctx, duration) {
			return
		}
	}
}

// videoPacing keeps live video at its framerate, measured from the first image.
type videoPacing struct {
	duration, reset time.Duration
	start           time.Time
	index           int
}

// wait holds the next live image until it is due. Drift of more than a few
// frames either way starts the pacing over from now, rather than rushing to
// catch up or holding images back. It reports false once ctx is done.
func (p *videoPacing) wait(ctx context.Context) bool {
	// The start time is taken as soon as the first image is in hand.
	if p.start.IsZero() {
		p.start = time.Now()
		p.index = 0
	}
	realElapsed := time.Since(p.start)
	realRender := time.Duration(p.index) * p.duration
	delay := p.duration + realRender - realElapsed
	switch {
	case delay > p.reset || delay < -p.reset:
		p.start = time.Now()
		p.index = 0
	case delay > 0:
		if !sleepCtx(ctx, delay) {
			return false
		}
		p.index++
	}
	return true
}

// drawImage sends one image as a frame of the outgoing video, resized to the
// video output size when it is not that size already. An image whose format is
// not one of the raw pixel formats cannot be resized, and is sent as it is.
func (s *mediaSender) drawImage(ctx context.Context, image *frames.OutputImageRawFrame) {
	frame := image
	width, height := s.out.params.VideoOutSize()
	if image.Format != "" && (image.Width != width || image.Height != height) {
		resized, err := imageenc.Resize(image.Image, image.Width, image.Height, image.Format, width, height)
		if err != nil {
			slog.Warn("sending an image the video output cannot resize as it is",
				"processor", s.out.Name(), "frame", image.String(), "err", err)
		} else {
			frame = frames.NewOutputImageRawFrame(resized, width, height, image.Format)
			frame.SetTransportDestination(s.destination)
		}
	}
	if _, err := s.out.self.WriteVideoFrame(ctx, frame); err != nil && !canceled(ctx) {
		slog.Error("write video to transport",
			"processor", s.out.Name(), "destination", s.destination, "err", err)
	}
}

// sleepCtx waits for d, and reports false if ctx was done first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
