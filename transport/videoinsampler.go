package transport

import (
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
)

// JitterTolerance is how early a frame may arrive and still count as due, at
// most half an interval. It covers jitter in the incoming stream, which is about
// the same at any framerate.
const JitterTolerance = 50 * time.Millisecond

// EveryFrame is the framerate that passes on every frame a video source sends.
const EveryFrame = -1

// CaptureFramerate is the sampler framerate for a CaptureParticipantVideo call's
// arguments. A sampler passes on only the frames that answer image requests at
// a framerate of 0, which is how onRequestOnly is applied.
func CaptureFramerate(framerate int, onRequestOnly bool) int {
	if onRequestOnly {
		return 0
	}
	return framerate
}

// VideoInSampler decides which frames of one incoming video source a transport
// passes on.
//
// A user's camera or screen share sends frames at its own rate, often 15 to 30
// per second, while a bot usually wants fewer: a vision bot may want one a
// second, or only a frame when it asks for one. An input transport keeps one
// sampler per video source and calls Sample for every frame that arrives; the
// sampler says whether to pass that frame on or drop it.
//
// The framerate decides which frames are passed on:
//
//   - EveryFrame: every frame.
//   - 0: only frames that answer an image request.
//   - N > 0: N frames per second, on the schedule described below.
//
// Whatever the framerate, an image request queued with AddRequest is answered
// by the next frame that arrives, which is passed on with the request.
//
// For N > 0, frames are due on a fixed schedule, one interval (1/N seconds)
// apart:
//
//   - On schedule. A frame is due once its scheduled time comes, or up to a
//     tolerance before it, since frames never arrive exactly on time. The next
//     frame is scheduled one interval after this frame's scheduled time, not its
//     arrival time, so the rate does not drift. A frame arriving late moves the
//     schedule to its arrival, so a stream slower than the framerate passes
//     every frame rather than falling further and further behind.
//   - Fresh start. The first frame, a frame that answers a request, and the
//     first frame after a stall (more than an interval past its scheduled time)
//     start a new schedule from their arrival. After a stall this passes on one
//     frame instead of a burst catching up on the missed ones.
//
// The tolerance is the smaller of JitterTolerance and half an interval. At low
// framerates the next frame after any frame is therefore close to a full
// interval later; at framerates near the stream's own rate it covers the
// stream's jitter, so every frame passes.
//
// A sampler is not safe for concurrent use; VideoInSamplers is.
type VideoInSampler struct {
	framerate int
	// nextTime is when the next frame is due. Zero before the first frame,
	// which then counts as arriving after a stall and starts the schedule.
	nextTime time.Time
	requests []*frames.UserImageRequestFrame
	// now reads the clock; tests replace it.
	now func() time.Time
}

// NewVideoInSampler builds a sampler passing on framerate frames per second.
// 0 passes on frames only to answer requests, and EveryFrame passes on every
// frame.
func NewVideoInSampler(framerate int) *VideoInSampler {
	return &VideoInSampler{framerate: framerate, now: time.Now}
}

// Framerate is the frames per second the sampler passes on.
func (s *VideoInSampler) Framerate() int { return s.framerate }

// SetFramerate changes the frames per second the sampler passes on.
func (s *VideoInSampler) SetFramerate(framerate int) { s.framerate = framerate }

// AddRequest queues an image request for the next frame to answer.
func (s *VideoInSampler) AddRequest(request *frames.UserImageRequestFrame) {
	s.requests = append(s.requests, request)
}

// TakeRequests removes and returns the image requests waiting for a frame,
// oldest first.
func (s *VideoInSampler) TakeRequests() []*frames.UserImageRequestFrame {
	requests := s.requests
	s.requests = nil
	return requests
}

// Sample decides whether to pass on the frame that just arrived. It reports
// whether to pass the frame on, and the image request the frame answers, or
// nil.
func (s *VideoInSampler) Sample() (bool, *frames.UserImageRequestFrame) {
	// A waiting request is always answered by this frame.
	var request *frames.UserImageRequestFrame
	if len(s.requests) > 0 {
		request = s.requests[0]
		s.requests = s.requests[1:]
	}

	if s.framerate < 0 {
		return true, request
	}
	if s.framerate == 0 {
		return request != nil, request
	}

	now := s.now()
	interval := time.Second / time.Duration(s.framerate)
	tolerance := min(JitterTolerance, interval/2)

	if request == nil && now.Before(s.nextTime.Add(-tolerance)) {
		// Too early: the next frame is not due yet.
		return false, nil
	}

	if request != nil || now.Sub(s.nextTime) > interval {
		// Fresh start (a request, the first frame, or a stall): the next frame
		// is due one interval from now.
		s.nextTime = now.Add(interval)
	} else {
		// On schedule: one interval after this frame's scheduled time, or after
		// its arrival if it came late.
		scheduled := s.nextTime
		if arrived := now.Add(-tolerance); arrived.After(scheduled) {
			scheduled = arrived
		}
		s.nextTime = scheduled.Add(interval)
	}
	return true, request
}

// videoInKey names one video source of one participant.
type videoInKey struct {
	participantID string
	videoSource   string
}

// VideoInSamplers are an input transport's video samplers, by participant and
// video source. They are safe for concurrent use.
type VideoInSamplers struct {
	mu       sync.Mutex
	samplers map[videoInKey]*VideoInSampler
	// order keeps the samplers in the order they were added, so requests are
	// taken back in a stable order.
	order []videoInKey
}

// NewVideoInSamplers builds a set with no sources being sampled.
func NewVideoInSamplers() *VideoInSamplers {
	return &VideoInSamplers{samplers: map[videoInKey]*VideoInSampler{}}
}

// Capture starts sampling a source, or changes its framerate. Changing the
// framerate keeps the source's sampler, so image requests already waiting on it
// are still answered.
func (v *VideoInSamplers) Capture(participantID, videoSource string, framerate int) {
	v.mu.Lock()
	defer v.mu.Unlock()
	key := videoInKey{participantID, videoSource}
	if s, ok := v.samplers[key]; ok {
		s.SetFramerate(framerate)
		return
	}
	v.samplers[key] = NewVideoInSampler(framerate)
	v.order = append(v.order, key)
}

// IsCapturing reports whether Capture was called for a source.
func (v *VideoInSamplers) IsCapturing(participantID, videoSource string) bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	_, ok := v.samplers[videoInKey{participantID, videoSource}]
	return ok
}

// AddRequest queues an image request for the next frame of a source to answer.
// It reports whether the request was queued: false when the source is not
// being sampled, so nothing would answer it.
func (v *VideoInSamplers) AddRequest(
	participantID, videoSource string, request *frames.UserImageRequestFrame,
) bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	s, ok := v.samplers[videoInKey{participantID, videoSource}]
	if ok {
		s.AddRequest(request)
	}
	return ok
}

// Sample decides whether to pass on the frame that just arrived from a source,
// as VideoInSampler.Sample does. A frame from a source that is not being
// sampled is never due.
func (v *VideoInSamplers) Sample(participantID, videoSource string) (bool, *frames.UserImageRequestFrame) {
	v.mu.Lock()
	defer v.mu.Unlock()
	s, ok := v.samplers[videoInKey{participantID, videoSource}]
	if !ok {
		return false, nil
	}
	return s.Sample()
}

// TakeRequests removes and returns the image requests waiting for a frame from
// a participant's source, keeping the samplers. An empty videoSource takes the
// requests waiting on every source of the participant.
func (v *VideoInSamplers) TakeRequests(participantID, videoSource string) []*frames.UserImageRequestFrame {
	v.mu.Lock()
	defer v.mu.Unlock()
	return v.takeRequests(participantID, videoSource)
}

// takeRequests is TakeRequests with the lock held.
func (v *VideoInSamplers) takeRequests(participantID, videoSource string) []*frames.UserImageRequestFrame {
	var requests []*frames.UserImageRequestFrame
	for _, key := range v.order {
		if key.participantID == participantID && (videoSource == "" || videoSource == key.videoSource) {
			requests = append(requests, v.samplers[key].TakeRequests()...)
		}
	}
	return requests
}

// RemoveParticipant stops sampling every video source of a participant, one who
// left say, and returns the image requests that were waiting on their video.
func (v *VideoInSamplers) RemoveParticipant(participantID string) []*frames.UserImageRequestFrame {
	v.mu.Lock()
	defer v.mu.Unlock()
	requests := v.takeRequests(participantID, "")
	kept := v.order[:0]
	for _, key := range v.order {
		if key.participantID == participantID {
			delete(v.samplers, key)
			continue
		}
		kept = append(kept, key)
	}
	v.order = kept
	return requests
}

// Clear stops sampling every video source, and returns the image requests that
// were waiting on any of them.
func (v *VideoInSamplers) Clear() []*frames.UserImageRequestFrame {
	v.mu.Lock()
	defer v.mu.Unlock()
	var requests []*frames.UserImageRequestFrame
	for _, key := range v.order {
		requests = append(requests, v.samplers[key].TakeRequests()...)
	}
	v.samplers = map[videoInKey]*VideoInSampler{}
	v.order = nil
	return requests
}
