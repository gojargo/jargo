package transport

import (
	"math"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// at is a moment secs seconds into a test's clock.
func at(secs float64) time.Time {
	return time.Unix(0, 0).Add(time.Duration(secs * float64(time.Second)))
}

func sampleAt(s *VideoInSampler, secs float64) (bool, *frames.UserImageRequestFrame) {
	s.now = func() time.Time { return at(secs) }
	return s.Sample()
}

func passed(s *VideoInSampler, times []float64) []float64 {
	var out []float64
	for _, secs := range times {
		if due, _ := sampleAt(s, secs); due {
			out = append(out, secs)
		}
	}
	return out
}

// stream is n frames from start, a frame every 1/fps seconds.
func stream(start float64, n int, fps float64) []float64 {
	times := make([]float64, n)
	for i := range n {
		times[i] = start + float64(i)/fps
	}
	return times
}

func TestEveryFramePassesEveryFrame(t *testing.T) {
	times := []float64{1.0, 1.01, 1.02}
	if got := passed(NewVideoInSampler(EveryFrame), times); !slices.Equal(got, times) {
		t.Errorf("passed %v, want %v", got, times)
	}
}

func TestZeroPassesOnlyRequestedFrames(t *testing.T) {
	s := NewVideoInSampler(0)
	if due, _ := sampleAt(s, 1.0); due {
		t.Error("an unrequested frame passed")
	}
	request := frames.NewUserImageRequestFrame("u")
	s.AddRequest(request)
	if due, got := sampleAt(s, 1.1); !due || got != request {
		t.Errorf("sample = (%v, %v), want the request answered", due, got)
	}
	if due, _ := sampleAt(s, 1.2); due {
		t.Error("a frame passed after the request was answered")
	}
}

func TestFramerateBelowInputRate(t *testing.T) {
	// A 30 fps stream sampled at 5 fps for ten seconds.
	got := passed(NewVideoInSampler(5), stream(10, 300, 30))
	if rate := float64(len(got)) / 10; math.Abs(rate-5) > 0.1 {
		t.Errorf("rate %v, want 5", rate)
	}
	// Every gap after the first is within a stream frame of the interval.
	for i := 2; i < len(got); i++ {
		if gap := got[i] - got[i-1]; math.Abs(gap-0.2) >= 1.0/30 {
			t.Errorf("gap %v, want about 0.2", gap)
		}
	}
}

func TestFramerateAtInputRatePassesEveryFrame(t *testing.T) {
	// A slightly slow 30 fps stream with jitter, sampled at 30 fps.
	var times []float64
	secs := 10.0
	for i := range 90 {
		jitter := 1.2
		if i%2 == 1 {
			jitter = 0.8
		}
		secs += (1 / 29.5) * jitter
		times = append(times, secs)
	}
	if got := passed(NewVideoInSampler(30), times); !slices.Equal(got, times) {
		t.Errorf("passed %d of %d frames", len(got), len(times))
	}
}

func TestNextFrameOneIntervalAfterTheFirst(t *testing.T) {
	got := passed(NewVideoInSampler(1), stream(10, 45, 30))
	if len(got) < 2 || got[1]-got[0] < 1.0-JitterTolerance.Seconds() {
		t.Errorf("passed %v, want the second a second after the first", got)
	}
}

func TestNoCloseFramesAfterALateFrame(t *testing.T) {
	// At 1 fps, the stream pauses for 1.2 s after a frame is passed on, so the
	// next frame arrives late but not long enough for a stall.
	times := append(stream(10, 31, 30), stream(11.2, 60, 30)...)
	got := passed(NewVideoInSampler(1), times)
	for i := 1; i < len(got); i++ {
		if gap := got[i] - got[i-1]; gap < 1.0-JitterTolerance.Seconds() {
			t.Errorf("gap %v, want at least an interval", gap)
		}
	}
}

func TestRequestAnsweredBetweenDueFrames(t *testing.T) {
	s := NewVideoInSampler(1)
	sampleAt(s, 10.0)
	request := frames.NewUserImageRequestFrame("u")
	s.AddRequest(request)
	if due, got := sampleAt(s, 10.2); !due || got != request {
		t.Errorf("sample = (%v, %v), want the request answered", due, got)
	}
	// The schedule restarts from the request: nothing until a second later.
	if due, _ := sampleAt(s, 11.0); due {
		t.Error("a frame passed before an interval after the request")
	}
	if due, _ := sampleAt(s, 11.2); !due {
		t.Error("no frame passed an interval after the request")
	}
}

func TestNoBurstAfterAStall(t *testing.T) {
	times := append(stream(10, 30, 30), stream(20, 30, 30)...)
	got := passed(NewVideoInSampler(1), times)
	n := 0
	for _, secs := range got {
		if secs >= 20 && secs < 20.9 {
			n++
		}
	}
	// One frame when the stream resumes, then the next an interval later.
	if n != 1 {
		t.Errorf("%d frames passed as the stream resumed, want 1", n)
	}
}

func TestRequestsAnsweredInOrder(t *testing.T) {
	s := NewVideoInSampler(0)
	first, second := frames.NewUserImageRequestFrame("a"), frames.NewUserImageRequestFrame("b")
	s.AddRequest(first)
	s.AddRequest(second)
	if _, got := sampleAt(s, 1.0); got != first {
		t.Error("the first request was not answered first")
	}
	if _, got := sampleAt(s, 1.1); got != second {
		t.Error("the second request was not answered second")
	}
}

func TestFramerateCanChange(t *testing.T) {
	s := NewVideoInSampler(EveryFrame)
	s.SetFramerate(0)
	if due, _ := sampleAt(s, 1.0); due {
		t.Error("a frame passed after the framerate changed to requests only")
	}
}

func TestUnsampledSourceIsNeverDue(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", EveryFrame)
	if due, req := v.Sample("p1", "screenVideo"); due || req != nil {
		t.Error("an unsampled source was due")
	}
	if due, req := v.Sample("p2", "camera"); due || req != nil {
		t.Error("an unsampled participant was due")
	}
}

func TestIsCapturing(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", 0)
	if !v.IsCapturing("p1", "camera") || v.IsCapturing("p1", "screenVideo") {
		t.Error("capturing the wrong sources")
	}
	v.RemoveParticipant("p1")
	if v.IsCapturing("p1", "camera") {
		t.Error("still capturing a removed participant")
	}
}

func TestTakeRequestsKeepsTheSamplers(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", 0)
	v.Capture("p1", "screenVideo", 0)
	camera := frames.NewUserImageRequestFrame("p1")
	screen := frames.NewUserImageRequestFrame("p1")
	screen.VideoSource = "screenVideo"
	v.AddRequest("p1", "camera", camera)
	v.AddRequest("p1", "screenVideo", screen)

	if got := v.TakeRequests("p1", "screenVideo"); !slices.Equal(got, []*frames.UserImageRequestFrame{screen}) {
		t.Errorf("took %v, want the screen request", got)
	}
	if got := v.TakeRequests("p1", ""); !slices.Equal(got, []*frames.UserImageRequestFrame{camera}) {
		t.Errorf("took %v, want the camera request", got)
	}
	if !v.IsCapturing("p1", "screenVideo") {
		t.Error("taking requests stopped the capture")
	}
}

func TestRemoveParticipantAndClearReturnWaitingRequests(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", 0)
	v.Capture("p2", "camera", 0)
	first, second := frames.NewUserImageRequestFrame("p1"), frames.NewUserImageRequestFrame("p2")
	v.AddRequest("p1", "camera", first)
	v.AddRequest("p2", "camera", second)

	if got := v.RemoveParticipant("p1"); !slices.Equal(got, []*frames.UserImageRequestFrame{first}) {
		t.Errorf("removing p1 returned %v", got)
	}
	if got := v.Clear(); !slices.Equal(got, []*frames.UserImageRequestFrame{second}) {
		t.Errorf("clearing returned %v", got)
	}
	if v.IsCapturing("p2", "camera") {
		t.Error("still capturing after a clear")
	}
}

func TestRequestForUnsampledSourceIsNotQueued(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", 0)
	request := frames.NewUserImageRequestFrame("p1")
	request.VideoSource = "screenVideo"
	if v.AddRequest("p1", "screenVideo", request) {
		t.Error("a request for an unsampled source was queued")
	}
}

func TestRequestAnsweredByItsSource(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", 0)
	request := frames.NewUserImageRequestFrame("p1")
	if !v.AddRequest("p1", "camera", request) {
		t.Fatal("the request was not queued")
	}
	if due, got := v.Sample("p1", "camera"); !due || got != request {
		t.Errorf("sample = (%v, %v), want the request answered", due, got)
	}
}

func TestCapturingAgainKeepsWaitingRequests(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", 0)
	request := frames.NewUserImageRequestFrame("p1")
	v.AddRequest("p1", "camera", request)

	v.Capture("p1", "camera", 1)

	if due, got := v.Sample("p1", "camera"); !due || got != request {
		t.Errorf("sample = (%v, %v), want the waiting request answered", due, got)
	}
}

func TestRemovingAParticipantStopsSamplingTheirSources(t *testing.T) {
	v := NewVideoInSamplers()
	v.Capture("p1", "camera", EveryFrame)
	v.Capture("p1", "screenVideo", EveryFrame)
	v.Capture("p2", "camera", EveryFrame)

	v.RemoveParticipant("p1")

	if due, _ := v.Sample("p1", "camera"); due {
		t.Error("a removed participant's camera was due")
	}
	if due, _ := v.Sample("p1", "screenVideo"); due {
		t.Error("a removed participant's screen was due")
	}
	if due, _ := v.Sample("p2", "camera"); !due {
		t.Error("another participant stopped being sampled")
	}
}

func TestCaptureFramerate(t *testing.T) {
	if got := CaptureFramerate(30, true); got != 0 {
		t.Errorf("on request only = %d, want 0", got)
	}
	if got := CaptureFramerate(5, false); got != 5 {
		t.Errorf("framerate = %d, want 5", got)
	}
}
