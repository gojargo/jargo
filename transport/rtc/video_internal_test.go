package rtc

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/transport"
	"github.com/pion/rtp"
	"github.com/pion/webrtc/v4"
)

// fakeTrack stands in for a client's video track: it returns its frames, then
// err, or waits once neither is left.
type fakeTrack struct {
	mu      sync.Mutex
	frames  []*VideoFrame
	err     error
	enabled bool
	calls   int
}

func (f *fakeTrack) Recv(ctx context.Context) (*VideoFrame, error) {
	f.mu.Lock()
	f.calls++
	if len(f.frames) > 0 {
		frame := f.frames[0]
		f.frames = f.frames[1:]
		f.mu.Unlock()
		return frame, nil
	}
	err := f.err
	f.mu.Unlock()
	if err != nil {
		return nil, err
	}
	<-ctx.Done()
	return nil, ctx.Err()
}

func (f *fakeTrack) IsEnabled() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.enabled
}

func (f *fakeTrack) recvCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

// pictures returns n pictures of 2 by 2 pixels.
func pictures(n int) []*VideoFrame {
	out := make([]*VideoFrame, n)
	for i := range out {
		out[i] = &VideoFrame{
			Width: 2, Height: 2, Format: FormatYUV420P,
			Planes: [][]byte{{16, 16, 16, 16}, {128}, {128}},
			PTS:    int64(i),
		}
	}
	return out
}

// videoInput is a transport over an unconnected connection, its input reading
// under a context the test ends, and the frames its input pushes.
type videoInput struct {
	tr     *Transport
	in     *inputTransport
	mu     sync.Mutex
	pushed []*frames.UserImageRawFrame
}

func newVideoInput(t *testing.T, params transport.Params) *videoInput {
	t.Helper()
	conn, err := NewConnection(WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	v := &videoInput{tr: NewTransport(conn, params)}
	v.in = v.tr.in
	v.in.pushVideoFrame = func(_ context.Context, f frames.InputImageFrame) {
		v.mu.Lock()
		defer v.mu.Unlock()
		image, ok := f.(*frames.UserImageRawFrame)
		if !ok {
			t.Errorf("pushed %v, want a user image", f)
			return
		}
		v.pushed = append(v.pushed, image)
	}
	ctx, cancel := context.WithCancel(context.Background())
	v.in.mu.Lock()
	v.in.readCtx = ctx
	v.in.mu.Unlock()
	t.Cleanup(func() {
		cancel()
		v.in.mu.Lock()
		v.in.readCtx = nil
		v.in.mu.Unlock()
		v.in.stopReceiveVideo()
	})
	return v
}

func (v *videoInput) frames() []*frames.UserImageRawFrame {
	v.mu.Lock()
	defer v.mu.Unlock()
	return append([]*frames.UserImageRawFrame{}, v.pushed...)
}

// waitPushed waits until n frames have been pushed.
func (v *videoInput) waitPushed(t *testing.T, n int) []*frames.UserImageRawFrame {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for len(v.frames()) < n {
		if time.Now().After(deadline) {
			t.Fatalf("%d frames pushed, want %d", len(v.frames()), n)
		}
		time.Sleep(5 * time.Millisecond)
	}
	return v.frames()
}

// recordedRequest is an image request whose result is recorded.
type recordedRequest struct {
	*frames.UserImageRequestFrame
	mu     sync.Mutex
	result string
}

func newRecordedRequest(userID, videoSource string) *recordedRequest {
	r := &recordedRequest{UserImageRequestFrame: frames.NewUserImageRequestFrame(userID)}
	r.VideoSource = videoSource
	r.ResultCallback = func(_ context.Context, result string, _ *frames.FunctionCallResultProperties) error {
		r.mu.Lock()
		defer r.mu.Unlock()
		r.result = result
		return nil
	}
	return r
}

// errorResult is the error the request was answered with, "" when it was not.
func (r *recordedRequest) errorResult(t *testing.T) string {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.result == "" {
		return ""
	}
	var result struct {
		Error string `json:"error"`
	}
	if err := json.Unmarshal([]byte(r.result), &result); err != nil {
		t.Fatal(err)
	}
	return result.Error
}

func videoParams() transport.Params {
	params := transport.DefaultParams()
	params.VideoInEnabled = true
	return params
}

func TestReadVideoFrameParksOnADeadTrack(t *testing.T) {
	for _, source := range []string{camVideoSource, screenVideoSource} {
		t.Run(source, func(t *testing.T) {
			v := newVideoInput(t, videoParams())
			track := &fakeTrack{err: ErrTrackEnded, enabled: true}
			c := v.tr.client
			if source == camVideoSource {
				c.videoInputTrack = track
			} else {
				c.screenVideoTrack = track
			}
			ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
			defer cancel()
			for range c.readVideoFrames(ctx, source) {
				t.Fatal("a dead track yielded a frame")
			}
			// One call: the dead track is forgotten rather than read again.
			if n := track.recvCalls(); n != 1 {
				t.Errorf("Recv called %d times, want 1", n)
			}
			if c.track(source) != nil {
				t.Error("the dead track was kept")
			}
		})
	}
}

func TestReadVideoFrameResumesOnANewTrack(t *testing.T) {
	v := newVideoInput(t, videoParams())
	c := v.tr.client
	dead := &fakeTrack{err: ErrTrackEnded, enabled: true}
	c.videoInputTrack = dead
	go func() {
		time.Sleep(50 * time.Millisecond)
		c.mu.Lock()
		c.videoInputTrack = &fakeTrack{frames: pictures(2), enabled: true}
		c.mu.Unlock()
	}()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	yielded := 0
	for frame := range c.readVideoFrames(ctx, camVideoSource) {
		if frame.Width != 2 || frame.Height != 2 || frame.Format != "RGB" || len(frame.Image) != 12 {
			t.Fatalf("frame = %v", frame)
		}
		if frame.TransportSource() != camVideoSource || frame.UserID != c.conn.PCID() {
			t.Fatalf("frame from %q of %q", frame.TransportSource(), frame.UserID)
		}
		yielded++
		if yielded == 2 {
			break
		}
	}
	if yielded != 2 || dead.recvCalls() != 1 {
		t.Fatalf("yielded %d frames, dead track read %d times", yielded, dead.recvCalls())
	}
}

func TestUnthrottledSourcePassesEveryFrame(t *testing.T) {
	v := newVideoInput(t, videoParams())
	v.tr.client.videoInputTrack = &fakeTrack{frames: pictures(3), enabled: true}
	v.in.videoSamplers.Capture(peerID, camVideoSource, transport.EveryFrame)
	v.in.startReceiveVideo(camVideoSource)
	v.waitPushed(t, 3)
}

func TestRequestRidesOnThePushedFrame(t *testing.T) {
	v := newVideoInput(t, videoParams())
	request := frames.NewUserImageRequestFrame("requester")
	request.Text = "describe"
	request.AppendToContext = true
	v.in.videoSamplers.Capture(peerID, camVideoSource, 0)
	v.in.videoSamplers.AddRequest(peerID, camVideoSource, request)
	v.tr.client.videoInputTrack = &fakeTrack{frames: pictures(2), enabled: true}
	v.in.startReceiveVideo(camVideoSource)

	pushed := v.waitPushed(t, 1)
	time.Sleep(50 * time.Millisecond)
	if n := len(v.frames()); n != 1 {
		t.Fatalf("%d frames pushed, want only the one answering the request", n)
	}
	frame := pushed[0]
	if frame.Request != request || frame.Text != "describe" || !frame.AppendToContext {
		t.Fatalf("frame = %v, request %v", frame, frame.Request)
	}
	if frame.UserID != v.tr.conn.PCID() {
		t.Fatalf("user %q, want the peer connection %q", frame.UserID, v.tr.conn.PCID())
	}
}

func TestWaitingRequestsAnsweredBySuccessiveFrames(t *testing.T) {
	v := newVideoInput(t, videoParams())
	requests := []*frames.UserImageRequestFrame{
		frames.NewUserImageRequestFrame("a"), frames.NewUserImageRequestFrame("b"),
	}
	v.in.videoSamplers.Capture(peerID, camVideoSource, 0)
	for _, r := range requests {
		v.in.videoSamplers.AddRequest(peerID, camVideoSource, r)
	}
	v.tr.client.videoInputTrack = &fakeTrack{frames: pictures(2), enabled: true}
	v.in.startReceiveVideo(camVideoSource)

	pushed := v.waitPushed(t, 2)
	for i, r := range requests {
		if pushed[i].Request != r {
			t.Fatalf("frame %d answers %v, want %v", i, pushed[i].Request, r)
		}
	}
}

func TestRequestStartsAnUncapturedSource(t *testing.T) {
	v := newVideoInput(t, videoParams())
	v.tr.client.videoInputTrack = &fakeTrack{enabled: true}
	request := frames.NewUserImageRequestFrame("requester")

	v.in.requestParticipantImage(context.Background(), request)

	// Captured at every frame, with the request waiting.
	if due, got := v.in.videoSamplers.Sample(peerID, camVideoSource); !due || got != request {
		t.Fatalf("first sample = %t, %v", due, got)
	}
	if due, got := v.in.videoSamplers.Sample(peerID, camVideoSource); !due || got != nil {
		t.Fatalf("second sample = %t, %v; want every frame passed", due, got)
	}
	v.in.mu.Lock()
	started := v.in.receiveVideo[camVideoSource] != nil
	v.in.mu.Unlock()
	if !started {
		t.Fatal("the camera is not being read")
	}
}

func TestCaptureSetsFramerateWithoutDroppingRequests(t *testing.T) {
	v := newVideoInput(t, videoParams())
	v.tr.client.videoInputTrack = &fakeTrack{enabled: true}
	request := frames.NewUserImageRequestFrame("requester")

	v.in.requestParticipantImage(context.Background(), request)
	v.in.captureParticipantMedia(camVideoSource, 1)

	if due, got := v.in.videoSamplers.Sample(peerID, camVideoSource); !due || got != request {
		t.Fatalf("first sample = %t, %v", due, got)
	}
	if due, _ := v.in.videoSamplers.Sample(peerID, camVideoSource); due {
		t.Fatal("a frame right after another was due at 1 frame per second")
	}
}

func TestCapturesConfiguredSourcesOnConnect(t *testing.T) {
	params := videoParams()
	params.VideoInSources = map[string]transport.VideoInSourceParams{
		camVideoSource:    {OnRequestOnly: true},
		screenVideoSource: {Framerate: 1},
	}
	v := newVideoInput(t, params)
	capturedFirst := make(chan bool, 1)
	v.tr.Events().Add(transport.EventClientConnected, func(context.Context, any, ...any) {
		capturedFirst <- v.in.videoSamplers.IsCapturing(peerID, camVideoSource) &&
			v.in.videoSamplers.IsCapturing(peerID, screenVideoSource)
	})

	v.tr.onClientConnected()

	if !<-capturedFirst {
		t.Fatal("the sources were not captured before the event handlers ran")
	}
	// The camera only on request: no frame is due without one.
	if due, _ := v.in.videoSamplers.Sample(peerID, camVideoSource); due {
		t.Error("a camera frame was due with no request")
	}
	// The screen share at 1 frame per second.
	if due, _ := v.in.videoSamplers.Sample(peerID, screenVideoSource); !due {
		t.Error("the first screen frame was not due")
	}
	if due, _ := v.in.videoSamplers.Sample(peerID, screenVideoSource); due {
		t.Error("a screen frame right after another was due at 1 frame per second")
	}
}

func TestUnsupportedSourceNotCaptured(t *testing.T) {
	params := videoParams()
	params.VideoInSources = map[string]transport.VideoInSourceParams{
		camVideoSource: {},
		micAudioSource: {},
	}
	v := newVideoInput(t, params)

	v.tr.onClientConnected()

	if v.in.videoSamplers.IsCapturing(peerID, micAudioSource) {
		t.Error("the microphone was captured as a video source")
	}
	// The camera at the default 30 frames per second.
	if due, _ := v.in.videoSamplers.Sample(peerID, camVideoSource); !due {
		t.Error("the first camera frame was not due")
	}
	if due, _ := v.in.videoSamplers.Sample(peerID, camVideoSource); due {
		t.Error("a camera frame right after another was due at 30 frames per second")
	}
}

func TestNoSourcesCapturesNothing(t *testing.T) {
	v := newVideoInput(t, videoParams())

	v.tr.onClientConnected()

	for _, source := range []string{camVideoSource, screenVideoSource} {
		if v.in.videoSamplers.IsCapturing(peerID, source) {
			t.Errorf("%s captured with no sources configured", source)
		}
	}
}

func TestRequestForASourceThatIsOff(t *testing.T) {
	v := newVideoInput(t, videoParams())
	request := newRecordedRequest("p1", screenVideoSource)

	v.in.requestParticipantImage(context.Background(), request.UserImageRequestFrame)

	if got := request.errorResult(t); !strings.Contains(got, "isn't sending") {
		t.Fatalf("error = %q", got)
	}
}

func TestWaitingRequestWhenTheSourceTurnsOff(t *testing.T) {
	v := newVideoInput(t, videoParams())
	v.tr.client.screenVideoTrack = &fakeTrack{enabled: true}
	request := newRecordedRequest("p1", screenVideoSource)
	v.in.requestParticipantImage(context.Background(), request.UserImageRequestFrame)
	if got := request.errorResult(t); got != "" {
		t.Fatalf("answered before the source turned off: %q", got)
	}

	v.in.stopVideo(context.Background(), screenVideoSource)

	if got := request.errorResult(t); !strings.Contains(got, "turned off") {
		t.Fatalf("error = %q", got)
	}
	if !v.in.videoSamplers.IsCapturing(peerID, screenVideoSource) {
		t.Fatal("the source stopped being captured")
	}
}

func TestTransportAnswersOnlyWhenVideoTurnsOff(t *testing.T) {
	v := newVideoInput(t, videoParams())
	v.tr.client.videoInputTrack = &fakeTrack{enabled: true}
	v.tr.client.screenVideoTrack = &fakeTrack{enabled: true}
	camera := newRecordedRequest("p1", camVideoSource)
	screen := newRecordedRequest("p1", screenVideoSource)
	v.in.requestParticipantImage(context.Background(), camera.UserImageRequestFrame)
	v.in.requestParticipantImage(context.Background(), screen.UserImageRequestFrame)

	v.tr.onTrackStatus(VideoTransceiverIndex, true)
	v.tr.onTrackStatus(AudioTransceiverIndex, false)
	v.tr.onTrackStatus(ScreenVideoTransceiverIndex, false)

	if got := camera.errorResult(t); got != "" {
		t.Errorf("camera request answered: %q", got)
	}
	if got := screen.errorResult(t); !strings.Contains(got, "turned off its screenVideo") {
		t.Errorf("screen request error = %q", got)
	}
}

func TestWaitingRequestsWhenThePeerDisconnects(t *testing.T) {
	v := newVideoInput(t, videoParams())
	v.tr.client.videoInputTrack = &fakeTrack{enabled: true}
	request := newRecordedRequest("p1", camVideoSource)
	v.in.requestParticipantImage(context.Background(), request.UserImageRequestFrame)

	v.tr.onClientClosed()

	if got := request.errorResult(t); !strings.Contains(got, "disconnected") {
		t.Fatalf("error = %q", got)
	}
	if v.in.videoSamplers.IsCapturing(peerID, camVideoSource) {
		t.Fatal("still sampling the video of a peer that left")
	}
}

func TestBotLeavingDoesNotAnswerAsAPeerDisconnection(t *testing.T) {
	v := newVideoInput(t, videoParams())
	v.tr.client.videoInputTrack = &fakeTrack{enabled: true}
	request := newRecordedRequest("p1", camVideoSource)
	v.in.requestParticipantImage(context.Background(), request.UserImageRequestFrame)
	v.tr.closing.Store(true)

	v.tr.onClientClosed()

	if got := request.errorResult(t); got != "" {
		t.Fatalf("answered %q when the bot left", got)
	}
}

func TestCaptureParticipantVideoTakesTheStandardArguments(t *testing.T) {
	v := newVideoInput(t, videoParams())
	ctx := context.Background()

	if err := v.tr.CaptureParticipantVideo(ctx, "pc-1", 30, screenVideoSource, false); err != nil {
		t.Fatal(err)
	}
	if err := v.tr.CaptureParticipantVideo(ctx, "pc-1", 1, "", true); err != nil {
		t.Fatal(err)
	}
	if !v.in.videoSamplers.IsCapturing(peerID, screenVideoSource) {
		t.Fatal("the screen share is not captured")
	}
	// An empty source is the camera, and on request only it passes no frame
	// without a request.
	if !v.in.videoSamplers.IsCapturing(peerID, camVideoSource) {
		t.Fatal("the camera is not captured")
	}
	if due, _ := v.in.videoSamplers.Sample(peerID, camVideoSource); due {
		t.Fatal("a camera frame was due with no request")
	}
	v.in.mu.Lock()
	reading := len(v.in.receiveVideo)
	v.in.mu.Unlock()
	if reading != 2 {
		t.Fatalf("%d sources being read, want 2", reading)
	}
}

func TestSupportsCameraAndScreenVideoSources(t *testing.T) {
	v := newVideoInput(t, videoParams())
	for source, want := range map[string]bool{
		camVideoSource: true, screenVideoSource: true, micAudioSource: false, "other": false,
	} {
		if got := v.in.SupportsVideoInSource(source); got != want {
			t.Errorf("SupportsVideoInSource(%q) = %t, want %t", source, got, want)
		}
	}
}

func TestConvertFrame(t *testing.T) {
	rgb, err := convertFrame(&VideoFrame{Width: 1, Height: 1, Format: "rgb24", Planes: [][]byte{{1, 2, 3}}})
	if err != nil || string(rgb) != "\x01\x02\x03" {
		t.Fatalf("rgb24 = %v, %v", rgb, err)
	}
	rgb, err = convertFrame(pictures(1)[0])
	if err != nil || len(rgb) != 12 || rgb[0] != 0 {
		t.Fatalf("yuv420p = %v, %v", rgb, err)
	}
	if _, err := convertFrame(&VideoFrame{Format: "nv21"}); err == nil {
		t.Fatal("an unsupported format converted")
	}
}

func TestVideoClockPacesAt30FramesPerSecond(t *testing.T) {
	var clock videoClock
	stop := make(chan struct{})
	start := time.Now()
	for i := range 4 {
		ts, ok := clock.nextTimestamp(stop)
		if !ok || ts != int64(i*3000) {
			t.Fatalf("timestamp %d = %d, %t", i, ts, ok)
		}
	}
	if elapsed := time.Since(start); elapsed < 90*time.Millisecond {
		t.Fatalf("3 frame periods passed in %v", elapsed)
	}
	close(stop)
	if _, ok := clock.nextTimestamp(stop); ok {
		t.Fatal("a stopped clock gave a timestamp")
	}
}

func TestWriteVideoFrameNeedsAnOutputTrack(t *testing.T) {
	v := newVideoInput(t, videoParams())
	ok, err := v.tr.out.WriteVideoFrame(context.Background(), frames.NewOutputImageRawFrame(nil, 1, 1, "RGB"))
	if ok || err != nil {
		t.Fatalf("WriteVideoFrame = %t, %v with no client", ok, err)
	}
}

// browserLikeOffer is an offer shaped like the browser client's: audio,
// camera and screen share transceivers plus a data channel, bundled onto one
// transport.
func browserLikeOffer(t *testing.T) (*webrtc.PeerConnection, *webrtc.RTPTransceiver) {
	t.Helper()
	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		t.Fatal(err)
	}
	pc, err := webrtc.NewAPI(webrtc.WithMediaEngine(m)).NewPeerConnection(webrtc.Configuration{
		BundlePolicy: webrtc.BundlePolicyMaxBundle,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pc.Close() })
	for _, kind := range []webrtc.RTPCodecType{webrtc.RTPCodecTypeAudio, webrtc.RTPCodecTypeVideo} {
		if _, err := pc.AddTransceiverFromKind(kind, webrtc.RTPTransceiverInit{
			Direction: webrtc.RTPTransceiverDirectionSendrecv,
		}); err != nil {
			t.Fatal(err)
		}
	}
	screen, err := pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionSendonly,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pc.CreateDataChannel("chat", nil); err != nil {
		t.Fatal(err)
	}
	offer, err := pc.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gathered := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gathered
	return pc, screen
}

// answeredConnection is a connection that answered a browser-like offer.
func answeredConnection(t *testing.T) *Connection {
	t.Helper()
	client, _ := browserLikeOffer(t)
	conn, err := NewConnection(WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	if err := conn.Initialize(*client.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	return conn
}

// TestScreenShareTransceiverSharesTheTransport covers every media line of a
// bundled offer sharing one transport, and the transceivers taking the
// directions they answer with: the first two send and receive, the screen
// share only receives.
func TestScreenShareTransceiverSharesTheTransport(t *testing.T) {
	conn := answeredConnection(t)
	transceivers := mediaTransceivers(conn.pc)
	if len(transceivers) != 3 {
		t.Fatalf("%d transceivers, want 3", len(transceivers))
	}
	transports := map[*webrtc.DTLSTransport]bool{}
	for _, tr := range transceivers {
		transports[tr.Receiver().Transport()] = true
	}
	if len(transports) != 1 {
		t.Fatalf("%d transports, want 1", len(transports))
	}
	for i, want := range []webrtc.RTPTransceiverDirection{
		webrtc.RTPTransceiverDirectionSendrecv,
		webrtc.RTPTransceiverDirectionSendrecv,
		webrtc.RTPTransceiverDirectionRecvonly,
	} {
		if got := transceivers[i].Direction(); got != want {
			t.Errorf("transceiver %d is %s, want %s", i, got, want)
		}
	}
	// The audio, the camera and the data channel send and receive.
	answer := conn.GetAnswer()
	if answer == nil || strings.Count(answer.SDP, "a=sendrecv") != 3 || strings.Count(answer.SDP, "a=recvonly") != 1 {
		t.Fatalf("answer directions wrong:\n%v", answer)
	}
}

// TestVideoJitterBuffer covers the screen share's receiver holding enough
// packets to assemble a frame of many of them, while the camera keeps the
// default.
func TestVideoJitterBuffer(t *testing.T) {
	conn := answeredConnection(t)
	if c := conn.VideoInputTrack().capacity; c != defaultJitterBufferCapacity {
		t.Errorf("camera capacity = %d", c)
	}
	screen := conn.ScreenVideoInputTrack()
	if screen.capacity != screenVideoJitterBufferCapacity {
		t.Fatalf("screen capacity = %d", screen.capacity)
	}

	// A keyframe of 300 packets, then the first packet of the next frame,
	// which completes it.
	r := frameReader{track: screen, mimeType: webrtc.MimeTypeVP8}
	r.newBuilder()
	for seq := range 301 {
		descriptor := byte(0)
		if seq == 0 || seq == 300 {
			descriptor = 0x10 // start of a partition
		}
		ts := uint32(0)
		if seq == 300 {
			ts = 3000
		}
		r.builder.Push(&rtp.Packet{
			SequenceNumber: uint16(seq), Timestamp: ts, Marker: seq == 299,
			Payload: []byte{descriptor, 'x'},
		})
	}
	sample := r.builder.Pop()
	if sample == nil || len(sample.Data) != 300 {
		t.Fatalf("assembled %v, want a frame of 300 bytes", sample)
	}
}

// TestTrackStatusTurnsTracksOnAndOff covers the client turning its sources on
// and off: the tracks record it, and each change is reported with the source
// it concerns.
func TestTrackStatusTurnsTracksOnAndOff(t *testing.T) {
	conn := answeredConnection(t)
	var mu sync.Mutex
	var got []string
	done := make(chan struct{})
	conn.OnTrackStatus(func(index int, enabled bool) {
		mu.Lock()
		defer mu.Unlock()
		state := "off"
		if enabled {
			state = "on"
		}
		got = append(got, sourcesByReceiver[index]+" "+state)
		if len(got) == 4 {
			close(done)
		}
	})
	for _, m := range []string{
		`{"type":"trackStatus","receiver_index":2,"enabled":true}`,
		`{"type":"trackStatus","receiver_index":2,"enabled":false}`,
		`{"type":"trackStatus","receiver_index":1,"enabled":false}`,
		`{"type":"trackStatus","receiver_index":0,"enabled":false}`,
	} {
		conn.handleSignalingMessage(json.RawMessage(m))
	}
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("track status changes not reported")
	}
	want := []string{"screenVideo on", "screenVideo off", "camera off", "microphone off"}
	mu.Lock()
	defer mu.Unlock()
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("reported %v, want %v", got, want)
	}
	if conn.VideoInputTrack().IsEnabled() || conn.ScreenVideoInputTrack().IsEnabled() {
		t.Fatal("a track turned off is still enabled")
	}
}

// TestDisabledVideoTrackReturnsNoPicture covers Recv on a track the client
// turned off, and on one that was stopped.
func TestDisabledVideoTrackReturnsNoPicture(t *testing.T) {
	conn := answeredConnection(t)
	track := conn.VideoInputTrack()
	track.SetEnabled(false)
	frame, err := track.Recv(context.Background())
	if frame != nil || err != nil {
		t.Fatalf("Recv = %v, %v on a disabled track", frame, err)
	}
	track.SetEnabled(true)
	track.Stop()
	if _, err := track.Recv(context.Background()); !errors.Is(err, ErrTrackEnded) {
		t.Fatalf("Recv = %v on a stopped track, want ErrTrackEnded", err)
	}
}

// TestIdleTrackStopsBeingConsumed covers a track nobody reads dropping its
// pictures.
func TestIdleTrackStopsBeingConsumed(t *testing.T) {
	conn := answeredConnection(t)
	track := conn.VideoInputTrack()
	track.idleTimeout = 20 * time.Millisecond
	ctx, cancel := context.WithTimeout(context.Background(), time.Millisecond)
	defer cancel()
	_, _ = track.Recv(ctx)
	track.push(pictures(1)[0])
	deadline := time.Now().Add(2 * time.Second)
	for track.isReceiving() {
		if time.Now().After(deadline) {
			t.Fatal("the idle track is still consumed")
		}
		time.Sleep(5 * time.Millisecond)
	}
	track.mu.Lock()
	queued := len(track.queue)
	track.mu.Unlock()
	if queued != 0 {
		t.Fatalf("%d pictures kept by an idle track", queued)
	}
}

// TestRestartStopsTheTracks covers a restart: the tracks of the old peer
// connection end, and the new one hands out its own.
func TestRestartStopsTheTracks(t *testing.T) {
	conn := answeredConnection(t)
	old := conn.VideoInputTrack()
	client, _ := browserLikeOffer(t)
	if err := conn.Renegotiate(*client.LocalDescription(), true); err != nil {
		t.Fatal(err)
	}
	if _, err := old.Recv(context.Background()); !errors.Is(err, ErrTrackEnded) {
		t.Fatalf("old track Recv = %v, want ErrTrackEnded", err)
	}
	if fresh := conn.VideoInputTrack(); fresh == nil || fresh == old {
		t.Fatal("the restarted connection did not hand out a new track")
	}
}
