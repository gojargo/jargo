package rtc_test

import (
	"context"
	"errors"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/videocodec"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
	"github.com/pion/webrtc/v4/pkg/media/samplebuilder"
)

const (
	videoWidth  = 64
	videoHeight = 48
)

// vp8EncoderOrSkip builds a VP8 encoder, or skips the test when libvpx is not
// installed.
func vp8EncoderOrSkip(t *testing.T) *videocodec.VP8Encoder {
	t.Helper()
	enc, err := videocodec.NewVP8Encoder("")
	if errors.Is(err, videocodec.ErrVPXNotAvailable) {
		t.Skipf("libvpx not available: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(enc.Close)
	return enc
}

// solid is a picture of one color.
func solid(r, g, b byte) []byte {
	rgb := make([]byte, 0, videoWidth*videoHeight*3)
	for range videoWidth * videoHeight {
		rgb = append(rgb, r, g, b)
	}
	return rgb
}

// near reports whether every pixel of rgb is within 10 of the color.
func near(rgb []byte, r, g, b byte) bool {
	within := func(got, want byte) bool { return int(got)-int(want) <= 10 && int(want)-int(got) <= 10 }
	for pixel := range len(rgb) / 3 {
		p := rgb[pixel*3 : pixel*3+3]
		if !within(p[0], r) || !within(p[1], g) || !within(p[2], b) {
			return false
		}
	}
	return len(rgb) > 0
}

// videoClient is a browser-like peer: a microphone, a camera and a screen
// share, and a data channel. Its camera sends VP8 from the start; the screen
// share sends once started.
type videoClient struct {
	pc     *webrtc.PeerConnection
	camera *webrtc.TrackLocalStaticSample
	screen *webrtc.RTPTransceiver
	// botVideo receives the pictures the bot sends on the camera's line.
	botVideo chan []byte
}

func newVideoClient(t *testing.T) *videoClient {
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
	c := &videoClient{pc: pc, botVideo: make(chan []byte, 64)}

	mic, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2}, "audio", "client")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pc.AddTrack(mic); err != nil {
		t.Fatal(err)
	}
	if c.camera, err = webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8, ClockRate: 90000}, "camera", "client"); err != nil {
		t.Fatal(err)
	}
	if _, err := pc.AddTrack(c.camera); err != nil {
		t.Fatal(err)
	}
	if c.screen, err = pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo, webrtc.RTPTransceiverInit{
		Direction: webrtc.RTPTransceiverDirectionSendonly,
	}); err != nil {
		t.Fatal(err)
	}
	if _, err := pc.CreateDataChannel("chat", nil); err != nil {
		t.Fatal(err)
	}

	pc.OnTrack(func(tr *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		if tr.Kind() != webrtc.RTPCodecTypeVideo {
			return
		}
		go c.decodeBotVideo(t, tr)
	})
	return c
}

// decodeBotVideo decodes the video the bot sends and hands each picture over
// as RGB.
func (c *videoClient) decodeBotVideo(t *testing.T, tr *webrtc.TrackRemote) {
	t.Helper()
	dec, err := videocodec.NewVP8Decoder("")
	if err != nil {
		t.Error(err)
		return
	}
	defer dec.Close()
	builder := samplebuilder.New(128, &codecs.VP8Packet{}, 90000)
	for {
		pkt, _, err := tr.ReadRTP()
		if err != nil {
			return
		}
		builder.Push(pkt)
		for s := builder.Pop(); s != nil; s = builder.Pop() {
			pictures, err := dec.Decode(s.Data)
			if err != nil {
				continue
			}
			for _, p := range pictures {
				select {
				case c.botVideo <- videocodec.I420ToRGB(p):
				default:
				}
			}
		}
	}
}

// connect offers to server and applies its answer.
func (c *videoClient) connect(t *testing.T, server *rtc.Connection) {
	t.Helper()
	offer, err := c.pc.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gathered := webrtc.GatheringCompletePromise(c.pc)
	if err := c.pc.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gathered
	answer, err := server.Answer(*c.pc.LocalDescription())
	if err != nil {
		t.Fatal(err)
	}
	if err := c.pc.SetRemoteDescription(answer); err != nil {
		t.Fatal(err)
	}
}

// send writes pictures of one color on track at 30 frames per second until
// stop closes, a keyframe every ten frames.
func send(t *testing.T, track *webrtc.TrackLocalStaticSample, rgb []byte, stop <-chan struct{}) {
	t.Helper()
	enc := vp8EncoderOrSkip(t)
	picture, err := videocodec.RGBToI420(rgb, videoWidth, videoHeight)
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		ticker := time.NewTicker(time.Second / 30)
		defer ticker.Stop()
		for i := int64(0); ; i++ {
			select {
			case <-stop:
				return
			case <-ticker.C:
			}
			data, err := enc.Encode(picture, i*3000, i%10 == 0)
			if err != nil {
				return
			}
			_ = track.WriteSample(media.Sample{Data: data, Duration: time.Second / 30})
		}
	}()
}

// TestScreenShareFramesReachTheScreenTrack covers a screen share started after
// the call connected: its pictures reach the connection's screen track, through
// real ICE, DTLS and RTP between two peers in this process.
func TestScreenShareFramesReachTheScreenTrack(t *testing.T) {
	vp8EncoderOrSkip(t)
	server, err := rtc.NewConnection(rtc.WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	client := newVideoClient(t)
	client.connect(t, server)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := server.WaitConnected(ctx); err != nil {
		t.Fatal(err)
	}

	// The user starts sharing their screen after the call connects.
	screenTrack, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeVP8, ClockRate: 90000}, "screen", "client")
	if err != nil {
		t.Fatal(err)
	}
	if err := client.screen.Sender().ReplaceTrack(screenTrack); err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	defer close(stop)
	send(t, screenTrack, solid(200, 40, 40), stop)

	track := server.ScreenVideoInputTrack()
	if track == nil {
		t.Fatal("no screen track")
	}
	for {
		frame, err := track.Recv(ctx)
		if err != nil {
			t.Fatalf("no screen picture: %v", err)
		}
		if frame == nil {
			continue
		}
		if frame.Width != videoWidth || frame.Height != videoHeight || frame.Format != rtc.FormatYUV420P {
			t.Fatalf("picture %dx%d %s", frame.Width, frame.Height, frame.Format)
		}
		return
	}
}

// imageRecorder records the user images that pass through it.
type imageRecorder struct {
	*processor.Base
	mu     sync.Mutex
	images []*frames.UserImageRawFrame
}

func newImageRecorder() *imageRecorder {
	r := &imageRecorder{}
	r.Base = processor.New("ImageRecorder", r)
	return r
}

func (r *imageRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if image, ok := f.(*frames.UserImageRawFrame); ok {
		r.mu.Lock()
		r.images = append(r.images, image)
		r.mu.Unlock()
	}
	return r.PushFrame(ctx, f, dir)
}

// answering waits for an image that answers request.
func (r *imageRecorder) answering(t *testing.T, request *frames.UserImageRequestFrame) *frames.UserImageRawFrame {
	t.Helper()
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		r.mu.Lock()
		for _, image := range r.images {
			if image.Request == request {
				r.mu.Unlock()
				return image
			}
		}
		r.mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("no image answered the request")
	return nil
}

// TestVideoBothWaysOverTheTransport runs a pipeline on the transport with
// video in and out: an image request is answered by a picture of the client's
// camera, and an image the bot shows reaches the client as video.
func TestVideoBothWaysOverTheTransport(t *testing.T) {
	vp8EncoderOrSkip(t)
	server, err := rtc.NewConnection(rtc.WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	client := newVideoClient(t)
	client.connect(t, server)
	stop := make(chan struct{})
	defer close(stop)
	send(t, client.camera, solid(51, 102, 204), stop)

	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	params.VideoInEnabled = true
	params.VideoInSources = map[string]transport.VideoInSourceParams{
		transport.CameraVideoSource: {OnRequestOnly: true},
	}
	params.VideoOutEnabled = true
	params.VideoOutWidth = videoWidth
	params.VideoOutHeight = videoHeight
	tr := rtc.NewTransport(server, params)
	rec := newImageRecorder()
	worker := pipeline.NewWorker(pipeline.New(tr.Input(), rec, tr.Output()), pipeline.WorkerConfig{
		Params: pipeline.Params{AudioInSampleRate: opus.SampleRate, AudioOutSampleRate: opus.SampleRate},
	})
	connected := make(chan struct{}, 1)
	tr.Events().Add(transport.EventClientConnected, func(context.Context, any, ...any) {
		connected <- struct{}{}
	})
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { _ = worker.Run(ctx); close(done) }()
	defer func() {
		cancel()
		<-done
	}()
	select {
	case <-connected:
	case <-time.After(15 * time.Second):
		t.Fatal("the client never connected")
	}

	request := frames.NewUserImageRequestFrame(tr.GetClientID(server))
	request.Text = "What is this?"
	request.AppendToContext = true
	worker.QueueFrame(request, processor.Upstream)
	image := rec.answering(t, request)
	if image.Text != "What is this?" || !image.AppendToContext || image.UserID != server.PCID() ||
		image.TransportSource() != transport.CameraVideoSource || image.Format != "RGB" {
		t.Fatalf("image = %v", image)
	}
	if image.Width != videoWidth || image.Height != videoHeight || !near(image.Image, 51, 102, 204) {
		t.Fatalf("image of %dx%d does not show the camera's color", image.Width, image.Height)
	}

	worker.QueueFrame(frames.NewOutputImageRawFrame(solid(220, 30, 30), videoWidth, videoHeight, "RGB"))
	deadline := time.After(15 * time.Second)
	for {
		select {
		case rgb := <-client.botVideo:
			if near(rgb, 220, 30, 30) {
				return
			}
		case <-deadline:
			t.Fatal("the bot's image never reached the client")
		}
	}
}

// TestH264CameraFramesAreDecoded covers a client whose camera sends H.264: its
// pictures are decoded through OpenH264.
func TestH264CameraFramesAreDecoded(t *testing.T) {
	dec, err := videocodec.NewH264Decoder("")
	if errors.Is(err, videocodec.ErrOpenH264NotAvailable) {
		t.Skipf("libopenh264 not available: %v", err)
	}
	if err != nil {
		t.Fatal(err)
	}
	dec.Close()
	// One keyframe of one color, (51, 102, 204), sent over and over.
	keyframe, err := os.ReadFile("../../internal/videocodec/testdata/color-64x48.h264")
	if err != nil {
		t.Fatal(err)
	}

	server, err := rtc.NewConnection(rtc.WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()
	client := newVideoClient(t)
	camera, err := webrtc.NewTrackLocalStaticSample(webrtc.RTPCodecCapability{
		MimeType: webrtc.MimeTypeH264, ClockRate: 90000,
		SDPFmtpLine: "level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f",
	}, "camera", "client")
	if err != nil {
		t.Fatal(err)
	}
	if err := client.pc.GetTransceivers()[1].Sender().ReplaceTrack(camera); err != nil {
		t.Fatal(err)
	}
	client.connect(t, server)
	stop := make(chan struct{})
	defer close(stop)
	go func() {
		ticker := time.NewTicker(time.Second / 30)
		defer ticker.Stop()
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				_ = camera.WriteSample(media.Sample{Data: keyframe, Duration: time.Second / 30})
			}
		}
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := server.WaitConnected(ctx); err != nil {
		t.Fatal(err)
	}
	track := server.VideoInputTrack()
	for {
		frame, err := track.Recv(ctx)
		if err != nil {
			t.Fatalf("no camera picture: %v", err)
		}
		if frame == nil {
			continue
		}
		rgb := videocodec.I420ToRGB(&videocodec.I420Frame{
			Width: frame.Width, Height: frame.Height, Y: frame.Planes[0], U: frame.Planes[1], V: frame.Planes[2],
		})
		if frame.Width != videoWidth || frame.Height != videoHeight || !near(rgb, 51, 102, 204) {
			t.Fatalf("picture of %dx%d does not show the camera's color", frame.Width, frame.Height)
		}
		return
	}
}
