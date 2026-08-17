package rtc_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/pion/webrtc/v4"
)

// frameRecorder records every frame that passes through it.
type frameRecorder struct {
	*processor.Base
	mu   sync.Mutex
	seen []frames.Frame
}

func newFrameRecorder() *frameRecorder {
	r := &frameRecorder{}
	r.Base = processor.New("Recorder", r)
	return r
}

func (r *frameRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	r.mu.Lock()
	r.seen = append(r.seen, f)
	r.mu.Unlock()
	return r.PushFrame(ctx, f, dir)
}

// count is how many frames of type T have passed through.
func count[T frames.Frame](r *frameRecorder) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	n := 0
	for _, f := range r.seen {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

// TestClientConnectedIsReportedWhenThePeerConnects covers the transport telling
// the pipeline a caller arrived. Nothing else says so: the audio track carries
// speech, not the fact of a connection, and a caller who connects and says
// nothing would otherwise be indistinguishable from no caller at all.
func TestClientConnectedIsReportedWhenThePeerConnects(t *testing.T) {
	server, err := rtc.NewConnection(rtc.WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()

	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		t.Fatal(err)
	}
	api := webrtc.NewAPI(webrtc.WithMediaEngine(m))
	client, err := api.NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	defer client.Close()

	clientTrack, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2},
		"audio", "client",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.AddTrack(clientTrack); err != nil {
		t.Fatal(err)
	}

	// Signaling: client offer, server answer, back to the client.
	offer, err := client.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gatherDone := webrtc.GatheringCompletePromise(client)
	if err := client.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gatherDone
	answer, err := server.Answer(*client.LocalDescription())
	if err != nil {
		t.Fatal(err)
	}
	if err := client.SetRemoteDescription(answer); err != nil {
		t.Fatal(err)
	}

	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	tr := rtc.NewTransport(server, params)
	rec := newFrameRecorder()
	task := pipeline.NewWorker(pipeline.New(tr.Input(), rec, tr.Output()), pipeline.WorkerConfig{
		Params: pipeline.Params{
			AudioInSampleRate:  opus.SampleRate,
			AudioOutSampleRate: opus.SampleRate,
		},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	taskDone := make(chan struct{})
	go func() { _ = task.Run(ctx); close(taskDone) }()

	deadline := time.Now().Add(15 * time.Second)
	for count[*frames.ClientConnectedFrame](rec) == 0 {
		if time.Now().After(deadline) {
			t.Fatal("the client connection was never reported to the pipeline")
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Once is enough: a connection that is up stays up, and a second report
	// would restart whatever is timing the setup.
	time.Sleep(100 * time.Millisecond)
	if n := count[*frames.ClientConnectedFrame](rec); n != 1 {
		t.Errorf("ClientConnectedFrames = %d, want exactly one", n)
	}

	cancel()
	select {
	case <-taskDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not shut down")
	}
}
