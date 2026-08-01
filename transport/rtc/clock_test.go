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

// stallGap is the pause injected between the two bursts of audio. Sentence
// boundaries produce pauses of roughly this order, and it is long enough that a
// timeline which dropped it could not be mistaken for scheduling noise.
const stallGap = 700 * time.Millisecond

// TestOutputTimelineSurvivesAStall is the property the transport exists to hold:
// RTP timestamps have to measure elapsed time, not packets sent.
//
// The source here does what a real one does at a sentence boundary — sends
// audio, stops for a while, then resumes. A sender that goes quiet during the
// pause and picks up where it left off timestamps the second burst as though the
// pause never happened, and the receiver reads that as delay: it conceals by
// repeating, then compresses once packets bunch up. So the assertion is on the
// span between the first and last timestamp, which must cover the wall clock the
// test actually spent, gap included.
func TestOutputTimelineSurvivesAStall(t *testing.T) {
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

	// Receive-only: the server does all the sending in this test.
	if _, err := client.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio,
		webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}); err != nil {
		t.Fatal(err)
	}

	type stamp struct {
		rtp uint32
		at  time.Time
	}
	// A mutex rather than a channel: the receive goroutine outlives the test
	// body, so there is no safe moment to close one.
	var stampMu sync.Mutex
	var stamps []stamp
	client.OnTrack(func(tr *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		go func() {
			for {
				pkt, _, err := tr.ReadRTP()
				if err != nil {
					return
				}
				stampMu.Lock()
				stamps = append(stamps, stamp{rtp: pkt.Timestamp, at: time.Now()})
				stampMu.Unlock()
			}
		}()
	})

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
	params.AudioInEnabled = false
	params.AudioInPassthrough = false
	params.AudioOutSampleRate = opus.SampleRate
	tr := rtc.NewTransport(server, params)

	src := newBurstSource()
	task := pipeline.NewTask(pipeline.New(src, tr.Output()), pipeline.TaskParams{
		AudioOutSampleRate: opus.SampleRate,
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	taskDone := make(chan struct{})
	go func() { _ = task.Run(ctx); close(taskDone) }()

	select {
	case <-src.done:
	case <-time.After(20 * time.Second):
		t.Fatal("source did not finish")
	}
	// Let the last frames reach the client before measuring.
	time.Sleep(300 * time.Millisecond)
	cancel()
	select {
	case <-taskDone:
	case <-time.After(5 * time.Second):
		t.Fatal("task did not shut down")
	}

	stampMu.Lock()
	got := append([]stamp(nil), stamps...)
	stampMu.Unlock()
	if len(got) < 2 {
		t.Fatalf("received %d packets, need at least 2 to measure a span", len(got))
	}
	first, last, n := got[0], got[len(got)-1], len(got)

	// Timestamps are 48 kHz sample counts, so a tick is 1/48000 s.
	spanned := time.Duration(last.rtp-first.rtp) * time.Second / opus.SampleRate
	elapsed := last.at.Sub(first.at)

	// The gap has to be in there. Allow generous slack for scheduling and for
	// however much of the burst landed either side of the measured window: the
	// failure this guards against loses the whole gap, not a few milliseconds.
	const slack = 150 * time.Millisecond
	if spanned+slack < elapsed {
		t.Fatalf("timeline lost %v: RTP spans %v over %v of wall clock (%d packets)",
			elapsed-spanned, spanned, elapsed, n)
	}
}

// burstSource emits a burst of audio, stalls the way a synthesizer does between
// sentences, then emits a second burst.
type burstSource struct {
	*processor.Base
	done chan struct{}
}

func newBurstSource() *burstSource {
	s := &burstSource{done: make(chan struct{})}
	s.Base = processor.New("BurstSource", s)
	return s
}

func (s *burstSource) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok {
		go s.emit(ctx)
	}
	return s.PushFrame(ctx, f, dir)
}

// emit sends 200 ms of audio, pauses, then sends 200 ms more.
func (s *burstSource) emit(ctx context.Context) {
	defer close(s.done)
	burst := func() {
		for range 10 {
			pcm := make([]byte, opus.FrameBytes(1))
			if err := s.PushFrame(ctx,
				frames.NewOutputAudioRawFrame(pcm, opus.SampleRate, 1), processor.Downstream); err != nil {
				return
			}
		}
	}
	burst()
	select {
	case <-time.After(stallGap):
	case <-ctx.Done():
		return
	}
	burst()
	// Give the sender time to drain the second burst before the test tears down.
	select {
	case <-time.After(500 * time.Millisecond):
	case <-ctx.Done():
	}
}
