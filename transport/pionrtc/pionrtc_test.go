package pionrtc_test

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// echoProcessor turns received audio frames into outgoing audio frames.
type echoProcessor struct {
	*processor.Base
}

func newEcho() *echoProcessor {
	e := &echoProcessor{}
	e.Base = processor.New("Echo", e)
	return e
}

func (e *echoProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if in, ok := f.(*frames.InputAudioRawFrame); ok {
		out := frames.NewOutputAudioRawFrame(in.Audio, in.SampleRate, in.NumChannels)
		return e.PushFrame(ctx, out, processor.Downstream)
	}
	return e.PushFrame(ctx, f, dir)
}

// TestWebRTCEchoLoopback runs the full transport between two in-process Pion
// peers: a client sends Opus audio, the jargo echo pipeline decodes, echoes and
// re-encodes it, and the client must receive audio back.
func TestWebRTCEchoLoopback(t *testing.T) {
	// Server side: a jargo connection with no STUN (host candidates only).
	server, err := pionrtc.NewConnection(pionrtc.WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	defer server.Close()

	// Client side: a plain Pion peer that sends and receives Opus.
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

	gotAudio := make(chan struct{})
	var once sync.Once
	client.OnTrack(func(tr *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		go func() {
			for {
				pkt, _, err := tr.ReadRTP()
				if err != nil {
					return
				}
				if len(pkt.Payload) > 0 {
					once.Do(func() { close(gotAudio) })
				}
			}
		}()
	})

	// Signaling: client offer -> server answer -> client.
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

	// Run the echo pipeline on the server connection.
	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	tr := pionrtc.NewTransport(server, params)
	task := pipeline.NewTask(pipeline.New(tr.Input(), newEcho(), tr.Output()), pipeline.TaskParams{
		AudioInSampleRate:  opus.SampleRate,
		AudioOutSampleRate: opus.SampleRate,
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	taskDone := make(chan struct{})
	go func() { _ = task.Run(ctx); close(taskDone) }()

	// Client streams Opus audio until the test ends.
	enc, err := opus.NewEncoder(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	go func() {
		ticker := time.NewTicker(opus.FrameDuration)
		defer ticker.Stop()
		phase := 0.0
		for {
			select {
			case <-stop:
				return
			case <-ticker.C:
				packet, err := enc.Encode(sine(&phase))
				if err != nil {
					return
				}
				_ = clientTrack.WriteSample(media.Sample{Data: packet, Duration: opus.FrameDuration})
			}
		}
	}()

	select {
	case <-gotAudio:
		// Echoed audio crossed the full transport.
	case <-time.After(15 * time.Second):
		t.Fatal("did not receive echoed audio within 15s")
	}

	close(stop)
	cancel()
	select {
	case <-taskDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not shut down")
	}
}

// TestRTVIHandshakeOverDataChannel runs a pipeline with an RTVI processor and a
// real WebRTC data channel: the client sends client-ready and must receive
// bot-ready back over the channel.
func TestRTVIHandshakeOverDataChannel(t *testing.T) {
	server, err := pionrtc.NewConnection(pionrtc.WithICEServers())
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

	dc, err := client.CreateDataChannel("rtvi", nil)
	if err != nil {
		t.Fatal(err)
	}
	botReady := make(chan rtvi.Incoming, 1)
	dc.OnOpen(func() {
		clientReady, _ := json.Marshal(rtvi.Message{
			Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady, ID: "c1",
			Data: map[string]any{"version": rtvi.ProtocolVersion},
		})
		_ = dc.SendText(string(clientReady))
	})
	dc.OnMessage(func(msg webrtc.DataChannelMessage) {
		if in, err := rtvi.ParseIncoming(msg.Data); err == nil && in.Type == rtvi.TypeBotReady {
			select {
			case botReady <- in:
			default:
			}
		}
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
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	tr := pionrtc.NewTransport(server, params)
	task := pipeline.NewTask(
		pipeline.New(tr.Input(), rtvi.NewProcessor(), newEcho(), tr.Output()),
		pipeline.TaskParams{AudioInSampleRate: opus.SampleRate, AudioOutSampleRate: opus.SampleRate},
	)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	taskDone := make(chan struct{})
	go func() { _ = task.Run(ctx); close(taskDone) }()

	select {
	case in := <-botReady:
		if in.ID != "c1" {
			t.Fatalf("bot-ready id = %q, want c1", in.ID)
		}
	case <-time.After(15 * time.Second):
		t.Fatal("did not receive bot-ready over the data channel")
	}

	cancel()
	select {
	case <-taskDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not shut down")
	}
}

// sine fills one 20 ms mono frame of S16LE PCM with a 440 Hz tone, advancing
// phase so successive frames are continuous.
func sine(phase *float64) []byte {
	pcm := make([]byte, opus.FrameBytes(1))
	const step = 2 * math.Pi * 440 / opus.SampleRate
	for i := range opus.FrameSamples {
		v := math.Sin(*phase) * 0.3 * math.MaxInt16
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(int16(v)))
		*phase += step
	}
	return pcm
}
