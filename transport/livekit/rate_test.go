package livekit

import (
	"context"
	"encoding/binary"
	"maps"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	lksdk "github.com/livekit/server-sdk-go/v2"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media"
)

// The codec runs at 48 kHz whatever the pipeline does, so the transport converts
// on the way in, to the rate the pipeline reads at, and on the way out, from the
// rate the pipeline writes at. The test below runs the pipeline at 16 kHz in and
// 24 kHz out, the rates of a typical speech recognizer and TTS service.
//
// It needs no LiveKit server: the room's media path is plain WebRTC, so the
// connection is given the track the SDK publishes on and the track a participant
// sends on, negotiated directly with a Pion peer standing in for that
// participant.

const (
	inRate  = 16000
	outRate = 24000
	toneHz  = 440
)

// rateRecorder records the shape of the audio frames reaching it and echoes them
// back out at the rate they arrived at.
type rateRecorder struct {
	*processor.Base

	mu      sync.Mutex
	rates   map[int]int
	samples int
}

func newRateRecorder() *rateRecorder {
	r := &rateRecorder{rates: map[int]int{}}
	r.Base = processor.New("RateRecorder", r)
	return r
}

func (r *rateRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if af, ok := f.(frames.InputAudioFrame); ok {
		in := af.AudioData()
		r.mu.Lock()
		r.rates[in.SampleRate]++
		r.samples += len(in.Audio) / 2
		r.mu.Unlock()
		out := frames.NewOutputAudioRawFrame(in.Audio, in.SampleRate, in.NumChannels)
		return r.PushFrame(ctx, out, processor.Downstream)
	}
	return r.PushFrame(ctx, f, dir)
}

func (r *rateRecorder) snapshot() (map[int]int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return maps.Clone(r.rates), r.samples
}

// listener decodes what the participant hears, keeping only the audible part of
// it: the gaps between packets say nothing about the pitch.
type listener struct {
	mu     sync.Mutex
	heard  []int16
	want   int
	enough chan struct{}
	once   sync.Once
}

func newListener(want int) *listener {
	return &listener{want: want, enough: make(chan struct{})}
}

func (l *listener) listen(t *testing.T, tr *webrtc.TrackRemote) {
	t.Helper()
	dec, err := opus.NewDecoder(1)
	if err != nil {
		t.Error(err)
		return
	}
	for {
		pkt, _, err := tr.ReadRTP()
		if err != nil {
			return
		}
		if len(pkt.Payload) == 0 {
			continue
		}
		pcm, err := dec.Decode(pkt.Payload)
		if err != nil || peak(pcm) < 3000 {
			continue
		}
		l.mu.Lock()
		for i := 0; i+1 < len(pcm); i += 2 {
			l.heard = append(l.heard, int16(binary.LittleEndian.Uint16(pcm[i:])))
		}
		if len(l.heard) >= l.want {
			l.once.Do(func() { close(l.enough) })
		}
		l.mu.Unlock()
	}
}

func (l *listener) pitch() float64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return dominantPitch(l.heard, opus.SampleRate)
}

// newPeer builds a Pion peer connection with the default codecs.
func newPeer(t *testing.T) *webrtc.PeerConnection {
	t.Helper()
	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		t.Fatal(err)
	}
	pc, err := webrtc.NewAPI(webrtc.WithMediaEngine(m)).NewPeerConnection(webrtc.Configuration{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = pc.Close() })
	return pc
}

// connectParticipant builds a Connection whose media runs to a Pion peer
// standing in for a participant in the room. The connection publishes on the
// SDK's own sample track, as it does in a room; whatever it publishes goes to
// onTrack, and the returned track is the one the participant sends on.
func connectParticipant(
	t *testing.T, onTrack func(*webrtc.TrackRemote),
) (*Connection, *webrtc.TrackLocalStaticSample) {
	t.Helper()
	c := &Connection{remoteCh: make(chan *webrtc.TrackRemote, 1), closed: make(chan struct{})}

	bot := newPeer(t)
	local, err := lksdk.NewLocalSampleTrack(webrtc.RTPCodecCapability{
		MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := bot.AddTrack(local); err != nil {
		t.Fatal(err)
	}
	c.localTrack = local
	bot.OnTrack(func(tr *webrtc.TrackRemote, _ *webrtc.RTPReceiver) {
		c.subscribedAudio(tr, "participant")
	})

	participant := newPeer(t)
	sendTrack, err := webrtc.NewTrackLocalStaticSample(
		webrtc.RTPCodecCapability{MimeType: webrtc.MimeTypeOpus, ClockRate: 48000, Channels: 2},
		"audio", "participant",
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := participant.AddTrack(sendTrack); err != nil {
		t.Fatal(err)
	}
	participant.OnTrack(func(tr *webrtc.TrackRemote, _ *webrtc.RTPReceiver) { go onTrack(tr) })

	negotiate(t, participant, bot)
	return c, sendTrack
}

// negotiate runs one offer and answer between two peers, with ICE gathered up
// front so no trickle is needed.
func negotiate(t *testing.T, offerer, answerer *webrtc.PeerConnection) {
	t.Helper()
	offer, err := offerer.CreateOffer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gathered := webrtc.GatheringCompletePromise(offerer)
	if err := offerer.SetLocalDescription(offer); err != nil {
		t.Fatal(err)
	}
	<-gathered
	if err := answerer.SetRemoteDescription(*offerer.LocalDescription()); err != nil {
		t.Fatal(err)
	}
	answer, err := answerer.CreateAnswer(nil)
	if err != nil {
		t.Fatal(err)
	}
	gathered = webrtc.GatheringCompletePromise(answerer)
	if err := answerer.SetLocalDescription(answer); err != nil {
		t.Fatal(err)
	}
	<-gathered
	if err := offerer.SetRemoteDescription(*answerer.LocalDescription()); err != nil {
		t.Fatal(err)
	}
}

// sendTone streams a 440 Hz tone on track, a 20 ms Opus packet at a time, until
// stop is closed. The returned channel closes once it has stopped.
func sendTone(t *testing.T, track *webrtc.TrackLocalStaticSample, stop <-chan struct{}) <-chan struct{} {
	t.Helper()
	enc, err := opus.NewEncoder(opus.EncoderConfig{Channels: 1})
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
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
				_ = track.WriteSample(media.Sample{Data: packet, Duration: opus.FrameDuration})
			}
		}
	}()
	return done
}

// sine returns one 20 ms frame of a 440 Hz tone at 48 kHz, continuing from
// phase.
func sine(phase *float64) []byte {
	pcm := make([]byte, opus.FrameBytes(1))
	step := 2 * math.Pi * toneHz / opus.SampleRate
	for i := range opus.FrameSamples {
		v := math.Sin(*phase) * 0.3 * math.MaxInt16
		binary.LittleEndian.PutUint16(pcm[i*2:], uint16(int16(v)))
		*phase += step
	}
	return pcm
}

// TestLiveKitConvertsToThePipelineRates sends a tone into a pipeline running at
// 16 kHz in and 24 kHz out, which echoes it back. The pipeline has to receive
// the tone at 16 kHz, labeled as such, and the participant has to hear the echo
// at the pitch it sent. A transport that ignored the rates would label 48 kHz
// audio as the pipeline's, and send 24 kHz audio as though it were 48 kHz, which
// the participant hears an octave up.
func TestLiveKitConvertsToThePipelineRates(t *testing.T) {
	ear := newListener(opus.SampleRate) // one second of echoed tone
	conn, sendTrack := connectParticipant(t, func(tr *webrtc.TrackRemote) { ear.listen(t, tr) })

	params := transport.DefaultParams()
	params.AudioInSampleRate = inRate
	params.AudioOutSampleRate = outRate
	tr := NewTransport(conn, params)
	rec := newRateRecorder()
	task := pipeline.NewWorker(pipeline.New(tr.Input(), rec, tr.Output()), pipeline.WorkerConfig{
		Params: pipeline.Params{AudioInSampleRate: inRate, AudioOutSampleRate: outRate},
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	taskDone := make(chan struct{})
	go func() { _ = task.Run(ctx); close(taskDone) }()

	stop := make(chan struct{})
	sent := sendTone(t, sendTrack, stop)
	select {
	case <-ear.enough:
	case <-time.After(20 * time.Second):
		t.Fatal("did not hear a second of echoed audio within 20s")
	}
	close(stop)
	<-sent

	rates, samples := rec.snapshot()
	if len(rates) != 1 || rates[inRate] == 0 {
		t.Errorf("the pipeline received audio labeled %v, want only %d Hz", rates, inRate)
	}
	// Each 20 ms frame decodes to 960 samples at 48 kHz, and has to reach the
	// pipeline as about 320 at 16 kHz.
	if n := rates[inRate]; n > 0 {
		if per := samples / n; per < 300 || per > 340 {
			t.Errorf("the pipeline received %d samples per frame, want about %d at %d Hz",
				per, inRate/50, inRate)
		}
	}

	if pitch := ear.pitch(); pitch < toneHz*0.85 || pitch > toneHz*1.15 {
		t.Errorf("the echo plays at %.0f Hz, want about %d Hz: the rates were not converted", pitch, toneHz)
	}

	cancel()
	select {
	case <-taskDone:
	case <-time.After(3 * time.Second):
		t.Fatal("task did not shut down")
	}
}

// peak is the largest absolute sample in S16LE PCM.
func peak(pcm []byte) int {
	most := 0
	for i := 0; i+1 < len(pcm); i += 2 {
		v := int(int16(binary.LittleEndian.Uint16(pcm[i:])))
		if v < 0 {
			v = -v
		}
		most = max(most, v)
	}
	return most
}

// dominantPitch estimates the frequency of a tone from its rising zero
// crossings.
func dominantPitch(samples []int16, rate int) float64 {
	if len(samples) < 2 {
		return 0
	}
	crossings := 0
	for i := 1; i < len(samples); i++ {
		if samples[i-1] < 0 && samples[i] >= 0 {
			crossings++
		}
	}
	return float64(crossings) * float64(rate) / float64(len(samples))
}
