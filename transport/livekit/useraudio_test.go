package livekit

import (
	"context"
	"encoding/binary"
	"math"
	"math/cmplx"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/utils/events"
)

// constPCM is constant-valued 16-bit mono PCM at 48 kHz, ms long.
func constPCM(value int16, ms int) []byte {
	pcm := make([]byte, 48000*ms/1000*2)
	for i := 0; i < len(pcm); i += 2 {
		binary.LittleEndian.PutUint16(pcm[i:], uint16(value))
	}
	return pcm
}

func pcmSamples(pcm []byte) []int16 {
	out := make([]int16, len(pcm)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(pcm[2*i:]))
	}
	return out
}

func allEqual(pcm []byte, want int16) bool {
	for _, s := range pcmSamples(pcm) {
		if s != want {
			return false
		}
	}
	return true
}

// mixerHeardFrom is a mixer that has already heard from participants.
func mixerHeardFrom(maxWaitMs int, participants ...string) *participantAudioMixer {
	m := newParticipantAudioMixer(48000, 1, mixChunkMs, maxWaitMs)
	for _, p := range participants {
		m.add(p, nil)
	}
	return m
}

// Participants' audio is mixed into one stream that keeps their pace.

func TestMixerPassesASingleParticipantThroughInChunks(t *testing.T) {
	chunks := mixerHeardFrom(mixMaxWaitMs).add("alice", constPCM(100, 30))
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want 3", len(chunks))
	}
	for _, c := range chunks {
		if len(c) != 960 || !allEqual(c, 100) {
			t.Errorf("chunk of %d bytes, want 960 bytes of 100", len(c))
		}
	}
}

func TestMixerHoldsPartialChunksUntilComplete(t *testing.T) {
	m := mixerHeardFrom(mixMaxWaitMs)
	if got := m.add("alice", constPCM(100, 5)); len(got) != 0 {
		t.Fatalf("chunks from 5 ms = %d, want none", len(got))
	}
	if got := m.add("alice", constPCM(100, 5)); len(got) != 1 {
		t.Fatalf("chunks from 10 ms = %d, want 1", len(got))
	}
}

func TestMixerSumsParticipantsAndKeepsRealTime(t *testing.T) {
	m := mixerHeardFrom(mixMaxWaitMs, "alice", "bob")
	var chunks [][]byte
	for range 100 { // one second of 10 ms frames from each participant
		chunks = append(chunks, m.add("alice", constPCM(1000, 10))...)
		chunks = append(chunks, m.add("bob", constPCM(2000, 10))...)
	}
	if len(chunks) != 100 {
		t.Fatalf("chunks = %d, want 100", len(chunks))
	}
	for _, c := range chunks {
		if !allEqual(c, 3000) {
			t.Fatal("a chunk is not the two participants summed")
		}
	}
}

func TestMixerWaitsForEveryParticipantBeforeMixing(t *testing.T) {
	m := mixerHeardFrom(mixMaxWaitMs, "alice", "bob")
	if got := m.add("alice", constPCM(1000, 30)); len(got) != 0 {
		t.Fatalf("chunks before bob sent = %d, want none", len(got))
	}
	chunks := m.add("bob", constPCM(2000, 30))
	if len(chunks) != 3 {
		t.Fatalf("chunks = %d, want 3", len(chunks))
	}
	for _, c := range chunks {
		if !allEqual(c, 3000) {
			t.Fatal("a chunk is not the two participants summed")
		}
	}
}

func TestMixerLeavesOutAParticipantWhoStopsSending(t *testing.T) {
	m := mixerHeardFrom(50, "alice", "bob")

	// Bob is quiet: alice's audio waits for him until 50 ms is buffered, then
	// comes out on its own, and keeps pace without him after that.
	if got := m.add("alice", constPCM(1000, 40)); len(got) != 0 {
		t.Fatalf("chunks after 40 ms = %d, want none", len(got))
	}
	chunks := m.add("alice", constPCM(1000, 10))
	if len(chunks) != 5 {
		t.Fatalf("chunks after 50 ms = %d, want 5", len(chunks))
	}
	for _, c := range chunks {
		if !allEqual(c, 1000) {
			t.Fatal("a chunk is not alice's audio alone")
		}
	}
	if got := m.add("alice", constPCM(1000, 10)); len(got) != 1 {
		t.Fatalf("chunks without bob = %d, want 1", len(got))
	}

	// Bob's audio is mixed in again as soon as it arrives.
	if got := m.add("bob", constPCM(2000, 10)); len(got) != 0 {
		t.Fatalf("chunks from bob alone = %d, want none", len(got))
	}
	chunks = m.add("alice", constPCM(1000, 10))
	if len(chunks) != 1 || !allEqual(chunks[0], 3000) {
		t.Fatalf("chunks = %d, want one with both participants summed", len(chunks))
	}
}

func TestMixerClipsMixedSamples(t *testing.T) {
	m := mixerHeardFrom(mixMaxWaitMs, "alice", "bob")
	m.add("alice", constPCM(30000, 10))
	chunks := m.add("bob", constPCM(30000, 10))
	if len(chunks) != 1 || !allEqual(chunks[0], 32767) {
		t.Fatal("the mixed samples are not clipped to the 16-bit range")
	}
}

// audioRecorder records the input audio frames reaching it.
type audioRecorder struct {
	*processor.Base
	mu  sync.Mutex
	got []frames.InputAudioFrame
}

func newAudioRecorder() *audioRecorder {
	r := &audioRecorder{}
	r.Base = processor.New("AudioRecorder", r)
	return r
}

func (r *audioRecorder) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := r.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if af, ok := f.(frames.InputAudioFrame); ok {
		r.mu.Lock()
		r.got = append(r.got, af)
		r.mu.Unlock()
	}
	return r.PushFrame(ctx, f, dir)
}

func (r *audioRecorder) frames() []frames.InputAudioFrame {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]frames.InputAudioFrame(nil), r.got...)
}

// pushParticipants runs a LiveKit input at 16 kHz in a pipeline, hands it
// every frame, and returns what reached the pipeline.
func pushParticipants(t *testing.T, opts options, audio []participantAudio) []frames.InputAudioFrame {
	t.Helper()
	params := transport.DefaultParams()
	params.AudioInSampleRate = 16000
	params.AudioInPassthrough = true
	in := newInput(&Connection{closed: make(chan struct{})}, params, opts)
	rec := newAudioRecorder()
	w := pipeline.NewWorker(pipeline.New(in, rec), pipeline.WorkerConfig{
		Params: pipeline.Params{AudioInSampleRate: 16000},
	})
	started := make(chan struct{})
	events.On(&w.Registry, pipeline.EventPipelineStarted, func(context.Context, *frames.StartFrame) { close(started) })
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline never started")
	}
	// The input starts reading once it has passed the StartFrame on, so the
	// pipeline can be started before it has.
	var ch chan participantAudio
	for deadline := time.Now().Add(3 * time.Second); ch == nil; {
		in.mu.Lock()
		ch = in.audio
		in.mu.Unlock()
		if ch == nil && time.Now().After(deadline) {
			t.Fatal("the input never started reading")
		}
		time.Sleep(time.Millisecond)
	}
	for _, a := range audio {
		ch <- a
	}
	// Everything handed over has been pushed once the audio goroutine and the
	// pipeline have gone quiet.
	time.Sleep(300 * time.Millisecond)
	w.Cancel(context.Background(), "")
	<-done
	return rec.frames()
}

// twoParticipants is one second of 10 ms frames from each of alice and bob,
// interleaved.
func twoParticipants() []participantAudio {
	var out []participantAudio
	for range 100 {
		out = append(out,
			participantAudio{participant: "alice", pcm: constPCM(1000, 10)},
			participantAudio{participant: "bob", pcm: constPCM(2000, 10)})
	}
	return out
}

// With user tracks on, each participant's audio enters the pipeline as their
// own UserAudioRawFrames.
func TestUserTracksPushEachParticipantsAudio(t *testing.T) {
	got := pushParticipants(t, options{audioInUserTracks: true}, twoParticipants())
	if len(got) == 0 {
		t.Fatal("no audio reached the pipeline")
	}
	users := map[string]bool{}
	for _, f := range got {
		u, ok := f.(*frames.UserAudioRawFrame)
		if !ok {
			t.Fatalf("got a %T, want UserAudioRawFrames", f)
		}
		users[u.UserID] = true
	}
	if len(users) != 2 || !users["alice"] || !users["bob"] {
		t.Errorf("users = %v, want alice and bob", users)
	}
}

// With user tracks off, the participants' audio is one real-time stream.
func TestMixedAudioIsOneRealTimeStream(t *testing.T) {
	got := pushParticipants(t, options{audioInUserTracks: false}, twoParticipants())
	var all []byte
	for _, f := range got {
		in, ok := f.(*frames.InputAudioRawFrame)
		if !ok {
			t.Fatalf("got a %T, want InputAudioRawFrames", f)
		}
		if in.SampleRate != 16000 {
			t.Fatalf("sample rate = %d, want 16000", in.SampleRate)
		}
		all = append(all, in.Audio...)
	}

	// One second from each of two participants is one second of audio, not
	// two (less what the resampler still holds).
	seconds := float64(len(all)) / 2 / 16000
	if seconds <= 0.9 || seconds > 1.0 {
		t.Errorf("mixed audio lasts %.3fs, want just under a second", seconds)
	}

	// The middle of the stream carries both participants' audio.
	samples := pcmSamples(all)
	if len(samples) < 12000 {
		t.Fatalf("only %d samples", len(samples))
	}
	for _, s := range samples[4000:12000] {
		if math.Abs(float64(s)-3000) > 3 {
			t.Fatalf("a sample in the middle is %d, want about 3000", s)
		}
	}
}

// toneFrame is the index-th 10 ms frame of a tone at hz, at 48 kHz.
func toneFrame(hz float64, index int) []byte {
	const n = 480
	pcm := make([]byte, 2*n)
	for i := range n {
		tt := float64(index*n+i) / 48000
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(int16(8000*math.Sin(2*math.Pi*hz*tt))))
	}
	return pcm
}

// magnitude is the windowed spectrum's magnitude at hz, for 16 kHz audio.
func magnitude(pcm []byte, hz float64) float64 {
	samples := pcmSamples(pcm)
	n := len(samples)
	bin := math.Round(hz * float64(n) / 16000)
	var sum complex128
	for i, s := range samples {
		window := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n-1))
		angle := -2 * math.Pi * bin * float64(i) / float64(n)
		sum += complex(float64(s)*window, 0) * cmplx.Exp(complex(0, angle))
	}
	return cmplx.Abs(sum)
}

// Audio from participants speaking at the same time must not bleed together:
// a stream resampler keeps filter state between calls, so one shared between
// them would build each output frame partly from the other's samples.
func TestEachParticipantKeepsOnlyTheirOwnAudio(t *testing.T) {
	tones := map[string]float64{"alice": 440, "bob": 1000}
	var audio []participantAudio
	for i := range 100 { // two participants talking at once, 1 s each
		for _, who := range []string{"alice", "bob"} {
			audio = append(audio, participantAudio{participant: who, pcm: toneFrame(tones[who], i)})
		}
	}
	got := pushParticipants(t, options{audioInUserTracks: true}, audio)

	heard := map[string][]byte{}
	for _, f := range got {
		u, ok := f.(*frames.UserAudioRawFrame)
		if !ok {
			t.Fatalf("got a %T, want UserAudioRawFrames", f)
		}
		heard[u.UserID] = append(heard[u.UserID], u.Audio...)
	}
	for _, pair := range [][2]string{{"alice", "bob"}, {"bob", "alice"}} {
		who, other := pair[0], pair[1]
		pcm := heard[who][len(heard[who])/4/2*2:] // past the resampler's start-up
		leaked := magnitude(pcm, tones[other]) / magnitude(pcm, tones[who])
		if leaked >= 0.01 {
			t.Errorf("%s's audio carries %s's voice (%.0f%%)", who, other, leaked*100)
		}
	}
}

// A participant whose audio track is gone has their resampler released.
func TestAParticipantWhoseAudioTrackIsGoneIsReleased(t *testing.T) {
	in := newInput(&Connection{}, transport.DefaultParams(), options{audioInUserTracks: true})
	in.resamplers = map[string]*resample.Resampler{}
	if _, err := in.resamplerFor("alice"); err != nil {
		t.Fatal(err)
	}

	in.handleAudio(t.Context(), participantAudio{participant: "alice", released: true})

	if _, ok := in.resamplers["alice"]; ok {
		t.Error("alice's resampler is still held after her track went")
	}
}
