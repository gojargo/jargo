package tts_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

// resampleServiceRate is the rate the service runs at in these tests.
const resampleServiceRate = 24000

// rateChunk is one chunk a provider sends, at the rate it sends it.
type rateChunk struct {
	audio []byte
	rate  int
}

// chunkSynth yields each utterance as the given chunks, pausing between them.
type chunkSynth struct {
	chunks []rateChunk
	delay  time.Duration
}

func (s *chunkSynth) SampleRate() int { return resampleServiceRate }

func (s *chunkSynth) RunTTS(ctx context.Context, _, _ string, yield func(frames.Frame) error) error {
	for _, c := range s.chunks {
		if err := yield(frames.NewTTSAudioRawFrame(c.audio, c.rate, 1)); err != nil {
			return err
		}
		if s.delay > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(s.delay):
			}
		}
	}
	return nil
}

// tone is a 220 Hz tone of n samples at rate.
func tone(rate, n int) []byte {
	out := make([]byte, 2*n)
	for i := range n {
		v := int16(math.Sin(2*math.Pi*220*float64(i)/float64(rate)) * 8000)
		binary.LittleEndian.PutUint16(out[2*i:], uint16(v))
	}
	return out
}

// splitPCM cuts audio into parts chunks of whole samples.
func splitPCM(audio []byte, parts int) [][]byte {
	size := len(audio) / parts / 2 * 2
	var out [][]byte
	for i := 0; i < len(audio); i += size {
		out = append(out, audio[i:min(i+size, len(audio))])
	}
	return out
}

// resampledWhole is a clip resampled in one go: what a context should come
// out as.
func resampledWhole(t *testing.T, clip []byte, rate int) []byte {
	t.Helper()
	r, err := resample.NewWithConfig(rate, resampleServiceRate, 1, resample.Config{ClearAfter: resample.NeverClear})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	return append(r.Process(clip), r.Flush()...)
}

// resampledChunks is the audio one resampler gives for the chunks fed in turn,
// with nothing dropped between them: what a context whose chunks arrive apart
// should come out as. It stands in for the whole clip resampled in one go where
// the chunking would make a difference, since this resampler's output length
// still depends on where the input is cut.
func resampledChunks(t *testing.T, rate int, chunks ...[]byte) []byte {
	t.Helper()
	r, err := resample.NewWithConfig(rate, resampleServiceRate, 1, resample.Config{ClearAfter: resample.NeverClear})
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()
	var out []byte
	for _, c := range chunks {
		out = append(out, r.Process(c)...)
	}
	return append(out, r.Flush()...)
}

// speakResampled speaks one sentence through a TTS service over syn and returns
// every frame that reached the end of the pipeline.
func speakResampled(t *testing.T, syn tts.Synthesizer, after ...frames.Frame) []frames.Frame {
	t.Helper()
	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(tts.New("ResampleTTS", syn)),
		pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		got = append(got, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewTTSSpeakFrame("Hello."))
	for _, f := range after {
		if d, ok := f.(*sleepFrame); ok {
			time.Sleep(d.d)
			continue
		}
		task.QueueFrame(f)
	}
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(got)
}

// sleepFrame stands for a pause between the frames speakResampled queues.
type sleepFrame struct {
	frames.BaseControlFrame
	d time.Duration
}

func audioOf(fs []frames.Frame) []*frames.TTSAudioRawFrame {
	var out []*frames.TTSAudioRawFrame
	for _, f := range fs {
		if a, ok := f.(*frames.TTSAudioRawFrame); ok {
			out = append(out, a)
		}
	}
	return out
}

func joined(as []*frames.TTSAudioRawFrame) []byte {
	var b bytes.Buffer
	for _, a := range as {
		b.Write(a.Audio)
	}
	return b.Bytes()
}

// TestContextAudioAtAnotherRateIsResampledInOneGo checks a context's audio at
// another rate comes out at the service rate, as if the whole clip had been
// resampled at once, however the provider cut it.
func TestContextAudioAtAnotherRateIsResampledInOneGo(t *testing.T) {
	clip := tone(16000, 4000)
	var chunks []rateChunk
	for _, c := range splitPCM(clip, 4) {
		chunks = append(chunks, rateChunk{c, 16000})
	}
	audio := audioOf(speakResampled(t, &chunkSynth{chunks: chunks}))
	for _, a := range audio {
		if a.SampleRate != resampleServiceRate {
			t.Fatalf("a frame came out at %d Hz, want the service rate", a.SampleRate)
		}
	}
	if !bytes.Equal(joined(audio), resampledWhole(t, clip, 16000)) {
		t.Error("the audio differs from the clip resampled in one go")
	}
}

// TestContextAudioPauseLeavesNoGap checks a pause between chunks does not cut a
// gap into the resampled audio.
func TestContextAudioPauseLeavesNoGap(t *testing.T) {
	clip := tone(16000, 4000)
	var chunks []rateChunk
	for _, c := range splitPCM(clip, 2) {
		chunks = append(chunks, rateChunk{c, 16000})
	}
	audio := audioOf(speakResampled(t, &chunkSynth{chunks: chunks, delay: 300 * time.Millisecond}))
	halves := splitPCM(clip, 2)
	if !bytes.Equal(joined(audio), resampledChunks(t, 16000, halves...)) {
		t.Error("the audio differs from the halves resampled back to back")
	}
}

// TestContextAudioTailComesOutBeforeTheStop checks the audio the resampler
// holds goes out ahead of the context's stopped frame.
func TestContextAudioTailComesOutBeforeTheStop(t *testing.T) {
	got := speakResampled(t, &chunkSynth{chunks: []rateChunk{{tone(16000, 4000), 16000}}})
	stopped, lastAudio := -1, -1
	for i, f := range got {
		switch f.(type) {
		case *frames.TTSStoppedFrame:
			if stopped < 0 {
				stopped = i
			}
		case *frames.TTSAudioRawFrame:
			lastAudio = i
		}
	}
	if stopped < 0 || lastAudio > stopped {
		t.Errorf("audio at %d, stopped at %d: want the held audio before the stop", lastAudio, stopped)
	}
	if n := len(audioOf(got)); n != 2 {
		t.Errorf("%d audio frames, want the chunk and the resampler's tail", n)
	}
}

// TestContextAudioRateChangeFlushesThePreviousRate checks audio at a new rate
// first ends the audio at the old one, so each comes out whole.
func TestContextAudioRateChangeFlushesThePreviousRate(t *testing.T) {
	first, second := tone(16000, 2000), tone(22050, 2000)
	audio := audioOf(speakResampled(t, &chunkSynth{chunks: []rateChunk{{first, 16000}, {second, 22050}}}))
	want := append(resampledWhole(t, first, 16000), resampledWhole(t, second, 22050)...)
	if !bytes.Equal(joined(audio), want) {
		t.Error("the audio differs from each rate's clip resampled in one go")
	}
}

// TestContextAudioBackAtTheServiceRateFollowsTheHeldAudio checks audio back at
// the service rate goes out after what the resampler held, untouched.
func TestContextAudioBackAtTheServiceRateFollowsTheHeldAudio(t *testing.T) {
	first, second := tone(16000, 2000), tone(resampleServiceRate, 3000)
	audio := audioOf(speakResampled(t, &chunkSynth{chunks: []rateChunk{{first, 16000}, {second, resampleServiceRate}}}))
	want := append(resampledWhole(t, first, 16000), second...)
	if !bytes.Equal(joined(audio), want) {
		t.Error("the audio differs from the resampled first part followed by the second as it was")
	}
}

// TestContextAudioCutMidSampleIsResampledWhole checks chunks cut in the middle
// of a sample still come out as the whole clip resampled.
func TestContextAudioCutMidSampleIsResampledWhole(t *testing.T) {
	clip := tone(16000, 4000)
	cuts := [][]byte{clip[:1001], clip[1001:4000], clip[4000:6999], clip[6999:]}
	var chunks []rateChunk
	for _, c := range cuts {
		chunks = append(chunks, rateChunk{c, 16000})
	}
	audio := audioOf(speakResampled(t, &chunkSynth{chunks: chunks}))
	// The split samples are carried forward, so the resampler is given whole
	// samples only.
	aligned := [][]byte{clip[:1000], clip[1000:4000], clip[4000:6998], clip[6998:]}
	if !bytes.Equal(joined(audio), resampledChunks(t, 16000, aligned...)) {
		t.Error("the audio differs from the realigned chunks resampled back to back")
	}
}

// TestContextAudioInterruptionDropsTheHeldAudio checks an interruption drops
// what the context's resampler holds, so the next reply starts clean and comes
// out whole.
func TestContextAudioInterruptionDropsTheHeldAudio(t *testing.T) {
	clip := tone(16000, 4000)
	var chunks []rateChunk
	for _, c := range splitPCM(clip, 4) {
		chunks = append(chunks, rateChunk{c, 16000})
	}
	got := speakResampled(t, &chunkSynth{chunks: chunks, delay: 50 * time.Millisecond},
		&sleepFrame{d: 70 * time.Millisecond},
		frames.NewInterruptionFrame(),
		&sleepFrame{d: 200 * time.Millisecond},
		frames.NewTTSSpeakFrame("Again."),
		&sleepFrame{d: 400 * time.Millisecond},
	)
	audio := audioOf(got)
	if len(audio) == 0 {
		t.Fatal("no audio came out")
	}
	last := audio[len(audio)-1].ContextID
	if audio[0].ContextID == last {
		t.Fatal("both replies came out on one context, want the interrupted one and the next")
	}
	var reply []*frames.TTSAudioRawFrame
	for _, a := range audio {
		if a.ContextID == last {
			reply = append(reply, a)
		}
	}
	if !bytes.Equal(joined(reply), resampledWhole(t, clip, 16000)) {
		t.Error("the reply after the interruption differs from its clip resampled in one go")
	}
}
