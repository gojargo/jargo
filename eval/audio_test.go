package eval_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/tts"
)

const audioRate = 16000

// fakeSynth is a canned tts.Synthesizer: it emits ~200ms of non-silent PCM for
// any text, so the harness has audio to stream and a downstream detector has
// energy to trip on.
type fakeSynth struct{}

func (fakeSynth) SampleRate() int { return audioRate }

func (fakeSynth) Synthesize(_ context.Context, _ string, emit func(pcm []byte) error) error {
	pcm := make([]byte, audioRate/5*2) // 200ms of 16-bit mono
	for i := range pcm {
		pcm[i] = 0x20 // non-zero → "speech"
	}
	return emit(pcm)
}

// fakeAudioSTT stands in for a VAD + STT stack without the ONNX runtime: it
// watches the streamed mic audio, marks the user speaking on the first non-silent
// frame, and — once a stretch of silence follows — ends the turn and emits a
// canned transcription. That drives the aggregator exactly as a real STT would.
type fakeAudioSTT struct {
	*processor.Base
	speaking bool
	silence  int
}

func newFakeAudioSTT() *fakeAudioSTT {
	s := &fakeAudioSTT{}
	s.Base = processor.New("FakeAudioSTT", s)
	return s
}

func (s *fakeAudioSTT) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := s.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	af, ok := f.(*frames.InputAudioRawFrame)
	if !ok {
		return s.PushFrame(ctx, f, dir)
	}
	if err := s.observe(ctx, af.Audio); err != nil {
		return err
	}
	return s.PushFrame(ctx, af, dir)
}

// observe tracks speech vs silence and drives the turn boundary.
func (s *fakeAudioSTT) observe(ctx context.Context, audio []byte) error {
	if !allZero(audio) {
		s.silence = 0
		if s.speaking {
			return nil
		}
		s.speaking = true
		return s.PushFrame(ctx, frames.NewUserStartedSpeakingFrame(), processor.Downstream)
	}
	if !s.speaking {
		return nil
	}
	s.silence++
	if s.silence < 10 { // wait for ~200ms of trailing silence
		return nil
	}
	return s.endTurn(ctx)
}

// endTurn ends the user's turn and emits a canned transcription.
func (s *fakeAudioSTT) endTurn(ctx context.Context) error {
	s.speaking = false
	s.silence = 0
	if err := s.PushFrame(ctx, frames.NewUserStoppedSpeakingFrame(), processor.Downstream); err != nil {
		return err
	}
	tf := frames.NewTranscriptionFrame("hello from audio", "user", "")
	tf.Finalized = true
	return s.PushFrame(ctx, tf, processor.Downstream)
}

func allZero(b []byte) bool {
	for _, x := range b {
		if x != 0 {
			return false
		}
	}
	return true
}

func buildAudioBot(in, out processor.Processor) *pipeline.Task {
	agg := aggregators.New(frames.NewLLMContext("test"))
	return pipeline.NewTask(pipeline.New(
		in, newFakeAudioSTT(), agg.User(), newFakeLLM(), rtvi.NewProcessor(), out, agg.Assistant(),
	), pipeline.TaskParams{AudioInSampleRate: audioRate})
}

// TestHarnessAudioMode drives the full real input path: the harness synthesizes
// the user turn, streams it as mic audio, and the bot's (fake) VAD/STT turn it
// into speaking + transcription events that reach the assertions.
func TestHarnessAudioMode(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: audio
turns:
  - user: "hello"
    expect:
      - event: user_started_speaking
      - event: user_transcription
        text_contains: "audio"
      - event: llm_response
        text_contains: "you said"
`))
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := eval.Host(ctx, scenario, buildAudioBot, eval.Options{
		UserTTS: tts.New("fakeTTS", fakeSynth{}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}
