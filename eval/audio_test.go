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
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/tts"
)

const audioRate = 16000

// fakeSynth is a canned tts.Synthesizer: it emits ~200ms of non-silent PCM for
// any text, so the harness has audio to stream and a downstream detector has
// energy to trip on.
type fakeSynth struct{}

func (fakeSynth) SampleRate() int { return audioRate }

func (f fakeSynth) RunTTS(_ context.Context, _, _ string, yield func(fr frames.Frame) error) error {
	emit := tts.PCMYielder(yield, f.SampleRate())
	pcm := make([]byte, audioRate/5*2) // 200ms of 16-bit mono
	for i := range pcm {
		pcm[i] = 0x20 // non-zero → "speech"
	}
	return emit(pcm)
}

// fakeAudioSTT stands in for a VAD + STT stack without the ONNX runtime: it
// watches the streamed mic audio, marks the user speaking on the first non-silent
// frame, and ends the turn once a stretch of silence follows, emitting a canned
// transcription. That drives the aggregator exactly as a real STT would.
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
	if _, ok := f.(*frames.StartFrame); ok {
		// It detects the turn boundaries itself and announces them, so the
		// aggregator is asked to defer to what it reports rather than running
		// its own detection alongside it. Without this the aggregator would open
		// the turn on the transcript at the end of the utterance and barge in
		// there, which is not where the turn began.
		md := frames.NewSTTMetadataFrame(0)
		md.ServiceName = s.Name()
		md.UserTurnStrategies = turns.ExternalStrategies(turns.ExternalStrategiesConfig{})
		if err := s.Broadcast(ctx, func() frames.Frame { return md }); err != nil {
			return err
		}
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

func buildAudioBot(in, out processor.Processor) *pipeline.Worker {
	agg := aggregators.New(frames.NewLLMContext("test"))
	rtviProc := rtvi.NewProcessor()
	return pipeline.NewWorker(pipeline.New(
		rtviProc, in, newFakeAudioSTT(), agg.User(), newFakeLLM(), out, agg.Assistant(),
	), pipeline.WorkerConfig{
		// The observer reports pipeline events; the processor carries them.
		Observers: []pipeline.Observer{rtvi.NewObserver(rtviProc)},
		Params: pipeline.Params{
			AudioInSampleRate: audioRate,
		},
	})
}

// buildSpeakingBot is the audio bot with a speech service in it, so the bot's
// reply is synthesized and reported as spoken. Only then is there a tts_response
// for a scenario to assert on.
func buildSpeakingBot(in, out processor.Processor) *pipeline.Worker {
	agg := aggregators.New(frames.NewLLMContext("test"))
	rtviProc := rtvi.NewProcessor()
	return pipeline.NewWorker(pipeline.New(
		in, newFakeAudioSTT(), agg.User(), newFakeLLM(),
		tts.New("fakeTTS", fakeSynth{}), rtviProc, out, agg.Assistant(),
	), pipeline.WorkerConfig{
		// The observer reports pipeline events; the processor carries them.
		Observers: []pipeline.Observer{rtvi.NewObserver(rtviProc)},
		Params: pipeline.Params{
			AudioInSampleRate: audioRate,
		},
	})
}

// TestHarnessTTSResponse asserts on the text that reached synthesis, which is a
// step past what the model wrote: llm_response passes on a reply the TTS never
// got, and tts_response does not.
func TestHarnessTTSResponse(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: spoken
turns:
  - user: "hello"
    expect:
      - event: llm_response
        text_contains: "you said"
      - event: tts_response
        text_contains: "you said"
`))
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := eval.Host(ctx, scenario, buildSpeakingBot, eval.Options{
		UserTTS: tts.New("userTTS", fakeSynth{}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Passed() {
		t.Fatalf("expected pass, got:\n%s", res)
	}
}

// unspeakable stands in for a chunk that filters down to nothing, the way a lone
// period split off an ellipsis does. The TTS base skips a chunk with nothing left
// to say, so the turn is silent and no text is ever reported as spoken.
type unspeakable struct{}

func (unspeakable) Filter(string) string { return "" }

// buildSilentBot answers every turn and speaks none of it.
func buildSilentBot(in, out processor.Processor) *pipeline.Worker {
	agg := aggregators.New(frames.NewLLMContext("test"))
	rtviProc := rtvi.NewProcessor()
	speech := tts.New("fakeTTS", fakeSynth{})
	speech.SetTextFilters(unspeakable{})
	return pipeline.NewWorker(pipeline.New(
		rtviProc, in, newFakeAudioSTT(), agg.User(), newFakeLLM(), speech, out, agg.Assistant(),
	), pipeline.WorkerConfig{
		Observers: []pipeline.Observer{rtvi.NewObserver(rtviProc)},
		Params: pipeline.Params{
			AudioInSampleRate: audioRate,
		},
	})
}

// TestHarnessTTSResponseCatchesSilentTurn is the fault a tts_response
// expectation exists for: the model answered, nothing speakable reached
// synthesis, and the turn was silent. llm_response passes on it, because the
// text was written; tts_response is what fails.
func TestHarnessTTSResponseCatchesSilentTurn(t *testing.T) {
	scenario, err := eval.Load(writeScenario(t, `
name: silent
turns:
  - user: "hello"
    expect:
      - event: llm_response
        text_contains: "you said"
      - event: tts_response
        text_contains: "you said"
        within_ms: 2000
`))
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := eval.Host(ctx, scenario, buildSilentBot, eval.Options{
		UserTTS: tts.New("userTTS", fakeSynth{}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Passed() {
		t.Fatal("expected a failure for the reply that was never spoken")
	}
	if len(res.Failures) != 1 || res.Failures[0].Event != eval.EventTTSResponse {
		t.Fatalf("expected the one failure to be on tts_response, got:\n%s", res)
	}
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
