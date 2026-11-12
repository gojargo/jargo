package tts_test

import (
	"context"
	"encoding/binary"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/utils/events"
)

const toneRate = 16000

// toneSynth answers each sentence with lead seconds of silence, then a tone.
type toneSynth struct{ lead float64 }

func (*toneSynth) SampleRate() int { return toneRate }

func (s *toneSynth) RunTTS(_ context.Context, _, _ string, yield func(f frames.Frame) error) error {
	emit := tts.PCMYielder(yield, toneRate)
	if s.lead > 0 {
		if err := emit(make([]byte, 2*int(toneRate*s.lead))); err != nil {
			return err
		}
	}
	n := int(toneRate * 0.2)
	b := make([]byte, 2*n)
	for i := range n {
		v := int16(math.Sin(2*math.Pi*220*float64(i)/toneRate) * 5000)
		binary.LittleEndian.PutUint16(b[2*i:], uint16(v))
	}
	return emit(b)
}

// arrival is a frame reaching the end of the pipeline, with when it did.
type arrival struct {
	at time.Time
	f  frames.Frame
}

// arrivals records every frame reaching the end of a pipeline.
type arrivals struct {
	mu  sync.Mutex
	all []arrival
}

func (r *arrivals) snapshot() []arrival {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]arrival(nil), r.all...)
}

// metricsArrived returns every metric of type T that reached the end, with when
// it did.
func metricsArrived[T frames.MetricsData](r *arrivals) ([]T, []time.Time) {
	var out []T
	var at []time.Time
	for _, a := range r.snapshot() {
		mf, ok := a.f.(*frames.MetricsFrame)
		if !ok {
			continue
		}
		for _, d := range mf.Data {
			if v, ok := d.(T); ok {
				out = append(out, v)
				at = append(at, a.at)
			}
		}
	}
	return out, at
}

// firstArrived reports when the first frame of type T reached the end.
func firstArrived[T frames.Frame](r *arrivals) (time.Time, bool) {
	for _, a := range r.snapshot() {
		if _, ok := a.f.(T); ok {
			return a.at, true
		}
	}
	return time.Time{}, false
}

// runRecorded runs svc in a pipeline with the given metrics switches, recording
// everything that reaches the end. stop ends the run.
func runRecorded(t *testing.T, svc processor.Processor, metrics, usage bool) (*pipeline.Worker, *arrivals, func()) {
	t.Helper()
	noInitial := false
	r := &arrivals{}
	w := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
		Params: pipeline.Params{
			EnableMetrics:           metrics,
			EnableUsageMetrics:      usage,
			SendInitialEmptyMetrics: &noInitial,
		},
	})
	events.On(&w.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		r.mu.Lock()
		r.all = append(r.all, arrival{time.Now(), f})
		r.mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- w.Run(context.Background()) }()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			w.StopWhenDone()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Error("the pipeline did not stop")
			}
		})
	}
	t.Cleanup(stop)
	return w, r, stop
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

// TestTTFBAndTTFAArePushedAtTheFirstAudio checks a reply's TTFB goes out with
// its first audio, and its TTFA once the onset is found, each in a frame of its
// own, rather than when the reply's context finishes. The reply here keeps its
// context open well after the first audio, as an LLM still streaming would.
func TestTTFBAndTTFAArePushedAtTheFirstAudio(t *testing.T) {
	w, r, stop := runRecorded(t, tts.New("FakeTTS", &toneSynth{lead: 0.1}), true, false)

	w.QueueFrame(frames.NewLLMFullResponseStartFrame())
	w.QueueFrame(frames.NewLLMTextFrame("Hello there. How"))
	waitFor(t, "the first audio", func() bool { _, ok := firstArrived[*frames.TTSAudioRawFrame](r); return ok })
	time.Sleep(400 * time.Millisecond) // The LLM is still producing the rest of the reply.
	w.QueueFrame(frames.NewLLMTextFrame(" are you?"))
	w.QueueFrame(frames.NewLLMFullResponseEndFrame())
	waitFor(t, "the stop frame", func() bool { _, ok := firstArrived[*frames.TTSStoppedFrame](r); return ok })
	stop()

	audio, _ := firstArrived[*frames.TTSAudioRawFrame](r)
	ttfb, ttfbAt := metricsArrived[frames.TTFBMetricsData](r)
	ttfa, ttfaAt := metricsArrived[frames.TTFAMetricsData](r)
	if len(ttfb) != 1 || len(ttfa) != 1 {
		t.Fatalf("got %d TTFB and %d TTFA, want one of each", len(ttfb), len(ttfa))
	}
	if lag := ttfbAt[0].Sub(audio); lag > 200*time.Millisecond {
		t.Errorf("the TTFB arrived %s after the first audio, want it with the first audio", lag)
	}
	if lag := ttfaAt[0].Sub(audio); lag > 200*time.Millisecond {
		t.Errorf("the TTFA arrived %s after the first audio, want it once the onset is found", lag)
	}
	if ls := ttfa[0].LeadingSilence; (ls - 100*time.Millisecond).Abs() > 20*time.Millisecond {
		t.Errorf("leading silence = %s, want about 100ms", ls)
	}
	if ttfa[0].TTFA != ttfa[0].TTFB+ttfa[0].LeadingSilence {
		t.Errorf("ttfa %s is not ttfb %s plus the leading silence %s", ttfa[0].TTFA, ttfa[0].TTFB, ttfa[0].LeadingSilence)
	}
}

// TestInterruptedReplyStillReportsTTFBAndTTFA checks a reply cut off while its
// first sentence plays has reported its TTFB and TTFA already.
func TestInterruptedReplyStillReportsTTFBAndTTFA(t *testing.T) {
	w, r, stop := runRecorded(t, tts.New("FakeTTS", &toneSynth{}), true, false)

	w.QueueFrame(frames.NewLLMFullResponseStartFrame())
	w.QueueFrame(frames.NewLLMTextFrame("Hello there. How"))
	waitFor(t, "the first audio", func() bool { _, ok := firstArrived[*frames.TTSAudioRawFrame](r); return ok })
	time.Sleep(100 * time.Millisecond)
	w.QueueFrame(frames.NewInterruptionFrame())
	time.Sleep(200 * time.Millisecond)
	stop()

	ttfb, _ := metricsArrived[frames.TTFBMetricsData](r)
	ttfa, _ := metricsArrived[frames.TTFAMetricsData](r)
	if len(ttfb) != 1 || len(ttfa) != 1 {
		t.Errorf("got %d TTFB and %d TTFA for a reply interrupted after its first audio, want 1 and 1", len(ttfb), len(ttfa))
	}
}

// stalledSynth never answers: it waits until the call is given up on.
type stalledSynth struct{}

func (stalledSynth) SampleRate() int { return toneRate }

func (stalledSynth) RunTTS(ctx context.Context, _, _ string, _ func(f frames.Frame) error) error {
	<-ctx.Done()
	return ctx.Err()
}

// A reply interrupted before its first audio still reports its time to first
// byte, measured as far as it got: the interruption stops the clock.
func TestReplyInterruptedBeforeItsAudioReportsTTFB(t *testing.T) {
	w, r, stop := runRecorded(t, tts.New("FakeTTS", stalledSynth{}), true, false)

	w.QueueFrame(frames.NewTTSSpeakFrame("Hello there."))
	time.Sleep(150 * time.Millisecond)
	w.QueueFrame(frames.NewInterruptionFrame())
	waitFor(t, "the TTFB of the interrupted reply", func() bool {
		ttfb, _ := metricsArrived[frames.TTFBMetricsData](r)
		return len(ttfb) > 0
	})
	time.Sleep(100 * time.Millisecond)
	stop()

	ttfb, _ := metricsArrived[frames.TTFBMetricsData](r)
	if len(ttfb) != 1 {
		t.Fatalf("got %d TTFB reports, want 1", len(ttfb))
	}
	if ttfb[0].Value < 100*time.Millisecond {
		t.Errorf("TTFB = %v, want the time until the interruption", ttfb[0].Value)
	}
}
