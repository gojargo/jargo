package turns

import (
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/frames"
)

// scriptedAnalyzer is a turn.Analyzer whose verdicts are set by the test: what
// AppendAudio returns, and what AnalyzeEndOfTurn reports.
type scriptedAnalyzer struct {
	slowAnalyzer
	appendState turn.EndOfTurnState
	state       turn.EndOfTurnState
	metrics     *turn.MetricsData
}

func (a *scriptedAnalyzer) AppendAudio([]byte, bool) turn.EndOfTurnState { return a.appendState }

func (a *scriptedAnalyzer) AnalyzeEndOfTurn() (turn.EndOfTurnState, *turn.MetricsData, error) {
	return a.state, a.metrics, nil
}

// turnMetrics returns the turn metrics the strategy pushed.
func (s *spy) turnMetrics() []frames.TurnMetricsData {
	s.mu.Lock()
	defer s.mu.Unlock()
	var out []frames.TurnMetricsData
	for _, f := range s.pushedFrame {
		mf, ok := f.(*frames.MetricsFrame)
		if !ok {
			continue
		}
		for _, d := range mf.Data {
			if tm, ok := d.(frames.TurnMetricsData); ok {
				out = append(out, tm)
			}
		}
	}
	return out
}

// A turn closed by the silence safety net was never judged by the model: the
// analyzer cleared its audio when the silence ran out, so it has no prediction
// to report. The turn still ends, and no turn metrics are invented for it.
func TestTurnAnalyzerSilenceTimeoutReportsNoPrediction(t *testing.T) {
	a := &scriptedAnalyzer{appendState: turn.Complete, state: turn.Incomplete}
	wait := false
	s := NewTurnAnalyzerStop(TurnAnalyzerConfig{Analyzer: a, WaitForTranscript: &wait})
	spy := attachStop(s)

	spy.sendStop(s, frames.NewInputAudioRawFrame(make([]byte, 640), 16000, 1))

	if got := spy.stops(); got != 1 {
		t.Fatalf("the silence safety net ended %d turns, want 1", got)
	}
	if got := spy.turnMetrics(); len(got) != 0 {
		t.Errorf("turn metrics %+v were reported for a turn the model never judged", got)
	}
}

// When the VAD stops and the analyzer has nothing to run the model on, there is
// no prediction and so no turn metrics.
func TestTurnAnalyzerVADStopWithoutAPredictionReportsNone(t *testing.T) {
	a := &scriptedAnalyzer{state: turn.Incomplete}
	s := NewTurnAnalyzerStop(TurnAnalyzerConfig{Analyzer: a})
	spy := attachStop(s)

	spy.sendStop(s, frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	spy.sendStop(s, frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	// Under the spy's mutex, as the controller cleans its strategies up: the
	// safety net's timer runs under it too.
	spy.mu.Lock()
	s.Cleanup()
	spy.mu.Unlock()

	if got := spy.turnMetrics(); len(got) != 0 {
		t.Errorf("turn metrics %+v were reported with no prediction made", got)
	}
}

// A prediction the model made is reported as the analyzer measured it,
// attributed to the analyzer.
func TestTurnAnalyzerReportsTheAnalyzersMetrics(t *testing.T) {
	want := turn.MetricsData{
		Processor:     "BaseSmartTurn",
		Complete:      true,
		Probability:   0.87,
		E2EProcessing: 42 * time.Millisecond,
	}
	a := &scriptedAnalyzer{state: turn.Complete, metrics: &want}
	s := NewTurnAnalyzerStop(TurnAnalyzerConfig{Analyzer: a})
	spy := attachStop(s)

	spy.sendStop(s, frames.NewVADUserStartedSpeakingFrame(0.2, time.Now()))
	spy.sendStop(s, frames.NewVADUserStoppedSpeakingFrame(0.2, time.Now()))
	// Under the spy's mutex, as the controller cleans its strategies up: the
	// safety net's timer runs under it too.
	spy.mu.Lock()
	s.Cleanup()
	spy.mu.Unlock()

	got := spy.turnMetrics()
	if len(got) != 1 {
		t.Fatalf("reported %d turn metrics, want 1", len(got))
	}
	m := got[0]
	if m.Processor != want.Processor || m.Complete != want.Complete ||
		m.Probability != want.Probability || m.E2EProcessing != want.E2EProcessing {
		t.Errorf("reported %+v, want the analyzer's %+v", m, want)
	}
}
