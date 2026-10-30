package llm_test

// The expression an LLM service is taught by the TTS service it feeds.

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/utils/events"
)

func calmHappy() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Delivery:     []expressive.ExpressionTerm{{Name: "calm"}, {Name: "happy"}},
		PauseRangeMS: &[2]int{100, 3000},
	}
}

func laughter() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds:            []expressive.ExpressionTerm{{Name: "laughter"}},
		ServiceVocabulary: map[expressive.UniversalName][]string{expressive.SoundLaugh: {"laughter"}},
	}
}

func appLLM(name string) *llm.Base {
	svc := llm.New(name, plainGenerator{})
	svc.SetSystemInstruction("APP")
	return svc
}

func expressiveParams(policy *expressive.ExpressionPolicy) pipeline.Params {
	return pipeline.Params{EnableExpressiveConversation: true, ExpressionPolicy: policy}
}

func ttsMetadata(expression *expressive.ExpressionCapabilities) frames.Frame {
	f := frames.NewTTSMetadataFrame("tts")
	f.Expression = expression
	return f
}

// runWith starts proc in a pipeline with params, sends down from its start and
// up from its end, as the processors around it would, and returns the frames
// that reached the start of the pipeline going upstream.
func runWith(
	t *testing.T, proc processor.Processor, params pipeline.Params, down, up []frames.Frame,
) []frames.Frame {
	t.Helper()
	var mu sync.Mutex
	var reached []frames.Frame
	task := pipeline.NewWorker(pipeline.New(proc), pipeline.WorkerConfig{
		Params:                params,
		ReachedUpstreamFilter: pipeline.AnyFrame,
	})
	started := make(chan struct{})
	var once sync.Once
	events.On(&task.Registry, pipeline.EventPipelineStarted, func(context.Context, frames.Frame) {
		once.Do(func() { close(started) })
	})
	events.On(&task.Registry, pipeline.EventFrameReachedUpstream, func(_ context.Context, f frames.Frame) {
		mu.Lock()
		reached = append(reached, f)
		mu.Unlock()
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not start")
	}
	for _, f := range down {
		task.QueueFrame(f)
	}
	for _, f := range up {
		task.QueueFrame(f, processor.Upstream)
		time.Sleep(20 * time.Millisecond)
	}
	time.Sleep(100 * time.Millisecond)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), reached...)
}

// sendUp sends frames up to the LLM, as a TTS service downstream of it would.
func sendUp(t *testing.T, svc processor.Processor, params pipeline.Params, up ...frames.Frame) []frames.Frame {
	t.Helper()
	return runWith(t, svc, params, nil, up)
}

// captureWarnings collects what is logged at warning level while run runs.
func captureWarnings(t *testing.T, run func()) []string {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)
	run()
	var lines []string
	for line := range strings.SplitSeq(buf.String(), "\n") {
		if strings.Contains(line, "expression policy") {
			lines = append(lines, line)
		}
	}
	return lines
}

func checkInstruction(t *testing.T, svc *llm.Base, want string) {
	t.Helper()
	if got := svc.SystemInstruction(); got != want {
		t.Errorf("system instruction =\n%q\nwant\n%q", got, want)
	}
}

func TestTheTTSExpressionIsTaughtAfterThePrompt(t *testing.T) {
	svc := appLLM("llm")

	up := sendUp(t, svc, expressiveParams(nil), ttsMetadata(calmHappy()))

	checkInstruction(t, svc, "APP\n\n"+expressive.ExpressionInstructions(calmHappy(), nil, nil))
	if countOf[*frames.TTSMetadataFrame](up) == 0 {
		t.Error("the TTS metadata did not travel on")
	}
}

func TestNothingIsTaughtWithoutExpressiveConversation(t *testing.T) {
	svc := appLLM("llm")

	sendUp(t, svc, pipeline.Params{}, ttsMetadata(calmHappy()))

	checkInstruction(t, svc, "APP")
}

func TestTheLatestTTSReplacesWhatIsTaught(t *testing.T) {
	svc := appLLM("llm")

	sendUp(t, svc, expressiveParams(nil), ttsMetadata(calmHappy()), ttsMetadata(laughter()))

	checkInstruction(t, svc, "APP\n\n"+expressive.ExpressionInstructions(laughter(), nil, nil))
}

func TestATTSThatPerformsNothingIsTaughtNothing(t *testing.T) {
	svc := appLLM("llm")

	sendUp(t, svc, expressiveParams(nil), ttsMetadata(calmHappy()), ttsMetadata(nil))

	checkInstruction(t, svc, "APP")
}

func TestThePolicyShapesWhatIsTaught(t *testing.T) {
	policy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle},
		Delivery: []expressive.ExpressionDelivery{expressive.DeliveryCalm},
	}
	svc := appLLM("llm")

	sendUp(t, svc, expressiveParams(policy), ttsMetadata(calmHappy()))

	checkInstruction(t, svc, "APP\n\n"+expressive.ExpressionInstructions(calmHappy(), policy, nil))
}

func TestEveryLLMInASwitcherIsTaughtThePolicy(t *testing.T) {
	policy := &expressive.ExpressionPolicy{Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle}}
	first, second := appLLM("first"), appLLM("second")
	sw, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{first, second}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Sent up as the TTS service after the switcher would, while the first is active.
	sendUp(t, sw, expressiveParams(policy), ttsMetadata(calmHappy()))

	taught := "APP\n\n" + expressive.ExpressionInstructions(calmHappy(), policy, nil)
	checkInstruction(t, first, taught)
	checkInstruction(t, second, taught)
}

func TestPolicyTagsTheTTSCantPerformAreWarnedAbout(t *testing.T) {
	policy := &expressive.ExpressionPolicy{
		Delivery: []expressive.ExpressionDelivery{"neutral", "calm", "relieved"},
		Sounds:   []expressive.ExpressionSound{"sighs"},
	}

	warnings := captureWarnings(t, func() {
		sendUp(t, appLLM("llm"), expressiveParams(policy), ttsMetadata(calmHappy()))
	})

	// Every service returns to its baseline, so neutral is never left out.
	want := "can't perform <<relieved>>, [[sighs]] from the expression policy"
	if len(warnings) != 1 || !strings.Contains(warnings[0], want) {
		t.Errorf("warnings = %q", warnings)
	}
}

func TestAPolicyTheTTSPerformsNoneOfIsWarnedAbout(t *testing.T) {
	policy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle},
		Sounds:   []expressive.ExpressionSound{"sighs"},
	}

	warnings := captureWarnings(t, func() {
		sendUp(t, appLLM("llm"), expressiveParams(policy), ttsMetadata(nil))
	})

	if len(warnings) != 1 || !strings.Contains(warnings[0], "performs none of the expression policy's tags") {
		t.Errorf("warnings = %q", warnings)
	}
}

func TestAPolicyWarningIsGivenOnceUntilItChanges(t *testing.T) {
	policy := &expressive.ExpressionPolicy{Delivery: []expressive.ExpressionDelivery{"calm", "relieved"}}

	warnings := captureWarnings(t, func() {
		sendUp(t, appLLM("llm"), expressiveParams(policy),
			ttsMetadata(calmHappy()), ttsMetadata(calmHappy()), ttsMetadata(laughter()))
	})

	if len(warnings) != 2 {
		t.Fatalf("warnings = %q, want 2", warnings)
	}
	if !strings.Contains(warnings[0], "can't perform <<relieved>> from") {
		t.Errorf("first warning = %q", warnings[0])
	}
	if !strings.Contains(warnings[1], "can't perform <<calm>>, <<relieved>> from") {
		t.Errorf("second warning = %q", warnings[1])
	}
}

func TestAPolicyWithoutExpressiveConversationIsWarnedAbout(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	defer slog.SetDefault(prev)

	pipeline.NewWorker(pipeline.New(appLLM("llm")), pipeline.WorkerConfig{
		Params: pipeline.Params{ExpressionPolicy: &expressive.ExpressionPolicy{}},
	})

	if !strings.Contains(buf.String(), "ExpressionPolicy has no effect") {
		t.Errorf("logged %q, want a warning that the policy has no effect", buf.String())
	}
}

// countOf is how many frames of a kind fs holds.
func countOf[T frames.Frame](fs []frames.Frame) int {
	n := 0
	for _, f := range fs {
		if _, ok := f.(T); ok {
			n++
		}
	}
	return n
}

func tone() expressive.PerceptionCapabilities {
	return expressive.PerceptionCapabilities{Signals: []expressive.PerceptionTerm{
		{Name: "tone", Labels: []string{"positive", "neutral", "frustrated"}},
	}}
}

func perceptionMetadata() frames.Frame { return frames.NewPerceptionMetadataFrame("tone", tone()) }

func TestThePerceptionUpstreamIsTaughtWithTheTTSExpression(t *testing.T) {
	svc := appLLM("llm")

	runWith(t, svc, expressiveParams(nil), []frames.Frame{perceptionMetadata()}, []frames.Frame{ttsMetadata(calmHappy())})

	perception := tone()
	checkInstruction(t, svc, "APP\n\n"+expressive.ExpressionInstructions(calmHappy(), nil, &perception))
}

func TestPerceptionWithoutTTSExpressionIsTaughtWithoutPresetsOrWarnings(t *testing.T) {
	svc := appLLM("llm")
	policy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle, expressive.GuidanceLine("Apologize once.")},
		Sounds:   []expressive.ExpressionSound{"sighs"},
	}

	warnings := captureWarnings(t, func() {
		runWith(t, svc, expressiveParams(policy), []frames.Frame{perceptionMetadata()}, nil)
	})

	perception := tone()
	checkInstruction(t, svc, "APP\n\n"+expressive.ExpressionInstructions(nil, nil, &perception)+"\n\nApologize once.")
	if len(warnings) != 0 {
		t.Errorf("warnings = %q, want none", warnings)
	}
}

func TestPerceptionIsNotTaughtWithoutExpressiveConversation(t *testing.T) {
	svc := appLLM("llm")

	runWith(t, svc, pipeline.Params{}, []frames.Frame{perceptionMetadata()}, nil)

	checkInstruction(t, svc, "APP")
}

func TestEveryLLMInASwitcherIsTaughtThePerception(t *testing.T) {
	first, second := appLLM("first"), appLLM("second")
	sw, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{first, second}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Sent down as a perception producer before the switcher would, while the first is active.
	runWith(t, sw, expressiveParams(nil), []frames.Frame{perceptionMetadata()}, nil)

	perception := tone()
	taught := "APP\n\n" + expressive.ExpressionInstructions(nil, nil, &perception)
	checkInstruction(t, first, taught)
	checkInstruction(t, second, taught)
}

func taggerMetadata() frames.Frame { return frames.NewExpressionMetadataFrame("tagger") }

func TestATaggerAfterTheLLMLeavesItTaughtNoTags(t *testing.T) {
	policy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle, expressive.GuidanceLine("Apologize once.")},
		Sounds:   []expressive.ExpressionSound{"sighs"},
	}
	svc := appLLM("llm")

	var up []frames.Frame
	warnings := captureWarnings(t, func() {
		up = runWith(t, svc, expressiveParams(policy),
			[]frames.Frame{perceptionMetadata()}, []frames.Frame{taggerMetadata(), ttsMetadata(calmHappy())})
	})

	// The tagger follows the policy's presets; the LLM keeps the developer's own lines.
	perception := tone()
	checkInstruction(t, svc, "APP\n\n"+expressive.ExpressionInstructions(nil, nil, &perception)+"\n\nApologize once.")
	if len(warnings) != 0 {
		t.Errorf("warnings = %q, want none", warnings)
	}
	if countOf[*frames.ExpressionMetadataFrame](up) == 0 {
		t.Error("the tagger's metadata did not travel on")
	}
}

func TestATaggerAnnouncedAfterTheTTSUnteachesItsTags(t *testing.T) {
	svc := appLLM("llm")

	sendUp(t, svc, expressiveParams(nil), ttsMetadata(calmHappy()), taggerMetadata())

	checkInstruction(t, svc, "APP")
}

func TestAnLLMBeforeATaggerWithoutAReadingIsGivenNoGuidance(t *testing.T) {
	policy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle, expressive.GuidanceLine("Apologize once.")},
	}
	svc := appLLM("llm")

	sendUp(t, svc, expressiveParams(policy), taggerMetadata(), ttsMetadata(calmHappy()))

	// With no tags to write and no reading to act on, the LLM has no part to play.
	checkInstruction(t, svc, "APP")
}

func TestEveryLLMInASwitcherLearnsOfTheTagger(t *testing.T) {
	first, second := appLLM("first"), appLLM("second")
	sw, err := pipeline.NewLLMSwitcher([]pipeline.LLMMember{first, second}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Sent up as a tagger and TTS service after the switcher would, while the first is active.
	sendUp(t, sw, expressiveParams(nil), taggerMetadata(), ttsMetadata(calmHappy()))

	checkInstruction(t, first, "APP")
	checkInstruction(t, second, "APP")
}
