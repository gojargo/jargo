package eval_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/classifier/zoom/decisions"
	"github.com/gojargo/jargo/eval"
)

func TestZoomClassifierTakesTheModelAndTheSuiteTimeout(t *testing.T) {
	t.Setenv("ZOOM_AI_API_KEY", "key")
	for _, tc := range []struct {
		config map[string]any
		model  string
	}{
		{map[string]any{"model": "zoom-decisions-pro"}, "zoom-decisions-pro"},
		{map[string]any{}, decisions.DefaultModel},
	} {
		built, err := eval.ZoomClassifier(tc.config)
		if err != nil {
			t.Fatal(err)
		}
		c, ok := built.(*decisions.Classifier)
		if !ok || c.Model() != tc.model || c.Client().Timeout() != eval.SuiteClassifierTimeout {
			t.Fatalf("classifier = %T %+v, want %s answering within the suite timeout", built, built, tc.model)
		}
		_ = c.Cleanup(t.Context())
	}
}

func TestZoomClassifierIsRegistered(t *testing.T) {
	t.Setenv("ZOOM_AI_API_KEY", "key")
	c, err := eval.ClassifierFromConfig(&eval.LLMServiceConfig{Factory: "evals.judges.zoom_classifier"}, "judge.eval")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := c.(*decisions.Classifier); !ok {
		t.Fatalf("classifier = %T, want Zoom's Decisions API", c)
	}
	_ = c.Cleanup(t.Context())
}

func TestZoomClassifierAMissingAPIKeyIsAnError(t *testing.T) {
	t.Setenv("ZOOM_AI_API_KEY", "")
	_, err := eval.NewEvalJudgeFromConfig(&eval.JudgeEvalConfig{
		Factory: "evals.judges.zoom_classifier",
	})
	if err == nil || !strings.Contains(err.Error(), "ZOOM_AI_API_KEY") {
		t.Fatalf("err = %v", err)
	}
}
