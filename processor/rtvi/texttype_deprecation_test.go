package rtvi

import (
	"testing"

	"github.com/gojargo/jargo/frames"
)

// TestSkipAggregatorTypesIsStillHonored checks the former name of SkipTextTypes
// still keeps text from the client.
func TestSkipAggregatorTypesIsStillHonored(t *testing.T) {
	o := NewObserverWithParams(nil, ObserverParams{
		SkipAggregatorTypes: []frames.TextType{"code"}, //nolint:staticcheck // the deprecated field under test
	})
	if !o.skipsAggregation("code") {
		t.Error("the deprecated SkipAggregatorTypes no longer keeps its types from the client")
	}
}

// TestBotOutputTransformerAggregatedByIsStillHonored checks the former name of
// TextType still selects what a rewrite applies to.
func TestBotOutputTransformerAggregatedByIsStillHonored(t *testing.T) {
	o := NewObserverWithParams(nil, ObserverParams{})
	o.AddBotOutputTransformer(BotOutputTransformer{
		AggregatedBy: frames.TextTypeSentence, //nolint:staticcheck // the deprecated field under test
		Transform:    func(BotOutputText) BotOutputTransformResult { return BotOutputTransformResult{Text: "rewritten"} },
	})
	if got := o.transform(BotOutputText{Text: "hello", TextType: frames.TextTypeSentence}).Text; got != "rewritten" {
		t.Errorf("text = %q, want the rewrite registered under the deprecated field", got)
	}
}
