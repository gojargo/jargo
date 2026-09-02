package observers_test

import (
	"testing"

	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/llm"
)

// The LLM logger reports only what passed through a model, which it recognizes
// by the marker interface. Frames from a switched service leave through the
// switcher, so the switcher has to carry the marker too: without it the logger
// is silent for every pipeline with a failover chain, which is every pipeline
// that has one.
func TestLLMSwitcherIsAnLLMService(t *testing.T) {
	var s any = &pipeline.LLMSwitcher{}
	marker, ok := s.(interface{ LLMService() })
	if !ok {
		t.Fatal("an LLM switcher does not look like an LLM service, so nothing it pushes is logged")
	}
	marker.LLMService()

	// And the thing it stands in for still does, which is what makes the two
	// interchangeable to an observer.
	var base any = &llm.Base{}
	if _, ok := base.(interface{ LLMService() }); !ok {
		t.Error("an LLM service does not carry the marker")
	}
}
