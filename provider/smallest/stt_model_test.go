package smallest

import (
	"testing"

	"github.com/gojargo/jargo/service/stt"
)

// TestSTTMetadataReportsTheModel checks the connector names the model it is
// transcribing with, which is what labels the service's metrics and spans.
func TestSTTMetadataReportsTheModel(t *testing.T) {
	c := &connector{cfg: STTConfig{Model: "pulse-custom"}}
	if got := c.Metadata().Model; got != "pulse-custom" {
		t.Errorf("Model = %q, want the configured model", got)
	}
	def := &connector{cfg: STTConfig{Model: defaultSTTModel}}
	if got := def.Metadata().Model; got != defaultSTTModel {
		t.Errorf("Model = %q, want the default %q", got, defaultSTTModel)
	}
	if c.Metadata().TTFSP99 != stt.SmallestTTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured figure", c.Metadata().TTFSP99)
	}
}
