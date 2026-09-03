package mistral

import (
	"testing"

	"github.com/gojargo/jargo/service/stt"
)

// TestSTTMetadataReportsTheModel checks the connector names the model it is
// transcribing with. The service labels its metrics and spans with it, and the
// usage is priced against it, so an empty one puts the cost against nothing.
func TestSTTMetadataReportsTheModel(t *testing.T) {
	c := newSTTConnector(STTConfig{Model: "voxtral-custom"})
	if got := c.Metadata().Model; got != "voxtral-custom" {
		t.Errorf("Model = %q, want the configured model", got)
	}

	// And the default a caller who names none is given.
	def := newSTTConnector(STTConfig{Model: sttDefaultModel})
	if got := def.Metadata().Model; got != sttDefaultModel {
		t.Errorf("Model = %q, want the default %q", got, sttDefaultModel)
	}

	// The latency the turn strategies size their wait by is still reported.
	if c.Metadata().TTFSP99 != stt.MistralTTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured figure", c.Metadata().TTFSP99)
	}
}
