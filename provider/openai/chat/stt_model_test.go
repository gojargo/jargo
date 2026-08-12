package chat

import (
	"testing"

	"github.com/gojargo/jargo/service/stt"
)

// TestSTTMetadataReportsTheModel checks the transcriber names the model it is
// transcribing with, which is what labels the service's metrics and spans.
func TestSTTMetadataReportsTheModel(t *testing.T) {
	tr := &transcriber{cfg: STTConfig{Model: "gpt-4o-mini-transcribe"}}
	if got := tr.Metadata().Model; got != "gpt-4o-mini-transcribe" {
		t.Errorf("Model = %q, want the configured model", got)
	}
	def := &transcriber{cfg: STTConfig{Model: defaultSTTModel}}
	if got := def.Metadata().Model; got != defaultSTTModel {
		t.Errorf("Model = %q, want the default %q", got, defaultSTTModel)
	}
	if tr.Metadata().TTFSP99 != stt.OpenAITTFSP99 {
		t.Errorf("TTFSP99 = %v, want the measured figure", tr.Metadata().TTFSP99)
	}
}
