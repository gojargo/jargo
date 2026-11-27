package tts_test

import (
	"testing"

	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
)

// wrappedTTS is a provider's own service type around the base.
type wrappedTTS struct {
	*tts.Base
}

// TestWithSelfMakesTheEmbeddingServiceTheProcessor checks a service type that
// embeds the base is the processor the base registers, so frames, observers
// and switchers know the service by the value its constructor returned.
func TestWithSelfMakesTheEmbeddingServiceTheProcessor(t *testing.T) {
	w := &wrappedTTS{}
	w.Base = tts.New("WrappedTTS", erroringSynth{}, tts.WithSelf(w))
	if got := w.Self(); got != processor.Processor(w) {
		t.Fatalf("Self() = %T, want the embedding service", got)
	}

	plain := tts.New("PlainTTS", erroringSynth{})
	if got := plain.Self(); got != processor.Processor(plain) {
		t.Fatalf("Self() = %T, want the base itself without WithSelf", got)
	}
}
