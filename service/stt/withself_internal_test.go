package stt

import (
	"testing"

	"github.com/gojargo/jargo/processor"
)

// TestWithSelfMakesTheEmbeddingServiceTheProcessor checks a service type that
// embeds a streaming or a segmented service is the processor the service
// registers, so frames, observers and switchers know it by the value its
// constructor returned.
func TestWithSelfMakesTheEmbeddingServiceTheProcessor(t *testing.T) {
	type wrappedStream struct{ *StreamService }
	ws := &wrappedStream{}
	ws.StreamService = NewStream("WrappedSTT", &describingConnector{}, 16000, WithSelf(ws))
	if got := ws.Self(); got != processor.Processor(ws) {
		t.Errorf("stream Self() = %T, want the embedding service", got)
	}

	type wrappedSegment struct{ *SegmentService }
	wg := &wrappedSegment{}
	wg.SegmentService = NewSegment("WrappedSTT", &describingTranscriber{}, 16000, WithSelf(wg))
	if got := wg.Self(); got != processor.Processor(wg) {
		t.Errorf("segment Self() = %T, want the embedding service", got)
	}

	plain := NewSegment("PlainSTT", &describingTranscriber{}, 16000)
	if got := plain.Self(); got != processor.Processor(plain) {
		t.Errorf("Self() = %T, want the service itself without WithSelf", got)
	}
}
