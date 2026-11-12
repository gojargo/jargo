package tts

import (
	"context"
	"sync"
	"unicode/utf8"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/telemetry/metrics"
)

// usageKey carries the service a RunTTS call is made for, so the provider can
// report what it sends. See StartUsageMetrics.
type usageKey struct{}

// withUsageReporter returns ctx carrying b, for the RunTTS call it is passed to.
func withUsageReporter(ctx context.Context, b *Base) context.Context {
	return context.WithValue(ctx, usageKey{}, b)
}

// StartUsageMetrics reports the characters of text a provider has just sent for
// synthesis, as the TTS usage the service is billed for. A provider calls it
// from RunTTS, with the context RunTTS was given, once the text is sent: after
// the request is accepted, or the message is written to its connection. Text
// that never reached the provider is not counted.
//
// The count is in runes, the unit providers bill in. It is recorded to
// OpenTelemetry, and pushed downstream as a TTSUsageMetricsData when usage
// metrics are enabled. Streaming tokens, the calls are not reported one by one:
// the service reports the whole reply's text once it ends.
//
// A context that did not come from a TTS service's RunTTS call reports nothing.
func StartUsageMetrics(ctx context.Context, text string) {
	if b, ok := ctx.Value(usageKey{}).(*Base); ok {
		b.StartTTSUsageMetrics(ctx, text)
	}
}

// StartTTSUsageMetrics reports text as TTS usage. See StartUsageMetrics, the way
// a provider reaches it from RunTTS.
func (b *Base) StartTTSUsageMetrics(ctx context.Context, text string) {
	if b.isStreamingTokens() {
		return
	}
	b.reportUsage(ctx, text)
}

// reportUsage records text's characters and, with usage metrics on, pushes
// them downstream.
func (b *Base) reportUsage(ctx context.Context, text string) {
	chars := utf8.RuneCountInString(text)
	model := b.model()
	metrics.RecordTTSCharacters(ctx, b.TypeName(), model, int64(chars))
	if !b.UsageMetricsEnabled() {
		return
	}
	base := frames.BaseMetricsData{Processor: b.Name(), Model: model}
	_ = b.PushFrame(ctx, frames.NewMetricsFrame(frames.TTSUsageMetricsData{BaseMetricsData: base, Value: chars}),
		processor.Downstream)
}

// streamedText gathers the text of a reply spoken token by token, whose usage
// is reported once, for the whole reply, when it ends.
type streamedText struct {
	mu   sync.Mutex
	text string
}

func (s *streamedText) add(text string) {
	s.mu.Lock()
	s.text += text
	s.mu.Unlock()
}

// take returns the gathered text and starts over.
func (s *streamedText) take() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	t := s.text
	s.text = ""
	return t
}

// reportStreamedUsage reports the usage of a reply spoken token by token, once
// it has ended.
func (b *Base) reportStreamedUsage(ctx context.Context) {
	if text := b.streamed.take(); text != "" {
		b.reportUsage(ctx, text)
	}
}
