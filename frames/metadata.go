package frames

import (
	"fmt"
	"time"

	"github.com/gojargo/jargo/expressive"
)

// ServiceMetadata is implemented by every metadata frame a service broadcasts at
// pipeline start. Downstream processors assert this interface to read the common
// fields; a concrete [STTMetadataFrame] or [LLMServiceMetadataFrame] carries
// more.
type ServiceMetadata interface {
	SystemFrame
	// Service is the name of the service that broadcast the metadata.
	Service() string
	// RecommendedUserTurnStrategies are the user turn strategies the service
	// recommends, or nil.
	RecommendedUserTurnStrategies() any
	// Copy returns the same metadata as a new frame with an id of its own, and
	// of the same concrete kind. It is how one service's metadata is handed to
	// several others and still told apart afterwards.
	Copy() ServiceMetadata
}

// ServiceMetadataFrame is broadcast by a service at pipeline start to share its
// configuration and performance characteristics with downstream processors. It
// is a system frame. [STTMetadataFrame] and [LLMServiceMetadataFrame] embed it to
// add their service-specific fields.
type ServiceMetadataFrame struct {
	BaseSystemFrame
	// ServiceName names the broadcasting service.
	ServiceName string
	// UserTurnStrategies are the user turn strategies the service recommends,
	// for example the external ones a service that does its own server-side
	// end-of-turn detection asks for. The user aggregator applies them unless
	// the application configured its own, which always win. Nil leaves whatever
	// is in place alone.
	//
	// It holds a turns.UserTurnStrategies. The type is not named here because
	// the turn strategies are built on this package, so naming it would be a
	// cycle.
	UserTurnStrategies any
}

// NewServiceMetadataFrame builds a ServiceMetadataFrame for the named service.
func NewServiceMetadataFrame(service string) *ServiceMetadataFrame {
	return &ServiceMetadataFrame{
		BaseSystemFrame: NewBaseSystemFrame("ServiceMetadataFrame"),
		ServiceName:     service,
	}
}

// Service implements ServiceMetadata.
func (f *ServiceMetadataFrame) Service() string { return f.ServiceName }

// RecommendedUserTurnStrategies implements ServiceMetadata.
func (f *ServiceMetadataFrame) RecommendedUserTurnStrategies() any { return f.UserTurnStrategies }

// Copy implements ServiceMetadata.
func (f *ServiceMetadataFrame) Copy() ServiceMetadata {
	c := *f
	c.BaseSystemFrame = NewBaseSystemFrame("ServiceMetadataFrame")
	return &c
}

// String implements fmt.Stringer.
func (f *ServiceMetadataFrame) String() string {
	return fmt.Sprintf("%s(service: %s, recommends turns: %t)",
		f.Name(), f.ServiceName, f.UserTurnStrategies != nil)
}

// LLMServiceMetadataFrame is the metadata an LLM service broadcasts. It reports
// whether the service is a realtime (speech-to-speech) LLM.
type LLMServiceMetadataFrame struct {
	ServiceMetadataFrame
	// Realtime reports whether the broadcasting LLM is a realtime
	// (speech-to-speech) service.
	Realtime bool
}

// NewLLMServiceMetadataFrame builds an LLMServiceMetadataFrame for the named service.
func NewLLMServiceMetadataFrame(service string) *LLMServiceMetadataFrame {
	return &LLMServiceMetadataFrame{
		BaseSystemFrame: NewBaseSystemFrame("LLMServiceMetadataFrame"),
		ServiceName:     service,
	}
}

// Copy implements ServiceMetadata.
func (f *LLMServiceMetadataFrame) Copy() ServiceMetadata {
	c := *f
	c.BaseSystemFrame = NewBaseSystemFrame("LLMServiceMetadataFrame")
	return &c
}

// STTMetadataFrame is the metadata an STT service broadcasts. Turn-stop
// strategies use the p99 time-to-final-speech latency to size their safety-net
// timeouts, and a service that does its own server-side endpointing recommends
// external turn strategies through UserTurnStrategies.
type STTMetadataFrame struct {
	ServiceMetadataFrame
	// TTFSP99Latency is the p99 latency from end of speech to a finalized
	// transcript. Zero means unknown; strategies fall back to a default.
	TTFSP99Latency time.Duration
}

// NewSTTMetadataFrame builds an STTMetadataFrame reporting the p99
// time-to-final-speech latency. Set ServiceName and UserTurns on the returned
// frame to describe the service further.
func NewSTTMetadataFrame(ttfsP99 time.Duration) *STTMetadataFrame {
	return &STTMetadataFrame{
		BaseSystemFrame: NewBaseSystemFrame("STTMetadataFrame"),
		TTFSP99Latency:  ttfsP99,
	}
}

// Copy implements ServiceMetadata.
func (f *STTMetadataFrame) Copy() ServiceMetadata {
	c := *f
	c.BaseSystemFrame = NewBaseSystemFrame("STTMetadataFrame")
	return &c
}

// String implements fmt.Stringer.
func (f *STTMetadataFrame) String() string {
	return fmt.Sprintf("%s(ttfs_p99: %s)", f.Name(), f.TTFSP99Latency)
}

// TTSMetadataFrame is the metadata a TTS service broadcasts at pipeline start.
type TTSMetadataFrame struct {
	ServiceMetadataFrame
	// Expression is the expression tags the service can perform, or nil when
	// it performs none. Part of expressive conversation (alpha).
	Expression *expressive.ExpressionCapabilities
}

// NewTTSMetadataFrame builds a TTSMetadataFrame for the named service.
func NewTTSMetadataFrame(service string) *TTSMetadataFrame {
	return &TTSMetadataFrame{
		BaseSystemFrame: NewBaseSystemFrame("TTSMetadataFrame"),
		ServiceName:     service,
	}
}

// Copy implements ServiceMetadata.
func (f *TTSMetadataFrame) Copy() ServiceMetadata {
	c := *f
	c.BaseSystemFrame = NewBaseSystemFrame("TTSMetadataFrame")
	return &c
}

// String implements fmt.Stringer.
func (f *TTSMetadataFrame) String() string {
	return fmt.Sprintf("%s(service: %s, expression: %t)", f.Name(), f.ServiceName, f.Expression != nil)
}

// PerceptionMetadataFrame is the metadata a processor that perceives how the
// user spoke broadcasts at start. The processor may be an STT service or a
// processor of its own, so the signals it reports have a frame of their own.
// Part of expressive conversation (alpha).
type PerceptionMetadataFrame struct {
	ServiceMetadataFrame
	// Perception is the signals the processor reports.
	Perception expressive.PerceptionCapabilities
}

// NewPerceptionMetadataFrame builds a PerceptionMetadataFrame for the named
// processor, reporting perception.
func NewPerceptionMetadataFrame(service string, perception expressive.PerceptionCapabilities) *PerceptionMetadataFrame {
	return &PerceptionMetadataFrame{
		BaseSystemFrame: NewBaseSystemFrame("PerceptionMetadataFrame"),
		ServiceName:     service,
		Perception:      perception,
	}
}

// Copy implements ServiceMetadata.
func (f *PerceptionMetadataFrame) Copy() ServiceMetadata {
	c := *f
	c.BaseSystemFrame = NewBaseSystemFrame("PerceptionMetadataFrame")
	return &c
}

// String implements fmt.Stringer.
func (f *PerceptionMetadataFrame) String() string {
	return fmt.Sprintf("%s(service: %s, signals: %d)", f.Name(), f.ServiceName, len(f.Perception.Signals))
}

// ExpressionMetadataFrame is the metadata a processor that tags the LLM's
// replies broadcasts, once it can. Part of expressive conversation (alpha).
//
// A tagger sits between the LLM and the TTS service and writes the expression
// tags itself. Upstream, the LLM is then taught no tags, so it writes plain
// words; downstream, the assistant aggregator writes each reply to the context
// as its words alone, keeping the tags out of what the LLM reads.
type ExpressionMetadataFrame struct {
	ServiceMetadataFrame
}

// NewExpressionMetadataFrame builds an ExpressionMetadataFrame for the named
// processor.
func NewExpressionMetadataFrame(service string) *ExpressionMetadataFrame {
	return &ExpressionMetadataFrame{
		BaseSystemFrame: NewBaseSystemFrame("ExpressionMetadataFrame"),
		ServiceName:     service,
	}
}

// Copy implements ServiceMetadata.
func (f *ExpressionMetadataFrame) Copy() ServiceMetadata {
	c := *f
	c.BaseSystemFrame = NewBaseSystemFrame("ExpressionMetadataFrame")
	return &c
}

// Compile-time interface checks.
var (
	_ SystemFrame     = (*ServiceMetadataFrame)(nil)
	_ ServiceMetadata = (*TTSMetadataFrame)(nil)
	_ ServiceMetadata = (*PerceptionMetadataFrame)(nil)
	_ ServiceMetadata = (*ExpressionMetadataFrame)(nil)
	_ ServiceMetadata = (*ServiceMetadataFrame)(nil)
	_ ServiceMetadata = (*LLMServiceMetadataFrame)(nil)
	_ SystemFrame     = (*STTMetadataFrame)(nil)
	_ ServiceMetadata = (*STTMetadataFrame)(nil)
)
