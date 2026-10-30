package frames

import (
	"fmt"
	"time"
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

// Compile-time interface checks.
var (
	_ SystemFrame     = (*ServiceMetadataFrame)(nil)
	_ ServiceMetadata = (*ServiceMetadataFrame)(nil)
	_ ServiceMetadata = (*LLMServiceMetadataFrame)(nil)
	_ SystemFrame     = (*STTMetadataFrame)(nil)
	_ ServiceMetadata = (*STTMetadataFrame)(nil)
)
