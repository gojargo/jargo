package frames

import "fmt"

// UserTurnRecommendation is a turn-taking strategy a service recommends to the
// user-turn aggregator through a metadata frame. The aggregator adopts the
// recommendation only when the application did not configure its own turn
// strategies, which always win.
type UserTurnRecommendation int

const (
	// UserTurnUnspecified makes no recommendation; the configured or default
	// strategies stay in place.
	UserTurnUnspecified UserTurnRecommendation = iota
	// UserTurnExternal recommends external turn strategies: the service performs
	// its own server-side end-of-turn detection and emits the user speaking
	// frames, so the pipeline relays them rather than running VAD-based turns.
	UserTurnExternal
)

// String implements fmt.Stringer.
func (r UserTurnRecommendation) String() string {
	if r == UserTurnExternal {
		return "external"
	}
	return "unspecified"
}

// ServiceMetadata is implemented by every metadata frame a service broadcasts at
// pipeline start. Downstream processors assert this interface to read the common
// fields; a concrete STTMetadataFrame or LLMServiceMetadataFrame carries more.
type ServiceMetadata interface {
	SystemFrame
	// Service is the name of the service that broadcast the metadata.
	Service() string
	// RecommendedUserTurns is the turn strategy the service recommends, or
	// UserTurnUnspecified.
	RecommendedUserTurns() UserTurnRecommendation
}

// ServiceMetadataFrame is broadcast by a service at pipeline start to share its
// configuration and performance characteristics with downstream processors. It
// is a system frame. LLMServiceMetadataFrame embeds it; STTMetadataFrame (see
// turns.go) carries the same common fields to satisfy ServiceMetadata.
type ServiceMetadataFrame struct {
	BaseSystemFrame
	// ServiceName names the broadcasting service.
	ServiceName string
	// UserTurns is the turn strategy the service recommends; the user aggregator
	// adopts it unless the application configured its own.
	UserTurns UserTurnRecommendation
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

// RecommendedUserTurns implements ServiceMetadata.
func (f *ServiceMetadataFrame) RecommendedUserTurns() UserTurnRecommendation { return f.UserTurns }

// String implements fmt.Stringer.
func (f *ServiceMetadataFrame) String() string {
	return fmt.Sprintf("%s(service: %s, turns: %s)", f.Name(), f.ServiceName, f.UserTurns)
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
		ServiceMetadataFrame: ServiceMetadataFrame{
			BaseSystemFrame: NewBaseSystemFrame("LLMServiceMetadataFrame"),
			ServiceName:     service,
		},
	}
}

// Compile-time interface checks.
var (
	_ SystemFrame     = (*ServiceMetadataFrame)(nil)
	_ ServiceMetadata = (*ServiceMetadataFrame)(nil)
	_ ServiceMetadata = (*LLMServiceMetadataFrame)(nil)
)
