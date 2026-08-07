package rtvi

import (
	"fmt"

	"github.com/gojargo/jargo/frames"
)

// ConfigureObserverFrame reconfigures a running Observer. It lets a trusted,
// server-side source adjust what the observer exposes at runtime without baking
// the setting into the bot, where it would apply to every client. Only the
// fields that are set are applied; a nil field leaves the current configuration
// unchanged.
//
// The eval harness pushes this, through the eval-only serializer, to raise the
// function-call report level for the calls a scenario asserts on, so production
// bots can keep the secure default.
type ConfigureObserverFrame struct {
	frames.BaseSystemFrame
	// FunctionCallReportLevel is the per-function report-level map to apply, or
	// nil to leave the observer's current map unchanged.
	FunctionCallReportLevel map[string]FunctionCallReportLevel
	// VADUserSpeakingEnabled turns the raw VAD speaking events on or off, or is
	// nil to leave the observer's current setting unchanged.
	VADUserSpeakingEnabled *bool
}

// NewConfigureObserverFrame builds a ConfigureObserverFrame.
func NewConfigureObserverFrame(
	level map[string]FunctionCallReportLevel, vadUserSpeaking *bool,
) *ConfigureObserverFrame {
	return &ConfigureObserverFrame{
		BaseSystemFrame:         frames.NewBaseSystemFrame("RTVIConfigureObserverFrame"),
		FunctionCallReportLevel: level,
		VADUserSpeakingEnabled:  vadUserSpeaking,
	}
}

// String implements fmt.Stringer.
func (f *ConfigureObserverFrame) String() string {
	return fmt.Sprintf("%s(function_call_report_level: %v, vad_user_speaking: %v)",
		f.Name(), f.FunctionCallReportLevel, f.VADUserSpeakingEnabled)
}
