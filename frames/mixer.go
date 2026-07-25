package frames

import "fmt"

// MixerControlFrame is the base for the frames that drive an output transport's
// audio mixer at runtime. Assert this interface to test whether a frame is a
// mixer control; embed MixerControlBase to define one. It is a control frame, so
// a mixer change is ordered against the audio around it.
type MixerControlFrame interface {
	ControlFrame
	isMixerControlFrame()
}

// MixerControlBase is embedded by mixer control frames.
type MixerControlBase struct{ BaseControlFrame }

func (*MixerControlBase) isMixerControlFrame() {}

// MixerUpdateSettingsFrame updates the mixer's settings — for example to change
// the background track or adjust its volume. The Settings are interpreted by the
// mixer implementation.
type MixerUpdateSettingsFrame struct {
	MixerControlBase
	// Settings are the mixer settings to apply (e.g. "volume", "track").
	Settings map[string]any
}

// NewMixerUpdateSettingsFrame builds a MixerUpdateSettingsFrame carrying settings.
func NewMixerUpdateSettingsFrame(settings map[string]any) *MixerUpdateSettingsFrame {
	return &MixerUpdateSettingsFrame{
		MixerControlBase: MixerControlBase{NewBaseControlFrame("MixerUpdateSettingsFrame")},
		Settings:         settings,
	}
}

// String implements fmt.Stringer.
func (f *MixerUpdateSettingsFrame) String() string {
	return fmt.Sprintf("%s(settings: %d)", f.Name(), len(f.Settings))
}

// MixerEnableFrame turns the mixer on or off at runtime, muting or restoring the
// auxiliary audio without changing its settings.
type MixerEnableFrame struct {
	MixerControlBase
	// Enable reports whether the mixer should be enabled.
	Enable bool
}

// NewMixerEnableFrame builds a MixerEnableFrame.
func NewMixerEnableFrame(enable bool) *MixerEnableFrame {
	return &MixerEnableFrame{
		MixerControlBase: MixerControlBase{NewBaseControlFrame("MixerEnableFrame")},
		Enable:           enable,
	}
}

// String implements fmt.Stringer.
func (f *MixerEnableFrame) String() string {
	return fmt.Sprintf("%s(enable: %t)", f.Name(), f.Enable)
}

// Compile-time interface checks.
var (
	_ MixerControlFrame = (*MixerUpdateSettingsFrame)(nil)
	_ MixerControlFrame = (*MixerEnableFrame)(nil)
)
