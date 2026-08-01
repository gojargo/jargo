package frames

import "fmt"

// FilterControlFrame is the base for the frames that drive an input transport's
// audio filter at runtime. Assert this interface to test whether a frame is a
// filter control; embed FilterControlBase to define one. It is a control frame,
// so a filter change is ordered against the audio around it.
type FilterControlFrame interface {
	ControlFrame
	isFilterControlFrame()
}

// FilterControlBase is embedded by filter control frames.
type FilterControlBase struct{ BaseControlFrame }

func (*FilterControlBase) isFilterControlFrame() {}

// FilterUpdateSettingsFrame updates the filter's settings, for example to change
// how strongly it suppresses noise. The Settings are interpreted by the filter
// implementation.
type FilterUpdateSettingsFrame struct {
	FilterControlBase
	// Settings are the filter settings to apply.
	Settings map[string]any
}

// NewFilterUpdateSettingsFrame builds a FilterUpdateSettingsFrame carrying
// settings.
func NewFilterUpdateSettingsFrame(settings map[string]any) *FilterUpdateSettingsFrame {
	return &FilterUpdateSettingsFrame{
		FilterControlBase: FilterControlBase{NewBaseControlFrame("FilterUpdateSettingsFrame")},
		Settings:          settings,
	}
}

// String implements fmt.Stringer.
func (f *FilterUpdateSettingsFrame) String() string {
	return fmt.Sprintf("%s(settings: %d)", f.Name(), len(f.Settings))
}

// FilterEnableFrame turns the filter on or off at runtime, passing incoming
// audio through untouched without tearing the filter down.
type FilterEnableFrame struct {
	FilterControlBase
	// Enable reports whether the filter should be enabled.
	Enable bool
}

// NewFilterEnableFrame builds a FilterEnableFrame.
func NewFilterEnableFrame(enable bool) *FilterEnableFrame {
	return &FilterEnableFrame{
		FilterControlBase: FilterControlBase{NewBaseControlFrame("FilterEnableFrame")},
		Enable:            enable,
	}
}

// String implements fmt.Stringer.
func (f *FilterEnableFrame) String() string {
	return fmt.Sprintf("%s(enable: %t)", f.Name(), f.Enable)
}

// Compile-time interface checks.
var (
	_ FilterControlFrame = (*FilterUpdateSettingsFrame)(nil)
	_ FilterControlFrame = (*FilterEnableFrame)(nil)
)
