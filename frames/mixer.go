package frames

// MixerControlFrame updates an output transport's audio mixer at runtime — for
// example to change the background track, adjust its volume, or mute it. The
// Settings are interpreted by the mixer implementation. It is a control frame.
type MixerControlFrame struct {
	BaseControlFrame
	// Settings are the mixer settings to apply (e.g. "volume", "enabled").
	Settings map[string]any
}

// NewMixerControlFrame builds a MixerControlFrame carrying settings.
func NewMixerControlFrame(settings map[string]any) *MixerControlFrame {
	return &MixerControlFrame{
		BaseControlFrame: NewBaseControlFrame("MixerControlFrame"),
		Settings:         settings,
	}
}

// Compile-time interface check.
var _ ControlFrame = (*MixerControlFrame)(nil)
