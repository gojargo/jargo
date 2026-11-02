package frames

import "fmt"

// VolumeFrame sets the volume of the output transport's audio.
//
// The volume holds until the next VolumeFrame, and starts at the transport's
// AudioOutVolume. Audio plays at the volume times the gain set by
// VolumeGainFrame. The frame takes effect in order with the audio before it,
// and an interruption doesn't drop it.
type VolumeFrame struct {
	BaseControlFrame
	UninterruptibleMixin
	// Volume is a multiplier on the audio's samples: 1.0 leaves it unchanged,
	// 0.5 halves it.
	Volume float64
}

// NewVolumeFrame builds a VolumeFrame.
func NewVolumeFrame(volume float64) *VolumeFrame {
	return &VolumeFrame{BaseControlFrame: NewBaseControlFrame("VolumeFrame"), Volume: volume}
}

// String implements fmt.Stringer.
func (f *VolumeFrame) String() string {
	return fmt.Sprintf("%s(volume: %g)", f.Name(), f.Volume)
}

// VolumeGainFrame sets a gain on the output transport's volume.
//
// A gain adjusts the volume for some audio and is reset afterwards, for example
// NewVolumeGainFrame(0.5) before a TTSSpeakFrame and NewVolumeGainFrame(1.0)
// after it, which plays that utterance at half the volume without changing the
// volume set by VolumeFrame. The frame takes effect in order with the audio
// before it, and an interruption doesn't drop it.
type VolumeGainFrame struct {
	BaseControlFrame
	UninterruptibleMixin
	// Gain is a multiplier on the volume: 1.0 leaves it unchanged, 0.5 halves
	// it.
	Gain float64
}

// NewVolumeGainFrame builds a VolumeGainFrame.
func NewVolumeGainFrame(gain float64) *VolumeGainFrame {
	return &VolumeGainFrame{BaseControlFrame: NewBaseControlFrame("VolumeGainFrame"), Gain: gain}
}

// String implements fmt.Stringer.
func (f *VolumeGainFrame) String() string {
	return fmt.Sprintf("%s(gain: %g)", f.Name(), f.Gain)
}
