// Package mixer mixes background audio into a transport's outgoing audio. Loop
// plays a PCM buffer on a loop, blending it under the bot's speech at a
// configurable volume; its volume and mute state can be changed at runtime
// through a frames.MixerUpdateSettingsFrame. It plugs into an output transport via
// Params.AudioOutMixer.
package mixer

import (
	"context"
	"encoding/binary"
	"sync"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
)

// defaultVolume scales the background when none is configured.
const defaultVolume = 0.3

// LoopConfig configures a Loop mixer.
type LoopConfig struct {
	// Background is 16-bit mono little-endian PCM to loop under the outgoing
	// audio. It should already be at the transport's output sample rate.
	Background []byte
	// Volume scales the background (0..1); 0 uses 0.3.
	Volume float64
	// Enabled starts the mixer active; a disabled mixer passes audio through
	// until a MixerEnableFrame enables it.
	Enabled bool
}

// Loop is an AudioMixer that loops a background PCM buffer, mixing it into each
// outgoing chunk of bot audio at a configurable volume. It mixes into audio the
// transport is already sending, so the background plays under the bot's speech.
type Loop struct {
	mu         sync.Mutex
	background []byte
	pos        int
	volume     float64
	enabled    bool
}

// NewLoop builds a Loop mixer.
func NewLoop(cfg LoopConfig) *Loop {
	v := cfg.Volume
	if v == 0 {
		v = defaultVolume
	}
	return &Loop{background: cfg.Background, volume: v, enabled: cfg.Enabled}
}

// Start implements audio.Mixer.
func (m *Loop) Start(context.Context, int) error { return nil }

// Stop implements audio.Mixer.
func (m *Loop) Stop(context.Context) error { return nil }

// Mix blends the looped background into pcm at the current volume when enabled,
// clipping to the 16-bit range, and advances the loop position.
func (m *Loop) Mix(_ context.Context, pcm []byte) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.enabled || len(m.background) < 2 {
		return pcm, nil
	}
	n := len(pcm) / 2
	out := make([]byte, len(pcm))
	for i := range n {
		fg := int32(int16(binary.LittleEndian.Uint16(pcm[2*i:])))
		bg := int32(float64(int16(binary.LittleEndian.Uint16(m.background[m.pos:]))) * m.volume)
		s := fg + bg
		if s > 32767 {
			s = 32767
		} else if s < -32768 {
			s = -32768
		}
		binary.LittleEndian.PutUint16(out[2*i:], uint16(int16(s)))
		m.pos += 2
		if m.pos+1 >= len(m.background) {
			m.pos = 0
		}
	}
	return out, nil
}

// ProcessFrame applies a runtime control frame. A MixerUpdateSettingsFrame
// carries "volume" (float64), "enabled" (bool) and "background" ([]byte, 16-bit
// mono PCM); a MixerEnableFrame turns mixing on or off without disturbing the
// settings.
func (m *Loop) ProcessFrame(_ context.Context, f frames.MixerControlFrame) error {
	switch fr := f.(type) {
	case *frames.MixerUpdateSettingsFrame:
		m.updateSettings(fr.Settings)
	case *frames.MixerEnableFrame:
		m.mu.Lock()
		m.enabled = fr.Enable
		m.mu.Unlock()
	}
	return nil
}

// updateSettings applies the settings a MixerUpdateSettingsFrame carries,
// leaving any it does not name untouched.
func (m *Loop) updateSettings(settings map[string]any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if v, ok := settings["volume"].(float64); ok {
		m.volume = v
	}
	if e, ok := settings["enabled"].(bool); ok {
		m.enabled = e
	}
	if bg, ok := settings["background"].([]byte); ok {
		m.background = bg
		m.pos = 0
	}
}

// Compile-time interface check.
var _ audio.Mixer = (*Loop)(nil)
