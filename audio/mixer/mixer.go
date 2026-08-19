// Package mixer mixes background audio into a transport's outgoing audio.
// Background holds a set of named sounds, plays one of them under the bot's
// speech at a configurable volume, and switches between them at runtime through
// a frames.MixerUpdateSettingsFrame. It plugs into an output transport via
// Params.AudioOutMixer.
package mixer

import (
	"context"
	"encoding/binary"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
)

// defaultVolume scales the background when none is configured.
const defaultVolume = 0.4

// Sound is one named piece of background audio.
type Sound struct {
	// PCM is 16-bit mono little-endian audio.
	PCM []byte
	// SampleRate is the rate PCM was recorded at, in Hz. A sound whose rate is
	// not the transport's output rate is dropped when the mixer starts, because
	// playing it anyway would pitch-shift it. 0 means "whatever the transport
	// runs at", which skips the check.
	SampleRate int
}

// Config configures a Background mixer.
type Config struct {
	// Sounds are the background sounds to choose between, by name. The
	// MixerUpdateSettingsFrame "sound" setting switches between them.
	Sounds map[string]Sound
	// Default names the sound to start with. An empty name, or one that is not
	// in Sounds, starts the mixer with nothing to play, and it passes audio
	// through until a "sound" setting names one.
	Default string
	// Volume scales the background (0..1); 0 uses 0.4.
	Volume float64
	// StartDisabled starts the mixer inactive, passing audio through until a
	// MixerEnableFrame enables it. A mixer is otherwise active from the start,
	// since mixing is what it was added to the pipeline to do.
	StartDisabled bool
	// NoLoop plays each sound once instead of on a loop. Once it runs out, the
	// outgoing audio passes through untouched until another sound is selected.
	NoLoop bool
}

// Background is an audio.Mixer that plays a background sound under the bot's
// speech. It mixes into audio the transport is already sending, so the
// background is heard behind whatever the bot is saying.
type Background struct {
	mu sync.Mutex
	// sounds holds only what survived the sample-rate check in Start, so a
	// sound that cannot be played is absent rather than silently wrong.
	sounds  map[string][]byte
	current string
	pos     int
	volume  float64
	enabled bool
	loop    bool

	cfg Config
}

// NewBackground builds a Background mixer.
func NewBackground(cfg Config) *Background {
	v := cfg.Volume
	if v == 0 {
		v = defaultVolume
	}
	return &Background{
		sounds:  map[string][]byte{},
		current: cfg.Default,
		volume:  v,
		enabled: !cfg.StartDisabled,
		loop:    !cfg.NoLoop,
		cfg:     cfg,
	}
}

// Start implements audio.Mixer. It takes up the sounds that can be played at the
// transport's output rate and drops the rest, since a sound at another rate
// would play at the wrong pitch and speed.
func (m *Background) Start(_ context.Context, sampleRate int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.sounds = make(map[string][]byte, len(m.cfg.Sounds))
	for name, s := range m.cfg.Sounds {
		if s.SampleRate != 0 && s.SampleRate != sampleRate {
			slog.Warn("mixer: dropping a sound recorded at another sample rate",
				"sound", name, "rate", s.SampleRate, "want", sampleRate)
			continue
		}
		m.sounds[name] = s.PCM
	}
	m.pos = 0
	return nil
}

// Stop implements audio.Mixer.
func (m *Background) Stop(context.Context) error { return nil }

// Mix blends the current sound into pcm at the current volume when enabled,
// clipping to the 16-bit range, and advances the play position. It returns pcm
// untouched when mixing is off, when no sound is selected, or when a
// non-looping sound has run out.
func (m *Background) Mix(_ context.Context, pcm []byte) ([]byte, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	sound, ok := m.sounds[m.current]
	if !m.enabled || !ok || len(sound) < 2 {
		return pcm, nil
	}

	// Take the next chunk of the sound whole, as long as the outgoing audio.
	// Wrapping part-way through would splice two unrelated points of the
	// waveform together, which is heard as a click.
	n := len(pcm) - len(pcm)%2
	if n == 0 {
		return pcm, nil
	}
	if m.pos+n > len(sound) {
		if !m.loop {
			return pcm, nil
		}
		m.pos = 0
	}
	// A sound shorter than one outgoing chunk cannot fill it even from the
	// start, so there is nothing to mix.
	if m.pos+n > len(sound) {
		return pcm, nil
	}
	chunk := sound[m.pos : m.pos+n]
	m.pos += n

	out := make([]byte, n)
	for i := 0; i+1 < n; i += 2 {
		fg := float64(int16(binary.LittleEndian.Uint16(pcm[i:])))
		bg := float64(int16(binary.LittleEndian.Uint16(chunk[i:])))
		binary.LittleEndian.PutUint16(out[i:], uint16(clamp(fg+bg*m.volume)))
	}
	return out, nil
}

// clamp saturates a mixed sample to the 16-bit range.
func clamp(v float64) int16 {
	switch {
	case v > 32767:
		return 32767
	case v < -32768:
		return -32768
	default:
		return int16(v)
	}
}

// ProcessFrame applies a runtime control frame. A MixerUpdateSettingsFrame
// carries "sound" (string, a name from Config.Sounds), "volume" (float64) and
// "loop" (bool); a MixerEnableFrame turns mixing on or off without disturbing
// the settings.
func (m *Background) ProcessFrame(_ context.Context, f frames.MixerControlFrame) error {
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
func (m *Background) updateSettings(settings map[string]any) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if name, ok := settings["sound"].(string); ok {
		m.changeSound(name)
	}
	if v, ok := settings["volume"].(float64); ok {
		m.volume = v
	}
	if l, ok := settings["loop"].(bool); ok {
		m.loop = l
	}
}

// changeSound selects another sound and restarts it, so a sound asked for is
// heard from its beginning rather than from wherever the last one had reached.
// A name that is not loaded is refused: switching to silence would be
// indistinguishable from a sound that failed to play.
func (m *Background) changeSound(name string) {
	if _, ok := m.sounds[name]; !ok {
		slog.Error("mixer: no such sound", "sound", name)
		return
	}
	m.current = name
	m.pos = 0
}

// Compile-time interface check.
var _ audio.Mixer = (*Background)(nil)
