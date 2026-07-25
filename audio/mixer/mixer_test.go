package mixer_test

import (
	"encoding/binary"
	"testing"

	"github.com/gojargo/jargo/audio/mixer"
)

// pcm encodes samples as 16-bit little-endian mono PCM, the format the mixer
// works in.
func pcm(samples ...int16) []byte {
	b := make([]byte, 2*len(samples))
	for i, s := range samples {
		binary.LittleEndian.PutUint16(b[2*i:], uint16(s))
	}
	return b
}

// samples decodes 16-bit little-endian PCM back to signed samples.
func samples(b []byte) []int16 {
	out := make([]int16, len(b)/2)
	for i := range out {
		out[i] = int16(binary.LittleEndian.Uint16(b[2*i:]))
	}
	return out
}

// mix runs one chunk through the mixer and returns the decoded result.
func mix(t *testing.T, m *mixer.Loop, in []byte) []int16 {
	t.Helper()
	out, err := m.Mix(t.Context(), in)
	if err != nil {
		t.Fatalf("Mix: %v", err)
	}
	return samples(out)
}

func equal(got []int16, want ...int16) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range want {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestDisabledPassesThrough checks a mixer that is off does not touch the audio,
// which is what lets an app add one and enable it later.
func TestDisabledPassesThrough(t *testing.T) {
	m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(1000, 1000), Volume: 1})
	if got := mix(t, m, pcm(100, 200)); !equal(got, 100, 200) {
		t.Errorf("samples = %v, want the input unchanged", got)
	}
}

// TestEmptyBackgroundPassesThrough checks a mixer with nothing to play is a
// no-op rather than a source of silence.
func TestEmptyBackgroundPassesThrough(t *testing.T) {
	for _, tt := range []struct {
		name string
		bg   []byte
	}{
		{"nil background", nil},
		{"single byte, less than one sample", []byte{0x01}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := mixer.NewLoop(mixer.LoopConfig{Background: tt.bg, Volume: 1, Enabled: true})
			if got := mix(t, m, pcm(100, 200)); !equal(got, 100, 200) {
				t.Errorf("samples = %v, want the input unchanged", got)
			}
		})
	}
}

// TestMixesAtVolume checks the background is scaled before being added, since
// the volume is what keeps it under the bot's speech rather than over it.
func TestMixesAtVolume(t *testing.T) {
	tests := []struct {
		name   string
		volume float64
		want   []int16
	}{
		{"full volume adds the background as-is", 1, []int16{1100, 1200}},
		{"half volume halves it", 0.5, []int16{600, 700}},
		{"silent background leaves the foreground", 0.0001, []int16{100, 200}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := mixer.NewLoop(mixer.LoopConfig{
				Background: pcm(1000, 1000),
				Volume:     tt.volume,
				Enabled:    true,
			})
			if got := mix(t, m, pcm(100, 200)); !equal(got, tt.want...) {
				t.Errorf("samples = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestDefaultVolume checks the documented 0.3 default applies when none is set,
// so an app that forgets the field still gets a background it can talk over.
func TestDefaultVolume(t *testing.T) {
	m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(1000), Enabled: true})
	// 0 + 1000*0.3 = 300.
	if got := mix(t, m, pcm(0)); !equal(got, 300) {
		t.Errorf("samples = %v, want the 0.3 default volume applied", got)
	}
}

// TestClipping checks the sum is clamped to the 16-bit range. Without it the sum
// wraps and a loud passage turns into audible noise rather than distortion.
func TestClipping(t *testing.T) {
	tests := []struct {
		name string
		fg   int16
		bg   int16
		want int16
	}{
		{"positive overflow clamps to max", 32000, 32000, 32767},
		{"negative overflow clamps to min", -32000, -32000, -32768},
		{"exactly at the ceiling", 32767, 0, 32767},
		{"exactly at the floor", -32768, 0, -32768},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(tt.bg), Volume: 1, Enabled: true})
			if got := mix(t, m, pcm(tt.fg)); !equal(got, tt.want) {
				t.Errorf("samples = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestLoopWraps checks the background restarts from the beginning when it runs
// out, which is what makes it a loop rather than a one-shot.
func TestLoopWraps(t *testing.T) {
	m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(10, 20), Volume: 1, Enabled: true})

	// Six silent samples over a two-sample background: 10, 20, 10, 20, 10, 20.
	got := mix(t, m, pcm(0, 0, 0, 0, 0, 0))
	if !equal(got, 10, 20, 10, 20, 10, 20) {
		t.Errorf("samples = %v, want the background looped", got)
	}
}

// TestLoopPositionPersistsAcrossChunks checks the loop keeps its place between
// calls; restarting each chunk would click at every audio-frame boundary.
func TestLoopPositionPersistsAcrossChunks(t *testing.T) {
	m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(10, 20, 30, 40), Volume: 1, Enabled: true})

	if got := mix(t, m, pcm(0, 0)); !equal(got, 10, 20) {
		t.Fatalf("first chunk = %v, want the start of the background", got)
	}
	if got := mix(t, m, pcm(0, 0)); !equal(got, 30, 40) {
		t.Errorf("second chunk = %v, want the background to continue, not restart", got)
	}
}

// TestControl covers the runtime updates a MixerControlFrame carries.
func TestControl(t *testing.T) {
	t.Run("enable and disable", func(t *testing.T) {
		m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(1000), Volume: 1})

		if got := mix(t, m, pcm(0)); !equal(got, 0) {
			t.Fatalf("samples = %v, want silence while disabled", got)
		}
		if err := m.Control(t.Context(), map[string]any{"enabled": true}); err != nil {
			t.Fatalf("Control: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 1000) {
			t.Errorf("samples = %v, want the background once enabled", got)
		}
		if err := m.Control(t.Context(), map[string]any{"enabled": false}); err != nil {
			t.Fatalf("Control: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 0) {
			t.Errorf("samples = %v, want silence once disabled again", got)
		}
	})

	t.Run("volume", func(t *testing.T) {
		m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(1000), Volume: 1, Enabled: true})
		if err := m.Control(t.Context(), map[string]any{"volume": 0.5}); err != nil {
			t.Fatalf("Control: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 500) {
			t.Errorf("samples = %v, want the new volume applied", got)
		}
	})

	t.Run("background swap restarts the loop", func(t *testing.T) {
		m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(10, 20), Volume: 1, Enabled: true})
		mix(t, m, pcm(0)) // advance into the current track

		if err := m.Control(t.Context(), map[string]any{"background": pcm(70, 80)}); err != nil {
			t.Fatalf("Control: %v", err)
		}
		// A new track must start at its beginning, not at the old offset.
		if got := mix(t, m, pcm(0, 0)); !equal(got, 70, 80) {
			t.Errorf("samples = %v, want the new background from its start", got)
		}
	})

	t.Run("unknown and mistyped settings are ignored", func(t *testing.T) {
		m := mixer.NewLoop(mixer.LoopConfig{Background: pcm(1000), Volume: 1, Enabled: true})
		err := m.Control(t.Context(), map[string]any{
			"volume":  "loud", // wrong type
			"enabled": 1,      // wrong type
			"unknown": true,
		})
		if err != nil {
			t.Fatalf("Control: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 1000) {
			t.Errorf("samples = %v, want the settings unchanged", got)
		}
	})
}

// TestStartStop checks the lifecycle hooks are usable no-ops; the transport
// calls them around every session.
func TestStartStop(t *testing.T) {
	m := mixer.NewLoop(mixer.LoopConfig{})
	if err := m.Start(t.Context(), 16000); err != nil {
		t.Errorf("Start: %v", err)
	}
	if err := m.Stop(t.Context()); err != nil {
		t.Errorf("Stop: %v", err)
	}
}
