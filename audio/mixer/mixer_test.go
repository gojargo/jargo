package mixer_test

import (
	"encoding/binary"
	"testing"

	"github.com/gojargo/jargo/audio/mixer"
	"github.com/gojargo/jargo/frames"
)

const rate = 16000

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

// one builds a started mixer holding a single sound named "bg", which is what
// most of these tests need.
func one(t *testing.T, cfg mixer.Config, background ...int16) *mixer.Background {
	t.Helper()
	if cfg.Sounds == nil {
		cfg.Sounds = map[string]mixer.Sound{"bg": {PCM: pcm(background...)}}
	}
	if cfg.Default == "" {
		cfg.Default = "bg"
	}
	m := mixer.NewBackground(cfg)
	if err := m.Start(t.Context(), rate); err != nil {
		t.Fatalf("Start: %v", err)
	}
	return m
}

// mix runs one chunk through the mixer and returns the decoded result.
func mix(t *testing.T, m *mixer.Background, in []byte) []int16 {
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

// TestMixesByDefault checks a mixer starts active. It was added to the pipeline
// to mix, so it should not need to be switched on first.
func TestMixesByDefault(t *testing.T) {
	m := one(t, mixer.Config{Volume: 1}, 1000, 1000)
	if got := mix(t, m, pcm(100, 200)); !equal(got, 1100, 1200) {
		t.Errorf("samples = %v, want the background mixed in", got)
	}
}

// TestStartDisabledPassesThrough checks a mixer configured to start off does not
// touch the audio, which is what lets an app add one and enable it later.
func TestStartDisabledPassesThrough(t *testing.T) {
	m := one(t, mixer.Config{Volume: 1, StartDisabled: true}, 1000, 1000)
	if got := mix(t, m, pcm(100, 200)); !equal(got, 100, 200) {
		t.Errorf("samples = %v, want the input unchanged", got)
	}
}

// TestNoSoundPassesThrough checks a mixer with nothing to play is a no-op rather
// than a source of silence.
func TestNoSoundPassesThrough(t *testing.T) {
	for _, tt := range []struct {
		name string
		cfg  mixer.Config
	}{
		{"no sounds at all", mixer.Config{Volume: 1, Sounds: map[string]mixer.Sound{}, Default: "bg"}},
		{"default names a sound that is not there", mixer.Config{
			Volume:  1,
			Sounds:  map[string]mixer.Sound{"other": {PCM: pcm(1000, 1000)}},
			Default: "missing",
		}},
		{"empty sound", mixer.Config{
			Volume:  1,
			Sounds:  map[string]mixer.Sound{"bg": {PCM: nil}},
			Default: "bg",
		}},
		{"sound shorter than one sample", mixer.Config{
			Volume:  1,
			Sounds:  map[string]mixer.Sound{"bg": {PCM: []byte{0x01}}},
			Default: "bg",
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := one(t, tt.cfg)
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
			m := one(t, mixer.Config{Volume: tt.volume}, 1000, 1000)
			if got := mix(t, m, pcm(100, 200)); !equal(got, tt.want...) {
				t.Errorf("samples = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestDefaultVolume checks the documented 0.4 default applies when none is set,
// so an app that forgets the field still gets a background it can talk over.
func TestDefaultVolume(t *testing.T) {
	m := one(t, mixer.Config{}, 1000)
	// 0 + 1000*0.4 = 400.
	if got := mix(t, m, pcm(0)); !equal(got, 400) {
		t.Errorf("samples = %v, want the 0.4 default volume applied", got)
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
			m := one(t, mixer.Config{Volume: 1}, tt.bg)
			if got := mix(t, m, pcm(tt.fg)); !equal(got, tt.want) {
				t.Errorf("samples = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestLoopWraps checks the background restarts from the beginning when the next
// chunk would run past its end, which is what makes it a loop rather than a
// one-shot. It restarts a whole chunk at a time: wrapping part-way through would
// splice two unrelated points of the waveform together, which is heard as a
// click at every wrap.
func TestLoopWraps(t *testing.T) {
	m := one(t, mixer.Config{Volume: 1}, 10, 20, 30, 40)

	if got := mix(t, m, pcm(0, 0)); !equal(got, 10, 20) {
		t.Fatalf("first chunk = %v, want the start of the background", got)
	}
	if got := mix(t, m, pcm(0, 0)); !equal(got, 30, 40) {
		t.Fatalf("second chunk = %v, want the background to continue", got)
	}
	if got := mix(t, m, pcm(0, 0)); !equal(got, 10, 20) {
		t.Errorf("third chunk = %v, want the background looped back to its start", got)
	}
}

// TestNoLoopStopsAtTheEnd checks a one-shot sound falls silent when it runs out
// instead of starting over.
func TestNoLoopStopsAtTheEnd(t *testing.T) {
	m := one(t, mixer.Config{Volume: 1, NoLoop: true}, 10, 20)

	if got := mix(t, m, pcm(0, 0)); !equal(got, 10, 20) {
		t.Fatalf("first chunk = %v, want the background", got)
	}
	if got := mix(t, m, pcm(0, 0)); !equal(got, 0, 0) {
		t.Errorf("second chunk = %v, want the input unchanged once the sound ran out", got)
	}
}

// TestChunkLongerThanTheSound checks a sound too short to fill one outgoing
// chunk is passed over rather than read past its end.
func TestChunkLongerThanTheSound(t *testing.T) {
	m := one(t, mixer.Config{Volume: 1}, 10, 20)
	if got := mix(t, m, pcm(1, 2, 3, 4, 5, 6)); !equal(got, 1, 2, 3, 4, 5, 6) {
		t.Errorf("samples = %v, want the input unchanged", got)
	}
}

// TestSampleRateMismatchDropsTheSound checks a sound recorded at another rate is
// refused. Playing it anyway would shift its pitch and speed.
func TestSampleRateMismatchDropsTheSound(t *testing.T) {
	m := mixer.NewBackground(mixer.Config{
		Volume:  1,
		Default: "bg",
		Sounds: map[string]mixer.Sound{
			"bg": {PCM: pcm(1000, 1000), SampleRate: 8000},
		},
	})
	if err := m.Start(t.Context(), rate); err != nil {
		t.Fatalf("Start: %v", err)
	}
	if got := mix(t, m, pcm(100, 200)); !equal(got, 100, 200) {
		t.Errorf("samples = %v, want the input unchanged", got)
	}
}

// TestUnstatedSampleRateIsAccepted checks a sound that does not declare a rate
// is taken to be at the transport's, which is the common case for audio an app
// already prepared for the pipeline.
func TestUnstatedSampleRateIsAccepted(t *testing.T) {
	m := one(t, mixer.Config{Volume: 1}, 1000)
	if got := mix(t, m, pcm(0)); !equal(got, 1000) {
		t.Errorf("samples = %v, want the background mixed in", got)
	}
}

// TestProcessFrame covers the runtime updates a MixerControlFrame carries.
func TestProcessFrame(t *testing.T) {
	t.Run("enable frame enables and disables", func(t *testing.T) {
		m := one(t, mixer.Config{Volume: 1, StartDisabled: true}, 1000)

		if got := mix(t, m, pcm(0)); !equal(got, 0) {
			t.Fatalf("samples = %v, want silence while disabled", got)
		}
		if err := m.ProcessFrame(t.Context(), frames.NewMixerEnableFrame(true)); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 1000) {
			t.Errorf("samples = %v, want the background once enabled", got)
		}
		if err := m.ProcessFrame(t.Context(), frames.NewMixerEnableFrame(false)); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 0) {
			t.Errorf("samples = %v, want silence once disabled again", got)
		}
	})

	t.Run("enable frame leaves the other settings alone", func(t *testing.T) {
		m := one(t, mixer.Config{Volume: 0.5}, 1000)
		if err := m.ProcessFrame(t.Context(), frames.NewMixerEnableFrame(true)); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 500) {
			t.Errorf("samples = %v, want the configured volume kept", got)
		}
	})

	t.Run("volume", func(t *testing.T) {
		m := one(t, mixer.Config{Volume: 1}, 1000)
		settings := frames.NewMixerUpdateSettingsFrame(map[string]any{"volume": 0.5})
		if err := m.ProcessFrame(t.Context(), settings); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 500) {
			t.Errorf("samples = %v, want the new volume applied", got)
		}
	})

	t.Run("loop can be turned off at runtime", func(t *testing.T) {
		m := one(t, mixer.Config{Volume: 1}, 10, 20)
		settings := frames.NewMixerUpdateSettingsFrame(map[string]any{"loop": false})
		if err := m.ProcessFrame(t.Context(), settings); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		if got := mix(t, m, pcm(0, 0)); !equal(got, 10, 20) {
			t.Fatalf("first chunk = %v, want the background", got)
		}
		if got := mix(t, m, pcm(0, 0)); !equal(got, 0, 0) {
			t.Errorf("second chunk = %v, want no loop back to the start", got)
		}
	})

	t.Run("sound switches track and restarts it", func(t *testing.T) {
		m := one(t, mixer.Config{
			Volume:  1,
			Default: "first",
			Sounds: map[string]mixer.Sound{
				"first":  {PCM: pcm(10, 20)},
				"second": {PCM: pcm(70, 80)},
			},
		})
		mix(t, m, pcm(0, 0)) // advance to the end of the current track

		settings := frames.NewMixerUpdateSettingsFrame(map[string]any{"sound": "second"})
		if err := m.ProcessFrame(t.Context(), settings); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		// A new track must start at its beginning, not at the old offset.
		if got := mix(t, m, pcm(0, 0)); !equal(got, 70, 80) {
			t.Errorf("samples = %v, want the new sound from its start", got)
		}
	})

	t.Run("an unknown sound name is refused", func(t *testing.T) {
		m := one(t, mixer.Config{Volume: 1}, 1000)
		settings := frames.NewMixerUpdateSettingsFrame(map[string]any{"sound": "nope"})
		if err := m.ProcessFrame(t.Context(), settings); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 1000) {
			t.Errorf("samples = %v, want the sound that was playing to keep playing", got)
		}
	})

	t.Run("unknown and mistyped settings are ignored", func(t *testing.T) {
		m := one(t, mixer.Config{Volume: 1}, 1000)
		settings := frames.NewMixerUpdateSettingsFrame(map[string]any{
			"volume":  "loud", // wrong type
			"sound":   1,      // wrong type
			"loop":    "yes",  // wrong type
			"unknown": true,
		})
		if err := m.ProcessFrame(t.Context(), settings); err != nil {
			t.Fatalf("ProcessFrame: %v", err)
		}
		if got := mix(t, m, pcm(0)); !equal(got, 1000) {
			t.Errorf("samples = %v, want the settings unchanged", got)
		}
	})
}

// TestStartStop checks the lifecycle hooks are usable on an empty mixer; the
// transport calls them around every session.
func TestStartStop(t *testing.T) {
	m := mixer.NewBackground(mixer.Config{})
	if err := m.Start(t.Context(), rate); err != nil {
		t.Errorf("Start: %v", err)
	}
	if err := m.Stop(t.Context()); err != nil {
		t.Errorf("Stop: %v", err)
	}
}
