package pionrtc

import (
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
)

func TestSilenceFill(t *testing.T) {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	tests := []struct {
		name     string
		nextSend time.Time
		now      time.Time
		frames   int
		restart  bool
	}{
		{
			name:     "clock not started restarts",
			nextSend: time.Time{},
			now:      base,
			restart:  true,
		},
		{
			name:     "on time needs no fill",
			nextSend: base,
			now:      base,
		},
		{
			name:     "early needs no fill",
			nextSend: base.Add(10 * time.Millisecond),
			now:      base,
		},
		{
			name:     "late by under a frame needs no fill",
			nextSend: base,
			now:      base.Add(15 * time.Millisecond),
		},
		{
			name:     "late by five frames fills five",
			nextSend: base,
			now:      base.Add(100 * time.Millisecond),
			frames:   5,
		},
		{
			name:     "fill is capped, past which the clock restarts",
			nextSend: base,
			now:      base.Add(maxSilenceFill + opus.FrameDuration),
			restart:  true,
		},
		{
			name:     "a long idle restarts rather than bursting",
			nextSend: base,
			now:      base.Add(30 * time.Second),
			restart:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			frames, restart := silenceFill(tc.now, tc.nextSend)
			if frames != tc.frames || restart != tc.restart {
				t.Fatalf("silenceFill = (%d, %v), want (%d, %v)",
					frames, restart, tc.frames, tc.restart)
			}
		})
	}
}

// A filled gap must advance the timeline by as much wall clock as it covers, or
// the audio after it is timestamped as though the gap never happened.
func TestSilenceFillCoversTheGap(t *testing.T) {
	base := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	gap := 200 * time.Millisecond

	frames, restart := silenceFill(base.Add(gap), base)
	if restart {
		t.Fatal("a gap within the cap should be filled, not restarted")
	}
	if covered := time.Duration(frames) * opus.FrameDuration; covered != gap {
		t.Fatalf("filled %v of a %v gap", covered, gap)
	}
}
