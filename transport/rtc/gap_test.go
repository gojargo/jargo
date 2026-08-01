package rtc

import (
	"strings"
	"testing"
)

// speech is a run of frames of real audio, quiet a run of silence frames sent
// because the queue was empty. Runs are built rather than written out so the
// lengths stay legible next to starvationFrames.
func speech(n int) string { return strings.Repeat("#", n) }
func quiet(n int) string  { return strings.Repeat(".", n) }

// play drives a tracker over a pattern of those runs.
func play(pattern string) gapTracker {
	var g gapTracker
	for _, c := range pattern {
		if c == '#' {
			g.real()
		} else {
			g.quiet()
		}
	}
	return g
}

// pause is longer than any gap can be, so it reads as the talker stopping.
func pause() string { return quiet(starvationFrames + 5) }

func TestGapTrackerChargesOnlySilenceInsideSpeech(t *testing.T) {
	cases := []struct {
		name    string
		pattern string
		starved int64
		gaps    int64
	}{
		// The counter this replaces charged every one of these tails, which made
		// starved track the number of utterances rather than any fault.
		{"an utterance and its trailing silence", speech(5) + pause(), 0, 0},
		{"two utterances, each with a tail", speech(3) + pause() + speech(3) + pause(), 0, 0},
		{"silence before anything is spoken", pause() + speech(5), 0, 0},
		{"a hiccup in the middle of a word", speech(5) + quiet(2) + speech(5), 2, 1},
		{"two hiccups in one utterance", speech(3) + quiet(1) + speech(3) + quiet(2) + speech(3), 3, 2},
		{"a hiccup at the edge of the window", speech(3) + quiet(starvationFrames) + speech(3), starvationFrames, 1},
		{"one frame past the window", speech(3) + quiet(starvationFrames+1) + speech(3), 0, 0},
		{"a hiccup, then the end of the utterance", speech(3) + quiet(2) + speech(3) + pause(), 2, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := play(tc.pattern)
			if g.starved != tc.starved || g.gaps != tc.gaps {
				t.Errorf("%q: starved=%d gaps=%d, want starved=%d gaps=%d",
					tc.pattern, g.starved, g.gaps, tc.starved, tc.gaps)
			}
		})
	}
}

func TestGapTrackerCountsEverySilentFrame(t *testing.T) {
	// silence stays the raw total, independent of the fault accounting, so the
	// two numbers together say how much of the silence was a fault.
	g := play(speech(3) + pause() + speech(3) + quiet(2) + speech(3))
	if want := int64(len(pause()) + 2); g.silence != want {
		t.Errorf("silence=%d, want %d", g.silence, want)
	}
	if g.starved != 2 || g.gaps != 1 {
		t.Errorf("starved=%d gaps=%d, want 2 and 1", g.starved, g.gaps)
	}
}
