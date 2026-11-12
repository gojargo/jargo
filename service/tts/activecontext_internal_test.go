package tts

// Which context a provider's unlabeled audio belongs to. Playback takes the
// contexts up on a goroutine of its own, so these tests set the playback cursor
// by hand where it would be had that goroutine run late, and check the audio is
// labeled with the context still taking it.

import "testing"

// TestTheActiveContextIsTheOnePlayingWhileItTakesAudio checks the context
// playing is the active one for as long as more audio is coming for it, even
// with a newer one open behind it.
func TestTheActiveContextIsTheOnePlayingWhileItTakesAudio(t *testing.T) {
	b := New("ActiveTTS", &chunkSynth{})
	b.CreateAudioContext("a")
	b.CreateAudioContext("b")
	b.setPlayingContextID("a")

	if got := b.ActiveAudioContextID(); got != "a" {
		t.Errorf("active context = %q, want the one playing", got)
	}
}

// TestAnEndedContextStillPlayingGivesWayToTheNextOne checks audio arriving
// while playback is still on a context the provider has already ended goes to
// the next context, which is the one it belongs to.
func TestAnEndedContextStillPlayingGivesWayToTheNextOne(t *testing.T) {
	b := New("ActiveTTS", &chunkSynth{})
	b.CreateAudioContext("a")
	b.setPlayingContextID("a")
	b.RemoveAudioContext("a")
	b.CreateAudioContext("b")

	if got := b.ActiveAudioContextID(); got != "b" {
		t.Errorf("active context = %q, want the next one taking audio", got)
	}
}

// TestAnOpenContextIsActiveBeforePlaybackTakesItUp checks a context opened for
// a fixed utterance is active once the utterance has been sent and the turn's
// context put back, though playback has not taken it up yet.
func TestAnOpenContextIsActiveBeforePlaybackTakesItUp(t *testing.T) {
	b := New("ActiveTTS", &chunkSynth{})
	b.CreateAudioContext("speak")

	if got := b.ActiveAudioContextID(); got != "speak" {
		t.Errorf("active context = %q, want the context still taking audio", got)
	}
}

// TestTheTurnContextIsActiveWhenNoneIsOpen checks the turn's context stands in
// at the start of a turn, before any context is open.
func TestTheTurnContextIsActiveWhenNoneIsOpen(t *testing.T) {
	b := New("ActiveTTS", &chunkSynth{})
	b.setTurnContextID("turn")

	if got := b.ActiveAudioContextID(); got != "turn" {
		t.Errorf("active context = %q, want the turn's", got)
	}
}

// TestAnEndedContextIsActiveWhenNothingElseIs checks the context playing is
// still reported once it has ended, when there is no other to report.
func TestAnEndedContextIsActiveWhenNothingElseIs(t *testing.T) {
	b := New("ActiveTTS", &chunkSynth{})
	b.CreateAudioContext("a")
	b.setPlayingContextID("a")
	b.RemoveAudioContext("a")

	if got := b.ActiveAudioContextID(); got != "a" {
		t.Errorf("active context = %q, want the one playing", got)
	}
}
