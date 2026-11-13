package gradium

import (
	"context"
	"slices"
	"testing"

	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

// Tests for turn detection from Gradium's end-pointing signal, and for the
// settings that tune it.

// turnStream builds a stream with no connection under it, read from a
// connector configured with turn detection on unless an option turns it off.
// With no connection a flush cannot be sent.
func turnStream(opts ...func(*STTConfig)) *sttStream {
	cfg := STTConfig{APIKey: "test-key", EnableTurnDetection: true}
	for _, o := range opts {
		o(&cfg)
	}
	s := newTestStream()
	s.owner = newSTTConnector(cfg)
	s.turnDetection = cfg.EnableTurnDetection
	return s
}

func withCooldown(n int) func(*STTConfig) {
	return func(c *STTConfig) { c.PostFlushCooldownFrames = &n }
}

// step is a step message reading inactivity on the default horizon.
func step(inactivity float64) sttMessage { return stepAt(defaultEOTHorizonS, inactivity) }

func stepAt(horizon, inactivity float64) sttMessage {
	return sttMessage{Type: msgStep, VAD: []vadEntry{{HorizonS: &horizon, InactivityProb: &inactivity}}}
}

// speechOf lists the speech boundaries the results propose.
func speechOf(res []stt.Result) []stt.SpeechState {
	var out []stt.SpeechState
	for _, r := range res {
		if r.Speech != stt.SpeechUnknown {
			out = append(out, r.Speech)
		}
	}
	return out
}

// steps feeds the steps to handleStep in turn and returns what they gave.
func (s *sttStream) steps(inactivity ...float64) []stt.Result {
	var res []stt.Result
	for _, p := range inactivity {
		res = append(res, s.handleStep(step(p))...)
	}
	return res
}

// Ported from upstream (test_gradium_recommends_external_strategies_only_with_turn_detection).
func TestExternalStrategiesAreRecommendedOnlyWithTurnDetection(t *testing.T) {
	on := newSTTConnector(STTConfig{APIKey: "k", EnableTurnDetection: true}).Metadata()
	strategies, ok := on.UserTurnStrategies.(turns.UserTurnStrategies)
	if !ok {
		t.Fatalf("recommended turns = %T, want the external strategies", on.UserTurnStrategies)
	}
	if _, external := strategies.ExternalInterruptions(); !external {
		t.Error("the recommended strategies are not the external ones")
	}

	if off := newSTTConnector(STTConfig{APIKey: "k"}).Metadata(); off.UserTurnStrategies != nil {
		t.Errorf("recommended turns = %v without turn detection, want the defaults left in place",
			off.UserTurnStrategies)
	}
}

// Ported from upstream (test_gradium_turn_detection_settings_are_set_only_with_turn_detection).
func TestTurnDetectionSettingsAreSetOnlyWithTurnDetection(t *testing.T) {
	on := newSTTSettings(STTConfig{EnableTurnDetection: true})
	horizon, _ := on.EOTHorizonS.Value()
	threshold, _ := on.EOTThreshold.Value()
	cooldown, _ := on.PostFlushCooldownFrames.Value()
	if horizon != 3.0 || threshold != 0.5 || cooldown != 8 {
		t.Errorf("with turn detection = (%v, %v, %v), want the defaults (3, 0.5, 8)", horizon, threshold, cooldown)
	}

	off := newSTTSettings(STTConfig{})
	for name, set := range map[string]bool{
		"eot_horizon_s":              hasValue(off.EOTHorizonS),
		"eot_threshold":              hasValue(off.EOTThreshold),
		"post_flush_cooldown_frames": hasValue(off.PostFlushCooldownFrames),
	} {
		if set {
			t.Errorf("%s has a value without turn detection, want none", name)
		}
	}
}

func hasValue[T any](o settings.Opt[T]) bool {
	_, ok := o.Value()
	return ok
}

// Ported from upstream (test_gradium_turn_detection_settings_passed_in_win_over_the_defaults).
func TestTurnDetectionSettingsGivenWinOverTheDefaults(t *testing.T) {
	threshold := 0.7
	s := newSTTSettings(STTConfig{EnableTurnDetection: true, EOTThreshold: &threshold})
	if got, _ := s.EOTThreshold.Value(); got != 0.7 {
		t.Errorf("eot_threshold = %v, want the 0.7 given", got)
	}
	if got, _ := s.EOTHorizonS.Value(); got != 3.0 {
		t.Errorf("eot_horizon_s = %v, want the default 3", got)
	}
}

// Ported from upstream (test_gradium_update_reconnects_only_for_connection_bound_settings).
// The turn detection fields are read as steps arrive, so changing them alone
// reopens nothing, while what the setup message carries reopens the session.
func TestAnUpdateReopensOnlyForConnectionBoundSettings(t *testing.T) {
	tests := []struct {
		name    string
		delta   *STTSettings
		reopens bool
	}{
		{"a turn detection threshold", &STTSettings{EOTThreshold: settings.Set(0.7)}, false},
		{"the decoder delay", &STTSettings{DelayInFrames: settings.Set(8)}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newSTTConnector(STTConfig{APIKey: "k", Model: defaultSTTModel, EnableTurnDetection: true})
			changed, err := settings.Apply(c.live, tt.delta)
			if err != nil {
				t.Fatalf("apply: %v", err)
			}
			reopen, err := c.UpdateSettings(context.Background(), changed)
			if err != nil {
				t.Fatalf("update settings: %v", err)
			}
			if reopen != tt.reopens {
				t.Errorf("reopen = %v, want %v", reopen, tt.reopens)
			}
		})
	}

	// A threshold changed without reopening is what the next step reads.
	s := turnStream()
	if _, err := settings.Apply(s.owner.live, &STTSettings{EOTThreshold: settings.Set(0.95)}); err != nil {
		t.Fatalf("apply: %v", err)
	}
	if _, err := s.owner.UpdateSettings(context.Background(), settings.Changed{"eot_threshold": 0.5}); err != nil {
		t.Fatalf("update settings: %v", err)
	}
	s.steps(0.9)
	if s.phase != phaseIdle {
		t.Errorf("a 0.9 reading armed the detector, want it below the 0.95 threshold now in force")
	}
}

// Ported from upstream (test_gradium_supports_ttfs_only_without_turn_detection).
func TestTTFSAppliesOnlyWithoutTurnDetection(t *testing.T) {
	on := newSTTConnector(STTConfig{APIKey: "k", EnableTurnDetection: true}).Metadata()
	if on.SupportsTTFS == nil || *on.SupportsTTFS {
		t.Error("TTFS is reported as applying with turn detection on")
	}
	off := newSTTConnector(STTConfig{APIKey: "k"}).Metadata()
	if off.SupportsTTFS != nil && !*off.SupportsTTFS {
		t.Error("TTFS is reported as not applying without turn detection")
	}
}

// Ported from upstream (test_gradium_a_vad_stop_flushes_only_without_turn_detection).
// Without turn detection the pipeline's VAD ends the utterance, and the server
// is flushed for it. With turn detection on the end-pointing signal decides, so
// the VAD flushes nothing: the next thing the server sees is the audio.
func TestAVADStopFlushesOnlyWithoutTurnDetection(t *testing.T) {
	for _, td := range []bool{false, true} {
		endpoint, seen := sttServer(t, nil, nil)
		session, err := sttConn(endpoint, func(c *STTConfig) { c.EnableTurnDetection = td }).
			Connect(context.Background(), 16000)
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		s, ok := session.(*sttStream)
		if !ok {
			t.Fatalf("Connect returned a %T", session)
		}
		seen.await(t) // the setup

		if err := s.Finalize(); err != nil {
			t.Fatalf("Finalize: %v", err)
		}
		if err := s.Send(make([]byte, s.chunkBytes)); err != nil {
			t.Fatalf("Send: %v", err)
		}

		next := seen.await(t)
		if td {
			if next[msgType] != msgAudio {
				t.Errorf("with turn detection the server saw %v first, want the audio", next)
			}
		} else if next[msgType] != msgFlush || next["flush_id"] != "1" {
			t.Errorf("without turn detection the server saw %v first, want the flush", next)
		}
		_ = s.Close()
	}
}

// Ported from upstream (test_gradium_turn_detection_the_signal_falling_below_the_threshold_proposes_a_turn_start).
// The service only proposes the start; the strategies own the interruption.
func TestTheSignalFallingBelowTheThresholdProposesATurnStart(t *testing.T) {
	s := turnStream()

	res := s.steps(0.9, 0.2)

	if got := speechOf(res); !slices.Equal(got, []stt.SpeechState{stt.SpeechStarted}) {
		t.Errorf("proposed %v, want one start", got)
	}
	if s.phase != phaseOpen {
		t.Errorf("phase = %v, want the turn open", s.phase)
	}
}

// Ported from upstream (test_gradium_turn_detection_a_signal_that_starts_low_opens_nothing).
// With nothing heard yet the estimate starts low and climbs; that is not
// speech. Only a dip after the signal has read inactive is.
func TestASignalThatStartsLowOpensNothing(t *testing.T) {
	s := turnStream()

	if res := s.steps(0.1, 0.3, 0.45); len(res) != 0 {
		t.Errorf("results = %+v, want nothing proposed", res)
	}
	if s.phase == phaseOpen {
		t.Error("a turn opened on a signal that only ever read low")
	}
}

// Ported from upstream (test_gradium_turn_detection_watches_the_horizon_closest_to_the_setting).
func TestTheHorizonClosestToTheSettingIsWatched(t *testing.T) {
	s := turnStream(func(c *STTConfig) { c.EOTHorizonS = new(1.0) })
	one, three := 1.0, 3.0
	high, low := 0.9, 0.1

	s.handleStep(sttMessage{Type: msgStep, VAD: []vadEntry{
		{HorizonS: &one, InactivityProb: &high},
		{HorizonS: &three, InactivityProb: &low},
	}})

	if s.phase == phaseOpen {
		t.Error("the turn opened on the 3 s horizon, want the 1 s one watched")
	}
}

// Ported from upstream (test_gradium_turn_detection_a_step_at_the_threshold_ends_an_open_turn_with_a_flush).
func TestAStepAtTheThresholdEndsAnOpenTurnWithAFlush(t *testing.T) {
	stepReply := func(p float64) map[string]any {
		return map[string]any{msgType: msgStep, "vad": []map[string]any{{"horizon_s": 3.0, "inactivity_prob": p}}}
	}
	endpoint, seen := sttServer(t, nil, []map[string]any{stepReply(0.9), stepReply(0.2), stepReply(0.5)})
	session, err := sttConn(endpoint, func(c *STTConfig) { c.EnableTurnDetection = true }).
		Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() { _ = session.Close() }()
	seen.await(t) // the setup

	start, err := session.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if got := speechOf(start); !slices.Equal(got, []stt.SpeechState{stt.SpeechStarted}) {
		t.Fatalf("first results propose %v, want the start", got)
	}
	stop, err := session.Recv()
	if err != nil {
		t.Fatalf("Recv: %v", err)
	}
	if got := speechOf(stop); !slices.Equal(got, []stt.SpeechState{stt.SpeechStopped}) {
		t.Errorf("results at the threshold propose %v, want the stop", got)
	}
	if m := seen.await(t); m[msgType] != msgFlush {
		t.Errorf("the server saw %v, want the flush", m)
	}
	if s, _ := session.(*sttStream); s.phase != phaseEnding {
		t.Errorf("phase = %v, want the turn ending", s.phase)
	}
}

// Ported from upstream (test_gradium_turn_detection_the_flush_ack_starts_the_cooldown_and_holds_the_ending_phase).
// The turn's transcript is still gathering after the acknowledgement, so the
// steps read meanwhile start nothing; they count against the cooldown the
// acknowledgement started.
func TestTheFlushAckStartsTheCooldownAndHoldsTheEndingPhase(t *testing.T) {
	s := turnStream()
	s.phase = phaseEnding
	s.accumulated = []string{"first turn"}
	s.feed(step(0.9), step(0.2))

	res := s.flushed()

	if len(res) != 1 || res[0].Text != "first turn" || len(speechOf(res)) != 0 {
		t.Errorf("results = %+v, want the transcript and nothing proposed", res)
	}
	if s.cooldown != defaultPostFlushCooldownFrames-2 {
		t.Errorf("cooldown = %d, want the %d frames less the two steps read", s.cooldown,
			defaultPostFlushCooldownFrames)
	}
	if s.phase != phaseArmed {
		t.Errorf("phase = %v, want the next turn armed once the transcript is out", s.phase)
	}
}

// Ported from upstream (test_gradium_turn_detection_speech_continuing_through_a_flush_opens_a_turn).
// After a flush the signal follows the audio: a user who keeps talking reads
// low straight away, without first reading inactive.
func TestSpeechContinuingThroughAFlushOpensATurn(t *testing.T) {
	s := turnStream(withCooldown(2))
	s.phase = phaseEnding
	s.accumulated = []string{"I'd like to book a"}

	res := s.flushed()
	res = append(res, s.steps(0.1, 0.1, 0.1)...)

	if len(res) != 2 || res[0].Text != "I'd like to book a" || res[1].Speech != stt.SpeechStarted {
		t.Errorf("results = %+v, want the transcript and then one start", res)
	}
	if s.phase != phaseOpen {
		t.Errorf("phase = %v, want the turn open", s.phase)
	}
}

// Ported from upstream (test_gradium_turn_detection_an_empty_turn_still_arms_the_next_turn).
func TestAnEmptyTurnStillArmsTheNextTurn(t *testing.T) {
	s := turnStream()
	s.phase = phaseEnding

	if res := s.flushed(); len(res) != 0 {
		t.Errorf("results = %+v, want nothing for an empty turn", res)
	}
	if s.phase != phaseArmed {
		t.Errorf("phase = %v, want the next turn armed", s.phase)
	}
}

// Ported from upstream (test_gradium_turn_detection_no_start_before_the_previous_transcript_is_pushed).
// With no cooldown, steps after the acknowledgement are read at once; the
// turn's transcript still goes out before the next start is proposed.
func TestNoStartBeforeThePreviousTranscriptIsOut(t *testing.T) {
	s := turnStream(withCooldown(0))
	s.phase = phaseEnding
	s.accumulated = []string{"first turn"}
	s.feed(step(0.1))

	res := s.flushed()
	res = append(res, s.steps(0.1)...)

	if len(res) != 2 || res[0].Text != "first turn" || res[1].Speech != stt.SpeechStarted {
		t.Errorf("results = %+v, want the transcript ahead of the start", res)
	}
}

// Ported from upstream (test_gradium_without_turn_detection_a_flush_ack_leaves_turns_alone).
func TestWithoutTurnDetectionAFlushAckLeavesTurnsAlone(t *testing.T) {
	s := turnStream(func(c *STTConfig) { c.EnableTurnDetection = false })
	s.accumulated = []string{"hello"}
	s.feed(step(0.9), step(0.1))

	res := s.flushed()

	if len(res) != 1 || res[0].Text != "hello" || len(speechOf(res)) != 0 {
		t.Errorf("results = %+v, want the transcript alone", res)
	}
	if s.phase != phaseIdle || s.cooldown != 0 {
		t.Errorf("phase = %v, cooldown = %d, want turn detection untouched", s.phase, s.cooldown)
	}

	s.feed(step(0.9), step(0.1), text("next"))
	if got := recvOne(t, s); got.Text != "next" || got.Speech != stt.SpeechUnknown {
		t.Errorf("result = %+v, want the steps ignored and the interim through", got)
	}
}

// Ported from upstream
// (test_gradium_turn_detection_the_transcript_finalizes_on_the_spot_when_the_flush_cannot_be_sent).
// No acknowledgement is coming for a flush that was never sent, so the
// transcript finalizes on what has arrived, behind the stop.
func TestTheTranscriptFinalizesOnTheSpotWhenTheFlushCannotBeSent(t *testing.T) {
	s := turnStream()
	s.phase = phaseOpen
	s.accumulated = []string{"so far"}

	res := s.endTurn()

	if len(res) != 2 || res[0].Speech != stt.SpeechStopped || res[1].Text != "so far" || !res[1].Final {
		t.Errorf("results = %+v, want the stop and then the transcript", res)
	}
	if s.phase != phaseIdle {
		t.Errorf("phase = %v, want idle", s.phase)
	}
}

// Ported from upstream (test_gradium_turn_detection_ignores_the_signal_during_the_post_flush_cooldown).
// The steps right after a flush can read as speech resuming; none of them may
// open a turn, and the first one past the cooldown may.
func TestTheSignalIsIgnoredDuringThePostFlushCooldown(t *testing.T) {
	s := turnStream(withCooldown(2))
	s.cooldown = 2

	if res := s.steps(0.9, 0.9, 0.2); len(res) != 0 {
		t.Errorf("results = %+v during the cooldown, want nothing", res)
	}
	res := s.steps(0.9, 0.2)
	if got := speechOf(res); !slices.Equal(got, []stt.SpeechState{stt.SpeechStarted}) {
		t.Errorf("proposed %v past the cooldown, want one start", got)
	}
}

// Ported from upstream (test_gradium_turn_detection_no_new_start_while_the_flush_is_unacknowledged).
func TestNoNewStartWhileTheFlushIsUnacknowledged(t *testing.T) {
	s := turnStream()
	s.phase = phaseEnding

	if res := s.steps(0.2); len(res) != 0 {
		t.Errorf("results = %+v, want nothing while the turn ends", res)
	}
	if s.phase == phaseOpen {
		t.Error("a turn opened before the flush was acknowledged")
	}
}

// reconnected opens the session that replaces one that dropped.
func reconnected(t *testing.T, c *sttConnector) *sttStream {
	t.Helper()
	session, err := c.Connect(context.Background(), 16000)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })
	s, ok := session.(*sttStream)
	if !ok {
		t.Fatalf("Connect returned a %T", session)
	}
	return s
}

// Ported from upstream (test_gradium_a_reconnect_pushes_the_text_so_far_and_keeps_an_open_turn_open).
// The user may still be speaking; the new session's signal ends the turn.
func TestAReconnectPushesTheTextSoFarAndKeepsAnOpenTurnOpen(t *testing.T) {
	endpoint, _ := sttServer(t, nil, nil)
	s := turnStream()
	s.owner = sttConn(endpoint, func(c *STTConfig) { c.EnableTurnDetection = true })
	s.phase = phaseOpen
	s.accumulated = []string{"book a table"}
	s.flushCounter = 3
	s.cooldown = 2
	s.drop()

	got := recvOne(t, s)
	if got.Text != "book a table" || !got.Final || got.Speech != stt.SpeechUnknown {
		t.Errorf("result = %+v, want the text so far and nothing proposed", got)
	}

	next := reconnected(t, s.owner)
	if next.phase != phaseOpen {
		t.Errorf("phase = %v on the new session, want the turn still open", next.phase)
	}
	if len(next.accumulated) != 0 || next.flushCounter != 0 || next.cooldown != 0 {
		t.Errorf("the new session starts with text %v, flush %d, cooldown %d, want a clean state",
			next.accumulated, next.flushCounter, next.cooldown)
	}
}

// Ported from upstream (test_gradium_a_reconnect_while_ending_pushes_the_transcript_without_a_second_stop).
func TestAReconnectWhileEndingPushesTheTranscriptWithoutASecondStop(t *testing.T) {
	endpoint, _ := sttServer(t, nil, nil)
	s := turnStream()
	s.owner = sttConn(endpoint, func(c *STTConfig) { c.EnableTurnDetection = true })
	s.phase = phaseEnding
	s.accumulated = []string{"the tail"}
	s.drop()

	got := recvOne(t, s)
	if got.Text != "the tail" || got.Speech != stt.SpeechUnknown {
		t.Errorf("result = %+v, want the transcript and no second stop", got)
	}
	if next := reconnected(t, s.owner); next.phase != phaseIdle {
		t.Errorf("phase = %v on the new session, want idle", next.phase)
	}
}

// Each flush a session sends carries the next number of its own.
func TestFlushesAreNumberedPerSession(t *testing.T) {
	endpoint, seen := sttServer(t, nil, nil)
	s := reconnected(t, sttConn(endpoint))
	seen.await(t) // the setup
	for _, want := range []string{"1", "2"} {
		if err := s.Finalize(); err != nil {
			t.Fatalf("Finalize: %v", err)
		}
		if m := seen.await(t); m[msgType] != msgFlush || m["flush_id"] != want {
			t.Errorf("the server saw %v, want flush %s", m, want)
		}
	}
}
