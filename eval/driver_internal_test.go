package eval

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/processor/rtvi"
)

// newSessionFor is a text-mode session of scenario against a bot that is never
// reached.
func newSessionFor(t *testing.T, scenario *ScriptScenario, params SessionParams) *ScriptSession {
	t.Helper()
	s, err := NewScriptSession(scenario, "ws://localhost:0", params, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// bareSession is a session of an empty scenario.
func bareSession(t *testing.T, botAudio bool) *ScriptSession {
	t.Helper()
	scenario := &ScriptScenario{Name: "t"}
	scenario.BotAudio = botAudio
	return newSessionFor(t, scenario, SessionParams{})
}

// A turn is not sent over a speaking bot.
func TestSendWaitsForTheBot(t *testing.T) {
	ctx := t.Context()
	t.Run("the send waits until the bot stops speaking", func(t *testing.T) {
		s := bareSession(t, false)
		s.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		go func() {
			time.Sleep(150 * time.Millisecond)
			s.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStoppedSpeaking, nil))
		}()
		started := time.Now()
		s.script.awaitBotQuiet(ctx)
		if time.Since(started) < 150*time.Millisecond {
			t.Fatal("the send did not wait for the bot")
		}
	})
	t.Run("the send waits until the bot's speech is transcribed", func(t *testing.T) {
		s := bareSession(t, true)
		// The bot is quiet, but the last segment of what it said is still with
		// the STT.
		s.stream.BotSegmentEnded()
		go func() {
			time.Sleep(150 * time.Millisecond)
			s.stream.BotSegmentTranscribed()
		}()
		started := time.Now()
		s.script.awaitBotQuiet(ctx)
		if time.Since(started) < 150*time.Millisecond {
			t.Fatal("the send did not wait for the transcription")
		}
	})
	t.Run("an observing turn lets the previous reply end first", func(t *testing.T) {
		// The bot is still speaking turn 1's reply when turn 2, which only
		// listens for the bot's next move, begins: the rest of that reply,
		// however late its transcription lands, is not turn 2's response.
		s := bareSession(t, true)
		s.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		s.stream.Append(map[string]any{keyType: EventResponse, "text": "Spring in Japan is gorgeous."})
		go func() {
			time.Sleep(100 * time.Millisecond)
			s.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStoppedSpeaking, nil))
		}()
		s.script.awaitPreviousReply(ctx)
		// The queued tail is gone, and one transcribed after the wait is stale.
		if _, err := s.stream.NextEvent(ctx, EventResponse, time.Now().Add(50*time.Millisecond), FromNow, nil); err == nil {
			t.Fatal("the previous reply's tail matched")
		}
		s.stream.BotTurnStopped("Do you have cities in mind?")
		if _, err := s.stream.NextEvent(ctx, EventResponse, time.Now().Add(50*time.Millisecond), FromNow, nil); err == nil {
			t.Fatal("a stale turn matched")
		}
	})
	t.Run("an observing turn takes the bot's next speech without its LLM", func(t *testing.T) {
		s := bareSession(t, true)
		s.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		go func() {
			time.Sleep(100 * time.Millisecond)
			s.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStoppedSpeaking, nil))
		}()
		s.script.awaitPreviousReply(ctx)
		s.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		s.stream.BotTurnStopped("Are you still there?")
		event, err := s.stream.NextEvent(ctx, EventResponse, time.Now().Add(50*time.Millisecond), FromNow, nil)
		if err != nil || event["text"] != "Are you still there?" {
			t.Fatalf("event = %v, %v", event, err)
		}
	})
	t.Run("an observing turn lets the previous reply finish transcribing", func(t *testing.T) {
		s := bareSession(t, true)
		s.stream.BotTurnStarted()
		s.stream.BotSegmentEnded()
		go func() {
			time.Sleep(100 * time.Millisecond)
			s.stream.BotSegmentTranscribed()
			s.stream.BotTurnStopped("Spring in Japan is gorgeous.")
		}()
		s.script.awaitPreviousReply(ctx)
		if _, err := s.stream.NextEvent(ctx, EventResponse, time.Now().Add(50*time.Millisecond), FromNow, nil); err == nil {
			t.Fatal("the previous reply matched")
		}
	})
	t.Run("a quiet bot holds nothing up", func(t *testing.T) {
		s := bareSession(t, false)
		started := time.Now()
		s.script.awaitBotQuiet(ctx)
		if time.Since(started) > 50*time.Millisecond {
			t.Fatal("the send waited on a quiet bot")
		}
	})
}

// repliesClient is a client whose bot answers each line sent with the events
// listed for it.
type repliesClient struct {
	stream  *EventStream
	replies map[string][]map[string]any
}

func (*repliesClient) HasUserTTS() bool { return false }

func (c *repliesClient) SendText(text string) {
	for _, event := range c.replies[text] {
		// As the client reports its own pipeline's errors.
		if event[keyType] == EventHarnessError {
			reason, _ := event["text"].(string)
			c.stream.HarnessFailed(reason)
		}
		c.stream.Append(event)
	}
}

func (*repliesClient) SendDTMF(string)                                 {}
func (*repliesClient) SendImage(string) error                          { return nil }
func (*repliesClient) Say(context.Context, string) error               { return nil }
func (*repliesClient) SayWithPauses(context.Context, []TurnPart) error { return nil }
func (*repliesClient) Play(context.Context, string) error              { return nil }

// testDriver is a script driver over stream for turns, recording its progress.
func testDriver(stream *EventStream, client turnSender, progress *[]Progress, turns ...ScriptTurn) *ScriptDriver {
	return newScriptDriver(&ScriptScenario{Name: "t", Turns: turns}, 5000, client, stream, nil, NewTrace(),
		func(p Progress) { *progress = append(*progress, p) })
}

// Once the bot ends the call, the driver fails what waits on it at once.
func TestBotEndedDriver(t *testing.T) {
	ctx := t.Context()
	t.Run("the driver fails the turn as the bot ended it", func(t *testing.T) {
		var progress []Progress
		stream := newStream(false)
		turn := ScriptTurn{User: "hi", Expect: []Expectation{{Event: EventLLMResponse}}}
		d := testDriver(stream, nil, &progress, turn)
		stream.BotEnded()

		failures, err := d.matchExpectations(ctx, &turn, 0)
		if err != nil || len(failures) != 1 || failures[0].Kind != FailureBotEnded ||
			!strings.Contains(failures[0].Reason, "the bot ended the call") {
			t.Fatalf("failures = %v, %v", failures, err)
		}
		if d.Turns[0].Expectations[0].Passed || progress[len(progress)-1].Status != StatusFailed {
			t.Fatalf("records = %+v, progress %+v", d.Turns[0], progress)
		}
	})
	t.Run("a send_after wait fails at once once the bot ended the call", func(t *testing.T) {
		var progress []Progress
		stream := newStream(false)
		sendAfter := SendAfter{Event: EventBotStartedSpeaking}
		d := testDriver(stream, nil, &progress, ScriptTurn{User: "hi", SendAfter: &sendAfter})
		stream.BotEnded()

		started := time.Now()
		f := d.awaitSendAfter(ctx, sendAfter, 0)
		if time.Since(started) > time.Second {
			t.Fatal("the wait did not fail at once")
		}
		if f == nil || f.Kind != FailureBotEnded || !strings.Contains(f.Reason, "the bot ended the call") ||
			progress[len(progress)-1].Status != StatusFailed {
			t.Fatalf("failure = %v, progress %+v", f, progress)
		}
	})
}

// A failure in the harness's own pipeline fails the run as an error, not the
// bot.
func TestHarnessErrorDriver(t *testing.T) {
	ctx := t.Context()
	t.Run("a harness error fails the turn at once as an error", func(t *testing.T) {
		var progress []Progress
		stream := newStream(false)
		client := &repliesClient{stream: stream, replies: map[string][]map[string]any{
			"hi": {{keyType: EventHarnessError, "text": "TTS failed"}},
		}}
		d := testDriver(stream, client, &progress,
			ScriptTurn{User: "hi", Expect: []Expectation{{Event: EventLLMResponse, WithinMS: 5000}}})

		started := time.Now()
		failures, err := d.run(ctx)
		if time.Since(started) > time.Second {
			t.Fatal("the turn did not fail at once")
		}
		if err != nil || len(failures) != 1 || failures[0].Kind != FailureError || failures[0].Reason != "TTS failed" {
			t.Fatalf("failures = %v, %v", failures, err)
		}
	})
	t.Run("a send_after wait fails at once as an error", func(t *testing.T) {
		var progress []Progress
		stream := newStream(false)
		sendAfter := SendAfter{Event: EventBotStartedSpeaking}
		d := testDriver(stream, &repliesClient{stream: stream}, &progress, ScriptTurn{User: "hi", SendAfter: &sendAfter})
		stream.HarnessFailed("TTS failed")

		started := time.Now()
		f := d.awaitSendAfter(ctx, sendAfter, 0)
		if time.Since(started) > time.Second {
			t.Fatal("the wait did not fail at once")
		}
		if f == nil || f.Kind != FailureError || f.Reason != "TTS failed" {
			t.Fatalf("failure = %v", f)
		}
	})
}

// sightAt records that an event of eventType was seen at when.
func sightAt(s *EventStream, eventType string, when time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.latestEventTimes[eventType] = when
	s.signalLocked()
}

// The send_after delay is measured from when its event was seen, not from when
// the turn asked: an event seen longer ago than the delay has already fired it.
func TestWaitSendAfterAnchorsOnAnEarlierSighting(t *testing.T) {
	var progress []Progress
	stream := newStream(false)
	d := testDriver(stream, nil, &progress)
	sightAt(stream, EventLLMStarted, time.Now().Add(-time.Second))

	start := time.Now()
	if err := d.waitSendAfter(t.Context(), SendAfter{Event: EventLLMStarted, DelayMS: 200}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed > 50*time.Millisecond {
		t.Fatalf("the delay had already elapsed, should not have waited: took %s", elapsed)
	}
}

// The send_after wait polls until its event arrives, and only then waits out
// its delay.
func TestWaitSendAfterWaitsForAnUnseenEvent(t *testing.T) {
	var progress []Progress
	stream := newStream(false)
	d := testDriver(stream, nil, &progress)
	const arrivesAfter = 200 * time.Millisecond
	go func() {
		time.Sleep(arrivesAfter)
		sightAt(stream, EventLLMStarted, time.Now())
	}()

	start := time.Now()
	if err := d.waitSendAfter(t.Context(), SendAfter{Event: EventLLMStarted, DelayMS: 100}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < arrivesAfter+100*time.Millisecond {
		t.Fatalf("should have waited for the event and then its delay: took %s", elapsed)
	}
}

// A send_after with no event waits out its delay from now, the previous turn's
// send.
func TestWaitSendAfterPureDelay(t *testing.T) {
	var progress []Progress
	d := testDriver(newStream(false), nil, &progress)
	start := time.Now()
	if err := d.waitSendAfter(t.Context(), SendAfter{DelayMS: 150}); err != nil {
		t.Fatal(err)
	}
	if elapsed := time.Since(start); elapsed < 150*time.Millisecond {
		t.Fatalf("should have waited out the delay: took %s", elapsed)
	}
}
