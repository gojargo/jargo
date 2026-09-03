package rtvi_test

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/utils/events"
)

// The bot-output events have no tests upstream, so the ones below are ours.
// What they pin is the behavior of upstream's implementation, which is what was
// ported.

// outputHarness runs frames through an observer and returns the messages it
// sent, with a client of the given protocol generation connected first. An
// empty version connects no client, which leaves the observer treating it as a
// client of this generation.
func outputHarness(
	t *testing.T, params rtvi.ObserverParams, version string, queue ...frames.Frame,
) []rtvi.Message {
	t.Helper()

	proc := rtvi.NewProcessor()
	out := make(chan rtvi.Message, 64)
	// The pipeline ends in a stand-in for the transport's output end: the bot's
	// output is reported from there, with the timing of what the caller hears.
	task := pipeline.NewWorker(pipeline.New(proc, newPlayback()), pipeline.WorkerConfig{
		Observers:               []pipeline.Observer{rtvi.NewObserverWithParams(proc, params)},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	events.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if m, ok := f.(*frames.OutputTransportMessageUrgentFrame); ok {
			if msg, ok := m.Message.(rtvi.Message); ok {
				out <- msg
			}
		}
	})

	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	if version != "" {
		raw, err := json.Marshal(rtvi.Message{
			Label: rtvi.MessageLabel,
			Type:  rtvi.TypeClientReady,
			ID:    "req-1",
			Data: rtvi.ClientReadyData{
				Version: version,
				About:   rtvi.AboutClientData{Library: "test-client"},
			},
		})
		if err != nil {
			t.Fatal(err)
		}
		task.QueueFrame(frames.NewInputTransportMessageFrame(raw))
		waitFor(t, out, rtvi.TypeBotReady)
	}

	for _, f := range queue {
		task.QueueFrame(f)
		// See observerHarness: the events bracketing playback are system frames
		// and would otherwise overtake the data frames they bracket.
		time.Sleep(10 * time.Millisecond)
	}
	// Settle, so a test expecting nothing waits as long as one expecting several.
	time.Sleep(300 * time.Millisecond)

	task.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}

	var msgs []rtvi.Message
	for {
		select {
		case m := <-out:
			msgs = append(msgs, m)
		default:
			return msgs
		}
	}
}

// waitFor drains messages until one of the given type arrives.
func waitFor(t *testing.T, out chan rtvi.Message, msgType string) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	for {
		select {
		case m := <-out:
			if m.Type == msgType {
				return
			}
		case <-deadline:
			t.Fatalf("timed out waiting for a %s message", msgType)
		}
	}
}

// outputs returns the payloads of the bot-output messages, in order.
func outputs(msgs []rtvi.Message) []rtvi.BotOutputData {
	var out []rtvi.BotOutputData
	for _, m := range msgs {
		if m.Type != rtvi.TypeBotOutput {
			continue
		}
		if d, ok := m.Data.(rtvi.BotOutputData); ok {
			out = append(out, d)
		}
	}
	return out
}

// speaking is the audio starting, which is what makes the bot's output audible
// and so reportable.
func speaking() frames.Frame { return frames.NewBotStartedSpeakingFrame() }

// segment builds one unit of the bot's output, announced before synthesis.
func segment(text string, by frames.AggregationType) *frames.AggregatedTextFrame {
	f := frames.NewAggregatedTextFrame(text, by)
	f.WillBeSpoken = true
	return f
}

// spokenText builds one unit of text the synthesizer has said.
func spokenText(text string, by frames.AggregationType) *frames.TTSTextFrame {
	f := frames.NewTTSTextFrame(text, by)
	f.WillBeSpoken = true
	return f
}

// A unit announced before synthesis is reported as new, with all of it still to
// be spoken: the client can render the whole segment and highlight nothing yet.
func TestBotOutputAnnouncesASegmentBeforeItIsSpoken(t *testing.T) {
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "",
		speaking(), segment("Hello there.", frames.AggregationSentence))

	got := outputs(msgs)
	if len(got) != 1 {
		t.Fatalf("got %d bot-output messages, want one: %v", len(got), types(msgs))
	}
	d := got[0]
	if d.Text != "Hello there." {
		t.Errorf("text = %q, want the segment", d.Text)
	}
	if d.AggregatedBy != frames.AggregationSentence {
		t.Errorf("aggregated by %q, want %q", d.AggregatedBy, frames.AggregationSentence)
	}
	if d.SpokenStatus != rtvi.SpokenNew {
		t.Errorf("status = %q, want %q", d.SpokenStatus, rtvi.SpokenNew)
	}
	if d.SpokenProgress == nil || d.SpokenProgress.AccumulatedText != "" ||
		d.SpokenProgress.RemainingText != "Hello there." {
		t.Errorf("progress = %+v, want all of it still to be spoken", d.SpokenProgress)
	}
	if d.WillBeSpoken == nil || !*d.WillBeSpoken {
		t.Error("the segment does not say it will be spoken")
	}
	if d.Spoken == nil || *d.Spoken {
		t.Error("the segment says it has been spoken, but synthesis has not run")
	}
	if d.SegmentID == nil {
		t.Error("the segment has no id, so progress could not be matched to it")
	}
	if sent(msgs, rtvi.TypeBotTTSText) {
		t.Error("a caption was sent for text the synthesizer has not spoken")
	}
}

// Text the synthesizer has spoken is reported as finished, and separately as the
// caption for it.
func TestBotOutputReportsSpokenTextAsCompleted(t *testing.T) {
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "",
		speaking(), spokenText("Hello there.", frames.AggregationSentence))

	got := outputs(msgs)
	if len(got) != 1 {
		t.Fatalf("got %d bot-output messages, want one: %v", len(got), types(msgs))
	}
	d := got[0]
	if d.SpokenStatus != rtvi.SpokenCompleted {
		t.Errorf("status = %q, want %q", d.SpokenStatus, rtvi.SpokenCompleted)
	}
	if d.SpokenProgress == nil || d.SpokenProgress.AccumulatedText != "Hello there." ||
		d.SpokenProgress.RemainingText != "" {
		t.Errorf("progress = %+v, want all of it spoken", d.SpokenProgress)
	}
	if d.Spoken == nil || !*d.Spoken {
		t.Error("the segment does not say it has been spoken")
	}
	if !sent(msgs, rtvi.TypeBotTTSText) {
		t.Errorf("no caption for the spoken text, got %v", types(msgs))
	}
}

// A unit nothing is going to speak has no playback to be anywhere in, so it
// carries no status and no progress.
func TestBotOutputSaysNothingAboutPlaybackForUnspokenText(t *testing.T) {
	unspoken := frames.NewAggregatedTextFrame("print('hi')", "code")
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "", speaking(), unspoken)

	got := outputs(msgs)
	if len(got) != 1 {
		t.Fatalf("got %d bot-output messages, want one: %v", len(got), types(msgs))
	}
	d := got[0]
	if d.SpokenStatus != "" {
		t.Errorf("status = %q, want none for text nothing will speak", d.SpokenStatus)
	}
	if d.SpokenProgress != nil {
		t.Errorf("progress = %+v, want none", d.SpokenProgress)
	}
	if d.WillBeSpoken == nil || *d.WillBeSpoken {
		t.Error("the segment claims it will be spoken")
	}
}

// Each word spoken moves the segment along, and the last one finishes it.
func TestBotOutputReportsProgressThroughASegment(t *testing.T) {
	seg := segment("Hello there.", frames.AggregationSentence)
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "",
		speaking(),
		seg,
		frames.NewAggregatedTextProgressFrame(seg.ID(), "", "Hello there.",
			frames.AggregationSentence, "Hello", " there."),
		frames.NewAggregatedTextProgressFrame(seg.ID(), "", "Hello there.",
			frames.AggregationSentence, "Hello there.", ""),
	)

	got := outputs(msgs)
	if len(got) != 3 {
		t.Fatalf("got %d bot-output messages, want the segment and two reports: %v",
			len(got), types(msgs))
	}
	if got[1].SpokenStatus != rtvi.SpokenInProgress {
		t.Errorf("status = %q, want %q while text remains", got[1].SpokenStatus, rtvi.SpokenInProgress)
	}
	if got[1].SpokenProgress.AccumulatedText != "Hello" ||
		got[1].SpokenProgress.RemainingText != " there." {
		t.Errorf("progress = %+v, want the split at the word being spoken", got[1].SpokenProgress)
	}
	if got[2].SpokenStatus != rtvi.SpokenCompleted {
		t.Errorf("status = %q, want %q once nothing remains", got[2].SpokenStatus, rtvi.SpokenCompleted)
	}
	for i, d := range got {
		if d.SegmentID == nil || *d.SegmentID != seg.ID() {
			t.Errorf("message %d names segment %v, want %d", i, d.SegmentID, seg.ID())
		}
	}
}

// A word is not a segment of its own to a client that understands progress: the
// segment is the sentence, and the word is reported as progress through it. The
// caption is a channel of its own and still carries every word.
func TestBotOutputSuppressesWordSegmentsForACurrentClient(t *testing.T) {
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "2.1.0",
		speaking(), spokenText("Hello", frames.AggregationWord))

	if got := outputs(msgs); len(got) != 0 {
		t.Errorf("got %d bot-output messages for a word, want none: %+v", len(got), got)
	}
	if !sent(msgs, rtvi.TypeBotTTSText) {
		t.Errorf("no caption for the spoken word, got %v", types(msgs))
	}
}

// A client of the older generation has no notion of progress within a segment,
// so each word is reported as output in its own right and no progress is sent.
func TestBotOutputGivesALegacyClientEachWord(t *testing.T) {
	seg := segment("Hello there.", frames.AggregationSentence)
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "1.4.0",
		speaking(),
		spokenText("Hello", frames.AggregationWord),
		frames.NewAggregatedTextProgressFrame(seg.ID(), "", "Hello there.",
			frames.AggregationSentence, "Hello", " there."),
	)

	got := outputs(msgs)
	if len(got) != 1 {
		t.Fatalf("got %d bot-output messages, want the word alone: %+v", len(got), got)
	}
	if got[0].Text != "Hello" || got[0].AggregatedBy != frames.AggregationWord {
		t.Errorf("output = %+v, want the word", got[0])
	}
}

// Segments produced before the bot is audible are held, and released in order
// once its audio starts: a client told about text before the audio carrying it
// would run ahead of what the caller hears.
func TestBotOutputHoldsSegmentsUntilTheBotIsAudible(t *testing.T) {
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "",
		segment("First.", frames.AggregationSentence),
		segment("Second.", frames.AggregationSentence),
		speaking(),
	)

	got := outputs(msgs)
	if len(got) != 2 {
		t.Fatalf("got %d bot-output messages, want both held segments: %v", len(got), types(msgs))
	}
	if got[0].Text != "First." || got[1].Text != "Second." {
		t.Errorf("released %q then %q, want them in the order they were produced",
			got[0].Text, got[1].Text)
	}
	// Held until the audio started, so the client hears about them after it.
	for i, m := range msgs {
		if m.Type == rtvi.TypeBotStartedSpeaking {
			if i == len(msgs)-1 {
				t.Errorf("the segments were sent before the bot started speaking: %v", types(msgs))
			}
			break
		}
	}
}

// An aggregation type the client is not meant to see is silenced outright, the
// caption for it included.
func TestBotOutputSkipsTheAggregationTypesItWasTold(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.SkipAggregatorTypes = []frames.AggregationType{"code"}
	msgs := outputHarness(t, params, "",
		speaking(),
		spokenText("print('hi')", "code"),
		spokenText("Done.", frames.AggregationSentence),
	)

	got := outputs(msgs)
	if len(got) != 1 || got[0].Text != "Done." {
		t.Fatalf("got %+v, want the skipped type silenced and the rest reported", got)
	}
	for _, m := range msgs {
		if m.Type != rtvi.TypeBotTTSText {
			continue
		}
		if d, ok := m.Data.(rtvi.TextData); ok && d.Text == "print('hi')" {
			t.Error("a caption was sent for a skipped aggregation type")
		}
	}
}

// Turning the bot's output off silences the segments and leaves the captions,
// which are a channel of their own.
func TestBotOutputCategoryCanBeTurnedOff(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.BotOutputEnabled = off()
	msgs := outputHarness(t, params, "",
		speaking(), spokenText("Hello there.", frames.AggregationSentence))

	if got := outputs(msgs); len(got) != 0 {
		t.Errorf("got %d bot-output messages, want none: %+v", len(got), got)
	}
	if !sent(msgs, rtvi.TypeBotTTSText) {
		t.Errorf("the caption was silenced too, got %v", types(msgs))
	}
}

// A segment is reported once, from the end that played it. The same frame is
// handed over several times on its way through the pipeline, and only the
// handover out of playback carries the timing of what the caller hears.
func TestBotOutputReportsASegmentOnce(t *testing.T) {
	msgs := outputHarness(t, rtvi.DefaultObserverParams(), "",
		speaking(), spokenText("Hello there.", frames.AggregationSentence))

	if got := outputs(msgs); len(got) != 1 {
		t.Errorf("got %d bot-output messages, want exactly one: %+v", len(got), got)
	}
	captions := 0
	for _, m := range msgs {
		if m.Type == rtvi.TypeBotTTSText {
			captions++
		}
	}
	if captions != 1 {
		t.Errorf("got %d captions, want exactly one", captions)
	}
}
