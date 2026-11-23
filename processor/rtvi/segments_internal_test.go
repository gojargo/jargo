package rtvi

import (
	"maps"
	"slices"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Tests for how the observer keeps a skipped segment from the client, its words
// and its progress included. They drive the observer's frame mapping directly,
// as it runs once a frame has gone through the output transport, so what the
// observer remembers about skipped segments can be checked too.

// segmentObserver is an observer that skips the "status" type, with the bot
// already speaking so text is reported at once instead of held.
type segmentObserver struct {
	*Observer
	sent []Message
}

func newSegmentObserver() *segmentObserver {
	params := DefaultObserverParams()
	params.SkipTextTypes = []frames.TextType{"status"}
	o := &segmentObserver{Observer: NewObserverWithParams(nil, params)}
	o.botSpeaking = true
	return o
}

func (o *segmentObserver) push(f frames.Frame) {
	o.sent = append(o.sent, o.messagesFor(f)...)
}

// texts is the text of every message sent that carries any: the bot's output
// and the captions for what it spoke.
func (o *segmentObserver) texts() []string {
	var out []string
	for _, m := range o.sent {
		switch d := m.Data.(type) {
		case BotOutputData:
			out = append(out, d.Text)
		case TextData:
			out = append(out, d.Text)
		}
	}
	return out
}

func (o *segmentObserver) skippedIDs() []uint64 {
	o.outputMu.Lock()
	defer o.outputMu.Unlock()
	return slices.Sorted(maps.Keys(o.skippedSegmentIDs))
}

// statusSegment is a segment the synthesizer will speak. All the segments of
// one response share a synthesis context.
func statusSegment(text string, by frames.TextType) *frames.AggregatedTextFrame {
	f := frames.NewAggregatedTextFrame(text, by)
	f.ContextID = "turn"
	f.WillBeSpoken = true
	return f
}

// word is one word spoken in seg.
func word(text string, seg *frames.AggregatedTextFrame) *frames.TTSTextFrame {
	f := frames.NewTTSTextFrame(text, frames.TextTypeWord)
	f.ContextID = "turn"
	f.SegmentID = seg.ID()
	return f
}

// progress is how far through seg the bot has spoken.
func progress(seg *frames.AggregatedTextFrame, accumulated, remaining string) *frames.AggregatedTextProgressFrame {
	return frames.NewAggregatedTextProgressFrame(seg.ID(), "turn", seg.Text, seg.TextType, accumulated, remaining)
}

func TestASkippedSegmentKeepsItsProgressAndWordsFromTheClient(t *testing.T) {
	o := newSegmentObserver()
	status := statusSegment("One moment, please.", "status")
	o.push(status)
	o.push(word("One", status))
	o.push(progress(status, "One", " moment, please."))

	if len(o.sent) != 0 {
		t.Fatalf("sent %+v, want nothing of a skipped segment", o.sent)
	}
}

func TestASegmentAfterASkippedOneIsSent(t *testing.T) {
	o := newSegmentObserver()
	status := statusSegment("One moment, please.", "status")
	sentence := statusSegment("That sounds fun.", frames.TextTypeSentence)
	o.push(status)
	o.push(word("One", status))
	o.push(sentence)
	o.push(word("That", sentence))

	if got, want := o.texts(), []string{"That sounds fun.", "That"}; !slices.Equal(got, want) {
		t.Fatalf("sent %q, want %q", got, want)
	}
}

func TestWordsFollowTheirSegmentWhenTheSegmentsComeFirst(t *testing.T) {
	// A turn's segments can all arrive before the first one is spoken.
	o := newSegmentObserver()
	status := statusSegment("One moment, please.", "status")
	sentence := statusSegment("That sounds fun.", frames.TextTypeSentence)
	o.push(status)
	o.push(sentence)
	o.push(word("One", status))
	o.push(word("That", sentence))

	if got, want := o.texts(), []string{"That sounds fun.", "That"}; !slices.Equal(got, want) {
		t.Fatalf("sent %q, want %q", got, want)
	}
}

func TestASkippedSegmentIsForgottenOnceSpoken(t *testing.T) {
	o := newSegmentObserver()
	status := statusSegment("One moment, please.", "status")
	o.push(status)
	o.push(progress(status, "One moment, please.", ""))

	if got := o.skippedIDs(); len(got) != 0 {
		t.Fatalf("still remembering %v after the segment was spoken", got)
	}
}

func TestASkippedSegmentCutOffIsForgottenWhenTheBotStops(t *testing.T) {
	o := newSegmentObserver()
	status := statusSegment("One moment, please.", "status")
	o.push(status)
	o.push(frames.NewInterruptionFrame())
	// A word the output transport pushed before it handled the interruption.
	o.push(word("One", status))
	o.push(frames.NewBotStoppedSpeakingFrame())

	if slices.Contains(o.texts(), "One") {
		t.Fatal("a word of the skipped segment was sent")
	}
	if got := o.skippedIDs(); len(got) != 0 {
		t.Fatalf("still remembering %v after the bot stopped", got)
	}
}

func TestASkippedSegmentCutOffInAPauseIsForgottenWhenTheBotSpeaks(t *testing.T) {
	o := newSegmentObserver()
	status := statusSegment("One moment, please.", "status")
	o.push(status)
	o.push(frames.NewBotStoppedSpeakingFrame())
	if got := o.skippedIDs(); !slices.Equal(got, []uint64{status.ID()}) {
		t.Fatalf("remembering %v, want the skipped segment %d", got, status.ID())
	}

	o.push(frames.NewInterruptionFrame())
	o.push(frames.NewBotStartedSpeakingFrame())
	if got := o.skippedIDs(); len(got) != 0 {
		t.Fatalf("still remembering %v after the bot spoke again", got)
	}
}

func TestASkippedSegmentSpokenInOnePieceIsForgotten(t *testing.T) {
	o := newSegmentObserver()
	status := statusSegment("One moment, please.", "status")
	spoken := frames.NewTTSTextFrame(status.Text, "status")
	spoken.SegmentID = status.ID()
	o.push(status)
	o.push(spoken)

	if len(o.sent) != 0 {
		t.Fatalf("sent %+v, want nothing of a skipped segment", o.sent)
	}
	if got := o.skippedIDs(); len(got) != 0 {
		t.Fatalf("still remembering %v after the segment was spoken", got)
	}
}

// connectClient stands in for a processor a client negotiated version with.
func (o *segmentObserver) connectClient(version [3]int) {
	sink := NewProcessor()
	sink.clientVersion = version
	sink.clientReady = true
	o.sink = sink
}

// types is the type of every message sent.
func (o *segmentObserver) types() []string {
	out := make([]string, 0, len(o.sent))
	for _, m := range o.sent {
		out = append(out, m.Type)
	}
	return out
}

// outputs is the data of every bot-output message sent.
func (o *segmentObserver) outputs() []BotOutputData {
	var out []BotOutputData
	for _, m := range o.sent {
		if d, ok := m.Data.(BotOutputData); ok {
			out = append(out, d)
		}
	}
	return out
}

func TestABackchannelIsSentAsOutputAndTTSText(t *testing.T) {
	o := newSegmentObserver()
	backchannel := statusSegment("Mm-hmm, go on.", frames.TextTypeBackchannel)
	o.push(backchannel)
	o.push(word("Mm-hmm,", backchannel))
	o.push(progress(backchannel, "Mm-hmm,", " go on."))

	if got, want := o.types(), []string{TypeBotOutput, TypeBotTTSText, TypeBotOutput}; !slices.Equal(got, want) {
		t.Fatalf("sent %v, want %v", got, want)
	}
	outs := o.outputs()
	if outs[0].OutputType != frames.TextTypeBackchannel || outs[0].SpokenStatus != SpokenNew ||
		outs[1].OutputType != frames.TextTypeBackchannel || outs[1].SpokenStatus != SpokenInProgress {
		t.Errorf("outputs %+v, want the backchannel new then in progress", outs)
	}
	if d, ok := o.sent[1].Data.(TextData); !ok || d.Text != "Mm-hmm," {
		t.Errorf("tts text %+v, want the word spoken", o.sent[1].Data)
	}
}

func TestABackchannelSpokenInOnePieceIsSentAsOutputAndTTSText(t *testing.T) {
	o := newSegmentObserver()
	backchannel := statusSegment("Mm-hmm.", frames.TextTypeBackchannel)
	spoken := frames.NewTTSTextFrame(backchannel.Text, frames.TextTypeBackchannel)
	spoken.SegmentID = backchannel.ID()
	spoken.WillBeSpoken = true
	o.push(backchannel)
	o.push(spoken)

	if got, want := o.types(), []string{TypeBotOutput, TypeBotOutput, TypeBotTTSText}; !slices.Equal(got, want) {
		t.Fatalf("sent %v, want %v", got, want)
	}
	outs := o.outputs()
	if outs[0].SpokenStatus != SpokenNew || outs[1].SpokenStatus != SpokenCompleted {
		t.Errorf("outputs %+v, want the backchannel new then completed", outs)
	}
}

func TestABackchannelIsKeptFromOlderClients(t *testing.T) {
	for _, version := range [][3]int{{2, 1, 0}, {1, 4, 0}} {
		o := newSegmentObserver()
		o.connectClient(version)
		backchannel := statusSegment("Mm-hmm, go on.", frames.TextTypeBackchannel)
		o.push(backchannel)
		o.push(word("Mm-hmm,", backchannel))
		o.push(progress(backchannel, "Mm-hmm,", " go on."))
		o.push(progress(backchannel, "Mm-hmm, go on.", ""))

		if len(o.sent) != 0 {
			t.Errorf("%v: sent %+v, want nothing of a backchannel", version, o.sent)
		}
		if got := o.skippedIDs(); len(got) != 0 {
			t.Errorf("%v: still remembering %v after the backchannel was spoken", version, got)
		}
	}
}

func TestABackchannelSpokenInOnePieceIsKeptFromOlderClients(t *testing.T) {
	o := newSegmentObserver()
	o.connectClient([3]int{2, 1, 0})
	backchannel := statusSegment("Mm-hmm.", frames.TextTypeBackchannel)
	spoken := frames.NewTTSTextFrame(backchannel.Text, frames.TextTypeBackchannel)
	spoken.SegmentID = backchannel.ID()
	spoken.WillBeSpoken = true
	o.push(backchannel)
	o.push(spoken)

	if len(o.sent) != 0 {
		t.Fatalf("sent %+v, want nothing of a backchannel", o.sent)
	}
	if got := o.skippedIDs(); len(got) != 0 {
		t.Fatalf("still remembering %v after the backchannel was spoken", got)
	}
}

func TestASegmentAfterABackchannelIsSentToOlderClients(t *testing.T) {
	o := newSegmentObserver()
	o.connectClient([3]int{2, 1, 0})
	backchannel := statusSegment("Mm-hmm.", frames.TextTypeBackchannel)
	sentence := statusSegment("That sounds fun.", frames.TextTypeSentence)
	o.push(backchannel)
	o.push(word("Mm-hmm.", backchannel))
	o.push(sentence)
	o.push(word("That", sentence))

	if got, want := o.texts(), []string{"That sounds fun.", "That"}; !slices.Equal(got, want) {
		t.Fatalf("sent %q, want %q", got, want)
	}
}
