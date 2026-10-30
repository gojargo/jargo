package rtvi_test

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/rtvi"
)

// A segment reports the delivery of each word as it is spoken, in expressive
// conversation.
func TestBotOutputReportsTheDeliveryOfEachWordAsItIsSpoken(t *testing.T) {
	seg := segment("I'm back", frames.TextTypeSentence)
	first := frames.NewAggregatedTextProgressFrame(seg.ID(), "", "I'm back", frames.TextTypeSentence, "I'm", " back")
	first.Delivery = []string{"calm"}
	second := frames.NewAggregatedTextProgressFrame(seg.ID(), "", "I'm back", frames.TextTypeSentence, "I'm back", "")
	second.Delivery = []string{"excited", "emphasis"}

	msgs := outputHarness(t, rtvi.DefaultObserverParams(), rtvi.ProtocolVersion,
		speaking(), seg, first, second)

	type report struct {
		status   rtvi.SpokenStatus
		delivery []string
	}
	var got []report
	for _, d := range outputs(msgs) {
		got = append(got, report{d.SpokenStatus, d.Delivery})
	}
	want := []report{
		{rtvi.SpokenNew, nil},
		{rtvi.SpokenInProgress, []string{"calm"}},
		{rtvi.SpokenCompleted, []string{"excited", "emphasis"}},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("reports = %+v, want %+v", got, want)
	}
}

// An expression sound is a segment of its own once performed.
func TestBotOutputReportsAnExpressionSoundAsASegmentOfItsOwn(t *testing.T) {
	laugh := spokenText("laugh", frames.TextTypeExpressionSound)
	laugh.ContextID = "turn"

	got := outputs(outputHarness(t, rtvi.DefaultObserverParams(), rtvi.ProtocolVersion, speaking(), laugh))

	if len(got) != 1 {
		t.Fatalf("got %d bot-output messages, want 1", len(got))
	}
	d := got[0]
	if d.Text != "laugh" || d.OutputType != frames.TextTypeExpressionSound || d.SpokenStatus != rtvi.SpokenCompleted ||
		d.SegmentID == nil || *d.SegmentID != laugh.ID() || d.Delivery != nil {
		t.Fatalf("output = %+v", d)
	}
}

// A unit that is only tags, announced and then spoken, has no words to show.
func TestBotOutputLeavesOutTextWithNoWordsToShow(t *testing.T) {
	seg := segment("", frames.TextTypeSentence)
	spoken := spokenText("", frames.TextTypeSentence)
	spoken.ContextID = "turn"
	spoken.RawText = "<<calm>>"

	msgs := outputHarness(t, rtvi.DefaultObserverParams(), rtvi.ProtocolVersion, speaking(), seg, spoken)
	if got := outputs(msgs); len(got) != 0 {
		t.Fatalf("sent %+v", got)
	}
}

// An expression sound is sent as output alone, not as TTS text, which a client
// takes for words the bot spoke.
func TestAnExpressionSoundIsSentAsOutputAlone(t *testing.T) {
	params := rtvi.DefaultObserverParams()
	params.TTSRawTextEnabled = true
	laugh := spokenText("laugh", frames.TextTypeExpressionSound)
	pause := spokenText("400", frames.TextTypeExpressionPause)

	msgs := outputHarness(t, params, "2.3.0", speaking(), laugh, pause)

	var got []string
	for _, m := range msgs {
		if m.Type == rtvi.TypeBotOutput || m.Type == rtvi.TypeTTSRawText || m.Type == rtvi.TypeBotTTSText {
			got = append(got, m.Type)
		}
	}
	if want := []string{rtvi.TypeBotOutput, rtvi.TypeBotOutput}; !reflect.DeepEqual(got, want) {
		t.Fatalf("messages = %q, want %q", got, want)
	}
}

// An expression sound is kept from a client older than protocol 2.3.0, which
// would take it for part of the bot's response.
func TestAnExpressionSoundIsKeptFromOlderClients(t *testing.T) {
	for _, version := range []string{"2.2.0", "1.4.0"} {
		t.Run(version, func(t *testing.T) {
			msgs := outputHarness(t, rtvi.DefaultObserverParams(), version, speaking(),
				spokenText("laugh", frames.TextTypeExpressionSound),
				spokenText("laugh", frames.TextTypeExpressionPause))

			for _, m := range msgs {
				switch m.Type {
				case rtvi.TypeBotOutput, rtvi.TypeTTSRawText, rtvi.TypeBotTTSText:
					t.Fatalf("sent %s: %+v", m.Type, m.Data)
				}
			}
		})
	}
}
