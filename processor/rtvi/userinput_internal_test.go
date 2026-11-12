package rtvi

import (
	"encoding/json"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Tests for the user-input message: what the user said or typed. They drive the
// observer's frame mapping directly, as it runs on a frame's first push.

func userInputObserver(params ObserverParams) *segmentObserver {
	return &segmentObserver{Observer: NewObserverWithParams(nil, params)}
}

func TestAUserBackchannelIsSentAsUserInput(t *testing.T) {
	o := userInputObserver(DefaultObserverParams())
	o.push(frames.NewUserBackchannelFrame("mhm", "user", "now"))

	if len(o.sent) != 1 || o.sent[0].Type != TypeUserInput {
		t.Fatalf("sent %+v, want one user-input", o.sent)
	}
	want := UserInputData{Text: "mhm", InputType: UserInputBackchannel, UserID: new("user"), Timestamp: "now", Final: true}
	if got := o.sent[0].Data; !reflect.DeepEqual(got, want) {
		t.Errorf("data = %+v, want %+v", got, want)
	}
}

func TestATranscriptionIsSentAsUserInputThenUserTranscription(t *testing.T) {
	o := userInputObserver(DefaultObserverParams())
	o.push(frames.NewInterimTranscriptionFrame("Hel", "user", "t1"))
	o.push(frames.NewTranscriptionFrame("Hello.", "user", "t2"))

	want := []string{TypeUserInput, TypeUserTranscription, TypeUserInput, TypeUserTranscription}
	if got := o.types(); !slices.Equal(got, want) {
		t.Fatalf("sent %v, want %v", got, want)
	}
	wantData := []UserInputData{
		{Text: "Hel", InputType: UserInputTranscription, UserID: new("user"), Timestamp: "t1", Final: false},
		{Text: "Hello.", InputType: UserInputTranscription, UserID: new("user"), Timestamp: "t2", Final: true},
	}
	if got := []any{o.sent[0].Data, o.sent[2].Data}; !reflect.DeepEqual(got[0], wantData[0]) ||
		!reflect.DeepEqual(got[1], wantData[1]) {
		t.Errorf("data = %+v, want %+v", got, wantData)
	}
}

func TestATranscriptionIsOnlySentAsUserInputWithoutUserTranscription(t *testing.T) {
	params := DefaultObserverParams()
	off := false
	params.UserTranscriptionEnabled = &off
	o := userInputObserver(params)
	o.push(frames.NewTranscriptionFrame("Hello.", "user", "now"))

	if got := o.types(); !slices.Equal(got, []string{TypeUserInput}) {
		t.Fatalf("sent %v, want only user-input", got)
	}
}

func TestTextFromTheClientIsAcknowledgedAsUserInput(t *testing.T) {
	o := userInputObserver(DefaultObserverParams())
	o.push(NewSendTextFrame("msg-1", "Hello.", ""))

	if len(o.sent) != 1 || o.sent[0].Type != TypeUserInput {
		t.Fatalf("sent %+v, want one user-input", o.sent)
	}
	d, _ := o.sent[0].Data.(UserInputData)
	if d.Text != "Hello." || d.InputType != UserInputChat || !d.Final {
		t.Errorf("data = %+v, want the text as final chat", d)
	}
	if d.MsgID != "msg-1" || d.UserID != nil {
		t.Errorf("data = %+v, want msg-1 from nobody named", d)
	}
	encoded, err := json.Marshal(o.sent[0])
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Data map[string]any `json:"data"`
	}
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	if _, ok := wire.Data["user_id"]; ok {
		t.Errorf("wire data %v carries a user_id for nobody named", wire.Data)
	}
}

func TestTextFromAKnownSenderIsAcknowledgedWithTheirID(t *testing.T) {
	o := userInputObserver(DefaultObserverParams())
	o.push(NewSendTextFrame("msg-1", "Hello.", "user"))

	if d, _ := o.sent[0].Data.(UserInputData); d.MsgID != "msg-1" || d.UserID == nil || *d.UserID != "user" {
		t.Errorf("data = %+v, want msg-1 from user", d)
	}
}

func TestUserInputIsKeptFromTheClientWhenDisabled(t *testing.T) {
	params := DefaultObserverParams()
	off := false
	params.UserInputEnabled = &off
	o := userInputObserver(params)
	o.push(frames.NewTranscriptionFrame("Hello.", "user", "now"))
	o.push(frames.NewUserBackchannelFrame("mhm", "user", "now"))
	o.push(NewSendTextFrame("msg-1", "Hello.", ""))

	if got := o.types(); !slices.Equal(got, []string{TypeUserTranscription}) {
		t.Fatalf("sent %v, want only user-transcription", got)
	}
}

// Ported from upstream (the speaker cases, which build every transcription
// with user_id ""). A transcription's user id is sent even when it is empty:
// only a chat message whose sender the transport does not know leaves it out.
func TestATranscriptionCarriesAnEmptyUserID(t *testing.T) {
	o := userInputObserver(DefaultObserverParams())
	o.push(frames.NewTranscriptionFrame("Hello.", "", "now"))

	encoded, err := json.Marshal(o.sent[0])
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(encoded), `"user_id":""`) {
		t.Errorf("wire %s, want the empty user_id sent", encoded)
	}
}

// A response that carried no text still reports its empty raw text and the
// markers, as an empty list when there were none.
func TestAMarkerReportWithoutTextCarriesRawAndMarkers(t *testing.T) {
	raw, err := json.Marshal(BotLLMMarker(BotLLMMarkerData{Markers: []string{}}).Data)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`"raw":""`, `"markers":[]`} {
		if !strings.Contains(string(raw), want) {
			t.Errorf("wire %s, want %s", raw, want)
		}
	}
}
