package stt

import (
	"context"
	"reflect"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// diarizingSettings stands in for a provider whose settings carry a speaker
// format.
type diarizingSettings struct {
	settings.STT

	SpeakerFormat settings.Opt[string] `settings:"speaker_format"`
}

// diarizingConnector holds diarizingSettings.
type diarizingConnector struct {
	store diarizingSettings
}

func (c *diarizingConnector) Connect(context.Context, int) (Stream, error) { return nil, errNoSession }

func (c *diarizingConnector) Settings() any { return &c.store }

// Ported from upstream (test_apply_speaker_format). The format is applied only
// when both a format and a speaker are present.
func TestApplySpeakerFormat(t *testing.T) {
	speakerA := &frames.TranscriptionSpeaker{Label: "A", Number: 2}
	cases := []struct {
		name    string
		format  string
		speaker *frames.TranscriptionSpeaker
		want    string
	}{
		{"label", "{speaker}: {text}", speakerA, "A: hello"},
		{"number", "Speaker {speaker_number}: {text}", speakerA, "Speaker 2: hello"},
		{"no speaker", "{speaker}: {text}", nil, "hello"},
		{"no format", "", speakerA, "hello"},
		{"escaped braces", "{{{speaker}}} {text}", speakerA, "{A} hello"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			conn := &diarizingConnector{}
			if tc.format != "" {
				conn.store.SpeakerFormat = settings.Set(tc.format)
			}
			svc := NewStream("DiarizingSTT", conn, 16000)
			if got := svc.ApplySpeakerFormat("hello", tc.speaker); got != tc.want {
				t.Errorf("ApplySpeakerFormat = %q, want %q", got, tc.want)
			}
		})
	}
}

// Ported from upstream (test_apply_speaker_format_without_the_setting). A
// service whose settings have no speaker_format leaves the text unchanged.
func TestApplySpeakerFormatWithoutTheSetting(t *testing.T) {
	speakerA := &frames.TranscriptionSpeaker{Label: "A", Number: 2}
	svc := NewStream("PlainSTT", &settingsConnector{}, 16000)
	if got := svc.ApplySpeakerFormat("hello", speakerA); got != "hello" {
		t.Errorf("ApplySpeakerFormat = %q, want the text unchanged", got)
	}
	none := NewStream("NoSettingsSTT", &diarizingConnectorWithoutSettings{}, 16000)
	if got := none.ApplySpeakerFormat("hello", speakerA); got != "hello" {
		t.Errorf("ApplySpeakerFormat = %q, want the text unchanged", got)
	}
}

// diarizingConnectorWithoutSettings keeps no settings at all.
type diarizingConnectorWithoutSettings struct{}

func (diarizingConnectorWithoutSettings) Connect(context.Context, int) (Stream, error) {
	return nil, errNoSession
}

// Ported from upstream (test_speaker_runs_from_word_pairs). Speaker runs are
// built from plain (text, speaker) pairs.
func TestSpeakerRunsFromWordPairs(t *testing.T) {
	cases := []struct {
		name  string
		words []SpeakerWord
		want  []SpeakerRun
	}{
		{
			"two speakers",
			[]SpeakerWord{{"Hi", 0}, {"there", 0}, {"hello", 1}},
			[]SpeakerRun{{0, "Hi there"}, {1, "hello"}},
		},
		{"a word with no speaker", []SpeakerWord{{"Hi", 0}, {"there", nil}}, nil},
		{"a word not in the transcript", []SpeakerWord{{"Hi", 0}, {"missing", 1}}, nil},
		{"an empty word", []SpeakerWord{{"Hi", 0}, {"", 1}}, nil},
		{"no words", nil, nil},
	}
	for _, tc := range cases {
		if got := SpeakerRuns(" Hi there hello ", tc.words); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s: SpeakerRuns = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// Ported from upstream (test_speaker_runs_locate_repeated_words_in_order). A
// repeated word is matched after the previous one, not at its first occurrence.
func TestSpeakerRunsLocateRepeatedWordsInOrder(t *testing.T) {
	got := SpeakerRuns("no, no, no.", []SpeakerWord{{"no", 0}, {"no", 1}, {"no", 0}})
	want := []SpeakerRun{{0, "no,"}, {1, "no,"}, {0, "no."}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("SpeakerRuns = %v, want %v", got, want)
	}
}

// Ported from upstream (test_speaker_runs_keep_provider_label_type). String
// labels pass through unchanged.
func TestSpeakerRunsKeepProviderLabelType(t *testing.T) {
	got := SpeakerRuns("Hi. Hello.", []SpeakerWord{{"Hi.", "A"}, {"Hello.", "B"}})
	want := []SpeakerRun{{"A", "Hi."}, {"B", "Hello."}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("SpeakerRuns = %v, want %v", got, want)
	}
}

// Ported from upstream (test_speaker_numbers_follow_order_of_first_appearance).
// Labels are numbered in the order speakers first appear, whatever the provider
// calls them, and both kinds of service number them the same way.
func TestSpeakerNumbersFollowOrderOfFirstAppearance(t *testing.T) {
	for name, d := range map[string]interface {
		Speaker(label any) *frames.TranscriptionSpeaker
	}{
		"stream":  NewStream("PlainSTT", &settingsConnector{}, 16000),
		"segment": NewSegment("PlainSTT", &settingsTranscriber{}, 16000),
	} {
		var got []frames.TranscriptionSpeaker
		for _, label := range []any{"S7", "A", "S7", 0, "A"} {
			got = append(got, *d.Speaker(label))
		}
		want := []frames.TranscriptionSpeaker{
			{Label: "S7", Number: 1},
			{Label: "A", Number: 2},
			{Label: "S7", Number: 1},
			{Label: "0", Number: 3},
			{Label: "A", Number: 2},
		}
		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s: speakers = %v, want %v", name, got, want)
		}
		if sp := d.Speaker(nil); sp != nil {
			t.Errorf("%s: Speaker(nil) = %v, want nil", name, sp)
		}
	}
}

// Ported from upstream (test_diarized_runs_give_each_run_its_speaker_and_format).
func TestDiarizedRunsGiveEachRunItsSpeakerAndFormat(t *testing.T) {
	conn := &diarizingConnector{}
	conn.store.SpeakerFormat = settings.Set("{speaker}: {text}")
	svc := NewStream("DiarizingSTT", conn, 16000)

	got := svc.DiarizedRuns("Hi. Hello.", []SpeakerWord{{"Hi.", 0}, {"Hello.", 1}})
	want := []DiarizedRun{
		{&frames.TranscriptionSpeaker{Label: "0", Number: 1}, "0: Hi."},
		{&frames.TranscriptionSpeaker{Label: "1", Number: 2}, "1: Hello."},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("DiarizedRuns = %v, want %v", got, want)
	}
}

// Ported from upstream (test_diarized_runs_fall_back_to_the_whole_transcript).
func TestDiarizedRunsFallBackToTheWholeTranscript(t *testing.T) {
	for name, words := range map[string][]SpeakerWord{
		"no speaker information": nil,
		"no words":               {},
		"a word not found":       {{"Hi.", 0}, {"missing", 1}},
	} {
		conn := &diarizingConnector{}
		conn.store.SpeakerFormat = settings.Set("{speaker}: {text}")
		svc := NewStream("DiarizingSTT", conn, 16000)

		got := svc.DiarizedRuns("Hi. Hello.", words)
		if want := []DiarizedRun{{nil, "Hi. Hello."}}; !reflect.DeepEqual(got, want) {
			t.Errorf("%s: DiarizedRuns = %v, want %v", name, got, want)
		}
	}
}
