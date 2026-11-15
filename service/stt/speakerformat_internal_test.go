package stt

import (
	"context"
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
			if got := svc.applySpeakerFormat("hello", tc.speaker); got != tc.want {
				t.Errorf("applySpeakerFormat = %q, want %q", got, tc.want)
			}
		})
	}
}

// Ported from upstream (test_apply_speaker_format_without_the_setting). A
// service whose settings have no speaker_format leaves the text unchanged.
func TestApplySpeakerFormatWithoutTheSetting(t *testing.T) {
	speakerA := &frames.TranscriptionSpeaker{Label: "A", Number: 2}
	svc := NewStream("PlainSTT", &settingsConnector{}, 16000)
	if got := svc.applySpeakerFormat("hello", speakerA); got != "hello" {
		t.Errorf("applySpeakerFormat = %q, want the text unchanged", got)
	}
	none := NewStream("NoSettingsSTT", &diarizingConnectorWithoutSettings{}, 16000)
	if got := none.applySpeakerFormat("hello", speakerA); got != "hello" {
		t.Errorf("applySpeakerFormat = %q, want the text unchanged", got)
	}
}

// diarizingConnectorWithoutSettings keeps no settings at all.
type diarizingConnectorWithoutSettings struct{}

func (diarizingConnectorWithoutSettings) Connect(context.Context, int) (Stream, error) {
	return nil, errNoSession
}
