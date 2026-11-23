package stt

import (
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// SpeakerWord is one word of a transcript with the provider's speaker label
// for it, as SpeakerRuns and DiarizedRuns take them.
type SpeakerWord struct {
	// Text is the word as it appears in the transcript.
	Text string
	// Speaker is the provider's speaker label, of whatever type the provider
	// uses, or nil when the word has none.
	Speaker any
}

// SpeakerRun is a contiguous stretch of a transcript spoken by one speaker, with
// the provider's label for that speaker.
type SpeakerRun struct {
	Speaker any
	Text    string
}

// DiarizedRun is a contiguous stretch of a transcript spoken by one speaker,
// with its text formatted by the service's speaker_format setting.
type DiarizedRun struct {
	// Speaker is the speaker of the run, or nil when the result could not be
	// told apart by speaker.
	Speaker *frames.TranscriptionSpeaker
	Text    string
}

// SpeakerRuns splits a transcript into contiguous speaker runs.
//
// Each word is located in transcript in order, and the transcript is cut where
// the speaker changes, so each run keeps the provider's text and punctuation.
// Runs are stripped of surrounding whitespace. Labels are compared as the
// provider gave them and returned unchanged.
//
// It returns nil when any word has no speaker or cannot be located in the
// transcript.
func SpeakerRuns(transcript string, words []SpeakerWord) []SpeakerRun {
	if transcript == "" || len(words) == 0 {
		return nil
	}

	// Where each run starts. The first starts at 0, so text before the first
	// word (an audio event, say) stays in it.
	type boundary struct {
		speaker any
		start   int
	}
	var boundaries []boundary
	cursor := 0
	for _, w := range words {
		if w.Speaker == nil || w.Text == "" {
			return nil
		}
		offset := strings.Index(transcript[cursor:], w.Text)
		if offset < 0 {
			return nil
		}
		start := cursor + offset
		switch {
		case len(boundaries) == 0:
			boundaries = append(boundaries, boundary{speaker: w.Speaker, start: 0})
		case w.Speaker != boundaries[len(boundaries)-1].speaker:
			boundaries = append(boundaries, boundary{speaker: w.Speaker, start: start})
		}
		cursor = start + len(w.Text)
	}

	var runs []SpeakerRun
	for i, b := range boundaries {
		end := len(transcript)
		if i+1 < len(boundaries) {
			end = boundaries[i+1].start
		}
		if text := strings.TrimSpace(transcript[b.start:end]); text != "" {
			runs = append(runs, SpeakerRun{Speaker: b.speaker, Text: text})
		}
	}
	return runs
}

// diarization is the speaker bookkeeping both kinds of transcription service
// share: the speakers a provider's diarization told apart, numbered in the
// order they first appeared in the session, and the speaker_format setting
// applied to what they said. Services that support diarization put each
// transcript's speaker, from Speaker, on the frame, splitting a final result
// into one frame per contiguous speaker run with DiarizedRuns.
type diarization struct {
	// set is the service's settings, read for the speaker_format setting.
	set *providerSettings
	// name identifies the service in a log line.
	name func() string

	mu       sync.Mutex
	speakers map[string]*frames.TranscriptionSpeaker
}

// Speaker is the speaker for a provider's diarization label, of any type, or
// nil for no label (nil, or an empty string). A label seen for the first time is
// numbered after the speakers before it, so numbers follow the order speakers
// first appeared in the session.
func (d *diarization) Speaker(label any) *frames.TranscriptionSpeaker {
	if label == nil {
		return nil
	}
	key := fmt.Sprint(label)
	if key == "" {
		return nil
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	if sp, ok := d.speakers[key]; ok {
		return sp
	}
	if d.speakers == nil {
		d.speakers = map[string]*frames.TranscriptionSpeaker{}
	}
	sp := &frames.TranscriptionSpeaker{Label: key, Number: len(d.speakers) + 1}
	d.speakers[key] = sp
	return sp
}

// ApplySpeakerFormat applies the service's speaker_format setting to a diarized
// transcript. The format uses {speaker} for the provider's speaker label,
// {speaker_number} for the speaker's number, which stays the same whatever
// labels the provider uses, and {text} for the transcript text (e.g.
// "Speaker {speaker_number}: {text}"). The text is returned unchanged when there
// is no speaker, no format, or a provider whose settings have no such setting.
func (d *diarization) ApplySpeakerFormat(text string, speaker *frames.TranscriptionSpeaker) string {
	if speaker == nil || d.set == nil {
		return text
	}
	format := d.set.speakerFormat()
	if format == "" {
		return text
	}
	return strings.NewReplacer(
		"{{", "{",
		"}}", "}",
		"{speaker}", speaker.Label,
		"{speaker_number}", strconv.Itoa(speaker.Number),
		"{text}", text,
	).Replace(format)
}

// DiarizedRuns splits a diarized result into runs by speaker, with each run's
// text formatted by speaker_format.
//
// words are the result's words in transcript order with the provider's speaker
// label for each, or nil when the result has no speaker information. A result
// with none, or whose words are empty or cannot be matched to the transcript
// (see SpeakerRuns), is a single run of the whole transcript with no speaker.
func (d *diarization) DiarizedRuns(transcript string, words []SpeakerWord) []DiarizedRun {
	if words == nil {
		return []DiarizedRun{{Text: transcript}}
	}
	runs := SpeakerRuns(transcript, words)
	if len(runs) == 0 {
		name := ""
		if d.name != nil {
			name = d.name()
		}
		slog.Debug("could not split the result by speaker, emitting it unsplit", "service", name)
		return []DiarizedRun{{Text: transcript}}
	}
	out := make([]DiarizedRun, 0, len(runs))
	for _, r := range runs {
		speaker := d.Speaker(r.Speaker)
		out = append(out, DiarizedRun{Speaker: speaker, Text: d.ApplySpeakerFormat(r.Text, speaker)})
	}
	return out
}
