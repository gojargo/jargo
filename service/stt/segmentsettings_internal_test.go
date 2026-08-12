package stt

import (
	"context"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/settings"
)

// Tests for changing a segmented service's settings while the pipeline runs.
// A segmented service transcribes each utterance on its own, so an update takes
// effect on the next segment rather than needing a session replaced.

// settingsTranscriber is a transcriber that holds settings, names languages its
// own way, and records what it was told changed.
type settingsTranscriber struct {
	mu      sync.Mutex
	store   STTSettings
	changed []settings.Changed
	reopen  bool
	meta    Metadata
}

func (t *settingsTranscriber) Transcribe(context.Context, []byte, int) (string, error) {
	return "", nil
}

func (t *settingsTranscriber) Metadata() Metadata { return t.meta }

func (t *settingsTranscriber) Settings() any { return &t.store }

func (t *settingsTranscriber) UpdateSettings(
	_ context.Context, changed settings.Changed,
) (bool, error) {
	t.mu.Lock()
	t.changed = append(t.changed, changed)
	t.mu.Unlock()
	return t.reopen, nil
}

// ServiceLanguage names a language the way this provider does.
func (t *settingsTranscriber) ServiceLanguage(l language.Language) string {
	if l.BaseCode() == "fr" {
		return "fr-provider"
	}
	return ""
}

func (t *settingsTranscriber) reported() []settings.Changed {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]settings.Changed(nil), t.changed...)
}

// updateFrame builds an update carrying the named field.
func updateFrame(t *testing.T, field string, value any) *frames.STTUpdateSettingsFrame {
	t.Helper()
	delta := &STTSettings{}
	if err := settings.SetNamed(delta, field, value); err != nil {
		t.Fatalf("SetNamed(%s): %v", field, err)
	}
	return frames.NewSTTUpdateSettingsFrame(delta)
}

// TestSegmentServiceAppliesASettingsUpdate checks an update reaches the
// provider's own store, so the next segment is transcribed with it.
func TestSegmentServiceAppliesASettingsUpdate(t *testing.T) {
	tr := &settingsTranscriber{meta: Metadata{Model: "whisper-1"}}
	s := NewSegment("TestSTT", tr, 0)

	s.updateSettings(context.Background(), updateFrame(t, "model", "whisper-2"))

	if v, _ := settings.Get(&tr.store, "model"); v != "whisper-2" {
		t.Errorf("the store holds %v, want the model the update named", v)
	}
	// And the label follows it, or the cost of what comes next lands on the
	// model no longer in use.
	if s.modelName() != "whisper-2" {
		t.Errorf("model = %q, want the one now in force", s.modelName())
	}

	reported := tr.reported()
	if len(reported) != 1 || !reported[0].Has("model") {
		t.Errorf("the provider was told %v changed, want the model", reported)
	}
}

// TestSegmentServiceNamesTheLanguageTheProvidersWay checks a neutrally named
// language is converted before it is stored, so the store holds the code the
// provider itself uses.
func TestSegmentServiceNamesTheLanguageTheProvidersWay(t *testing.T) {
	tr := &settingsTranscriber{}
	s := NewSegment("TestSTT", tr, 0)

	s.updateSettings(context.Background(), updateFrame(t, "language", "fr"))

	if v, _ := settings.Get(&tr.store, "language"); v != "fr-provider" {
		t.Errorf("the store holds %v, want the provider's own code", v)
	}
}

// TestSegmentServiceIgnoresAReopenRequest checks a provider asking for its
// session to be replaced is not an error here. There is no session: the next
// segment simply reads the settings as they now stand.
func TestSegmentServiceIgnoresAReopenRequest(t *testing.T) {
	tr := &settingsTranscriber{reopen: true}
	s := NewSegment("TestSTT", tr, 0)

	s.updateSettings(context.Background(), updateFrame(t, "model", "whisper-2"))

	if v, _ := settings.Get(&tr.store, "model"); v != "whisper-2" {
		t.Errorf("the store holds %v, want the update applied regardless", v)
	}
}

// TestSegmentServiceReportsNothingForAnUnchangedSetting checks re-sending what
// the provider already holds is not reported as a change, so a provider that
// reacts to one is not made to react to a non-change.
func TestSegmentServiceReportsNothingForAnUnchangedSetting(t *testing.T) {
	tr := &settingsTranscriber{}
	if err := settings.SetNamed(&tr.store, "model", "whisper-1"); err != nil {
		t.Fatal(err)
	}
	s := NewSegment("TestSTT", tr, 0)

	s.updateSettings(context.Background(), updateFrame(t, "model", "whisper-1"))

	if got := tr.reported(); len(got) != 0 {
		t.Errorf("the provider was told %v changed, want nothing", got)
	}
}

// TestSegmentServiceWithoutSettingsIsQuiet checks an update aimed at a provider
// that holds no settings does nothing rather than failing. Most transcribers
// are built once and never change.
func TestSegmentServiceWithoutSettingsIsQuiet(t *testing.T) {
	s := NewSegment("TestSTT", &describingTranscriber{meta: Metadata{Model: "fixed"}}, 0)

	s.updateSettings(context.Background(), updateFrame(t, "model", "other"))

	if s.modelName() != "fixed" {
		t.Errorf("model = %q, want the one it was built with", s.modelName())
	}
}

// TestSegmentServiceLeavesAnUpdateForAnotherService checks an update naming a
// different service is not applied here. It travels on instead, so the service
// it was meant for gets it.
func TestSegmentServiceLeavesAnUpdateForAnotherService(t *testing.T) {
	tr := &settingsTranscriber{}
	s := NewSegment("TestSTT", tr, 0)

	f := updateFrame(t, "model", "whisper-2")
	f.Service = otherService("someone else")

	// The routing predicate ProcessFrame reads.
	if f.TargetsService(s) {
		t.Fatal("an update naming another service was read as meant for this one")
	}

	if err := s.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatalf("ProcessFrame: %v", err)
	}
	if v, _ := settings.Get(&tr.store, "model"); v != nil {
		t.Errorf("the store holds %v, want the update left for its target", v)
	}
	if got := tr.reported(); len(got) != 0 {
		t.Errorf("the provider was told %v changed, want nothing", got)
	}
}

// TestSegmentServiceTraceSettingsCarryTheStore checks the span covering a
// transcription reports what the provider is configured with.
func TestSegmentServiceTraceSettingsCarryTheStore(t *testing.T) {
	tr := &settingsTranscriber{}
	if err := settings.SetNamed(&tr.store, "model", "whisper-1"); err != nil {
		t.Fatal(err)
	}
	s := NewSegment("TestSTT", tr, 0)

	given := s.set.traceSettings()
	if given["model"] != "whisper-1" {
		t.Errorf("the span carries %v, want the model the provider holds", given)
	}
}
