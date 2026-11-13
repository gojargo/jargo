package stt

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
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

// The update is the caller's: naming its language the provider's way is done on
// the service's own copy, so the caller's reads afterwards as it was sent.
func TestALanguageUpdateLeavesTheCallersDeltaUnchanged(t *testing.T) {
	tr := &settingsTranscriber{}
	s := NewSegment("TestSTT", tr, 0)
	f := updateFrame(t, "language", "fr")

	s.updateSettings(context.Background(), f)

	if v, _ := settings.Get(f.Delta, "language"); v != "fr" {
		t.Errorf("the caller's update holds %v after it was applied, want fr as sent", v)
	}
	if v, _ := settings.Get(&tr.store, "language"); v != "fr-provider" {
		t.Errorf("the store holds %v, want the provider's own code", v)
	}
}

// preparingTranscriber moves a language it is given into a field of its own
// before the service sees it, the way a provider whose language is spelled as
// hints does, and records the language as it was handed over.
type preparingTranscriber struct {
	settingsTranscriber
	prepared []any
}

func (t *preparingTranscriber) PrepareSettings(delta any) error {
	v, _ := settings.Get(delta, "language")
	t.prepared = append(t.prepared, v)
	return settings.SetNamed(delta, "language", nil)
}

// A provider that prepares an update sees it before its language is named, and
// what it does to it is what the service then applies.
func TestAProviderSeesTheUpdateBeforeItsLanguageIsNamed(t *testing.T) {
	tr := &preparingTranscriber{}
	s := NewSegment("TestSTT", tr, 0)
	f := updateFrame(t, "language", "fr")

	s.updateSettings(context.Background(), f)

	if len(tr.prepared) != 1 || tr.prepared[0] != "fr" {
		t.Errorf("the provider was handed %v, want the language as sent", tr.prepared)
	}
	if v, ok := settings.Get(&tr.store, "language"); ok {
		t.Errorf("the store holds %v, want what the provider prepared: no language", v)
	}
	if v, _ := settings.Get(f.Delta, "language"); v != "fr" {
		t.Errorf("the caller's update holds %v, want fr as sent", v)
	}
}

// baseCodeTranscriber names the languages it lists by their base code, the way
// a provider that takes no regional variants does.
type baseCodeTranscriber struct {
	settingsTranscriber
	named []language.Language
}

func (t *baseCodeTranscriber) ServiceLanguage(l language.Language) string {
	t.mu.Lock()
	t.named = append(t.named, l)
	t.mu.Unlock()
	codes := map[language.Language]string{language.German: "de", language.English: "en", language.French: "fr"}
	return language.Resolve(l, codes, true)
}

// A regional language the provider does not list is named by the base code it
// does list, whether the update names it by its constant or writes it out.
func TestARegionalLanguageIsNamedByTheBaseCodeTheProviderLists(t *testing.T) {
	for _, code := range []string{string(language.FrenchFR), "fr-FR"} {
		tr := &baseCodeTranscriber{}
		s := NewSegment("TestSTT", tr, 0)

		s.updateSettings(context.Background(), updateFrame(t, "language", code))

		if v, _ := settings.Get(&tr.store, "language"); v != "fr" {
			t.Errorf("updating to %q stored %v, want fr", code, v)
		}
	}
}

// A code naming no language jargo knows may still be one the provider takes, so
// it is stored as it came, without the provider being asked to name it, and
// the log says so.
func TestAnUnrecognizedLanguageIsPassedToTheProviderAsItIs(t *testing.T) {
	var logs bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	tr := &baseCodeTranscriber{}
	s := NewSegment("TestSTT", tr, 0)

	s.updateSettings(context.Background(), updateFrame(t, "language", "klingon"))

	if v, _ := settings.Get(&tr.store, "language"); v != "klingon" {
		t.Errorf("the store holds %v, want klingon as it came", v)
	}
	if len(tr.named) != 0 {
		t.Errorf("the provider was asked to name %v, want nothing asked of it", tr.named)
	}
	var notes []string
	for line := range strings.Lines(logs.String()) {
		if strings.Contains(line, "not a recognized language code") {
			notes = append(notes, line)
		}
	}
	if len(notes) != 1 || !strings.Contains(notes[0], "klingon") {
		t.Errorf("logged %q, want one debug line naming klingon", notes)
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
