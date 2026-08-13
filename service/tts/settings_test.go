package tts_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/tts"
)

// settingsSynth holds settings the way a provider does, names languages its own
// way, and records what it was told changed.
type settingsSynth struct {
	mu      sync.Mutex
	store   settings.TTS
	changed []settings.Changed
	// names says whether this provider has its own language codes.
	names bool
}

func (s *settingsSynth) SampleRate() int { return 24000 }

func (s *settingsSynth) RunTTS(context.Context, string, string, func(frames.Frame) error) error {
	return nil
}

func (s *settingsSynth) Settings() any { return &s.store }

func (s *settingsSynth) UpdateSettings(_ context.Context, changed settings.Changed) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.changed = append(s.changed, changed)
	return nil
}

// ServiceLanguage names a language the way this provider does: the base code
// only, so "fr-CA" and "fr" both become "fr".
func (s *settingsSynth) ServiceLanguage(l language.Language) string {
	if !s.names {
		return ""
	}
	return l.BaseCode()
}

func (s *settingsSynth) updates() []settings.Changed {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]settings.Changed(nil), s.changed...)
}

func (s *settingsSynth) voice() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.store.Voice.Value()
}

func (s *settingsSynth) lang() (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.store.Language.Value()
}

func runTTSService(t *testing.T, svc *tts.Base) (*pipeline.Worker, func()) {
	t.Helper()
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	return task, func() {
		task.StopWhenDone()
		select {
		case <-done:
		case <-time.After(3 * time.Second):
			t.Error("pipeline did not shut down")
		}
	}
}

func waitForTTSUpdates(t *testing.T, syn *settingsSynth, n int) {
	t.Helper()
	deadline := time.After(3 * time.Second)
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	for len(syn.updates()) < n {
		select {
		case <-tick.C:
		case <-deadline:
			t.Fatalf("timed out waiting for %d settings updates, saw %d", n, len(syn.updates()))
		}
	}
}

func TestTTSSettingsUpdateReachesTheProvider(t *testing.T) {
	syn := &settingsSynth{store: settings.TTS{Voice: settings.Set("alice")}}
	svc := tts.New("FakeTTS", syn)
	task, stop := runTTSService(t, svc)
	defer stop()

	task.QueueFrame(frames.NewTTSUpdateSettingsFrame(&settings.TTS{Voice: settings.Set("bob")}))
	waitForTTSUpdates(t, syn, 1)

	changed := syn.updates()[0]
	if !changed.Has("voice") {
		t.Errorf("changed = %v, want voice reported", changed)
	}
	if changed["voice"] != "alice" {
		t.Errorf("previous voice = %v, want alice", changed["voice"])
	}
	if v, _ := syn.voice(); v != "bob" {
		t.Errorf("stored voice = %q, want bob", v)
	}
}

// A language is stored the way the provider names it, so the store and the next
// update speak the same vocabulary.
func TestTTSSettingsUpdateNamesTheLanguageTheProvidersWay(t *testing.T) {
	syn := &settingsSynth{names: true}
	svc := tts.New("FakeTTS", syn)
	task, stop := runTTSService(t, svc)
	defer stop()

	task.QueueFrame(frames.NewTTSUpdateSettingsFrame(&settings.TTS{
		Language: settings.Set(string(language.FrenchCA)),
	}))
	waitForTTSUpdates(t, syn, 1)

	if v, _ := syn.lang(); v != "fr" {
		t.Errorf("stored language = %q, want fr: the provider names it by base code", v)
	}
}

// Naming happens before the comparison, so a neutral name meaning the language
// already stored is not mistaken for a change.
func TestTTSSettingsUpdateToTheSameLanguageIsNotAChange(t *testing.T) {
	syn := &settingsSynth{names: true, store: settings.TTS{Language: settings.Set("fr")}}
	svc := tts.New("FakeTTS", syn)
	task, stop := runTTSService(t, svc)
	defer stop()

	// "fr-CA" names the same language this provider already holds as "fr".
	task.QueueFrame(frames.NewTTSUpdateSettingsFrame(&settings.TTS{
		Language: settings.Set(string(language.FrenchCA)),
	}))
	// A real change behind it, so there is something to wait for.
	task.QueueFrame(frames.NewTTSUpdateSettingsFrame(&settings.TTS{Voice: settings.Set("bob")}))
	waitForTTSUpdates(t, syn, 1)
	time.Sleep(50 * time.Millisecond)

	got := syn.updates()
	if len(got) != 1 {
		t.Fatalf("updates = %v, want only the voice change", got)
	}
	if got[0].Has("language") {
		t.Errorf("changed = %v, want no language change: it was already that language", got[0])
	}
}

// voice_id is the spelling a provider's own API uses, and it means voice.
func TestTTSSettingsUpdateResolvesTheVoiceAlias(t *testing.T) {
	syn := &settingsSynth{store: settings.TTS{Voice: settings.Set("alice")}}
	svc := tts.New("FakeTTS", syn)
	task, stop := runTTSService(t, svc)
	defer stop()

	f := frames.NewTTSUpdateSettingsFrame(nil)
	f.Settings = map[string]any{"voice_id": "bob"}
	task.QueueFrame(f)
	waitForTTSUpdates(t, syn, 1)

	if v, _ := syn.voice(); v != "bob" {
		t.Errorf("stored voice = %q, want bob", v)
	}
}

func TestTTSSettingsUpdateForAnotherServiceIsNotApplied(t *testing.T) {
	syn := &settingsSynth{store: settings.TTS{Voice: settings.Set("alice")}}
	svc := tts.New("FakeTTS", syn)
	task, stop := runTTSService(t, svc)
	defer stop()

	f := frames.NewTTSUpdateSettingsFrame(&settings.TTS{Voice: settings.Set("bob")})
	f.Service = otherTTS("SomeOtherTTS")
	task.QueueFrame(f)
	time.Sleep(100 * time.Millisecond)

	if got := syn.updates(); len(got) != 0 {
		t.Errorf("updates = %v, want none: the frame named another service", got)
	}
	if v, _ := syn.voice(); v != "alice" {
		t.Errorf("stored voice = %q, want alice", v)
	}
}

// otherTTS stands in for a different service the frame could name.
type otherTTS string

func (o otherTTS) Name() string { return string(o) }
