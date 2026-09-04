package settings_test

import (
	"reflect"
	"testing"

	"github.com/gojargo/jargo/service/settings"
)

// store builds a TTS store: a service holding a voice and a language.
func store(voice, language string) *settings.TTS {
	return &settings.TTS{Voice: settings.Set(voice), Language: settings.Set(language)}
}

func TestApplyChangesOnlyWhatTheDeltaGives(t *testing.T) {
	t.Parallel()

	current := store("alice", "en")
	changed, err := settings.Apply(current, &settings.TTS{Voice: settings.Set("bob")})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if got := changed.Names(); !reflect.DeepEqual(got, []string{"voice"}) {
		t.Errorf("changed = %v, want [voice]", got)
	}
	if changed["voice"] != "alice" {
		t.Errorf("previous voice = %v, want alice", changed["voice"])
	}
	if v, _ := current.Voice.Value(); v != "bob" {
		t.Errorf("voice = %v, want bob", v)
	}
	if v, _ := current.Language.Value(); v != "en" {
		t.Errorf("language = %v, want en: a field the delta left alone must not move", v)
	}
}

func TestApplyReportsNothingWhenTheValueIsTheSame(t *testing.T) {
	t.Parallel()

	current := store("alice", "en")
	changed, err := settings.Apply(current, &settings.TTS{Voice: settings.Set("alice")})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if len(changed) != 0 {
		t.Errorf("changed = %v, want nothing: the value was already that", changed)
	}
	if v, _ := current.Voice.Value(); v != "alice" {
		t.Errorf("voice = %v, want alice", v)
	}
}

func TestApplyLeavesEverythingWhenNothingIsGiven(t *testing.T) {
	t.Parallel()

	current := store("alice", "en")
	changed, err := settings.Apply(current, &settings.TTS{})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if len(changed) != 0 {
		t.Errorf("changed = %v, want nothing", changed)
	}
	if v, _ := current.Voice.Value(); v != "alice" {
		t.Errorf("voice = %v, want alice", v)
	}
	if v, _ := current.Language.Value(); v != "en" {
		t.Errorf("language = %v, want en", v)
	}
}

func TestApplyChangesSeveralFieldsAtOnce(t *testing.T) {
	t.Parallel()

	current := &settings.LLM{Temperature: settings.Set(0.7), MaxTokens: settings.Set(100)}
	changed, err := settings.Apply(current, &settings.LLM{
		Temperature: settings.Set(0.9),
		MaxTokens:   settings.Set(200),
		TopP:        settings.Set(0.95),
	})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	want := []string{"max_tokens", "temperature", "top_p"}
	if got := changed.Names(); !reflect.DeepEqual(got, want) {
		t.Errorf("changed = %v, want %v", got, want)
	}
	if changed["temperature"] != 0.7 {
		t.Errorf("previous temperature = %v, want 0.7", changed["temperature"])
	}
	if changed["max_tokens"] != 100 {
		t.Errorf("previous max_tokens = %v, want 100", changed["max_tokens"])
	}
	if v, _ := current.Temperature.Value(); v != 0.9 {
		t.Errorf("temperature = %v, want 0.9", v)
	}
	if v, _ := current.TopP.Value(); v != 0.95 {
		t.Errorf("top_p = %v, want 0.95", v)
	}
}

func TestApplyMergesExtra(t *testing.T) {
	t.Parallel()

	current := &settings.TTS{Voice: settings.Set("alice")}
	current.Extra = map[string]any{"speed": 1.0, "stability": 0.5}
	changed, err := settings.Apply(current, &settings.TTS{
		Extra: map[string]any{"speed": 1.2},
	})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if !changed.Has("speed") {
		t.Fatalf("changed = %v, want speed reported", changed)
	}
	if changed["speed"] != 1.0 {
		t.Errorf("previous speed = %v, want 1.0", changed["speed"])
	}
	want := map[string]any{"speed": 1.2, "stability": 0.5}
	if !reflect.DeepEqual(current.Extra, want) {
		t.Errorf("extra = %v, want %v: a key the delta left out must survive", current.Extra, want)
	}
}

func TestApplyReportsNothingForAnUnchangedExtra(t *testing.T) {
	t.Parallel()

	current := &settings.TTS{Voice: settings.Set("alice")}
	current.Extra = map[string]any{"speed": 1.0}
	changed, err := settings.Apply(current, &settings.TTS{
		Extra: map[string]any{"speed": 1.0},
	})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if len(changed) != 0 {
		t.Errorf("changed = %v, want nothing", changed)
	}
}

func TestApplyChangesTheModel(t *testing.T) {
	t.Parallel()

	current := &settings.STT{Model: settings.Set("old-model")}
	changed, err := settings.Apply(current, &settings.STT{
		Model: settings.Set("new-model"),
	})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if got := changed.Names(); !reflect.DeepEqual(got, []string{"model"}) {
		t.Errorf("changed = %v, want [model]", got)
	}
	if changed["model"] != "old-model" {
		t.Errorf("previous model = %v, want old-model", changed["model"])
	}
	if v, _ := current.Model.Value(); v != "new-model" {
		t.Errorf("model = %v, want new-model", v)
	}
}

// Clearing a setting is a change, and a different request from not mentioning
// it: it asks the service to drop the value it holds.
func TestApplyClearingASettingIsAChange(t *testing.T) {
	t.Parallel()

	current := &settings.TTS{Language: settings.Set("en")}
	changed, err := settings.Apply(current, &settings.TTS{Language: settings.Cleared[string]()})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if !changed.Has("language") {
		t.Fatalf("changed = %v, want language reported", changed)
	}
	if changed["language"] != "en" {
		t.Errorf("previous language = %v, want en", changed["language"])
	}
	if _, ok := current.Language.Value(); ok {
		t.Error("language still holds a value after being cleared")
	}
	if !current.Language.IsGiven() {
		t.Error("a cleared setting reads as never given")
	}
}

func TestApplyGivingAValueToAClearedSetting(t *testing.T) {
	t.Parallel()

	current := &settings.TTS{Language: settings.Cleared[string]()}
	changed, err := settings.Apply(current, &settings.TTS{Language: settings.Set("en")})
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	if !changed.Has("language") {
		t.Fatalf("changed = %v, want language reported", changed)
	}
	if changed["language"] != nil {
		t.Errorf("previous language = %v, want nothing", changed["language"])
	}
	if v, _ := current.Language.Value(); v != "en" {
		t.Errorf("language = %v, want en", v)
	}
}

func TestApplyRefusesADeltaOfAnotherType(t *testing.T) {
	t.Parallel()

	if _, err := settings.Apply(&settings.TTS{}, &settings.STT{}); err == nil {
		t.Error("applying STT settings to a TTS store was accepted")
	}
}

// From plain data.

func TestFromMapFillsDeclaredFields(t *testing.T) {
	t.Parallel()

	var delta settings.TTS
	if err := settings.FromMap(&delta, map[string]any{"voice": "alice", "language": "en"}); err != nil {
		t.Fatalf("from map: %v", err)
	}

	if v, _ := delta.Voice.Value(); v != "alice" {
		t.Errorf("voice = %v, want alice", v)
	}
	if v, _ := delta.Language.Value(); v != "en" {
		t.Errorf("language = %v, want en", v)
	}
	if delta.Model.IsGiven() {
		t.Error("model was given by a map that never mentioned it")
	}
}

func TestFromMapResolvesAnAlias(t *testing.T) {
	t.Parallel()

	var delta settings.TTS
	if err := settings.FromMap(&delta, map[string]any{"voice_id": "alice"}); err != nil {
		t.Fatalf("from map: %v", err)
	}

	if v, _ := delta.Voice.Value(); v != "alice" {
		t.Errorf("voice = %v, want alice: voice_id is the provider spelling of it", v)
	}
}

func TestFromMapPutsUnknownKeysInExtra(t *testing.T) {
	t.Parallel()

	var delta settings.TTS
	err := settings.FromMap(&delta, map[string]any{"voice": "alice", "speed": 1.2, "stability": 0.5})
	if err != nil {
		t.Fatalf("from map: %v", err)
	}

	if v, _ := delta.Voice.Value(); v != "alice" {
		t.Errorf("voice = %v, want alice", v)
	}
	want := map[string]any{"speed": 1.2, "stability": 0.5}
	if !reflect.DeepEqual(delta.Extra, want) {
		t.Errorf("extra = %v, want %v", delta.Extra, want)
	}
}

func TestFromMapAcceptsAWholeNumberForAnIntField(t *testing.T) {
	t.Parallel()

	var delta settings.LLM
	// Numbers that arrived as JSON are floats, whole ones included.
	if err := settings.FromMap(&delta, map[string]any{"max_tokens": float64(1000)}); err != nil {
		t.Fatalf("from map: %v", err)
	}

	if v, _ := delta.MaxTokens.Value(); v != 1000 {
		t.Errorf("max_tokens = %v, want 1000", v)
	}
}

func TestFromMapRefusesAValueThatDoesNotFit(t *testing.T) {
	t.Parallel()

	var delta settings.LLM
	err := settings.FromMap(&delta, map[string]any{"max_tokens": "many"})
	if err == nil {
		t.Fatal("a value of the wrong type was accepted")
	}

	var fractional settings.LLM
	if err := settings.FromMap(&fractional, map[string]any{"max_tokens": 1.5}); err == nil {
		t.Error("a fractional value was accepted for a whole-number field")
	}
}

func TestFromMapEmpty(t *testing.T) {
	t.Parallel()

	var delta settings.LLM
	if err := settings.FromMap(&delta, map[string]any{}); err != nil {
		t.Fatalf("from map: %v", err)
	}

	given, err := settings.Given(&delta)
	if err != nil {
		t.Fatalf("given: %v", err)
	}
	if len(given) != 0 {
		t.Errorf("given = %v, want nothing", given)
	}
}

func TestFromMapAllUnknown(t *testing.T) {
	t.Parallel()

	var delta settings.STT
	if err := settings.FromMap(&delta, map[string]any{"foo": 1, "bar": 2}); err != nil {
		t.Fatalf("from map: %v", err)
	}

	if delta.Model.IsGiven() {
		t.Error("model was given by a map that never mentioned it")
	}
	want := map[string]any{"foo": 1, "bar": 2}
	if !reflect.DeepEqual(delta.Extra, want) {
		t.Errorf("extra = %v, want %v", delta.Extra, want)
	}
}

func TestGivenListsWhatTheDeltaCarries(t *testing.T) {
	t.Parallel()

	delta := &settings.LLM{
		Model: settings.Set("a-model"), Extra: map[string]any{"custom": true},
		Temperature: settings.Set(0.5),
	}

	given, err := settings.Given(delta)
	if err != nil {
		t.Fatalf("given: %v", err)
	}
	want := map[string]any{"model": "a-model", "temperature": 0.5, "custom": true}
	if !reflect.DeepEqual(given, want) {
		t.Errorf("given = %v, want %v", given, want)
	}
}

// Data in, delta applied: the round trip a settings update actually takes.
func TestFromMapThenApply(t *testing.T) {
	t.Parallel()

	current := store("alice", "en")

	var delta settings.TTS
	if err := settings.FromMap(&delta, map[string]any{"voice_id": "bob", "speed": 1.2}); err != nil {
		t.Fatalf("from map: %v", err)
	}
	changed, err := settings.Apply(current, &delta)
	if err != nil {
		t.Fatalf("apply: %v", err)
	}

	want := []string{"speed", "voice"}
	if got := changed.Names(); !reflect.DeepEqual(got, want) {
		t.Errorf("changed = %v, want %v", got, want)
	}
	if v, _ := current.Voice.Value(); v != "bob" {
		t.Errorf("voice = %v, want bob", v)
	}
	if v, _ := current.Language.Value(); v != "en" {
		t.Errorf("language = %v, want en", v)
	}
	if current.Extra["speed"] != 1.2 {
		t.Errorf("extra speed = %v, want 1.2", current.Extra["speed"])
	}
}

func TestOptStates(t *testing.T) {
	t.Parallel()

	var missing settings.Opt[string]
	if missing.IsGiven() {
		t.Error("the zero value reads as given")
	}
	if _, ok := missing.Value(); ok {
		t.Error("the zero value holds a value")
	}
	if got := missing.Or("fallback"); got != "fallback" {
		t.Errorf("Or = %q, want the fallback", got)
	}

	cleared := settings.Cleared[string]()
	if !cleared.IsGiven() {
		t.Error("a cleared setting reads as not given")
	}
	if _, ok := cleared.Value(); ok {
		t.Error("a cleared setting holds a value")
	}

	set := settings.Set("value")
	if !set.IsGiven() {
		t.Error("a set setting reads as not given")
	}
	if v, ok := set.Value(); !ok || v != "value" {
		t.Errorf("Value = %q, %v, want value, true", v, ok)
	}
	if got := set.Or("fallback"); got != "value" {
		t.Errorf("Or = %q, want the value it holds", got)
	}
}
