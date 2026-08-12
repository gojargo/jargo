package settings_test

import (
	"errors"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// Tests for reading an update that arrived from outside the process, where the
// settings come as plain names and values and have to be given a shape before a
// service can apply them, and for reading a single field by name.

// TestResolvePrefersTheTypedDelta checks that an update carrying a typed value
// is handed straight on: it already has the shape the service wants.
func TestResolvePrefersTheTypedDelta(t *testing.T) {
	delta := &settings.TTS{Voice: settings.Set("nova")}
	f := frames.NewTTSUpdateSettingsFrame(delta)
	// Plain settings alongside a typed delta are ignored; the delta wins.
	f.Settings = map[string]any{"voice": "ignored"}

	got, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, &settings.TTS{})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !ok {
		t.Fatal("Resolve reported nothing to do for an update carrying a delta")
	}
	if got != any(delta) {
		t.Fatalf("Resolve returned %v, want the delta the frame carries", got)
	}
}

// TestResolveBuildsADeltaFromPlainSettings checks the other route: names and
// values with no typed value become a delta of the store's own type, so an
// update arriving over RTVI is applied exactly like one built in process.
func TestResolveBuildsADeltaFromPlainSettings(t *testing.T) {
	store := &settings.TTS{Voice: settings.Set("old"), Language: settings.Set("en")}

	f := frames.NewTTSUpdateSettingsFrame(nil)
	f.Settings = map[string]any{"voice": "nova", "speed": 1.5}

	delta, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, store)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !ok {
		t.Fatal("Resolve reported nothing to do for an update carrying settings")
	}

	typed, isTTS := delta.(*settings.TTS)
	if !isTTS {
		t.Fatalf("Resolve built a %T, want the store's own type", delta)
	}
	if v, has := typed.Voice.Value(); !has || v != "nova" {
		t.Errorf("the built delta gives voice %q (%v), want nova", v, has)
	}
	if typed.Language.IsGiven() {
		t.Error("the built delta gives a language the update never mentioned")
	}

	// Applying it moves only what the update named, and the setting with no
	// field of its own survives the trip.
	changed, err := settings.Apply(store, typed)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}
	if !changed.Has("voice") || !changed.Has("speed") {
		t.Errorf("changed = %v, want the voice and the extra setting", changed)
	}
}

// TestResolveReportsAnEmptyUpdate checks that an update asking for nothing is
// reported as such rather than as an empty delta a service would go on to apply.
func TestResolveReportsAnEmptyUpdate(t *testing.T) {
	f := frames.NewTTSUpdateSettingsFrame(nil)

	delta, ok, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, &settings.TTS{})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if ok {
		t.Fatalf("Resolve reported work to do for an empty update, returning %v", delta)
	}
}

// TestResolveRefusesAValueThatDoesNotFit checks that a value of the wrong type
// is reported rather than skipped: applying half an update silently is worse
// than refusing it.
func TestResolveRefusesAValueThatDoesNotFit(t *testing.T) {
	f := frames.NewLLMUpdateSettingsFrame(nil)
	f.Settings = map[string]any{"max_tokens": "many"}

	if _, _, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, &settings.LLM{}); err == nil {
		t.Fatal("Resolve accepted a string for an int field")
	} else if !errors.Is(err, settings.ErrType) {
		t.Fatalf("Resolve reported %v, want an ErrType", err)
	}
}

// TestResolveRefusesAStoreThatIsNotSettings checks the store is validated: a
// delta cannot be shaped after something that is not a settings struct.
func TestResolveRefusesAStoreThatIsNotSettings(t *testing.T) {
	f := frames.NewTTSUpdateSettingsFrame(nil)
	f.Settings = map[string]any{"voice": "nova"}

	if _, _, err := settings.Resolve(&f.ServiceUpdateSettingsFrame, "not settings"); err == nil {
		t.Fatal("Resolve accepted a store that is not a settings struct")
	} else if !errors.Is(err, settings.ErrNotSettings) {
		t.Fatalf("Resolve reported %v, want an ErrNotSettings", err)
	}
}

// TestNewDelta checks that an empty delta of the store's type gives nothing at
// all, so applying it before it is filled changes nothing.
func TestNewDelta(t *testing.T) {
	delta, err := settings.NewDelta(&settings.LLM{Temperature: settings.Set(0.7)})
	if err != nil {
		t.Fatalf("NewDelta: %v", err)
	}
	typed, ok := delta.(*settings.LLM)
	if !ok {
		t.Fatalf("NewDelta built a %T, want the store's own type", delta)
	}
	if typed.Temperature.IsGiven() {
		t.Error("a fresh delta carries a value from the store")
	}

	given, err := settings.Given(typed)
	if err != nil {
		t.Fatalf("Given: %v", err)
	}
	if len(given) != 0 {
		t.Errorf("a fresh delta gives %v, want nothing", given)
	}

	if _, err := settings.NewDelta(nil); err == nil {
		t.Error("NewDelta accepted a store that is not a settings struct")
	}
}

// TestGet checks reading one field by name, which is how a service asks what it
// currently holds without knowing the shape of its own settings type.
func TestGet(t *testing.T) {
	store := &settings.TTS{
		Base:     settings.Base{Model: settings.Set("sonic")},
		Voice:    settings.Set("nova"),
		Language: settings.Cleared[string](),
	}

	if v, ok := settings.Get(store, "voice"); !ok || v != "nova" {
		t.Errorf("Get(voice) = (%v, %v), want (nova, true)", v, ok)
	}
	// A field of the embedded Base is reached the same way as the service's own.
	if v, ok := settings.Get(store, "model"); !ok || v != "sonic" {
		t.Errorf("Get(model) = (%v, %v), want (sonic, true)", v, ok)
	}
	// A field given no value carries nothing, like one never given at all.
	if v, ok := settings.Get(store, "language"); ok {
		t.Errorf("Get(language) = (%v, true), want no value for a cleared field", v)
	}
	if v, ok := settings.Get(store, "voice_id"); ok {
		t.Errorf("Get(voice_id) = (%v, true), want no such field", v)
	}
	if v, ok := settings.Get("not settings", "voice"); ok {
		t.Errorf("Get on a non-settings value returned (%v, true)", v)
	}
}

// TestSetNamed checks writing one field by name, converting where the
// conversion loses nothing, which is how a setting that arrived as plain data
// reaches a typed field.
func TestSetNamed(t *testing.T) {
	store := &settings.LLM{}

	if err := settings.SetNamed(store, "temperature", 0.4); err != nil {
		t.Fatalf("SetNamed(temperature): %v", err)
	}
	if v, ok := store.Temperature.Value(); !ok || v != 0.4 {
		t.Errorf("Temperature = (%v, %v), want (0.4, true)", v, ok)
	}

	// Data that arrived as JSON carries every number as a float, so a whole one
	// is usable as an int.
	if err := settings.SetNamed(store, "max_tokens", 128.0); err != nil {
		t.Fatalf("SetNamed(max_tokens): %v", err)
	}
	if v, ok := store.MaxTokens.Value(); !ok || v != 128 {
		t.Errorf("MaxTokens = (%v, %v), want (128, true)", v, ok)
	}

	// A fractional one is not: that would be a silent truncation.
	if err := settings.SetNamed(store, "max_tokens", 12.5); err == nil {
		t.Error("SetNamed accepted a fractional value for an int field")
	}

	// A nil clears the setting, which is a different request from not
	// mentioning it.
	if err := settings.SetNamed(store, "temperature", nil); err != nil {
		t.Fatalf("SetNamed(temperature, nil): %v", err)
	}
	if !store.Temperature.IsGiven() {
		t.Error("clearing a setting left it not given")
	}
	if _, ok := store.Temperature.Value(); ok {
		t.Error("a cleared setting still carries a value")
	}

	if err := settings.SetNamed(store, "nonesuch", 1); err == nil {
		t.Error("SetNamed accepted a field the settings type does not declare")
	}
	if err := settings.SetNamed("not settings", "temperature", 1.0); err == nil {
		t.Error("SetNamed accepted a value that is not a settings struct")
	}
}

// TestChangedExcept checks the report a service makes about an update it could
// not fully act on: a caller who asked for a change that cannot take effect
// should hear so rather than meet silence.
func TestChangedExcept(t *testing.T) {
	store := &settings.LLM{}
	delta := &settings.LLM{
		Base:        settings.Base{Model: settings.Set("gpt-4")},
		Temperature: settings.Set(0.4),
		TopP:        settings.Set(0.9),
	}

	changed, err := settings.Apply(store, delta)
	if err != nil {
		t.Fatalf("Apply: %v", err)
	}

	rest := changed.Except("model", "temperature")
	if len(rest) != 1 || rest[0] != "top_p" {
		t.Errorf("Except(model, temperature) = %v, want [top_p]", rest)
	}

	if rest := changed.Except("model", "temperature", "top_p"); len(rest) != 0 {
		t.Errorf("Except of everything = %v, want nothing left", rest)
	}
	if rest := changed.Except(); len(rest) != 3 {
		t.Errorf("Except of nothing = %v, want all three fields", rest)
	}

	// The names read in a stable order, so a log line does not shuffle.
	if got := changed.String(); got != "model, temperature, top_p" {
		t.Errorf("String() = %q, want the names in order", got)
	}
}

// TestOptString checks the three states a field carries read differently, since
// a setting given no value and one never mentioned are different requests.
func TestOptString(t *testing.T) {
	tests := []struct {
		name string
		opt  settings.Opt[string]
		want string
	}{
		{name: "not given", opt: settings.Opt[string]{}, want: "<not given>"},
		{name: "cleared", opt: settings.Cleared[string](), want: "<cleared>"},
		{name: "given a value", opt: settings.Set("nova"), want: "nova"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.opt.String(); got != tt.want {
				t.Errorf("String() = %q, want %q", got, tt.want)
			}
		})
	}

	if got := settings.Set(42).String(); got != "42" {
		t.Errorf("String() on an int setting = %q, want 42", got)
	}
}

// TestSetNamedConversions checks which conversions are allowed on the way into
// a typed field. A conversion that loses nothing is made; one that would
// truncate, or that Go would allow for unrelated reasons, is refused.
func TestSetNamedConversions(t *testing.T) {
	tests := []struct {
		name  string
		field string
		value any
		ok    bool
	}{
		{name: "an int for an int field", field: "max_tokens", value: int64(64), ok: true},
		{name: "a whole float for an int field", field: "max_tokens", value: 64.0, ok: true},
		{name: "a fractional float for an int field", field: "max_tokens", value: 64.5},
		{name: "a string for an int field", field: "max_tokens", value: "64"},
		{name: "a float for a float field", field: "temperature", value: float32(0.5), ok: true},
		{name: "an int for a float field", field: "temperature", value: 1, ok: true},
		{name: "a string for a float field", field: "temperature", value: "0.5"},
		{name: "a string for a string field", field: "model", value: "gpt-4", ok: true},
		// Go converts an int to a string as a rune, which is never what a
		// caller naming a model meant.
		{name: "an int for a string field", field: "model", value: 65},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := settings.SetNamed(&settings.LLM{}, tt.field, tt.value)
			if tt.ok && err != nil {
				t.Fatalf("SetNamed(%s, %v) = %v, want it accepted", tt.field, tt.value, err)
			}
			if !tt.ok && err == nil {
				t.Fatalf("SetNamed(%s, %v) was accepted, want it refused", tt.field, tt.value)
			}
		})
	}
}
