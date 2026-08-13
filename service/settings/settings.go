// Package settings holds the part of a service's configuration that can change
// while the pipeline is running: the model, the voice, the language, the
// sampling knobs an LLM exposes. It is not the whole of a service's
// configuration. What a service is built with stays in its own Config; what may
// change mid-call lives here.
//
// A settings value is read in one of two ways.
//
// As a store, held by a running service, every field carries the current state.
// A field with no value means the service has no such setting.
//
// As a delta, carried by an update frame, only the fields the caller wants to
// change are given. The rest are left alone, which is what lets a caller change
// the voice without restating the model.
//
// Both are the same Go type, because a field carries the three states that
// needs: not given, given with a value, and given with no value. The last is how
// a caller clears a setting, which is a different request from not mentioning
// it, and the two would be indistinguishable if a field were a plain pointer.
package settings

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/gojargo/jargo/frames"
)

// Errors reported when a settings value does not fit where it is being put.
//
//nolint:gochecknoglobals // sentinel errors
var (
	// ErrType is returned when a value cannot be stored in a field without
	// losing something.
	ErrType = errors.New("settings: value does not fit the field")
	// ErrMismatch is returned when a delta and a store are different types.
	ErrMismatch = errors.New("settings: delta and store are different types")
	// ErrNotSettings is returned for anything that is not a settings struct.
	ErrNotSettings = errors.New("settings: not a pointer to a settings struct")
)

// Opt is one runtime-updatable setting. Its zero value is not given, which is
// what every field of a delta starts as.
type Opt[T any] struct {
	given bool
	value *T
}

// Set is a setting given the value v.
func Set[T any](v T) Opt[T] { return Opt[T]{given: true, value: &v} }

// Cleared is a setting given no value, for a caller asking a service to drop
// one: to stop naming a language and go back to detecting it, say. It differs
// from leaving the field alone, which asks for no change at all.
func Cleared[T any]() Opt[T] { return Opt[T]{given: true} }

// IsGiven reports whether the setting was mentioned at all.
func (o Opt[T]) IsGiven() bool { return o.given }

// Value is the value held, and whether there is one. A setting that was never
// given, and one given no value, both report false.
func (o Opt[T]) Value() (T, bool) {
	if o.value == nil {
		var zero T
		return zero, false
	}
	return *o.value, true
}

// Or is the value held, or alt when there is none.
func (o Opt[T]) Or(alt T) T {
	if v, ok := o.Value(); ok {
		return v
	}
	return alt
}

// AsAny is the value held as an any, or nil when there is none.
func (o Opt[T]) AsAny() any {
	if o.value == nil {
		return nil
	}
	return *o.value
}

// SetAny stores v, converting it where the conversion loses nothing. A nil v
// clears the setting. It is how a setting that arrived as plain data, over RTVI
// say, reaches a typed field.
func (o *Opt[T]) SetAny(v any) error {
	if v == nil {
		o.given, o.value = true, nil
		return nil
	}
	want := reflect.TypeFor[T]()
	rv := reflect.ValueOf(v)
	switch {
	case rv.Type() == want:
	case rv.Type().ConvertibleTo(want) && exactConversion(rv, want):
		rv = rv.Convert(want)
	default:
		return fmt.Errorf("%w: cannot use %T as %s", ErrType, v, want)
	}
	val, ok := rv.Interface().(T)
	if !ok {
		return fmt.Errorf("%w: cannot use %T as %s", ErrType, v, want)
	}
	o.given, o.value = true, &val
	return nil
}

// String implements fmt.Stringer.
func (o Opt[T]) String() string {
	if !o.given {
		return "<not given>"
	}
	if o.value == nil {
		return "<cleared>"
	}
	return fmt.Sprint(*o.value)
}

// option is the reflected view of a settings field. Every Opt satisfies it,
// which is how a field is told apart from anything else on a settings struct.
type option interface {
	IsGiven() bool
	AsAny() any
}

//nolint:gochecknoglobals // the reflected interface, resolved once
var optionType = reflect.TypeFor[option]()

// Changed maps each field a delta altered to the value it held before, so a
// service can see both what moved and what it is moving away from.
type Changed map[string]any

// Names lists the changed fields in a stable order.
func (c Changed) Names() []string {
	names := make([]string, 0, len(c))
	for name := range c {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Has reports whether field changed.
func (c Changed) Has(field string) bool {
	_, ok := c[field]
	return ok
}

// Except lists the changed fields a service does not act on, given the ones it
// does. A service warns about these rather than staying silent: a caller who
// asked for a change that cannot take effect should hear so.
func (c Changed) Except(handled ...string) []string {
	skip := make(map[string]struct{}, len(handled))
	for _, name := range handled {
		skip[name] = struct{}{}
	}
	var rest []string
	for _, name := range c.Names() {
		if _, ok := skip[name]; !ok {
			rest = append(rest, name)
		}
	}
	return rest
}

// String implements fmt.Stringer.
func (c Changed) String() string { return strings.Join(c.Names(), ", ") }

// Aliaser is implemented by a settings type that accepts more than one spelling
// for a field, mapping each alternative to the canonical name. It is consulted
// when a delta is built from plain data, where the names come from outside.
type Aliaser interface {
	Aliases() map[string]string
}

// Base is the part every service's settings share. Embed it.
type Base struct {
	// Model is the service's model identifier.
	Model Opt[string] `settings:"model"`
	// Extra carries settings a provider understands that have no field of their
	// own. They are merged and compared like declared fields, so a change to one
	// is reported the same way.
	Extra map[string]any `settings:"-"`
}

// LLM is the runtime-updatable settings of a language model service.
type LLM struct {
	Base
	// SystemInstruction is the system prompt the model is given.
	SystemInstruction Opt[string] `settings:"system_instruction"`
	// Temperature is the sampling temperature.
	Temperature Opt[float64] `settings:"temperature"`
	// MaxTokens bounds how much the model generates.
	MaxTokens Opt[int] `settings:"max_tokens"`
	// TopP is the nucleus sampling probability.
	TopP Opt[float64] `settings:"top_p"`
	// TopK is the top-k sampling cutoff.
	TopK Opt[int] `settings:"top_k"`
	// FrequencyPenalty discourages repeating the same tokens.
	FrequencyPenalty Opt[float64] `settings:"frequency_penalty"`
	// PresencePenalty discourages repeating the same subjects.
	PresencePenalty Opt[float64] `settings:"presence_penalty"`
	// Seed makes generation reproducible where the provider supports it.
	Seed Opt[int] `settings:"seed"`
	// FilterIncompleteUserTurns gates each reply on the model's own verdict of
	// whether the user had finished speaking. It is set by the turn strategy that
	// drives the protocol rather than by an application directly.
	FilterIncompleteUserTurns Opt[bool] `settings:"filter_incomplete_user_turns"`
	// UserTurnCompletionConfig configures that gating: the protocol taught to the
	// model, the waits before re-prompting, and the re-prompts themselves. It is
	// set by the same turn strategy.
	//
	// It is typed loosely because the type belongs to the LLM service, which is
	// built on this package and so cannot be named here. Set it to an
	// llm.UserTurnCompletionConfig; anything else is reported and ignored.
	UserTurnCompletionConfig Opt[any] `settings:"user_turn_completion_config"`
}

// TTS is the runtime-updatable settings of a speech synthesis service.
type TTS struct {
	Base
	// Voice is the voice identifier or name.
	Voice Opt[string] `settings:"voice"`
	// Language is the language spoken, as the service names it. A caller may
	// give a neutral tag; the service converts it before it is stored.
	Language Opt[string] `settings:"language"`
}

// Aliases accepts the spelling a provider's own API uses for the voice.
func (TTS) Aliases() map[string]string { return map[string]string{"voice_id": "voice"} }

// STT is the runtime-updatable settings of a transcription service.
type STT struct {
	Base
	// Language is the language transcribed, as the service names it. A caller
	// may give a neutral tag; the service converts it before it is stored.
	Language Opt[string] `settings:"language"`
}

// Resolve turns an update into a delta of the store's own type: the typed one
// the update carries, or one built from the plain names and values it carries
// instead. It reports false when the update asks for nothing at all.
func Resolve(f *frames.ServiceUpdateSettingsFrame, store any) (delta any, ok bool, err error) {
	if f.Delta != nil {
		return f.Delta, true, nil
	}
	if len(f.Settings) == 0 {
		return nil, false, nil
	}
	built, err := NewDelta(store)
	if err != nil {
		return nil, false, err
	}
	if err := FromMap(built, f.Settings); err != nil {
		return nil, false, err
	}
	return built, true, nil
}

// Get is what a settings value holds for the named field: the value, and
// whether the field carries one at all. A field a provider's own settings
// declare is reached the same way as one of the shared fields.
func Get(v any, name string) (any, bool) {
	rv, err := settingsValue(v)
	if err != nil {
		return nil, false
	}
	var value any
	var found bool
	eachField(rv, func(fieldName string, field reflect.Value, _ []int) {
		if fieldName != name || found {
			return
		}
		o, _ := field.Interface().(option)
		if o.IsGiven() {
			value = o.AsAny()
		}
		found = value != nil
	})
	return value, found
}

// SetNamed gives the named field a value, converting it where the conversion
// loses nothing. It reports whether the field exists.
func SetNamed(v any, name string, value any) error {
	rv, err := settingsValue(v)
	if err != nil {
		return err
	}
	var target reflect.Value
	eachField(rv, func(fieldName string, field reflect.Value, _ []int) {
		if fieldName == name && !target.IsValid() {
			target = field
		}
	})
	if !target.IsValid() {
		return fmt.Errorf("%w: no field %q", ErrNotSettings, name)
	}
	results := target.Addr().MethodByName("SetAny").
		Call([]reflect.Value{reflect.ValueOf(&value).Elem()})
	if err, _ := results[0].Interface().(error); err != nil {
		return fmt.Errorf("settings: field %q: %w", name, err)
	}
	return nil
}

// NewDelta builds an empty delta of the same type as store, for an update that
// arrived as plain data and has to be given a shape before it can be applied.
func NewDelta(store any) (any, error) {
	sv, err := settingsValue(store)
	if err != nil {
		return nil, err
	}
	return reflect.New(sv.Type()).Interface(), nil
}

// Apply merges delta into store and reports what changed. Only the fields the
// delta gives are considered, and a field changes only when what it carries
// differs from what is already there, so re-sending a service what it already
// has reports nothing and a service that reconnects on a change is not made to
// reconnect for a non-change.
//
// store and delta must be pointers to the same settings type.
func Apply(store, delta any) (Changed, error) {
	sv, dv, err := settingsValues(store, delta)
	if err != nil {
		return nil, err
	}

	changed := Changed{}
	eachField(dv, func(name string, deltaField reflect.Value, index []int) {
		d, _ := deltaField.Interface().(option)
		if !d.IsGiven() {
			return
		}
		storeField := sv.FieldByIndex(index)
		s, _ := storeField.Interface().(option)
		if reflect.DeepEqual(s.AsAny(), d.AsAny()) {
			return
		}
		old := s.AsAny()
		storeField.Set(deltaField)
		changed[name] = old
	})
	applyExtra(sv, dv, changed)
	return changed, nil
}

// applyExtra merges the provider-specific settings that have no field.
func applyExtra(store, delta reflect.Value, changed Changed) {
	deltaExtra := extraField(delta)
	if !deltaExtra.IsValid() || deltaExtra.Len() == 0 {
		return
	}
	storeExtra := extraField(store)
	if !storeExtra.IsValid() {
		return
	}
	if storeExtra.IsNil() {
		storeExtra.Set(reflect.MakeMap(storeExtra.Type()))
	}
	for _, key := range deltaExtra.MapKeys() {
		newVal := deltaExtra.MapIndex(key).Interface()
		var old any
		if cur := storeExtra.MapIndex(key); cur.IsValid() {
			old = cur.Interface()
			if reflect.DeepEqual(old, newVal) {
				continue
			}
		}
		storeExtra.SetMapIndex(key, reflect.ValueOf(newVal))
		changed[key.String()] = old
	}
}

// Given lists the fields a delta gives, mapped to their values, with the extra
// settings alongside. It is what a service reads when it wants the update as
// plain data rather than reacting field by field. A field given no value is
// present with a nil value.
func Given(delta any) (map[string]any, error) {
	dv, err := settingsValue(delta)
	if err != nil {
		return nil, err
	}
	given := map[string]any{}
	eachField(dv, func(name string, field reflect.Value, _ []int) {
		o, _ := field.Interface().(option)
		if o.IsGiven() {
			given[name] = o.AsAny()
		}
	})
	if extra := extraField(dv); extra.IsValid() {
		for _, key := range extra.MapKeys() {
			given[key.String()] = extra.MapIndex(key).Interface()
		}
	}
	return given, nil
}

// FromMap fills delta from plain setting names and values, for an update that
// arrived as data rather than as a typed value. A name the settings type
// declares, or one of its aliases, gives that field; anything else lands in
// Extra, so a provider-specific setting survives the trip rather than being
// dropped on the way.
//
// A value whose type does not fit its field is reported rather than skipped:
// applying half an update silently is worse than refusing it.
func FromMap(delta any, values map[string]any) error {
	dv, err := settingsValue(delta)
	if err != nil {
		return err
	}

	var aliases map[string]string
	if a, ok := delta.(Aliaser); ok {
		aliases = a.Aliases()
	}

	byName := map[string][]int{}
	eachField(dv, func(name string, _ reflect.Value, index []int) {
		byName[name] = index
	})

	var extra map[string]any
	for key, value := range values {
		name := key
		if canonical, ok := aliases[key]; ok {
			name = canonical
		}
		index, declared := byName[name]
		if !declared {
			if extra == nil {
				extra = map[string]any{}
			}
			extra[key] = value
			continue
		}
		results := dv.FieldByIndex(index).Addr().MethodByName("SetAny").
			Call([]reflect.Value{reflect.ValueOf(&value).Elem()})
		if err, _ := results[0].Interface().(error); err != nil {
			return fmt.Errorf("settings: field %q: %w", name, err)
		}
	}

	if extra != nil {
		if f := extraField(dv); f.IsValid() {
			f.Set(reflect.ValueOf(extra))
		}
	}
	return nil
}

// exactConversion reports whether converting rv to want loses nothing. Data that
// arrived as JSON carries every number as a float, so a whole one has to be
// usable as an int without letting a fractional one through as a truncation.
func exactConversion(rv reflect.Value, want reflect.Type) bool {
	switch want.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		switch rv.Kind() {
		case reflect.Float32, reflect.Float64:
			f := rv.Float()
			return f == float64(int64(f))
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return true
		default:
			return false
		}
	case reflect.Float32, reflect.Float64:
		switch rv.Kind() {
		case reflect.Float32, reflect.Float64,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			return true
		default:
			return false
		}
	default:
		return rv.Type() == want
	}
}

// eachField visits every declared settings field, walking into the embedded Base
// so the shared fields and a service's own look alike to a caller.
func eachField(v reflect.Value, fn func(name string, field reflect.Value, index []int)) {
	var walk func(t reflect.Type, prefix []int)
	walk = func(t reflect.Type, prefix []int) {
		for i := range t.NumField() {
			f := t.Field(i)
			index := append(append([]int(nil), prefix...), i)
			if !f.Type.Implements(optionType) {
				if f.Anonymous && f.Type.Kind() == reflect.Struct {
					walk(f.Type, index)
				}
				continue
			}
			name := f.Tag.Get("settings")
			if name == "" || name == "-" {
				continue
			}
			fn(name, v.FieldByIndex(index), index)
		}
	}
	walk(v.Type(), nil)
}

// extraField finds the embedded Base's Extra map, if the type has one.
func extraField(v reflect.Value) reflect.Value {
	if v.Type() == reflect.TypeFor[Base]() {
		return v.FieldByName("Extra")
	}
	for i := range v.NumField() {
		f := v.Type().Field(i)
		if f.Anonymous && f.Type.Kind() == reflect.Struct {
			if found := extraField(v.Field(i)); found.IsValid() {
				return found
			}
		}
	}
	return reflect.Value{}
}

// settingsValues resolves a store and a delta of the same type.
func settingsValues(store, delta any) (reflect.Value, reflect.Value, error) {
	sv, err := settingsValue(store)
	if err != nil {
		return reflect.Value{}, reflect.Value{}, err
	}
	dv, err := settingsValue(delta)
	if err != nil {
		return reflect.Value{}, reflect.Value{}, err
	}
	if sv.Type() != dv.Type() {
		return reflect.Value{}, reflect.Value{}, fmt.Errorf(
			"%w: cannot apply %s to %s", ErrMismatch, dv.Type(), sv.Type())
	}
	return sv, dv, nil
}

// settingsValue resolves a pointer to a settings struct.
func settingsValue(v any) (reflect.Value, error) {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Pointer || rv.IsNil() || rv.Elem().Kind() != reflect.Struct {
		return reflect.Value{}, fmt.Errorf("%w: got %T", ErrNotSettings, v)
	}
	return rv.Elem(), nil
}
