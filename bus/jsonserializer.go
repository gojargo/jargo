package bus

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"reflect"
	"sync"
)

// ErrUnknownType reports a wire name this process has no type registered for,
// which is what a peer carrying something this one does not know looks like.
var ErrUnknownType = errors.New("bus: no type registered for that name")

// ErrNotAMessage reports bytes that carried something other than a bus message.
var ErrNotAMessage = errors.New("bus: the payload is not a bus message")

// JSONSerializer writes bus messages as JSON, tagging every value whose type
// cannot be read back from the JSON alone.
//
// A value is written as it stands when JSON already describes it: a string, a
// number, a bool. Anything else is written as {"__type__":…,"__data__":…}, and
// the name is what the far end looks up to know what to build. Raw bytes are
// tagged too, base64 inside, since JSON has no way to carry them.
//
// A type whose state the serializer cannot reach, because it is private or
// because its wire form is deliberately not its field layout, is handled by a
// TypeAdapter registered against it. The adapters for a conversation and its
// toolset are registered by default.
type JSONSerializer struct {
	types *TypeRegistry

	mu       sync.RWMutex
	adapters map[reflect.Type]TypeAdapter
}

// NewJSONSerializer builds a serializer over types, with the default adapters
// registered. A nil registry uses the package's own, which every built-in bus
// message and frame is registered in.
func NewJSONSerializer(types *TypeRegistry) *JSONSerializer {
	if types == nil {
		types = Types
	}
	s := &JSONSerializer{types: types, adapters: map[reflect.Type]TypeAdapter{}}
	s.RegisterAdapter(&frameContextSample{}, contextAdapter{})
	return s
}

// RegisterAdapter records the adapter that converts values of the type sample
// points at. sample is a zero value given as a pointer, as for the registry.
func (s *JSONSerializer) RegisterAdapter(sample any, a TypeAdapter) {
	t := reflect.TypeOf(sample)
	if t == nil || t.Kind() != reflect.Pointer {
		//nolint:forbidigo // a mistake in the program, refused where it is made
		panic("bus: an adapter must be registered against a pointer to its type")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.adapters[t] = a
}

// adapterFor returns the adapter registered for a value's type, if any.
func (s *JSONSerializer) adapterFor(t reflect.Type) (TypeAdapter, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	a, ok := s.adapters[t]
	return a, ok
}

// Serialize converts a message to the bytes to send.
func (s *JSONSerializer) Serialize(m Message) ([]byte, error) {
	v, err := s.serializeValue(m)
	if err != nil {
		return nil, err
	}
	return json.Marshal(v)
}

// Deserialize reconstructs the message those bytes carried.
func (s *JSONSerializer) Deserialize(data []byte) (Message, error) {
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	v, err := s.deserializeValue(raw)
	if err != nil {
		return nil, err
	}
	m, ok := v.(Message)
	if !ok {
		return nil, fmt.Errorf("%w: %T", ErrNotAMessage, v)
	}
	return m, nil
}

// serializeValue converts one value to its wire form, walking into whatever it
// holds. A value it cannot write is left out rather than failing the message:
// a handler or a connection sitting in a field is not something the far end
// could have used anyway, and dropping the whole message over one would lose
// the work it was carrying.
func (s *JSONSerializer) serializeValue(v any) (any, error) {
	if v == nil {
		return nil, nil //nolint:nilnil // nothing to write is not a failure
	}
	rv := reflect.ValueOf(v)

	// A nil pointer, map or slice carries nothing, and dereferencing it below
	// would panic.
	switch rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		if rv.IsNil() {
			return nil, nil //nolint:nilnil // nothing to write is not a failure
		}
	default:
	}

	// Raw JSON crosses as the JSON it already holds, not as the bytes it is
	// made of. A tool's parameters and a call's arguments are carried this way,
	// and writing them as opaque bytes would leave the far end holding a blob
	// where it expects an object.
	if raw, ok := v.(json.RawMessage); ok {
		var parsed any
		if err := json.Unmarshal(raw, &parsed); err != nil {
			return nil, fmt.Errorf("%w: %w", ErrNotJSON, err)
		}
		return parsed, nil
	}

	if b, ok := v.([]byte); ok {
		return map[string]any{
			typeKey: bytesTypeName,
			dataKey: base64.StdEncoding.EncodeToString(b),
		}, nil
	}

	if adapter, ok := s.adapterFor(rv.Type()); ok {
		name, known := s.types.NameOf(v)
		if !known {
			return nil, fmt.Errorf("%w: %s", ErrUnnamedType, rv.Type())
		}
		data, err := adapter.Serialize(v, s.serializeValue)
		if err != nil {
			return nil, err
		}
		return map[string]any{typeKey: name, dataKey: data}, nil
	}

	if name, known := s.types.NameOf(v); known {
		fields, err := s.serializeStruct(rv)
		if err != nil {
			return nil, err
		}
		return map[string]any{typeKey: name, dataKey: fields}, nil
	}

	return s.serializeUntagged(rv)
}

// serializeUntagged writes a value that needs no name of its own: JSON already
// describes what it is.
func (s *JSONSerializer) serializeUntagged(rv reflect.Value) (any, error) {
	if scalar, ok := serializeScalar(rv); ok {
		return scalar, nil
	}
	switch rv.Kind() {
	case reflect.Pointer, reflect.Interface:
		return s.serializeValue(rv.Elem().Interface())
	case reflect.Slice, reflect.Array:
		out := make([]any, rv.Len())
		for i := range rv.Len() {
			el, err := s.serializeValue(rv.Index(i).Interface())
			if err != nil {
				return nil, err
			}
			out[i] = el
		}
		return out, nil
	case reflect.Map:
		out := map[string]any{}
		for _, k := range rv.MapKeys() {
			el, err := s.serializeValue(rv.MapIndex(k).Interface())
			if err != nil {
				return nil, err
			}
			out[fmt.Sprint(k.Interface())] = el
		}
		return out, nil
	case reflect.Struct:
		return s.serializeStruct(rv)
	case reflect.Func, reflect.Chan, reflect.UnsafePointer:
		// Nothing the far end could call, so it is left out rather than failing
		// the message it was a field of.
		return nil, nil //nolint:nilnil // a field that cannot cross is dropped, not an error
	default:
		slog.Warn("bus: a field of this kind cannot be serialized and is being left out",
			"kind", rv.Kind().String())
		return nil, nil //nolint:nilnil // a field that cannot cross is dropped, not an error
	}
}

// serializeScalar writes the kinds JSON describes on their own, reporting false
// for a kind that needs walking into.
func serializeScalar(rv reflect.Value) (any, bool) {
	switch rv.Kind() {
	case reflect.String:
		return rv.String(), true
	case reflect.Bool:
		return rv.Bool(), true
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return rv.Uint(), true
	case reflect.Float32, reflect.Float64:
		return rv.Float(), true
	default:
		return nil, false
	}
}

// serializeStruct writes a struct's exported fields, skipping the ones carrying
// nothing so the wire form stays close to what was actually set. An embedded
// struct is flattened, which is how a message's own fields sit beside the ones
// every message has.
func (s *JSONSerializer) serializeStruct(rv reflect.Value) (map[string]any, error) {
	if rv.Kind() == reflect.Pointer {
		rv = rv.Elem()
	}
	out := map[string]any{}
	rt := rv.Type()
	for i := range rt.NumField() {
		f := rt.Field(i)
		if !f.IsExported() {
			continue
		}
		fv := rv.Field(i)
		if f.Anonymous && fv.Kind() == reflect.Struct {
			nested, err := s.serializeStruct(fv)
			if err != nil {
				return nil, err
			}
			maps.Copy(out, nested)
			continue
		}
		if fv.IsZero() {
			continue
		}
		el, err := s.serializeValue(fv.Interface())
		if err != nil {
			return nil, err
		}
		if el == nil {
			continue
		}
		out[f.Name] = el
	}
	return out, nil
}

// deserializeValue rebuilds one value from its wire form.
func (s *JSONSerializer) deserializeValue(v any) (any, error) {
	switch t := v.(type) {
	case nil, string, bool, float64:
		return v, nil
	case []any:
		out := make([]any, len(t))
		for i, el := range t {
			d, err := s.deserializeValue(el)
			if err != nil {
				return nil, err
			}
			out[i] = d
		}
		return out, nil
	case map[string]any:
		name, tagged := t[typeKey].(string)
		if !tagged {
			out := map[string]any{}
			for k, el := range t {
				d, err := s.deserializeValue(el)
				if err != nil {
					return nil, err
				}
				out[k] = d
			}
			return out, nil
		}
		return s.deserializeTagged(name, t[dataKey])
	default:
		return v, nil
	}
}

// deserializeTagged rebuilds a value the sender named.
func (s *JSONSerializer) deserializeTagged(name string, data any) (any, error) {
	if name == bytesTypeName {
		encoded, ok := data.(string)
		if !ok {
			return nil, fmt.Errorf("%w: %s carried %T, want a base64 string", ErrWrongShape, bytesTypeName, data)
		}
		return base64.StdEncoding.DecodeString(encoded)
	}

	built, known := s.types.New(name)
	if !known {
		return nil, fmt.Errorf("%w: %q", ErrUnknownType, name)
	}

	if adapter, ok := s.adapterFor(reflect.TypeOf(built)); ok {
		fields, _ := data.(map[string]any)
		return adapter.Deserialize(fields, s.deserializeValue)
	}

	fields, ok := data.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("%w: %s carried %T, want an object", ErrWrongShape, name, data)
	}
	if err := s.fillStruct(reflect.ValueOf(built).Elem(), fields); err != nil {
		return nil, err
	}
	return built, nil
}

// fillStruct writes the fields a wire object carried into the value built for
// it. A field the sender did not write is left at its zero value, and one this
// process does not know is skipped: a peer running a newer build may carry
// fields this one has never heard of, and that is not a reason to lose the
// message.
func (s *JSONSerializer) fillStruct(rv reflect.Value, fields map[string]any) error {
	rt := rv.Type()
	for i := range rt.NumField() {
		f := rt.Field(i)
		if !f.IsExported() {
			continue
		}
		fv := rv.Field(i)
		if f.Anonymous && fv.Kind() == reflect.Struct {
			if err := s.fillStruct(fv, fields); err != nil {
				return err
			}
			continue
		}
		raw, present := fields[f.Name]
		if !present {
			continue
		}
		if err := s.assign(fv, raw); err != nil {
			return fmt.Errorf("bus: %s.%s: %w", rt.Name(), f.Name, err)
		}
	}
	return nil
}

// assign writes one deserialized value into a field, converting the numbers
// JSON hands back and stepping into whatever the field holds.
func (s *JSONSerializer) assign(fv reflect.Value, raw any) error {
	v, err := s.deserializeValue(raw)
	if err != nil {
		return err
	}
	if v == nil {
		return nil
	}
	val := reflect.ValueOf(v)

	if val.Type().AssignableTo(fv.Type()) {
		fv.Set(val)
		return nil
	}
	if val.Type().ConvertibleTo(fv.Type()) && fv.Kind() != reflect.Interface {
		fv.Set(val.Convert(fv.Type()))
		return nil
	}

	// Anything left is a shape json.Unmarshal already knows how to fill: a
	// nested struct, a slice of them, a map. Round-tripping it through JSON is
	// what saves this from re-implementing that.
	encoded, err := json.Marshal(v)
	if err != nil {
		return err
	}
	target := reflect.New(fv.Type())
	if err := json.Unmarshal(encoded, target.Interface()); err != nil {
		return fmt.Errorf("cannot fill %s from %T: %w", fv.Type(), v, err)
	}
	fv.Set(target.Elem())
	return nil
}

// reencode converts a deserialized value to a concrete Go type by writing it
// back to JSON and reading it into T. It is how a value that came back as maps
// and slices becomes the struct a field is typed as.
func reencode[T any](v any) (T, error) {
	var out T
	encoded, err := json.Marshal(v)
	if err != nil {
		return out, err
	}
	if err := json.Unmarshal(encoded, &out); err != nil {
		return out, fmt.Errorf("cannot read a %T back as %T: %w", v, out, err)
	}
	return out, nil
}
