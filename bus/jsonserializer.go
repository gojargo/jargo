package bus

import (
	"bytes"
	"encoding"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/settings"
)

// baseKey is the field a frame's base state crosses under: its id, name,
// timestamp, metadata and transport routing. No frame can have a field of that
// name, since Base is the method every frame has.
const baseKey = "Base"

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
	raw, err := decodeJSON(data)
	if err != nil {
		return nil, err
	}
	if top, ok := raw.(map[string]any); ok {
		if name, tagged := top[typeKey].(string); tagged && name != bytesTypeName {
			if _, known := s.types.New(name); !known {
				return nil, fmt.Errorf("%w: %q", ErrUnknownType, name)
			}
		}
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

	// A settings value crosses as the fields it gives, a cleared one as null, so
	// an update arrives saying what it said: a field it left alone stays
	// alone. One of a type the far end may not know crosses untagged, and
	// arrives as the plain settings it is.
	if settings.IsSettings(v) {
		return s.serializeSettings(v)
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
		fields, err := s.serializeNamed(v, rv)
		if err != nil {
			return nil, err
		}
		return map[string]any{typeKey: name, dataKey: fields}, nil
	}

	return s.serializeUntagged(rv)
}

// serializeNamed writes the fields of a value of a registered type. A frame
// carries its base state too, so it arrives as itself rather than as a fresh
// frame with the same fields.
func (s *JSONSerializer) serializeNamed(v any, rv reflect.Value) (map[string]any, error) {
	fields, err := s.serializeStruct(rv)
	if err != nil {
		return nil, err
	}
	if f, isFrame := v.(frames.Frame); isFrame {
		state, err := s.serializeValue(frameState(f.Base().State()))
		if err != nil {
			return nil, err
		}
		fields[baseKey] = state
	}
	return fields, nil
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
		return s.serializePlainStruct(rv)
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
	case json.Number:
		return number(t), nil
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
		// Something nested the far end carried and this process has no type
		// for is left out, rather than losing the message it sat in.
		slog.Warn("bus: a value of a type this process does not know is being left out", "type", name)
		return nil, nil //nolint:nilnil // a value that cannot be rebuilt is dropped, not an error
	}

	if settings.IsSettings(built) {
		values, _ := data.(map[string]any)
		decoded := make(map[string]any, len(values))
		for k, raw := range values {
			d, err := s.deserializeValue(raw)
			if err != nil {
				return nil, err
			}
			decoded[k] = d
		}
		if err := settings.FromMap(built, decoded); err != nil {
			return nil, err
		}
		return built, nil
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
	if f, isFrame := built.(frames.Frame); isFrame {
		if err := s.restoreState(f, fields[baseKey]); err != nil {
			return nil, err
		}
	}
	return built, nil
}

// restoreState sets a frame's base state back from what was carried for it,
// if anything was.
func (s *JSONSerializer) restoreState(f frames.Frame, raw any) error {
	if raw == nil {
		return nil
	}
	state, err := s.deserializeValue(raw)
	if err != nil {
		return err
	}
	if m, ok := state.(map[string]any); ok {
		f.Base().SetState(frameStateFrom(m))
	}
	return nil
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
		// A field that cannot be filled is left out rather than losing the
		// message: the rest of what it carried is still worth delivering.
		if err := s.assign(fv, raw); err != nil {
			slog.Warn("bus: a field could not be read back and is being left out",
				"type", rt.Name(), "field", f.Name, "err", err)
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

// decodeJSON reads JSON keeping its numbers exact, so a whole number arrives as
// an integer, as it was sent, rather than as a float.
func decodeJSON(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var raw any
	if err := dec.Decode(&raw); err != nil {
		return nil, err
	}
	return raw, nil
}

// number turns a JSON number into the Go value it reads as: an int when it is
// whole, a uint64 past the range of an int, and a float64 otherwise.
func number(n json.Number) any {
	if i, err := strconv.Atoi(string(n)); err == nil {
		return i
	}
	if u, err := strconv.ParseUint(string(n), 10, 64); err == nil {
		return u
	}
	f, _ := n.Float64()
	return f
}

// normalizeNumbers walks plain decoded JSON turning each json.Number into the
// Go value it reads as.
func normalizeNumbers(v any) any {
	switch t := v.(type) {
	case json.Number:
		return number(t)
	case []any:
		for i, el := range t {
			t[i] = normalizeNumbers(el)
		}
		return t
	case map[string]any:
		for k, el := range t {
			t[k] = normalizeNumbers(el)
		}
		return t
	default:
		return v
	}
}

// serializeSettings writes a settings value as the fields it gives, tagged with
// its name when it has one.
func (s *JSONSerializer) serializeSettings(v any) (any, error) {
	given, err := settings.Given(v)
	if err != nil {
		return nil, err
	}
	data := make(map[string]any, len(given))
	for k, val := range given {
		el, err := s.serializeValue(val)
		if err != nil {
			return nil, err
		}
		data[k] = el
	}
	if name, known := s.types.NameOf(v); known {
		return map[string]any{typeKey: name, dataKey: data}, nil
	}
	return data, nil
}

// serializePlainStruct writes a struct that has no name of its own on the
// wire, the way encoding/json would, so the far end reads it back into its
// type, or holds it as the object it would have decoded from JSON. A type that
// writes its own JSON (a time, say) is written that way; any other has its
// exported fields walked under their JSON names, so a value nested in it that
// needs a name of its own still gets one.
func (s *JSONSerializer) serializePlainStruct(rv reflect.Value) (any, error) {
	if writesOwnJSON(rv.Type()) || !hasExportedField(rv.Type()) {
		return marshalPlain(rv)
	}
	out := map[string]any{}
	rt := rv.Type()
	for i := range rt.NumField() {
		f := rt.Field(i)
		if !f.IsExported() {
			continue
		}
		name, omit := jsonName(f)
		if omit {
			continue
		}
		fv := rv.Field(i)
		if f.Anonymous && fv.Kind() == reflect.Struct && name == "" {
			nested, err := s.serializePlainStruct(fv)
			if err != nil {
				return nil, err
			}
			if m, ok := nested.(map[string]any); ok {
				maps.Copy(out, m)
			}
			continue
		}
		if name == "" {
			name = f.Name
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
		out[name] = el
	}
	return out, nil
}

// marshalPlain writes a value through encoding/json and reads it back as the
// plain object, list or scalar it encodes to.
func marshalPlain(rv reflect.Value) (any, error) {
	ptr := reflect.New(rv.Type())
	ptr.Elem().Set(rv)
	encoded, err := json.Marshal(ptr.Interface())
	if err != nil {
		slog.Warn("bus: a value could not be serialized and is being left out",
			"type", rv.Type().String(), "err", err)
		return nil, nil //nolint:nilnil // a field that cannot cross is dropped, not an error
	}
	decoded, err := decodeJSON(encoded)
	if err != nil {
		return nil, err
	}
	return normalizeNumbers(decoded), nil
}

// jsonName is the name encoding/json writes a field under, empty when its tag
// names none, and whether the tag leaves it out altogether.
func jsonName(f reflect.StructField) (string, bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", true
	}
	name, _, _ := strings.Cut(tag, ",")
	return name, false
}

// writesOwnJSON reports whether a type marshals itself.
func writesOwnJSON(t reflect.Type) bool {
	ptr := reflect.PointerTo(t)
	for _, iface := range []reflect.Type{
		reflect.TypeFor[json.Marshaler](), reflect.TypeFor[encoding.TextMarshaler](),
	} {
		if t.Implements(iface) || ptr.Implements(iface) {
			return true
		}
	}
	return false
}

// hasExportedField reports whether a struct type has a field encoding/json
// could see.
func hasExportedField(t reflect.Type) bool {
	for f := range t.Fields() {
		if f.IsExported() {
			return true
		}
	}
	return false
}

// frameState lays a frame's base state out as plain fields, leaving out what
// it does not carry.
func frameState(st frames.BaseState) map[string]any {
	m := map[string]any{"ID": st.ID, "TypeName": st.TypeName}
	if st.PTS != nil {
		m["PTS"] = *st.PTS
	}
	if st.BroadcastSiblingID != nil {
		m["BroadcastSiblingID"] = *st.BroadcastSiblingID
	}
	if len(st.Metadata) > 0 {
		m["Metadata"] = st.Metadata
	}
	if st.TransportSource != "" {
		m["TransportSource"] = st.TransportSource
	}
	if st.TransportDestination != "" {
		m["TransportDestination"] = st.TransportDestination
	}
	if st.Interruptible != nil {
		m["Interruptible"] = *st.Interruptible
	}
	return m
}

// frameStateFrom reads a frame's base state back from the plain fields
// frameState wrote.
func frameStateFrom(m map[string]any) frames.BaseState {
	var st frames.BaseState
	if id, ok := asUint64(m["ID"]); ok {
		st.ID = id
	}
	st.TypeName, _ = m["TypeName"].(string)
	if pts, ok := asUint64(m["PTS"]); ok {
		st.PTS = new(int64(pts)) //nolint:gosec // carried from an int64
	} else if pts, ok := m["PTS"].(int); ok {
		st.PTS = new(int64(pts))
	}
	if sib, ok := asUint64(m["BroadcastSiblingID"]); ok {
		st.BroadcastSiblingID = &sib
	}
	if md, ok := m["Metadata"].(map[string]any); ok {
		st.Metadata = md
	}
	st.TransportSource, _ = m["TransportSource"].(string)
	st.TransportDestination, _ = m["TransportDestination"].(string)
	if in, ok := m["Interruptible"].(bool); ok {
		st.Interruptible = &in
	}
	return st
}

// asUint64 reads a non-negative whole number back from what JSON decoding gave.
func asUint64(v any) (uint64, bool) {
	switch n := v.(type) {
	case int:
		if n < 0 {
			return 0, false
		}
		return uint64(n), true
	case uint64:
		return n, true
	default:
		return 0, false
	}
}
