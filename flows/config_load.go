package flows

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/utils/yamlutil"
	"gopkg.in/yaml.v3"
)

// errJSONTrailingData reports JSON text with something after its value.
//
//nolint:gochecknoglobals // sentinel error
var errJSONTrailingData = errors.New("unexpected data after the top-level value")

// FlowConfigFromYAML loads a config from YAML text.
//
// baseDir is the directory that the document's `!include` paths resolve
// against; an included file's own includes resolve against its directory. When
// it is empty, `!include` is unavailable. An include can read any file the
// process can read, so leave baseDir empty for text from an untrusted source.
func FlowConfigFromYAML(text, baseDir string) (*FlowConfig, error) {
	var (
		node *yaml.Node
		err  error
	)
	if baseDir != "" {
		node, err = yamlutil.Load([]byte(text), baseDir)
	} else {
		node, err = yamlutil.Parse([]byte(text))
	}
	if err != nil {
		return nil, fmt.Errorf("flow config: %w", err)
	}
	return flowConfigFromValue(yamlutil.Value(node, yamlutil.ValueOptions{
		NewMapping: func(keys, values []any) any { return &object{keys: keys, values: values} },
	}))
}

// FlowConfigFromJSON loads a config from JSON text.
func FlowConfigFromJSON(text string) (*FlowConfig, error) {
	value, err := decodeJSON(text)
	if err != nil {
		return nil, fmt.Errorf("flow config: %w", err)
	}
	return flowConfigFromValue(value)
}

// FlowConfigFromFile loads a config from a .yaml, .yml, or .json file.
//
// YAML files may use `!include` with paths relative to the directory of the
// file that names them. Included paths are not confined to that directory, so
// load only files you control. For a config from another source, such as a
// request body, use FlowConfigFromYAML without a base directory or
// FlowConfigFromJSON.
func FlowConfigFromFile(path string) (*FlowConfig, error) {
	text, err := os.ReadFile(path) //nolint:gosec // the config path is application-supplied
	if err != nil {
		return nil, fmt.Errorf("flow config: %w", err)
	}
	if filepath.Ext(path) == ".json" {
		return FlowConfigFromJSON(string(text))
	}
	return FlowConfigFromYAML(string(text), filepath.Dir(path))
}

// FlowConfigFromMap loads a config from a document that is already parsed: maps
// with string keys, slices, strings, booleans and numbers. The cases of a
// branch may be a map[any]any whose keys are booleans or numbers.
func FlowConfigFromMap(data map[string]any) (*FlowConfig, error) {
	return flowConfigFromValue(fromGo(data))
}

// UnmarshalJSON loads the config from JSON text, as FlowConfigFromJSON does.
func (c *FlowConfig) UnmarshalJSON(data []byte) error {
	loaded, err := FlowConfigFromJSON(string(data))
	if err != nil {
		return err
	}
	*c = *loaded
	return nil
}

// MarshalJSON writes the config as a document, its nodes and subflows in the
// order they were written in.
func (c *FlowConfig) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"initial_node":`)
	if err := writeJSON(&b, c.InitialNode); err != nil {
		return nil, err
	}
	b.WriteString(`,"nodes":`)
	if err := writeOrdered(&b, c.Nodes, c.nodeNames()); err != nil {
		return nil, err
	}
	if len(c.GlobalFunctions) > 0 {
		b.WriteString(`,"global_functions":`)
		if err := writeJSON(&b, c.GlobalFunctions); err != nil {
			return nil, err
		}
	}
	if len(c.Flows) > 0 {
		b.WriteString(`,"flows":`)
		if err := writeOrdered(&b, c.Flows, c.flowNames()); err != nil {
			return nil, err
		}
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// MarshalJSON writes the subflow as a document, its nodes in the order they
// were written in.
func (s *FlowConfigSubflow) MarshalJSON() ([]byte, error) {
	var b bytes.Buffer
	b.WriteString(`{"initial_node":`)
	if err := writeJSON(&b, s.InitialNode); err != nil {
		return nil, err
	}
	b.WriteString(`,"nodes":`)
	if err := writeOrdered(&b, s.Nodes, s.nodeNames()); err != nil {
		return nil, err
	}
	b.WriteByte('}')
	return b.Bytes(), nil
}

// writeJSON appends value as JSON.
func writeJSON(b *bytes.Buffer, value any) error {
	raw, err := json.Marshal(value)
	if err != nil {
		return err
	}
	b.Write(raw)
	return nil
}

// writeOrdered appends m as a JSON object whose keys come in the order given.
func writeOrdered[V any](b *bytes.Buffer, m map[string]V, names []string) error {
	b.WriteByte('{')
	for i, name := range names {
		if i > 0 {
			b.WriteByte(',')
		}
		if err := writeJSON(b, name); err != nil {
			return err
		}
		b.WriteByte(':')
		if err := writeJSON(b, m[name]); err != nil {
			return err
		}
	}
	b.WriteByte('}')
	return nil
}

// object is a mapping of a parsed document: its keys as the document resolved
// them, which for YAML need not be strings, and its values, in document order.
type object struct {
	keys   []any
	values []any
}

// get returns the value under a string key.
func (o *object) get(name string) (any, bool) {
	for i, key := range o.keys {
		if key == name {
			return o.values[i], true
		}
	}
	return nil, false
}

// stringKeys reports whether every key is a string.
func (o *object) stringKeys() bool {
	for _, key := range o.keys {
		if _, ok := key.(string); !ok {
			return false
		}
	}
	return true
}

// decodeJSON parses JSON text into the values a document is made of, keeping
// the order of each object's keys.
func decodeJSON(text string) (any, error) {
	dec := json.NewDecoder(strings.NewReader(text))
	dec.UseNumber()
	value, err := jsonValue(dec)
	if err != nil {
		return nil, err
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, errJSONTrailingData
	}
	return value, nil
}

// jsonValue reads the next JSON value.
func jsonValue(dec *json.Decoder) (any, error) {
	token, err := dec.Token()
	if err != nil {
		return nil, err
	}
	switch t := token.(type) {
	case json.Delim:
		if t == '{' {
			return jsonObject(dec)
		}
		list := []any{}
		for dec.More() {
			item, err := jsonValue(dec)
			if err != nil {
				return nil, err
			}
			list = append(list, item)
		}
		_, err := dec.Token()
		return list, err
	case json.Number:
		if n, err := strconv.Atoi(t.String()); err == nil {
			return n, nil
		}
		f, err := t.Float64()
		return f, err
	default:
		return token, nil
	}
}

// jsonObject reads the rest of a JSON object whose opening brace was read. A
// key written twice keeps its first place and its last value.
func jsonObject(dec *json.Decoder) (any, error) {
	obj := &object{}
	for dec.More() {
		token, err := dec.Token()
		if err != nil {
			return nil, err
		}
		value, err := jsonValue(dec)
		if err != nil {
			return nil, err
		}
		var key any = token
		if seen := slices.Index(obj.keys, key); seen >= 0 {
			obj.values[seen] = value
			continue
		}
		obj.keys = append(obj.keys, key)
		obj.values = append(obj.values, value)
	}
	_, err := dec.Token()
	return obj, err
}

// fromGo converts a document made of Go maps and slices into the values a
// document is made of. A map's keys come in sorted order.
func fromGo(value any) any {
	switch v := value.(type) {
	case map[string]any:
		obj := &object{}
		for _, key := range slices.Sorted(maps.Keys(v)) {
			obj.keys = append(obj.keys, key)
			obj.values = append(obj.values, fromGo(v[key]))
		}
		return obj
	case map[string]string:
		obj := &object{}
		for _, key := range slices.Sorted(maps.Keys(v)) {
			obj.keys = append(obj.keys, key)
			obj.values = append(obj.values, v[key])
		}
		return obj
	case map[any]any:
		keys := slices.Collect(maps.Keys(v))
		slices.SortFunc(keys, func(a, b any) int { return strings.Compare(fmt.Sprint(a), fmt.Sprint(b)) })
		obj := &object{keys: keys}
		for _, key := range keys {
			obj.values = append(obj.values, fromGo(v[key]))
		}
		return obj
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = fromGo(item)
		}
		return out
	case []map[string]any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = fromGo(item)
		}
		return out
	case []string:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = item
		}
		return out
	default:
		return value
	}
}

// plain converts a document value into plain Go values: a mapping becomes a
// map[string]any, or a map[any]any when a key is not a string.
func plain(value any) any {
	switch v := value.(type) {
	case *object:
		if !v.stringKeys() {
			out := make(map[any]any, len(v.keys))
			for i, key := range v.keys {
				out[key] = plain(v.values[i])
			}
			return out
		}
		out := make(map[string]any, len(v.keys))
		for i, key := range v.keys {
			out[fmt.Sprint(key)] = plain(v.values[i])
		}
		return out
	case []any:
		out := make([]any, len(v))
		for i, item := range v {
			out[i] = plain(item)
		}
		return out
	default:
		return value
	}
}

// caseKey is the canonical string a branch matches a case key or result value
// on.
//
// Booleans, and strings spelling one in any case, become "true" and "false";
// everything else is its string form. So `true:`, `"True":`, and a result of
// true all meet at the same case. A null is "None", a number with a fraction or
// an exponent is written as a float ("7.0"), and a list or a mapping is its
// JSON text.
func caseKey(value any) string {
	var text string
	switch v := value.(type) {
	case bool:
		return strconv.FormatBool(v)
	case nil:
		return "None"
	case string:
		text = v
	case float64:
		return floatText(v)
	case float32:
		return floatText(float64(v))
	case json.Number:
		text = v.String()
		if strings.ContainsAny(text, ".eE") {
			if f, err := v.Float64(); err == nil {
				return floatText(f)
			}
		}
		if text == "-0" {
			return "0"
		}
		return text
	case []any, map[string]any:
		raw, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprint(v)
		}
		return string(raw)
	default:
		return fmt.Sprint(v)
	}
	if lowered := strings.ToLower(text); lowered == "true" || lowered == "false" {
		return lowered
	}
	return text
}

// floatText writes a float the way a result that holds one reads: 7.0, 0.5,
// 1e+16.
func floatText(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	}
	if abs := math.Abs(f); abs != 0 && (abs < 1e-4 || abs >= 1e16) {
		return strconv.FormatFloat(f, 'e', -1, 64)
	}
	text := strconv.FormatFloat(f, 'f', -1, 64)
	if !strings.Contains(text, ".") {
		text += ".0"
	}
	return text
}

// flowConfigFromValue validates a parsed document as a config.
func flowConfigFromValue(value any) (*FlowConfig, error) {
	if _, ok := value.(*object); !ok {
		return nil, &FlowConfigError{Details: []FlowConfigErrorDetail{{Msg: msgTopLevel}}}
	}
	d := &configDecoder{}
	config := d.config(value)
	if len(d.details) > 0 {
		return nil, &FlowConfigError{Details: d.details}
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	return config, nil
}
