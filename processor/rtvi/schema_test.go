package rtvi_test

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// The JSON Schema checks of the conformance tests. The protocol documents
// describe their payloads in JSON Schema draft 7, the dialect AsyncAPI's
// schemas follow, and a message is checked against the document itself, never
// against the Go types that build it.
//
// The checks cover the keywords the documents use. A keyword they do not know
// fails the check rather than being skipped, so a document that starts using a
// new one cannot pass unchecked. Like a draft 7 validator by default, they
// treat format as an annotation, and ignore the keywords next to a $ref.

// annotations are the keywords that describe a value without constraining it.
//
//nolint:gochecknoglobals // fixed table, read only
var annotations = map[string]bool{
	"description": true, "title": true, "default": true, "deprecated": true, "examples": true,
	"format": true, "readOnly": true, "writeOnly": true, "$comment": true,
}

// schemaChecker checks JSON values against the schemas of one document, whose
// components a $ref points into.
type schemaChecker struct {
	document map[string]any
}

// errors are what is wrong with value under schema, each prefixed with where
// in value it is. A value without problems has none.
func (c schemaChecker) errors(schema, value any, path string) []string {
	switch s := schema.(type) {
	case bool:
		if s {
			return nil
		}
		return []string{fmt.Sprintf("%s: no value is allowed", pathOrRoot(path))}
	case map[string]any:
		if ref, ok := s["$ref"].(string); ok {
			target, ok := c.resolveRef(ref)
			if !ok {
				return []string{fmt.Sprintf("%s: %q points at nothing in the document", pathOrRoot(path), ref)}
			}
			return c.errors(target, value, path)
		}
		var problems []string
		for key, arg := range s {
			problems = append(problems, c.keyword(s, key, arg, value, path)...)
		}
		slices.Sort(problems)
		return problems
	default:
		return []string{fmt.Sprintf("%s: the schema %v is neither an object nor a boolean", pathOrRoot(path), schema)}
	}
}

// valid reports whether value has no problems under schema.
func (c schemaChecker) valid(schema, value any) bool {
	return len(c.errors(schema, value, "")) == 0
}

// keyword checks value against one keyword of schema.
func (c schemaChecker) keyword(schema map[string]any, key string, arg, value any, path string) []string {
	at := pathOrRoot(path)
	switch key {
	case "type":
		return c.checkType(arg, value, at)
	case "properties", "required", "additionalProperties":
		return c.checkObject(schema, key, arg, value, path)
	case "items", "minItems":
		return c.checkArray(key, arg, value, path)
	case "enum":
		options, _ := arg.([]any)
		if !slices.ContainsFunc(options, func(option any) bool { return jsonEqual(option, value) }) {
			return []string{fmt.Sprintf("%s: %s is not one of %s", at, show(value), show(arg))}
		}
	case "const":
		if !jsonEqual(arg, value) {
			return []string{fmt.Sprintf("%s: %s was expected, got %s", at, show(arg), show(value))}
		}
	case "anyOf", "oneOf":
		return c.checkAlternatives(key, arg, value, at)
	case "allOf":
		parts, _ := arg.([]any)
		var problems []string
		for _, part := range parts {
			problems = append(problems, c.errors(part, value, path)...)
		}
		return problems
	case "not":
		if c.valid(arg, value) {
			return []string{fmt.Sprintf("%s: %s should not be valid under %s", at, show(value), show(arg))}
		}
	default:
		if !annotations[key] {
			return []string{fmt.Sprintf("%s: the checks do not know the keyword %q", at, key)}
		}
	}
	return nil
}

// checkAlternatives checks value against anyOf, valid under one alternative
// or more, or oneOf, valid under exactly one.
func (c schemaChecker) checkAlternatives(key string, arg, value any, at string) []string {
	parts, _ := arg.([]any)
	matched := 0
	for _, part := range parts {
		if c.valid(part, value) {
			matched++
		}
	}
	switch {
	case matched == 0:
		return []string{fmt.Sprintf("%s: %s is not valid under any of the given schemas", at, show(value))}
	case key == "oneOf" && matched > 1:
		return []string{fmt.Sprintf("%s: %s is valid under each of %d schemas", at, show(value), matched)}
	}
	return nil
}

// checkType checks value is of one of the JSON types arg names.
func (c schemaChecker) checkType(arg, value any, at string) []string {
	var types []string
	switch t := arg.(type) {
	case string:
		types = []string{t}
	case []any:
		for _, item := range t {
			name, _ := item.(string)
			types = append(types, name)
		}
	}
	for _, name := range types {
		if isJSONType(name, value) {
			return nil
		}
	}
	return []string{fmt.Sprintf("%s: %s is not of type %s", at, show(value), strings.Join(types, ", "))}
}

// checkObject checks an object's properties, the ones it requires, and the
// ones it doesn't list. A value that isn't an object passes.
func (c schemaChecker) checkObject(schema map[string]any, key string, arg, value any, path string) []string {
	obj, ok := value.(map[string]any)
	if !ok {
		return nil
	}
	props, _ := schema["properties"].(map[string]any)
	var problems []string
	switch key {
	case "properties":
		for name, sub := range props {
			if v, ok := obj[name]; ok {
				problems = append(problems, c.errors(sub, v, path+"/"+name)...)
			}
		}
	case "required":
		names, _ := arg.([]any)
		for _, name := range names {
			if _, ok := obj[name.(string)]; !ok {
				problems = append(problems, fmt.Sprintf("%s: %q is a required property", pathOrRoot(path), name))
			}
		}
	case "additionalProperties":
		for name, v := range obj {
			if _, listed := props[name]; !listed {
				problems = append(problems, c.errors(arg, v, path+"/"+name)...)
			}
		}
	}
	return problems
}

// checkArray checks an array's items and its length. A value that isn't an
// array passes.
func (c schemaChecker) checkArray(key string, arg, value any, path string) []string {
	items, ok := value.([]any)
	if !ok {
		return nil
	}
	switch key {
	case "items":
		if _, tuple := arg.([]any); tuple {
			return []string{fmt.Sprintf("%s: the checks do not know items as a list", pathOrRoot(path))}
		}
		var problems []string
		for i, item := range items {
			problems = append(problems, c.errors(arg, item, path+"/"+strconv.Itoa(i))...)
		}
		return problems
	case "minItems":
		minimum, _ := toFloat(arg)
		if float64(len(items)) < minimum {
			return []string{fmt.Sprintf("%s: %s should be non-empty", pathOrRoot(path), show(value))}
		}
	}
	return nil
}

// resolveRef is the schema a $ref names, a JSON pointer into the document,
// reporting false when it names nothing there. A reference outside the
// document names nothing.
func (c schemaChecker) resolveRef(ref string) (any, bool) {
	pointer, ok := strings.CutPrefix(ref, "#/")
	if !ok {
		return nil, false
	}
	var node any = c.document
	for part := range strings.SplitSeq(pointer, "/") {
		part = strings.ReplaceAll(strings.ReplaceAll(part, "~1", "/"), "~0", "~")
		obj, isObject := node.(map[string]any)
		if !isObject {
			return nil, false
		}
		if node, ok = obj[part]; !ok {
			return nil, false
		}
	}
	return node, true
}

// isJSONType reports whether value, decoded with numbers as json.Number, is of
// the named JSON type. An integer is a number with no fractional part.
func isJSONType(name string, value any) bool {
	switch name {
	case "object":
		_, ok := value.(map[string]any)
		return ok
	case "array":
		_, ok := value.([]any)
		return ok
	case "string":
		_, ok := value.(string)
		return ok
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "null":
		return value == nil
	case "number":
		_, ok := toFloat(value)
		return ok
	case "integer":
		f, ok := toFloat(value)
		return ok && f == math.Trunc(f)
	default:
		return false
	}
}

// toFloat reads a JSON number.
func toFloat(value any) (float64, bool) {
	switch n := value.(type) {
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	case float64:
		return n, true
	case int:
		return float64(n), true
	default:
		return 0, false
	}
}

// jsonEqual reports whether two decoded JSON values are equal, numbers being
// equal when their values are, and a boolean never equal to a number.
func jsonEqual(a, b any) bool {
	af, aNumber := toFloat(a)
	bf, bNumber := toFloat(b)
	if aNumber || bNumber {
		return aNumber && bNumber && af == bf
	}
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for key, value := range av {
			other, ok := bv[key]
			if !ok || !jsonEqual(value, other) {
				return false
			}
		}
		return true
	case []any:
		bv, ok := b.([]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if !jsonEqual(av[i], bv[i]) {
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(a, b)
	}
}

// show writes a value as JSON, for a problem's description.
func show(value any) string {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Sprint(value)
	}
	return string(raw)
}

// pathOrRoot names where in a message a problem is.
func pathOrRoot(path string) string {
	if path == "" {
		return "(message)"
	}
	return strings.TrimPrefix(path, "/")
}

// TestSchemaChecker checks the schema checks themselves, on the keywords the
// documents use, so a conformance test passing means what it says.
func TestSchemaChecker(t *testing.T) {
	c := schemaChecker{document: map[string]any{"components": map[string]any{"schemas": map[string]any{
		"Key": map[string]any{"type": "string", "enum": []any{"1", "#"}},
	}}}}
	schema := map[string]any{
		"type":     "object",
		"required": []any{"keys", "kind"},
		"properties": map[string]any{
			"keys": map[string]any{
				"type": "array", "minItems": json.Number("1"),
				"items": map[string]any{"$ref": "#/components/schemas/Key"},
			},
			"kind":  map[string]any{"const": "press"},
			"count": map[string]any{"type": "integer"},
			"note":  map[string]any{"anyOf": []any{map[string]any{"type": "string"}, map[string]any{"type": "null"}}},
			"name":  map[string]any{"allOf": []any{map[string]any{"type": "string"}}, "not": map[string]any{"enum": []any{"x"}}},
			"extra": map[string]any{"type": "object", "additionalProperties": false, "properties": map[string]any{}},
		},
	}
	valid := map[string]any{
		"keys": []any{"1", "#"}, "kind": "press", "count": json.Number("2.0"), "note": nil, "name": "y",
		"extra": map[string]any{}, "unknown": true,
	}
	if problems := c.errors(schema, valid, ""); len(problems) != 0 {
		t.Errorf("a valid value has problems: %v", problems)
	}
	for name, invalid := range map[string]map[string]any{
		"a required field missing": {"keys": []any{"1"}},
		"of the wrong type":        {"keys": []any{"1"}, "kind": "press", "count": json.Number("2.5")},
		"not in an enum":           {"keys": []any{"A"}, "kind": "press"},
		"too few items":            {"keys": []any{}, "kind": "press"},
		"not the const":            {"keys": []any{"1"}, "kind": "release"},
		"no alternative matching":  {"keys": []any{"1"}, "kind": "press", "note": json.Number("1")},
		"a forbidden value":        {"keys": []any{"1"}, "kind": "press", "name": "x"},
		"an unlisted property":     {"keys": []any{"1"}, "kind": "press", "extra": map[string]any{"a": true}},
	} {
		if c.valid(schema, invalid) {
			t.Errorf("%s: %v passes", name, invalid)
		}
	}
	if c.valid(map[string]any{"patternProperties": map[string]any{}}, map[string]any{}) {
		t.Error("a keyword the checks do not know passes")
	}
	if c.valid(map[string]any{"$ref": "#/components/schemas/Missing"}, "1") {
		t.Error("a reference to nothing passes")
	}
}
