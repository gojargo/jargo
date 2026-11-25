package eval

import (
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/gojargo/jargo/internal/yamlutil"
	"go.yaml.in/yaml/v3"
)

// ScenarioError is a scenario file that is malformed, or a scenario in it that
// is invalid for its kind. Its message names the file and, where there is one,
// the turn and expectation at fault, both counted from 0.
type ScenarioError struct {
	msg string
}

// Error returns the message.
func (e *ScenarioError) Error() string { return e.msg }

// invalidf builds a ScenarioError.
func invalidf(format string, args ...any) error {
	return &ScenarioError{msg: fmt.Sprintf(format, args...)}
}

// loadMapping reads a scenario file's top-level mapping, resolving !include
// tags relative to the file.
//
// Scalars resolve as a YAML 1.1 reader resolves them, except integers, which
// are plain decimal only: `012` and `0x10` stay strings, so a DTMF sequence
// keeps its digits rather than being read as octal or hex.
func loadMapping(path string) (map[string]any, error) {
	node, err := loadNode(path)
	if err != nil {
		return nil, err
	}
	data, ok := nodeValue(node).(map[string]any)
	if !ok {
		return nil, invalidf("%s: top level must be a mapping", path)
	}
	return data, nil
}

// loadNode parses a YAML file and resolves the !include tags in it. An included
// file's own includes resolve against its directory, and a file that includes
// itself is an error.
func loadNode(path string) (*yaml.Node, error) {
	data, err := os.ReadFile(path) //nolint:gosec // scenario path is operator-supplied
	if err != nil {
		return nil, fmt.Errorf("eval: read scenario: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	if err := yamlutil.ResolveIncludes(&doc, filepath.Dir(path)); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	return &doc, nil
}

// nodeValue converts a YAML node into plain Go values: map[string]any, []any,
// string, int, float64, bool, or nil. Integers are plain decimal only, and a
// mapping key that is not a string is written out as text.
func nodeValue(node *yaml.Node) any {
	return yamlutil.Value(node, yamlutil.ValueOptions{
		DecimalIntegers: true,
		NewMapping: func(keys, values []any) any {
			out := make(map[string]any, len(keys))
			for i, key := range keys {
				out[keyString(key)] = values[i]
			}
			return out
		},
	})
}

// keyString is a mapping key as a string.
func keyString(key any) string {
	if s, ok := key.(string); ok {
		return s
	}
	return pyStr(key)
}

// truthy reports whether a value reads as true, as a condition reads it: false,
// zero, an empty string, an empty mapping or list, and nil are false.
func truthy(v any) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case int:
		return x != 0
	case float64:
		return x != 0
	case string:
		return x != ""
	case map[string]any:
		return len(x) > 0
	case []any:
		return len(x) > 0
	default:
		return true
	}
}

// pyStr renders a value as text, the way a message quotes a value inline.
func pyStr(v any) string {
	switch x := v.(type) {
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case string:
		return x
	case float64:
		return formatFloat(x)
	case map[string]any:
		parts := make([]string, 0, len(x))
		for _, k := range slices.Sorted(maps.Keys(x)) {
			parts = append(parts, pyRepr(k)+": "+pyRepr(x[k]))
		}
		return "{" + strings.Join(parts, ", ") + "}"
	case []any:
		parts := make([]string, 0, len(x))
		for _, item := range x {
			parts = append(parts, pyRepr(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	default:
		return fmt.Sprint(x)
	}
}

// pyRepr renders a value as a message quotes it: a string in quotes, anything
// else as pyStr does.
func pyRepr(v any) string {
	s, ok := v.(string)
	if !ok {
		return pyStr(v)
	}
	quote := "'"
	if strings.Contains(s, "'") && !strings.Contains(s, `"`) {
		quote = `"`
	}
	escaped := strings.ReplaceAll(s, `\`, `\\`)
	if quote == "'" {
		escaped = strings.ReplaceAll(escaped, "'", `\'`)
	}
	escaped = strings.NewReplacer("\n", `\n`, "\t", `\t`, "\r", `\r`).Replace(escaped)
	return quote + escaped + quote
}

// formatFloat renders a float as a value is written: 1.0, 0.5, 1e-07.
func formatFloat(f float64) string {
	switch {
	case math.IsInf(f, 1):
		return "inf"
	case math.IsInf(f, -1):
		return "-inf"
	case math.IsNaN(f):
		return "nan"
	}
	s := strconv.FormatFloat(f, 'g', -1, 64)
	if !strings.ContainsAny(s, ".eEn") {
		s += ".0"
	}
	return s
}

// isInt reports whether v is an integer, which a boolean is not.
func isInt(v any) (int, bool) {
	n, ok := v.(int)
	return n, ok
}

// isNumber reports whether v is a number, which a boolean is not.
func isNumber(v any) (float64, bool) {
	switch x := v.(type) {
	case int:
		return float64(x), true
	case float64:
		return x, true
	default:
		return 0, false
	}
}
