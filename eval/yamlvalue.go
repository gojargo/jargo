package eval

import (
	"errors"
	"fmt"
	"maps"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Errors reading a scenario file's YAML, before any of its keys are looked at.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errIncludeNotPath = errors.New("!include expects a file path")
	errIncludeTooDeep = errors.New("!include nested too deep, check for a cycle")
)

// includeDepth bounds how deep !include may nest, so a file that includes
// itself is reported rather than exhausting the stack.
const includeDepth = 16

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
	node, err := loadNode(path, includeDepth)
	if err != nil {
		return nil, err
	}
	data, ok := nodeValue(node).(map[string]any)
	if !ok {
		return nil, invalidf("%s: top level must be a mapping", path)
	}
	return data, nil
}

// loadNode parses a YAML file and resolves the !include tags in it.
func loadNode(path string, depth int) (*yaml.Node, error) {
	data, err := os.ReadFile(path) //nolint:gosec // scenario path is operator-supplied
	if err != nil {
		return nil, fmt.Errorf("eval: read scenario: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	if err := resolveIncludes(&doc, filepath.Dir(path), depth); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	return &doc, nil
}

// resolveIncludes replaces every `!include <path>` node in the tree with the
// contents of the file it names, resolved against dir. Included files are parsed
// the same way, so an include may itself include, against its own directory.
func resolveIncludes(node *yaml.Node, dir string, depth int) error {
	if node.Tag == "!include" {
		if depth <= 0 {
			return errIncludeTooDeep
		}
		if node.Kind != yaml.ScalarNode {
			return errIncludeNotPath
		}
		target := node.Value
		if !filepath.IsAbs(target) {
			target = filepath.Join(dir, target)
		}
		included, err := loadNode(target, depth-1)
		if err != nil {
			return err
		}
		*node = *unwrapDocument(included)
		return nil
	}
	for _, child := range node.Content {
		if err := resolveIncludes(child, dir, depth); err != nil {
			return err
		}
	}
	return nil
}

// unwrapDocument returns the value a parsed document wraps.
func unwrapDocument(node *yaml.Node) *yaml.Node {
	if node.Kind == yaml.DocumentNode && len(node.Content) == 1 {
		return node.Content[0]
	}
	return node
}

// The plain scalars a scenario file resolves to something other than a string.
//
//nolint:gochecknoglobals // compiled once, read only
var (
	yamlNull  = regexp.MustCompile(`^(?:~|null|Null|NULL|)$`)
	yamlBool  = regexp.MustCompile(`^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$`)
	yamlInt   = regexp.MustCompile(`^[-+]?(?:0|[1-9][0-9_]*)$`)
	yamlFloat = regexp.MustCompile(`^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?` +
		`|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$`)
)

// quotedStyles are the scalar styles whose value is always a string.
const quotedStyles = yaml.DoubleQuotedStyle | yaml.SingleQuotedStyle | yaml.LiteralStyle | yaml.FoldedStyle

// nodeValue converts a YAML node into plain Go values: map[string]any, []any,
// string, int, float64, bool, or nil.
func nodeValue(node *yaml.Node) any {
	switch node.Kind {
	case yaml.DocumentNode:
		if len(node.Content) == 0 {
			return nil
		}
		return nodeValue(node.Content[0])
	case yaml.AliasNode:
		return nodeValue(node.Alias)
	case yaml.MappingNode:
		return mappingNodeValue(node)
	case yaml.SequenceNode:
		out := make([]any, 0, len(node.Content))
		for _, item := range node.Content {
			out = append(out, nodeValue(item))
		}
		return out
	default:
		return scalarValue(node)
	}
}

// mappingNodeValue converts a mapping node, applying `<<` merge keys.
func mappingNodeValue(node *yaml.Node) map[string]any {
	out := make(map[string]any, len(node.Content)/2)
	var merged []map[string]any
	for i := 0; i+1 < len(node.Content); i += 2 {
		key, value := node.Content[i], node.Content[i+1]
		if key.Tag == "!!merge" {
			switch v := nodeValue(value).(type) {
			case map[string]any:
				merged = append(merged, v)
			case []any:
				for _, item := range v {
					if m, ok := item.(map[string]any); ok {
						merged = append(merged, m)
					}
				}
			}
			continue
		}
		out[keyString(nodeValue(key))] = nodeValue(value)
	}
	for _, m := range merged {
		for k, v := range m {
			if _, set := out[k]; !set {
				out[k] = v
			}
		}
	}
	return out
}

// keyString is a mapping key as a string.
func keyString(key any) string {
	if s, ok := key.(string); ok {
		return s
	}
	return pyStr(key)
}

// scalarValue resolves a scalar: a quoted one is a string, an explicitly tagged
// one follows its tag, and a plain one resolves by its text.
func scalarValue(node *yaml.Node) any {
	if node.Style&quotedStyles != 0 {
		return node.Value
	}
	if node.Style&yaml.TaggedStyle != 0 {
		var v any
		if err := node.Decode(&v); err == nil {
			return v
		}
		return node.Value
	}
	return plainScalar(node.Value)
}

// plainScalar resolves an unquoted, untagged scalar by its text.
func plainScalar(text string) any {
	switch {
	case yamlNull.MatchString(text):
		return nil
	case yamlBool.MatchString(text):
		switch strings.ToLower(text) {
		case "yes", "true", "on":
			return true
		default:
			return false
		}
	case yamlInt.MatchString(text):
		if n, err := strconv.Atoi(strings.ReplaceAll(text, "_", "")); err == nil {
			return n
		}
		return text
	case yamlFloat.MatchString(text):
		return parseYAMLFloat(text)
	default:
		return text
	}
}

// parseYAMLFloat reads a scalar yamlFloat matched.
func parseYAMLFloat(text string) float64 {
	clean := strings.ToLower(strings.ReplaceAll(text, "_", ""))
	switch strings.TrimLeft(clean, "+-") {
	case ".inf":
		if strings.HasPrefix(clean, "-") {
			return math.Inf(-1)
		}
		return math.Inf(1)
	case ".nan":
		return math.NaN()
	}
	f, err := strconv.ParseFloat(clean, 64)
	if err != nil {
		return 0
	}
	return f
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
