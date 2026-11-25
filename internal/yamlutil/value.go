package yamlutil

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"go.yaml.in/yaml/v3"
)

// ValueOptions says how Value reads a document.
type ValueOptions struct {
	// DecimalIntegers resolves only plain decimal integers: `012` and `0x10`
	// stay strings, so a sequence of digits keeps them rather than being read
	// as octal or hex. Left false, integers resolve as a YAML 1.1 reader
	// resolves them: binary, octal, decimal, hexadecimal and sexagesimal.
	DecimalIntegers bool
	// NewMapping builds the value of a mapping from its keys and values, in
	// document order. Nil builds a map[string]any, a key that is not a string
	// being written out as text.
	NewMapping func(keys, values []any) any
}

// The plain scalars a document resolves to something other than a string.
//
//nolint:gochecknoglobals // compiled once, read only
var (
	yamlNull    = regexp.MustCompile(`^(?:~|null|Null|NULL|)$`)
	yamlBool    = regexp.MustCompile(`^(?:yes|Yes|YES|no|No|NO|true|True|TRUE|false|False|FALSE|on|On|ON|off|Off|OFF)$`)
	yamlDecimal = regexp.MustCompile(`^[-+]?(?:0|[1-9][0-9_]*)$`)
	yamlInt     = regexp.MustCompile(`^(?:[-+]?0b[0-1_]+|[-+]?0[0-7_]+|[-+]?(?:0|[1-9][0-9_]*)` +
		`|[-+]?0x[0-9a-fA-F_]+|[-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+)$`)
	yamlFloat = regexp.MustCompile(`^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?` +
		`|\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$`)
)

// quotedStyles are the scalar styles whose value is always a string.
const quotedStyles = yaml.DoubleQuotedStyle | yaml.SingleQuotedStyle | yaml.LiteralStyle | yaml.FoldedStyle

// mergeTag is the tag of a `<<` merge key.
const mergeTag = "!!merge"

// Value converts a YAML node into plain Go values: a mapping (a map[string]any
// unless opts builds something else), []any, string, int, float64, bool, or
// nil.
//
// Scalars resolve as a YAML 1.1 reader resolves them: `yes`, `no`, `on` and
// `off` are booleans. A quoted scalar is a string, and an explicitly tagged one
// follows its tag. `<<` merge keys are applied.
func Value(node *yaml.Node, opts ValueOptions) any {
	switch node.Kind {
	case yaml.DocumentNode:
		if len(node.Content) == 0 {
			return nil
		}
		return Value(node.Content[0], opts)
	case yaml.AliasNode:
		return Value(node.Alias, opts)
	case yaml.MappingNode:
		keys, values := mappingEntries(node, opts)
		if opts.NewMapping != nil {
			return opts.NewMapping(keys, values)
		}
		out := make(map[string]any, len(keys))
		for i, key := range keys {
			name, ok := key.(string)
			if !ok {
				name = fmt.Sprint(key)
			}
			out[name] = values[i]
		}
		return out
	case yaml.SequenceNode:
		out := make([]any, 0, len(node.Content))
		for _, item := range node.Content {
			out = append(out, Value(item, opts))
		}
		return out
	default:
		return scalarValue(node, opts)
	}
}

// mappingEntries is a mapping node's keys and values, with `<<` merge keys
// applied: a merged mapping's entries come first, an earlier merged mapping
// wins over a later one, and the mapping's own entries win over both.
func mappingEntries(node *yaml.Node, opts ValueOptions) (keys, values []any) {
	index := make(map[string]int)
	set := func(key, value any, override bool) {
		id := fmt.Sprintf("%T\x00%v", key, key)
		if at, seen := index[id]; seen {
			if override {
				values[at] = value
			}
			return
		}
		index[id] = len(keys)
		keys = append(keys, key)
		values = append(values, value)
	}

	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Tag != mergeTag {
			continue
		}
		for _, merged := range mergedMappings(node.Content[i+1]) {
			mergedKeys, mergedValues := mappingEntries(merged, opts)
			for j, key := range mergedKeys {
				set(key, mergedValues[j], false)
			}
		}
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Tag == mergeTag {
			continue
		}
		set(Value(node.Content[i], opts), Value(node.Content[i+1], opts), true)
	}
	return keys, values
}

// mergedMappings is the mapping nodes a merge key's value names: a mapping, or
// a sequence of them, each possibly reached through an alias.
func mergedMappings(node *yaml.Node) []*yaml.Node {
	if node.Kind == yaml.AliasNode {
		node = node.Alias
	}
	switch node.Kind {
	case yaml.MappingNode:
		return []*yaml.Node{node}
	case yaml.SequenceNode:
		var out []*yaml.Node
		for _, item := range node.Content {
			if item.Kind == yaml.AliasNode {
				item = item.Alias
			}
			if item.Kind == yaml.MappingNode {
				out = append(out, item)
			}
		}
		return out
	default:
		return nil
	}
}

// scalarValue resolves a scalar: a quoted one is a string, an explicitly tagged
// one follows its tag, and a plain one resolves by its text.
func scalarValue(node *yaml.Node, opts ValueOptions) any {
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
	return plainScalar(node.Value, opts)
}

// plainScalar resolves an unquoted, untagged scalar by its text.
func plainScalar(text string, opts ValueOptions) any {
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
	case opts.DecimalIntegers && yamlDecimal.MatchString(text),
		!opts.DecimalIntegers && yamlInt.MatchString(text):
		if n, ok := parseYAMLInt(text); ok {
			return n
		}
		return text
	case yamlFloat.MatchString(text):
		return parseYAMLFloat(text)
	default:
		return text
	}
}

// parseYAMLInt reads a scalar yamlInt matched. It reports false for a number
// that does not fit an int.
func parseYAMLInt(text string) (int, bool) {
	clean := strings.ReplaceAll(text, "_", "")
	sign := 1
	switch {
	case strings.HasPrefix(clean, "-"):
		sign, clean = -1, clean[1:]
	case strings.HasPrefix(clean, "+"):
		clean = clean[1:]
	}

	var (
		n   int64
		err error
	)
	switch {
	case clean == "0":
	case strings.HasPrefix(clean, "0b"):
		n, err = strconv.ParseInt(clean[2:], 2, 64)
	case strings.HasPrefix(clean, "0x"):
		n, err = strconv.ParseInt(clean[2:], 16, 64)
	case strings.HasPrefix(clean, "0"):
		n, err = strconv.ParseInt(clean, 8, 64)
	case strings.Contains(clean, ":"):
		// Sexagesimal: each part is a base-60 digit.
		for part := range strings.SplitSeq(clean, ":") {
			digit, perr := strconv.ParseInt(part, 10, 64)
			if perr != nil || n > (math.MaxInt64-digit)/60 {
				return 0, false
			}
			n = n*60 + digit
		}
	default:
		n, err = strconv.ParseInt(clean, 10, 64)
	}
	if err != nil || n > math.MaxInt {
		return 0, false
	}
	return sign * int(n), true
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
