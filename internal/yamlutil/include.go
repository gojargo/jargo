// Package yamlutil holds the YAML loading helpers shared by the framework's
// YAML-based configuration.
//
// The main helper resolves an `!include <relative-path>` tag, so a document can
// pull in other files:
//
//	judge: !include judge_audio.yaml
//	task_messages: !include prompts/greeting.yaml
//
// A path resolves relative to the file that names it, so an included file's own
// includes resolve against its directory. Paths are not confined to that
// directory: `..` and absolute paths reach anywhere the process can read. Use
// these helpers only on trusted documents.
package yamlutil

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"go.yaml.in/yaml/v3"
)

// includeTag is the tag that names a file to pull in.
const includeTag = "!include"

// Errors resolving the !include tags of a document.
//
//nolint:gochecknoglobals // sentinel errors
var (
	// ErrIncludeNotPath reports an !include whose value is not a scalar.
	ErrIncludeNotPath = errors.New("!include expects a file path")
	// ErrIncludeCycle reports a file that includes itself, directly or through
	// the files it includes.
	ErrIncludeCycle = errors.New("!include cycle")
	// ErrIncludeUnavailable reports an !include in a document parsed without
	// include support.
	ErrIncludeUnavailable = errors.New("could not determine a constructor for the tag '!include'")
)

// Load parses a YAML document and resolves its !include tags against baseDir,
// typically the directory of the document being loaded. See ResolveIncludes.
func Load(data []byte, baseDir string) (*yaml.Node, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if err := ResolveIncludes(&doc, baseDir); err != nil {
		return nil, err
	}
	return &doc, nil
}

// Parse parses a YAML document without include support: an !include tag in it
// is an ErrIncludeUnavailable. It is how a document from an untrusted source is
// read, since an include can read any file the process can read.
func Parse(data []byte) (*yaml.Node, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if node := findInclude(&doc); node != nil {
		return nil, fmt.Errorf("%w (line %d)", ErrIncludeUnavailable, node.Line)
	}
	return &doc, nil
}

// ResolveIncludes replaces every `!include <path>` node in the tree with the
// contents of the file it names.
//
// Paths resolve against baseDir and may reach outside it. An included file is
// parsed the same way and its own includes resolve against that file's
// directory, so a fragment's includes work wherever the fragment is included
// from. A file that includes itself, directly or through the files it
// includes, is an ErrIncludeCycle. A file may be included more than once.
func ResolveIncludes(node *yaml.Node, baseDir string) error {
	return resolveIncludes(node, baseDir, nil)
}

// resolveIncludes resolves the includes under node. including holds the files
// being loaded on the way to this one: a file among them that is included again
// would include itself without end.
func resolveIncludes(node *yaml.Node, baseDir string, including []string) error {
	if node.Tag != includeTag {
		for _, child := range node.Content {
			if err := resolveIncludes(child, baseDir, including); err != nil {
				return err
			}
		}
		return nil
	}

	if node.Kind != yaml.ScalarNode {
		return fmt.Errorf("%w (line %d)", ErrIncludeNotPath, node.Line)
	}
	path := node.Value
	if !filepath.IsAbs(path) {
		path = filepath.Join(baseDir, path)
	}
	resolved := resolvedPath(path)
	chain := append(slices.Clone(including), resolved)
	if slices.Contains(including, resolved) {
		return fmt.Errorf("%w: %s (line %d)", ErrIncludeCycle, strings.Join(chain, " -> "), node.Line)
	}

	data, err := os.ReadFile(path) //nolint:gosec // include paths come from a trusted document
	if err != nil {
		return fmt.Errorf("!include: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return fmt.Errorf("!include %s: %w", path, err)
	}
	if err := resolveIncludes(&doc, filepath.Dir(path), chain); err != nil {
		return err
	}
	*node = *unwrapDocument(&doc)
	return nil
}

// resolvedPath is path made absolute with its symbolic links followed, which is
// what tells two spellings of one file apart from two files. A path that cannot
// be followed, such as one that does not exist, is only made absolute.
func resolvedPath(path string) string {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = filepath.Clean(path)
	}
	if followed, err := filepath.EvalSymlinks(abs); err == nil {
		return followed
	}
	return abs
}

// findInclude returns the first !include node in the tree, or nil.
func findInclude(node *yaml.Node) *yaml.Node {
	if node.Tag == includeTag {
		return node
	}
	for _, child := range node.Content {
		if found := findInclude(child); found != nil {
			return found
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
