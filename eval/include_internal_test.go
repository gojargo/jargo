package eval

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// loadIncluded writes files under a temporary directory, loads top from it and
// decodes what it resolves to.
func loadIncluded(t *testing.T, files map[string]string, top string) any {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	node, err := loadNode(filepath.Join(dir, top), includeDepth)
	if err != nil {
		t.Fatal(err)
	}
	var out any
	if err := node.Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

// test_nested_include_resolves_against_including_file: an included file's own
// includes resolve against its directory.
func TestNestedIncludeResolvesAgainstTheIncludingFile(t *testing.T) {
	got := loadIncluded(t, map[string]string{
		"leaf.yaml":       "leaf: top\n",
		"sub/leaf.yaml":   "leaf: sub\n",
		"sub/middle.yaml": "inner: !include leaf.yaml\n",
		"top.yaml":        "outer: !include sub/middle.yaml\n",
	}, "top.yaml")
	want := map[string]any{"outer": map[string]any{"inner": map[string]any{"leaf": "sub"}}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("loaded %v, want %v", got, want)
	}
}

// test_nested_includes_chain_across_directories: each hop resolves from where it is.
func TestNestedIncludesChainAcrossDirectories(t *testing.T) {
	got := loadIncluded(t, map[string]string{
		"shared/rubric.yaml":      "criterion: polite\n",
		"shared/judges/text.yaml": "rubric: !include ../rubric.yaml\n",
		"shared/judge.yaml":       "text: !include judges/text.yaml\n",
		"scenarios/top.yaml":      "judge: !include ../shared/judge.yaml\n",
	}, "scenarios/top.yaml")
	rubric := map[string]any{"rubric": map[string]any{"criterion": "polite"}}
	want := map[string]any{"judge": map[string]any{"text": rubric}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("loaded %v, want %v", got, want)
	}
}
