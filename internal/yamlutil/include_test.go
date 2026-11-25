package yamlutil_test

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/gojargo/jargo/internal/yamlutil"
)

// write creates files under a new temporary directory and returns it.
func write(t *testing.T, files map[string]string) string {
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
	return dir
}

// load reads a document with its includes resolved against baseDir.
func load(t *testing.T, text, baseDir string, opts yamlutil.ValueOptions) any {
	t.Helper()
	node, err := yamlutil.Load([]byte(text), baseDir)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	return yamlutil.Value(node, opts)
}

func assertLoaded(t *testing.T, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("loaded %#v, want %#v", got, want)
	}
}

func TestIncludeResolvesRelativeToBaseDir(t *testing.T) {
	dir := write(t, map[string]string{"fragment.yaml": "a: 1\nb: two\n"})

	got := load(t, "top: !include fragment.yaml\n", dir, yamlutil.ValueOptions{})
	assertLoaded(t, got, map[string]any{"top": map[string]any{"a": 1, "b": "two"}})
}

func TestNestedIncludeResolvesAgainstTheIncludingFile(t *testing.T) {
	dir := write(t, map[string]string{
		"leaf.yaml":       "leaf: top\n",
		"sub/leaf.yaml":   "leaf: sub\n",
		"sub/middle.yaml": "inner: !include leaf.yaml\n",
	})

	got := load(t, "outer: !include sub/middle.yaml\n", dir, yamlutil.ValueOptions{})
	assertLoaded(t, got, map[string]any{"outer": map[string]any{"inner": map[string]any{"leaf": "sub"}}})
}

func TestNestedIncludesChainAcrossDirectories(t *testing.T) {
	dir := write(t, map[string]string{
		"scenarios/.keep":         "",
		"shared/rubric.yaml":      "criterion: polite\n",
		"shared/judges/text.yaml": "rubric: !include ../rubric.yaml\n",
		"shared/judge.yaml":       "text: !include judges/text.yaml\n",
	})

	got := load(t, "judge: !include ../shared/judge.yaml\n", filepath.Join(dir, "scenarios"), yamlutil.ValueOptions{})
	rubric := map[string]any{"rubric": map[string]any{"criterion": "polite"}}
	assertLoaded(t, got, map[string]any{"judge": map[string]any{"text": rubric}})
}

func TestFragmentResolvesTheSameFromAnyIncluder(t *testing.T) {
	dir := write(t, map[string]string{
		"a/deep/.keep":         "",
		"shared/leaf.yaml":     "leaf: shared\n",
		"shared/fragment.yaml": "x: !include leaf.yaml\n",
	})

	top := load(t, "f: !include shared/fragment.yaml\n", dir, yamlutil.ValueOptions{})
	deep := load(t, "f: !include ../../shared/fragment.yaml\n", filepath.Join(dir, "a", "deep"), yamlutil.ValueOptions{})
	assertLoaded(t, top, map[string]any{"f": map[string]any{"x": map[string]any{"leaf": "shared"}}})
	assertLoaded(t, deep, top)
}

func TestIncludeScalarDocument(t *testing.T) {
	dir := write(t, map[string]string{"prompt.yaml": "Say hello, then ask for the order.\n"})

	got := load(t, "content: !include prompt.yaml\n", dir, yamlutil.ValueOptions{})
	assertLoaded(t, got, map[string]any{"content": "Say hello, then ask for the order."})
}

func TestIncludeOfAMissingFileIsAnError(t *testing.T) {
	_, err := yamlutil.Load([]byte("x: !include nope.yaml\n"), t.TempDir())
	if !errors.Is(err, fs.ErrNotExist) {
		t.Fatalf("got %v, want a file-not-found error", err)
	}
}

func TestIncludeCycleIsAnError(t *testing.T) {
	dir := write(t, map[string]string{
		"a.yaml": "b: !include b.yaml\n",
		"b.yaml": "a: !include a.yaml\n",
	})

	_, err := yamlutil.Load([]byte("top: !include a.yaml\n"), dir)
	if !errors.Is(err, yamlutil.ErrIncludeCycle) {
		t.Fatalf("got %v, want ErrIncludeCycle", err)
	}
	for _, want := range []string{"!include cycle", "a.yaml", "b.yaml"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

func TestFileThatIncludesItselfIsAnError(t *testing.T) {
	dir := write(t, map[string]string{"self.yaml": "me: !include self.yaml\n"})

	_, err := yamlutil.Load([]byte("top: !include self.yaml\n"), dir)
	if !errors.Is(err, yamlutil.ErrIncludeCycle) {
		t.Fatalf("got %v, want ErrIncludeCycle", err)
	}
	if !strings.Contains(err.Error(), "!include cycle") {
		t.Errorf("error %q does not say it is a cycle", err)
	}
}

func TestFileMayBeIncludedMoreThanOnce(t *testing.T) {
	dir := write(t, map[string]string{
		"leaf.yaml":   "x: 1\n",
		"middle.yaml": "leaf: !include leaf.yaml\n",
	})

	got := load(t, "a: !include leaf.yaml\nb: !include leaf.yaml\nc: !include middle.yaml\n",
		dir, yamlutil.ValueOptions{})
	leaf := map[string]any{"x": 1}
	assertLoaded(t, got, map[string]any{"a": leaf, "b": leaf, "c": map[string]any{"leaf": leaf}})
}

func TestIncludeOfANonScalarIsAnError(t *testing.T) {
	_, err := yamlutil.Load([]byte("x: !include [a, b]\n"), t.TempDir())
	if !errors.Is(err, yamlutil.ErrIncludeNotPath) {
		t.Fatalf("got %v, want ErrIncludeNotPath", err)
	}
	if !strings.Contains(err.Error(), "expects a file path") {
		t.Errorf("error %q does not say what it expects", err)
	}
}

// A document parsed without include support rejects the tag.
func TestParseRejectsInclude(t *testing.T) {
	write(t, map[string]string{"fragment.yaml": "a: 1\n"})

	_, err := yamlutil.Parse([]byte("x: !include fragment.yaml\n"))
	if !errors.Is(err, yamlutil.ErrIncludeUnavailable) {
		t.Fatalf("got %v, want ErrIncludeUnavailable", err)
	}
	if _, err := yamlutil.Parse([]byte("x: 1\n")); err != nil {
		t.Fatalf("Parse of a document without includes: %v", err)
	}
}

// The options a document is read with apply to what it includes: a
// leading-zero token stays a string under DecimalIntegers.
func TestIncludedFileIsReadWithTheSameOptions(t *testing.T) {
	dir := write(t, map[string]string{"digits.yaml": "dtmf: 012\n"})

	got := load(t, "t: !include digits.yaml\n", dir, yamlutil.ValueOptions{DecimalIntegers: true})
	assertLoaded(t, got, map[string]any{"t": map[string]any{"dtmf": "012"}})

	got = load(t, "t: !include digits.yaml\n", dir, yamlutil.ValueOptions{})
	assertLoaded(t, got, map[string]any{"t": map[string]any{"dtmf": 10}})
}

func TestNestedIncludeIsReadWithTheSameOptions(t *testing.T) {
	dir := write(t, map[string]string{
		"sub/digits.yaml": "dtmf: 012\n",
		"sub/middle.yaml": "inner: !include digits.yaml\n",
	})

	got := load(t, "outer: !include sub/middle.yaml\n", dir, yamlutil.ValueOptions{DecimalIntegers: true})
	assertLoaded(t, got, map[string]any{"outer": map[string]any{"inner": map[string]any{"dtmf": "012"}}})
}

// ResolveIncludes works on a tree the caller parsed, and leaves a later load
// resolving against the directory it was given.
func TestResolveIncludesOnAParsedTree(t *testing.T) {
	dir := write(t, map[string]string{
		"fragment.yaml":   "k: v\n",
		"sub/leaf.yaml":   "leaf: sub\n",
		"sub/middle.yaml": "inner: !include leaf.yaml\n",
	})
	fragment := map[string]any{"x": map[string]any{"k": "v"}}

	assertLoaded(t, load(t, "x: !include fragment.yaml\n", dir, yamlutil.ValueOptions{}), fragment)
	assertLoaded(t, load(t, "x: !include sub/middle.yaml\n", dir, yamlutil.ValueOptions{}),
		map[string]any{"x": map[string]any{"inner": map[string]any{"leaf": "sub"}}})
	assertLoaded(t, load(t, "x: !include fragment.yaml\n", dir, yamlutil.ValueOptions{}), fragment)
}

func TestValueResolvesScalars(t *testing.T) {
	text := "yes: on\nn: ~\nhex: 0x10\noctal: 012\nbinary: 0b11\nminutes: 1:30\nbig: 1_000\n" +
		"ratio: 1.5\nquoted: \"012\"\nexp: 1e3\n"
	node, err := yamlutil.Parse([]byte(text))
	if err != nil {
		t.Fatal(err)
	}

	assertLoaded(t, yamlutil.Value(node, yamlutil.ValueOptions{}), map[string]any{
		"true": true, "n": nil, "hex": 16, "octal": 10, "binary": 3, "minutes": 90, "big": 1000,
		"ratio": 1.5, "quoted": "012", "exp": "1e3",
	})
	assertLoaded(t, yamlutil.Value(node, yamlutil.ValueOptions{DecimalIntegers: true}), map[string]any{
		"true": true, "n": nil, "hex": "0x10", "octal": "012", "binary": "0b11", "minutes": "1:30",
		"big": 1000, "ratio": 1.5, "quoted": "012", "exp": "1e3",
	})
}

func TestValueAppliesMergeKeys(t *testing.T) {
	text := "base: &base {a: 1, b: 2}\nother: &other {b: 3, c: 4}\n" +
		"one: {<<: *base, b: 9}\nmany: {<<: [*base, *other]}\n"
	node, err := yamlutil.Parse([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	got, ok := yamlutil.Value(node, yamlutil.ValueOptions{}).(map[string]any)
	if !ok {
		t.Fatal("top level is not a mapping")
	}
	assertLoaded(t, got["one"], map[string]any{"a": 1, "b": 9})
	assertLoaded(t, got["many"], map[string]any{"a": 1, "b": 2, "c": 4})
}

// NewMapping sees a mapping's keys as they resolve, in document order.
func TestValueBuildsMappingsWithNewMapping(t *testing.T) {
	node, err := yamlutil.Parse([]byte("z: 1\ntrue: 2\n3: 3\n"))
	if err != nil {
		t.Fatal(err)
	}
	var seen []any
	yamlutil.Value(node, yamlutil.ValueOptions{NewMapping: func(keys, _ []any) any {
		seen = keys
		return nil
	}})
	assertLoaded(t, seen, []any{"z", true, 3})
}
