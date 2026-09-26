package sentencex

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestLanguageCorpus runs each language's corpus through that language's
// rules. Lines starting with # are comments wherever they appear, a
// malformed case is skipped, and both sides are compared with whitespace
// runs collapsed to one space.
//
// he.txt is not run: no language uses it upstream, and the upstream rules do
// not produce its expected output either.
func TestLanguageCorpus(t *testing.T) {
	// Each corpus file and the language whose rules it tests. lists.txt
	// exercises the list detector through the English rules.
	corpusLanguages := []struct {
		file string
		code string
	}{
		{"am.txt", "am"},
		{"ar.txt", "ar"},
		{"bg.txt", "bg"},
		{"bn.txt", "bn"},
		{"ca.txt", "ca"},
		{"da.txt", "da"},
		{"de.txt", "de"},
		{"el.txt", "el"},
		{"en.txt", "en"},
		{"es.txt", "es"},
		{"fi.txt", "fi"},
		{"fr.txt", "fr"},
		{"gu.txt", "gu"},
		{"hi.txt", "hi"},
		{"hy.txt", "hy"},
		{"it.txt", "it"},
		{"ja.txt", "ja"},
		{"kk.txt", "kk"},
		{"kn.txt", "kn"},
		{"ml.txt", "ml"},
		{"mr.txt", "mr"},
		{"my.txt", "my"},
		{"nl.txt", "nl"},
		{"pa.txt", "pa"},
		{"pl.txt", "pl"},
		{"pt.txt", "pt"},
		{"ru.txt", "ru"},
		{"sk.txt", "sk"},
		{"ta.txt", "ta"},
		{"te.txt", "te"},
		{"uk.txt", "uk"},
		{"lists.txt", "en"},
	}

	for _, c := range corpusLanguages {
		t.Run(strings.TrimSuffix(c.file, ".txt"), func(t *testing.T) {
			lang, ok := languages[c.code]
			if !ok {
				t.Fatalf("no rules for %q", c.code)
			}
			runLanguageTests(t, lang(), filepath.Join("testdata", c.file))
		})
	}
}

// TestSegmentCorpus runs the corpora that go through Segment with a language
// code that resolves by fallback. A case starting with # is skipped, a
// malformed case fails, and both sides are compared trimmed.
func TestSegmentCorpus(t *testing.T) {
	for _, code := range []string{"ur", "zh"} {
		t.Run(code, func(t *testing.T) {
			runLanguageTestsForLanguage(t, code, filepath.Join("testdata", code+".txt"))
		})
	}
}

func runLanguageTests(t *testing.T, lang *language, testFile string) {
	t.Helper()

	raw, err := os.ReadFile(testFile) //nolint:gosec // a corpus file under testdata
	if err != nil {
		t.Fatalf("read %s: %v", testFile, err)
	}
	content := strings.ReplaceAll(string(raw), "\r\n", "\n")

	normalise := func(s string) string { return strings.Join(strings.Fields(s), " ") }

	ran := 0
	for testCase := range strings.SplitSeq(content, "===") {
		var kept []string
		for _, line := range lines(testCase) {
			if !strings.HasPrefix(trimStart(line), "#") {
				kept = append(kept, line)
			}
		}
		cleaned := strings.TrimSpace(strings.Join(kept, "\n"))
		if cleaned == "" {
			continue
		}

		parts := nonEmptyTrimmedParts(cleaned)
		if len(parts) != 2 {
			continue // Skip malformed test cases.
		}

		input := parts[0]
		var expected []string
		for _, line := range lines(parts[1]) {
			if s := normalise(line); s != "" {
				expected = append(expected, s)
			}
		}

		var actual []string
		for _, item := range lang.segment(input) {
			if s := normalise(item); s != "" {
				actual = append(actual, s)
			}
		}

		ran++
		if !slices.Equal(actual, expected) {
			t.Errorf("failed for input:\n%s\ngot:  %q\nwant: %q", input, actual, expected)
		}
	}
	t.Logf("%d cases", ran)
}

func runLanguageTestsForLanguage(t *testing.T, code, testFile string) {
	t.Helper()

	raw, err := os.ReadFile(testFile) //nolint:gosec // a corpus file under testdata
	if err != nil {
		t.Fatalf("read %s: %v", testFile, err)
	}

	ran := 0
	for testCase := range strings.SplitSeq(string(raw), "===") {
		testCase = strings.TrimSpace(testCase)
		if testCase == "" || strings.HasPrefix(testCase, "#") {
			continue // Skip comment and empty cases.
		}
		parts := nonEmptyTrimmedParts(testCase)
		if len(parts) == 0 {
			continue // Skip cases with empty lines.
		}
		if len(parts) != 2 {
			t.Fatalf("malformed test case:\n%s", testCase)
		}

		input := parts[0]
		var expected []string
		for _, line := range lines(parts[1]) {
			if s := strings.TrimSpace(line); s != "" {
				expected = append(expected, s)
			}
		}

		result := Segment(code, input)
		trimmed := make([]string, 0, len(result))
		for _, item := range result {
			trimmed = append(trimmed, strings.TrimSpace(item))
		}

		ran++
		if !slices.Equal(trimmed, expected) {
			t.Errorf("failed for input:\n%s\ngot:  %q\nwant: %q", input, trimmed, expected)
		}
	}
	t.Logf("%d cases", ran)
}

// nonEmptyTrimmedParts splits a case on ---, trims each part and drops the
// empty ones.
func nonEmptyTrimmedParts(testCase string) []string {
	var parts []string
	for part := range strings.SplitSeq(testCase, "---") {
		if part = strings.TrimSpace(part); part != "" {
			parts = append(parts, part)
		}
	}
	return parts
}
