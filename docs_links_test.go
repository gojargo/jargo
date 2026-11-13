package jargo_test

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestDocsLinks fails on a relative link in the documentation (docs/ and the
// README) that points at nothing. The pages are read on GitHub and imported by
// gojargo.dev, and both resolve a relative link against the file tree, so a
// link to a missing file or directory is broken in both places. Links inside
// code (fenced blocks and inline spans) are not links and are skipped.
func TestDocsLinks(t *testing.T) {
	var (
		fence      = regexp.MustCompile("^\\s*(`{3,}|~{3,})")
		inlineCode = regexp.MustCompile("`+[^`]*`+")
		inline     = regexp.MustCompile(`\]\(([^)\s]+)(?:\s+"[^"]*")?\)`)
		reference  = regexp.MustCompile(`^\s*\[[^\]]+\]:\s*(\S+)`)
		scheme     = regexp.MustCompile(`^[a-zA-Z][a-zA-Z0-9+.-]*:`)
	)

	files := []string{"README.md"}
	err := filepath.WalkDir("docs", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, ".md") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	checked := 0
	for _, file := range files {
		data, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		open := "" // the fence of the code block we are in, if any
		for i, line := range strings.Split(string(data), "\n") {
			if m := fence.FindStringSubmatch(line); m != nil {
				switch {
				case open == "":
					open = m[1]
				case m[1][0] == open[0] && len(m[1]) >= len(open):
					open = ""
				}
				continue
			}
			if open != "" {
				continue
			}
			line = inlineCode.ReplaceAllString(line, "")

			var targets []string
			for _, m := range inline.FindAllStringSubmatch(line, -1) {
				targets = append(targets, m[1])
			}
			if m := reference.FindStringSubmatch(line); m != nil {
				targets = append(targets, m[1])
			}
			for _, target := range targets {
				if scheme.MatchString(target) || strings.HasPrefix(target, "#") || strings.HasPrefix(target, "/") {
					continue // a URL, an anchor on the same page, or site-absolute
				}
				path, _, _ := strings.Cut(target, "#")
				path, _, _ = strings.Cut(path, "?")
				if path == "" {
					continue
				}
				checked++
				resolved := filepath.Join(filepath.Dir(file), filepath.FromSlash(path))
				if _, err := os.Stat(resolved); err != nil {
					t.Errorf("%s:%d: link %q: %s does not exist", file, i+1, target, resolved)
				}
			}
		}
	}
	if checked == 0 {
		t.Fatal("no relative links found; the link pattern is broken")
	}
}
