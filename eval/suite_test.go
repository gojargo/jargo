package eval_test

import (
	"context"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gojargo/jargo/eval"
)

func writeFile(t *testing.T, dir, name, body string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestRunSuite(t *testing.T) {
	srv1 := httptest.NewServer(eval.Handler(buildFakeBot))
	defer srv1.Close()
	srv2 := httptest.NewServer(eval.Handler(buildFakeBot))
	defer srv2.Close()
	ws := func(s *httptest.Server) string { return "ws" + strings.TrimPrefix(s.URL, "http") }

	dir := t.TempDir()
	writeFile(t, dir, "pass.yaml", `name: pass
turns:
  - user: "hello world"
    expect:
      - event: llm_response
        text_contains: "hello world"
`)
	writeFile(t, dir, "fail.yaml", `name: fail
turns:
  - user: "hello"
    expect:
      - event: llm_response
        text_contains: "goodbye"
`)
	writeFile(t, dir, "manifest.yaml", fmt.Sprintf(`concurrency: 2
suite:
  - bot_url: %s
    scenarios: [pass.yaml, fail.yaml]
  - bot_url: %s
    scenarios: [pass.yaml]
`, ws(srv1), ws(srv2)))

	m, err := eval.LoadManifest(filepath.Join(dir, "manifest.yaml"))
	if err != nil {
		t.Fatal(err)
	}

	results := eval.RunSuite(context.Background(), m, nil)
	if len(results) != 3 {
		t.Fatalf("want 3 results, got %d", len(results))
	}
	passed := 0
	for _, r := range results {
		if r.Err != nil {
			t.Fatalf("unexpected run error for %s: %v", r.Scenario, r.Err)
		}
		if r.Passed() {
			passed++
		}
	}
	if passed != 2 { // pass.yaml runs on both bots; fail.yaml fails
		t.Fatalf("want 2 passed of 3, got %d\n%+v", passed, results)
	}
}

func TestLoadManifestRejectsInvalid(t *testing.T) {
	cases := map[string]struct {
		body string
		want string
	}{
		"no suite":     {"concurrency: 2\n", "no suite entries"},
		"no bot_url":   {"suite:\n  - scenarios: [a.yaml]\n", "no bot_url"},
		"no scenarios": {"suite:\n  - bot_url: ws://x\n", "no scenarios"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			writeFile(t, dir, "m.yaml", tc.body)
			_, err := eval.LoadManifest(filepath.Join(dir, "m.yaml"))
			if err == nil {
				t.Fatal("expected an error")
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.want)
			}
		})
	}
}
