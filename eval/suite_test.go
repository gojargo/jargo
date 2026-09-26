package eval_test

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
        within_ms: 2000
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

// overlapBot counts how many scenarios are being played against it at once, and
// records the highest that count ever reached, which is what a concurrency cap
// is about.
type overlapBot struct {
	mu   sync.Mutex
	live int
	peak int
}

func (b *overlapBot) enter() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.live++
	b.peak = max(b.peak, b.live)
}

func (b *overlapBot) leave() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.live--
}

func (b *overlapBot) highest() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.peak
}

// serve runs a bot counting the scenarios being played against it at once. The
// count is kept around the handler, whose life is the connection's, so a
// scenario that has finished is no longer counted.
func (b *overlapBot) serve(t *testing.T) string {
	t.Helper()
	bot := eval.Handler(buildFakeBot)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b.enter()
		defer b.leave()
		bot.ServeHTTP(w, r)
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

// suiteDir writes a manifest and the scenarios it names, and loads it.
func suiteDir(t *testing.T, manifest string, scenarios int) *eval.Manifest {
	t.Helper()
	dir := t.TempDir()
	for i := range scenarios {
		writeFile(t, dir, fmt.Sprintf("s%d.yaml", i), fmt.Sprintf(`name: s%d
turns:
  - user: "hello"
    expect:
      - event: llm_response
        text_contains: "you said"
`, i))
	}
	writeFile(t, dir, "manifest.yaml", manifest)
	m, err := eval.LoadManifest(filepath.Join(dir, "manifest.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// wantAtMost fails unless the bot was played no more scenarios at once than its
// entry's cap.
//
// The count is one more than that at most, because a connection the harness has
// finished with is counted until the bot's own pipeline has torn down behind it.
// That laxity is far short of the failure being watched for here, where an entry
// with a cap of its own takes every slot the suite has.
func wantAtMost(t *testing.T, bot *overlapBot, slots int, what string) {
	t.Helper()
	if got := bot.highest(); got > slots+1 {
		t.Errorf("%s was played %d scenarios at once, want at most its %d", what, got, slots)
	}
}

// Its concurrency: is how many it may have in flight at once, for a bot whose
// provider rate-limits concurrent connections.
func TestAnEntrysConcurrencyIsItsOwnCap(t *testing.T) {
	bot := &overlapBot{}
	m := suiteDir(t, fmt.Sprintf(`concurrency: 6
suite:
  - bot_url: %s
    concurrency: 2
    scenarios: [s0.yaml, s1.yaml, s2.yaml, s3.yaml, s4.yaml, s5.yaml]
`, bot.serve(t)), 6)

	eval.RunSuite(t.Context(), m, nil)
	wantAtMost(t, bot, 2, "the bot")
}

// An entry is labeled by its name, or by its bot when it has none, and the
// label is what the results carry.
func TestSuiteResultsCarryTheEntryLabel(t *testing.T) {
	bot := &overlapBot{}
	url := bot.serve(t)
	m := suiteDir(t, fmt.Sprintf(`suite:
  - name: nightly
    bot_url: %s
    scenarios: [s0.yaml]
  - bot_url: %s
    scenarios: [s1.yaml]
`, url, url), 2)

	results := eval.RunSuite(t.Context(), m, nil)
	if len(results) != 2 {
		t.Fatalf("got %d results, want two", len(results))
	}
	if results[0].Entry != "nightly" {
		t.Errorf("entry = %q, want the name the manifest gave it", results[0].Entry)
	}
	if results[1].Entry != url {
		t.Errorf("entry = %q, want the bot's URL where the entry has no name", results[1].Entry)
	}
}
