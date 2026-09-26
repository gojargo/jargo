package eval

import (
	"context"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"
)

// The suite keeps Concurrency scenarios going; an entry's own cap limits its
// scenarios in flight.

// suiteOf builds a manifest from entries, each already holding its scenarios.
func suiteOf(concurrency int, entries ...SuiteEntry) *Manifest {
	return &Manifest{Concurrency: concurrency, Suite: entries}
}

// entry is a suite entry with n scenarios and a cap of its own.
func entry(url string, n, concurrency int) SuiteEntry {
	e := SuiteEntry{BotURL: url, Concurrency: concurrency}
	for i := range n {
		e.Scenarios = append(e.Scenarios, fmt.Sprintf("s%d", i))
	}
	return e
}

// overlap stands in for the bot and the harness, recording the most scenarios
// held at once per bot and in all, and the order they started in.
type overlap struct {
	mu      sync.Mutex
	active  map[string]int
	peak    map[string]int
	started []string
}

func newOverlap() *overlap {
	return &overlap{active: map[string]int{}, peak: map[string]int{}}
}

func (o *overlap) run(hold time.Duration) func(context.Context, job, Judge) SuiteResult {
	return func(_ context.Context, j job, _ Judge) SuiteResult {
		o.mu.Lock()
		o.active[j.botURL]++
		o.peak[j.botURL] = max(o.peak[j.botURL], o.active[j.botURL])
		total := 0
		for _, n := range o.active {
			total += n
		}
		o.peak["total"] = max(o.peak["total"], total)
		o.started = append(o.started, j.botURL+" "+j.path)
		o.mu.Unlock()

		time.Sleep(hold)

		o.mu.Lock()
		o.active[j.botURL]--
		o.mu.Unlock()
		return SuiteResult{Entry: j.entry, BotURL: j.botURL, Scenario: j.path}
	}
}

// withJobs roots the manifest's scenario paths. The files need not exist: a
// path that will not load is still one job, which is all the stub runs.
func withJobs(m *Manifest) *Manifest {
	m.dir = "."
	return m
}

func runStubbed(t *testing.T, m *Manifest, o *overlap, hold time.Duration) []SuiteResult {
	t.Helper()
	results := withJobs(m).runSuite(t.Context(), nil, o.run(hold))
	for i, r := range results {
		if r.BotURL == "" {
			t.Fatalf("scenario %d was never run", i)
		}
	}
	return results
}

func TestSuiteOneEntryFillsEverySlot(t *testing.T) {
	o := newOverlap()
	results := runStubbed(t, suiteOf(4, entry("a", 10, 0)), o, 50*time.Millisecond)

	if o.peak["a"] != 4 {
		t.Fatalf("peak = %d, want the suite's 4 slots filled", o.peak["a"])
	}
	if len(results) != 10 {
		t.Fatalf("got %d results, want 10", len(results))
	}
}

func TestSuiteAnEntryCapLimitsItsRunsInFlight(t *testing.T) {
	o := newOverlap()
	runStubbed(t, suiteOf(4, entry("capped", 3, 1), entry("plain", 3, 0)), o, 50*time.Millisecond)

	if o.peak["capped"] != 1 {
		t.Fatalf("capped peak = %d, want its cap of 1", o.peak["capped"])
	}
	if o.peak["total"] != 4 {
		t.Fatalf("total peak = %d, want the suite's 4 slots filled", o.peak["total"])
	}
}

func TestSuiteRunsAreTakenInManifestOrder(t *testing.T) {
	o := newOverlap()
	runStubbed(t, suiteOf(1, entry("a", 2, 0), entry("b", 1, 0), entry("c", 2, 0)), o, 0)

	// One at a time, each entry's scenarios run together, in manifest order.
	want := []string{"a s0", "a s1", "b s0", "c s0", "c s1"}
	if !slices.Equal(o.started, want) {
		t.Fatalf("started %v, want %v", o.started, want)
	}
}

func TestSuiteEntryQueues(t *testing.T) {
	t.Run("one queue per entry in manifest order", func(t *testing.T) {
		m := withJobs(suiteOf(0, entry("a", 3, 0), entry("b", 1, 0), entry("c", 1, 0)))
		js := m.jobs()
		var got [][]string
		var caps []int
		for _, q := range m.entryQueues(js) {
			var items []string
			for _, i := range q.items {
				items = append(items, js[i].botURL+" "+js[i].path)
			}
			got = append(got, items)
			caps = append(caps, q.cap)
		}
		want := [][]string{{"a s0", "a s1", "a s2"}, {"b s0"}, {"c s0"}}
		if !slices.EqualFunc(got, want, slices.Equal) {
			t.Fatalf("queues %v, want %v", got, want)
		}
		if !slices.Equal(caps, []int{0, 0, 0}) {
			t.Fatalf("caps %v, want none", caps)
		}
	})

	t.Run("an entry cap is the lowest given and none without one", func(t *testing.T) {
		a1 := SuiteEntry{Name: "a", BotURL: "a", Concurrency: 3, Scenarios: []string{"s1"}}
		a2 := SuiteEntry{Name: "a", BotURL: "a", Concurrency: 2, Scenarios: []string{"s2"}}
		b := SuiteEntry{Name: "b", BotURL: "b", Scenarios: []string{"s1"}}
		m := withJobs(suiteOf(0, a1, a2, b))
		var caps []int
		for _, q := range m.entryQueues(m.jobs()) {
			caps = append(caps, q.cap)
		}
		if !slices.Equal(caps, []int{2, 0}) {
			t.Fatalf("caps %v, want [2 0]", caps)
		}
	})
}
