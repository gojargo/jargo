package eval

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"gopkg.in/yaml.v3"
)

// defaultConcurrency is how many scenarios run at once when a manifest sets none.
const defaultConcurrency = 4

// Manifest validation errors.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errNoSuite      = errors.New("manifest has no suite entries")
	errNoBotURL     = errors.New("suite entry has no bot_url")
	errNoScenarios  = errors.New("suite entry has no scenarios")
	errDuplicateRun = errors.New("two suite entries run one scenario under one label")
)

// Manifest lists the bots to test and the scenarios to run against each. Scenario
// paths are resolved relative to the manifest file, so a manifest is portable.
type Manifest struct {
	// Concurrency is how many scenarios run at once; zero uses a default.
	Concurrency int `yaml:"concurrency"`
	// Suite is the list of bots and their scenarios.
	Suite []SuiteEntry `yaml:"suite"`

	dir string // directory of the manifest file, for resolving scenario paths
}

// SuiteEntry is one bot and the scenarios to play against it.
type SuiteEntry struct {
	// Name labels the entry in the results. Empty uses the bot's URL, which is
	// what tells apart two entries differing only in the scenarios they play.
	Name string `yaml:"name"`
	// BotURL is the bot's RTVI WebSocket endpoint (ws:// or wss://).
	BotURL string `yaml:"bot_url"`
	// Concurrency is how many of this entry's scenarios run at once, under the
	// manifest's own; zero runs them one after another. It is for a bot whose
	// provider rate-limits concurrent connections.
	Concurrency int `yaml:"concurrency"`
	// Scenarios are scenario file paths, resolved relative to the manifest.
	Scenarios []string `yaml:"scenarios"`
}

// label is what the entry is called in the results.
func (e SuiteEntry) label() string {
	if e.Name != "" {
		return e.Name
	}
	return e.BotURL
}

// slots is how many of the entry's scenarios may run at once.
func (e SuiteEntry) slots() int {
	if e.Concurrency > 0 {
		return e.Concurrency
	}
	return 1
}

// SuiteResult is the outcome of running one scenario against one bot.
type SuiteResult struct {
	// Entry is the label of the manifest entry the scenario ran under, which is
	// the entry's name or, having none, its bot's URL.
	Entry string
	// BotURL is the bot the scenario ran against.
	BotURL string
	// Scenario is the scenario file path.
	Scenario string
	// Result is the scenario result; zero-valued when Err is set.
	Result Result
	// Err is non-nil when the scenario could not be loaded or run.
	Err error
}

// Passed reports whether the scenario ran and every expectation was met.
func (r SuiteResult) Passed() bool { return r.Err == nil && r.Result.Passed() }

// LoadManifest reads and validates a suite manifest.
func LoadManifest(path string) (*Manifest, error) {
	data, err := os.ReadFile(path) //nolint:gosec // manifest path is operator-supplied
	if err != nil {
		return nil, fmt.Errorf("eval: read manifest: %w", err)
	}
	var m Manifest
	if err := yaml.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	if err := m.validate(); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	m.dir = filepath.Dir(path)
	return &m, nil
}

// validate checks the manifest is well-formed.
func (m *Manifest) validate() error {
	if len(m.Suite) == 0 {
		return errNoSuite
	}
	seen := make(map[string]bool)
	for i, e := range m.Suite {
		if e.BotURL == "" {
			return fmt.Errorf("suite entry %d: %w", i+1, errNoBotURL)
		}
		if len(e.Scenarios) == 0 {
			return fmt.Errorf("suite entry %d: %w", i+1, errNoScenarios)
		}
		// Two entries running one scenario under one label are two runs nothing
		// can tell apart in the results, which is a manifest mistake rather than
		// something to report twice.
		for _, sc := range e.Scenarios {
			key := e.label() + "\x00" + sc
			if seen[key] {
				return fmt.Errorf("suite entry %d: %w: %s runs %s twice",
					i+1, errDuplicateRun, e.label(), sc)
			}
			seen[key] = true
		}
	}
	return nil
}

// concurrency returns the effective worker count.
func (m *Manifest) concurrency() int {
	if m.Concurrency > 0 {
		return m.Concurrency
	}
	return defaultConcurrency
}

// job is one scenario to run against one bot: the scenario as loaded, or the
// reason the file holding it could not be read.
type job struct {
	entry    string
	botURL   string
	path     string
	scenario *Scenario
	loadErr  error
}

// jobs flattens the manifest into scenario/bot pairs, resolving scenario paths
// and reading each file, since one file may hold several scenarios and each is
// run and reported on its own.
//
// A file that will not load becomes one job carrying the failure, so it is
// reported against that file rather than ending the suite.
func (m *Manifest) jobs() []job {
	var js []job
	for _, e := range m.Suite {
		for _, sc := range e.Scenarios {
			path := sc
			if !filepath.IsAbs(path) {
				path = filepath.Join(m.dir, path)
			}
			file, err := LoadFile(path)
			if err != nil {
				js = append(js, job{entry: e.label(), botURL: e.BotURL, path: path, loadErr: err})
				continue
			}
			for _, scenario := range file.Scenarios {
				js = append(js, job{
					entry: e.label(), botURL: e.BotURL, path: path, scenario: scenario,
				})
			}
		}
	}
	return js
}

// RunSuite runs every scenario in the manifest against its bot and returns one
// result per scenario, in manifest order.
//
// Each entry gets a queue of its own, and its scenarios go one after another on
// the slots it holds, one by default and as many as its Concurrency says. The
// manifest's own Concurrency caps how many run at once across every entry, and
// the queues take those slots in manifest order, so the suite is spread over
// every bot from the start rather than putting every slot on one entry's
// scenarios until they are done. That is what keeps a slow or rate-limited
// provider from holding the whole suite.
//
// newJudge builds the judge for one scenario, and is called once per scenario
// rather than once for the suite: a judge holds the conversation it grades
// against, so scenarios running at the same time cannot share one. It may
// return nil, and may itself be nil, when no scenario uses `judge:`.
func RunSuite(ctx context.Context, m *Manifest, newJudge func() Judge) []SuiteResult {
	js := m.jobs()
	results := make([]SuiteResult, len(js))

	slots := make(chan struct{}, m.concurrency())
	var wg sync.WaitGroup
	for _, q := range m.entryQueues(js) {
		for range q.lanes() {
			wg.Add(1)
			go func(q *entryQueue) {
				defer wg.Done()
				q.drain(ctx, slots, results, newJudge)
			}(q)
		}
	}
	wg.Wait()
	return results
}

// entryQueue is one entry's scenarios, in manifest order, and how many of them
// may run at once.
type entryQueue struct {
	slots int

	mu    sync.Mutex
	next  int
	items []int // indices into the suite's job list
	jobs  []job
}

// lanes is how many of this entry's scenarios to run at once: its slots, or
// fewer when it holds fewer scenarios than that.
func (q *entryQueue) lanes() int {
	return min(q.slots, len(q.items))
}

// take claims the next scenario in the queue, reporting false once it is empty.
func (q *entryQueue) take() (idx int, j job, ok bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.next >= len(q.items) {
		return 0, job{}, false
	}
	i := q.next
	q.next++
	return q.items[i], q.jobs[i], true
}

// drain holds one of the suite's slots and runs this entry's queue on it, one
// scenario after another.
func (q *entryQueue) drain(
	ctx context.Context, slots chan struct{}, results []SuiteResult, newJudge func() Judge,
) {
	slots <- struct{}{}
	defer func() { <-slots }()
	for {
		idx, j, ok := q.take()
		if !ok {
			return
		}
		var judge Judge
		if newJudge != nil {
			judge = newJudge()
		}
		results[idx] = runOne(ctx, j, judge)
	}
}

// entryQueues splits the job list into one queue per entry, in manifest order,
// each carrying the slots that entry may hold.
func (m *Manifest) entryQueues(js []job) []*entryQueue {
	slots := make(map[string]int, len(m.Suite))
	for _, e := range m.Suite {
		// Two entries under one label share their slots, and the lower cap wins.
		if have, ok := slots[e.label()]; !ok || e.slots() < have {
			slots[e.label()] = e.slots()
		}
	}
	var order []string
	queues := make(map[string]*entryQueue, len(m.Suite))
	for i, j := range js {
		q, ok := queues[j.entry]
		if !ok {
			q = &entryQueue{slots: slots[j.entry]}
			queues[j.entry] = q
			order = append(order, j.entry)
		}
		q.items = append(q.items, i)
		q.jobs = append(q.jobs, j)
	}
	out := make([]*entryQueue, 0, len(order))
	for _, label := range order {
		out = append(out, queues[label])
	}
	return out
}

// runOne plays a single scenario against its bot.
func runOne(ctx context.Context, j job, judge Judge) SuiteResult {
	sr := SuiteResult{Entry: j.entry, BotURL: j.botURL, Scenario: j.path}
	if j.loadErr != nil {
		sr.Err = j.loadErr
		return sr
	}
	sr.Result, sr.Err = RunURL(ctx, j.scenario, j.botURL, judge)
	return sr
}
