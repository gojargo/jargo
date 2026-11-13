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
	errConcurrency  = errors.New("suite entry concurrency must be a positive integer")
)

// Manifest lists the bots to test and the scenarios to run against each. Scenario
// paths are resolved relative to the manifest file, so a manifest is portable.
type Manifest struct {
	// Concurrency is how many scenarios run at once; zero uses a default.
	Concurrency int `yaml:"concurrency"`
	// Suite is the list of bots and their scenarios.
	Suite []SuiteEntry `yaml:"suite"`

	dir  string // directory of the manifest file, for resolving scenario paths
	runs []job  // the scenarios to run, read when the manifest is loaded
}

// SuiteEntry is one bot and the scenarios to play against it.
type SuiteEntry struct {
	// Name labels the entry in the results. Empty uses the bot's URL, which is
	// what tells apart two entries differing only in the scenarios they play.
	Name string `yaml:"name"`
	// BotURL is the bot's RTVI WebSocket endpoint (ws:// or wss://).
	BotURL string `yaml:"bot_url"`
	// Concurrency caps how many of this entry's scenarios may be in flight at
	// once, under the manifest's own; zero, which leaving it out gives, sets no
	// cap of its own. It is for a bot whose provider rate-limits concurrent
	// connections. Written out, it must be a positive integer. The cap is the
	// entry's alone: another entry for the same bot is not held to it.
	Concurrency int `yaml:"concurrency"`
	// Scenarios are scenario file paths, resolved relative to the manifest.
	Scenarios []string `yaml:"scenarios"`
}

// UnmarshalYAML decodes a suite entry, refusing a `concurrency:` that is not a
// positive integer rather than reading it as no cap.
func (e *SuiteEntry) UnmarshalYAML(node *yaml.Node) error {
	if v := mappingValue(node, "concurrency"); v != nil {
		var n int
		if v.Kind != yaml.ScalarNode || v.Tag != "!!int" || v.Decode(&n) != nil || n < 1 {
			return fmt.Errorf("%w, got %q", errConcurrency, v.Value)
		}
	}
	type raw SuiteEntry // a distinct type, so decoding does not recurse here
	var r raw
	if err := node.Decode(&r); err != nil {
		return err
	}
	*e = SuiteEntry(r)
	return nil
}

// label is what the entry is called in the results.
func (e SuiteEntry) label() string {
	if e.Name != "" {
		return e.Name
	}
	return e.BotURL
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
	// Result is the scenario result; zero-valued when Err is set. A bot that
	// never became ready, or a harness that failed mid-run, is a failure in it.
	Result Result
	// Err is non-nil when the scenario could not be loaded, or the run's
	// context ended.
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
	m.runs = m.jobs()
	if err := checkDuplicateRuns(m.runs); err != nil {
		return nil, fmt.Errorf("eval: %s: %w", path, err)
	}
	return &m, nil
}

// checkDuplicateRuns refuses two runs of one scenario under one label. They are
// two runs nothing can tell apart in the results, which is a manifest mistake
// rather than something to report twice. Runs are compared by the scenario's
// name once the files are read, so two paths to one file collide.
func checkDuplicateRuns(js []job) error {
	seen := make(map[[2]string]bool, len(js))
	for _, j := range js {
		key := [2]string{j.entry, j.name()}
		if seen[key] {
			return fmt.Errorf("%w: %q runs %q twice; give one of the entries a name", errDuplicateRun, j.entry, j.name())
		}
		seen[key] = true
	}
	return nil
}

// validate checks the manifest is well-formed.
func (m *Manifest) validate() error {
	if len(m.Suite) == 0 {
		return errNoSuite
	}
	for i, e := range m.Suite {
		if e.BotURL == "" {
			return fmt.Errorf("suite entry %d: %w", i+1, errNoBotURL)
		}
		if len(e.Scenarios) == 0 {
			return fmt.Errorf("suite entry %d: %w", i+1, errNoScenarios)
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
	entryIdx int // the entry's place in the manifest, which its cap counts against
	botURL   string
	path     string
	scenario *Scenario
	loadErr  error
}

// name is what the run is called: its scenario's name, or, for a file that
// would not load, its path.
func (j job) name() string {
	if j.scenario != nil {
		return j.scenario.Name
	}
	return j.path
}

// jobs flattens the manifest into scenario/bot pairs, resolving scenario paths
// and reading each file, since one file may hold several scenarios and each is
// run and reported on its own.
//
// A file that will not load becomes one job carrying the failure, so it is
// reported against that file rather than ending the suite.
func (m *Manifest) jobs() []job {
	var js []job
	for i, e := range m.Suite {
		for _, sc := range e.Scenarios {
			path := sc
			if !filepath.IsAbs(path) {
				path = filepath.Join(m.dir, path)
			}
			file, err := LoadFile(path)
			if err != nil {
				js = append(js, job{entry: e.label(), entryIdx: i, botURL: e.BotURL, path: path, loadErr: err})
				continue
			}
			for _, scenario := range file.Scenarios {
				js = append(js, job{
					entry: e.label(), entryIdx: i, botURL: e.BotURL, path: path, scenario: scenario,
				})
			}
		}
	}
	return js
}

// RunSuite runs every scenario in the manifest against its bot and returns one
// result per scenario, in manifest order.
//
// The manifest's Concurrency is how many scenarios run at once. The suite keeps
// that many going, taking the next scenario from the first entry in manifest
// order that still has one, so an entry's scenarios finish together and no slot
// waits while any entry still has scenarios. An entry whose provider rate-limits
// sets its own Concurrency, and never has more than that many scenarios in
// flight; a worker that finds it full takes the next entry's scenario.
//
// newJudge builds the judge for one scenario, and is called once per scenario
// rather than once for the suite: a judge holds the conversation it grades
// against, so scenarios running at the same time cannot share one. It may
// return nil, and may itself be nil, when no scenario uses `judge:`.
func RunSuite(ctx context.Context, m *Manifest, newJudge func() Judge) []SuiteResult {
	return m.runSuite(ctx, newJudge, runOne)
}

// runSuite is RunSuite with the scenario runner given, so the scheduling can be
// tested without a bot behind it.
func (m *Manifest) runSuite(
	ctx context.Context, newJudge func() Judge, run func(context.Context, job, Judge) SuiteResult,
) []SuiteResult {
	js := m.runs
	if js == nil {
		js = m.jobs()
	}
	results := make([]SuiteResult, len(js))

	queue := newRunQueue(m.entryQueues(js))
	var wg sync.WaitGroup
	for range min(m.concurrency(), len(js)) {
		wg.Go(func() {
			// Take scenarios from the queue and run them, one after another,
			// until none is left.
			for {
				idx, entry, ok := queue.take()
				if !ok {
					return
				}
				var judge Judge
				if newJudge != nil {
					judge = newJudge()
				}
				results[idx] = run(ctx, js[idx], judge)
				queue.done(entry)
			}
		})
	}
	wg.Wait()
	return results
}

// entryQueue is one entry's scenarios, in manifest order, with the entry's cap
// on scenarios in flight.
type entryQueue struct {
	// entry is the entry's place in the manifest.
	entry int
	// cap is the most of the entry's scenarios in flight at once; zero is no cap.
	cap   int
	items []int // indices into the suite's job list
}

// runQueue is the suite's scenarios, handed out to workers in manifest order.
//
// Each queue is one entry's scenarios, in the order the entries were given. A
// worker takes the next scenario from the first queue whose entry is under its
// cap; when every queue with scenarios left is at its cap, it waits for a
// scenario to finish.
type runQueue struct {
	mu       sync.Mutex
	changed  *sync.Cond
	queues   []*entryQueue
	inFlight map[int]int
}

func newRunQueue(queues []*entryQueue) *runQueue {
	q := &runQueue{queues: queues, inFlight: make(map[int]int, len(queues))}
	q.changed = sync.NewCond(&q.mu)
	return q
}

// take claims the next scenario to run, and the entry it counts against,
// reporting false once none is left.
func (q *runQueue) take() (idx, entry int, ok bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	for q.anyLeft() {
		if idx, entry, ok := q.pick(); ok {
			return idx, entry, true
		}
		q.changed.Wait()
	}
	return 0, 0, false
}

// done counts a scenario of entry as finished, so the entry may take another.
func (q *runQueue) done(entry int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.inFlight[entry]--
	q.changed.Broadcast()
}

// anyLeft reports whether any queue still has a scenario to hand out.
func (q *runQueue) anyLeft() bool {
	for _, eq := range q.queues {
		if len(eq.items) > 0 {
			return true
		}
	}
	return false
}

// pick takes the first queued scenario whose entry is under its cap.
func (q *runQueue) pick() (idx, entry int, ok bool) {
	for _, eq := range q.queues {
		if len(eq.items) > 0 && (eq.cap == 0 || q.inFlight[eq.entry] < eq.cap) {
			q.inFlight[eq.entry]++
			idx := eq.items[0]
			eq.items = eq.items[1:]
			return idx, eq.entry, true
		}
	}
	return 0, 0, false
}

// entryQueues splits the job list into one queue per manifest entry, in
// manifest order, each with the entry's cap. An entry's cap is its own
// Concurrency, and none when it sets none; another entry for the same bot, or
// under the same label, has a queue and a cap of its own.
func (m *Manifest) entryQueues(js []job) []*entryQueue {
	var out []*entryQueue
	queues := make(map[int]*entryQueue, len(m.Suite))
	for i, j := range js {
		q, ok := queues[j.entryIdx]
		if !ok {
			q = &entryQueue{entry: j.entryIdx, cap: max(m.Suite[j.entryIdx].Concurrency, 0)}
			queues[j.entryIdx] = q
			out = append(out, q)
		}
		q.items = append(q.items, i)
	}
	return out
}

// runOne plays a single scenario against its bot.
func runOne(ctx context.Context, j job, judge Judge) SuiteResult {
	sr := SuiteResult{Entry: j.entry, BotURL: j.botURL, Scenario: j.path}
	if j.loadErr != nil {
		// A run that is skipped still closes its judge.
		closeJudge(ctx, judge)
		sr.Err = j.loadErr
		return sr
	}
	sr.Result, sr.Err = RunURL(ctx, j.scenario, j.botURL, Options{Judge: judge})
	return sr
}
