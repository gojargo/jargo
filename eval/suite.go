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
	errNoSuite     = errors.New("manifest has no suite entries")
	errNoBotURL    = errors.New("suite entry has no bot_url")
	errNoScenarios = errors.New("suite entry has no scenarios")
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
	// BotURL is the bot's RTVI WebSocket endpoint (ws:// or wss://).
	BotURL string `yaml:"bot_url"`
	// Scenarios are scenario file paths, resolved relative to the manifest.
	Scenarios []string `yaml:"scenarios"`
}

// SuiteResult is the outcome of running one scenario against one bot.
type SuiteResult struct {
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

// job is one scenario to run against one bot.
type job struct {
	botURL   string
	scenario string
}

// jobs flattens the manifest into scenario/bot pairs, resolving scenario paths.
func (m *Manifest) jobs() []job {
	var js []job
	for _, e := range m.Suite {
		for _, sc := range e.Scenarios {
			path := sc
			if !filepath.IsAbs(path) {
				path = filepath.Join(m.dir, path)
			}
			js = append(js, job{botURL: e.BotURL, scenario: path})
		}
	}
	return js
}

// RunSuite runs every scenario in the manifest against its bot, up to
// Concurrency at once, and returns one result per scenario in manifest order.
// judge may be nil when no scenario uses `judge:`.
func RunSuite(ctx context.Context, m *Manifest, judge Judge) []SuiteResult {
	js := m.jobs()
	results := make([]SuiteResult, len(js))

	sem := make(chan struct{}, m.concurrency())
	var wg sync.WaitGroup
	for i, j := range js {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, j job) {
			defer wg.Done()
			defer func() { <-sem }()
			results[i] = runOne(ctx, j, judge)
		}(i, j)
	}
	wg.Wait()
	return results
}

// runOne loads and plays a single scenario against its bot.
func runOne(ctx context.Context, j job, judge Judge) SuiteResult {
	sr := SuiteResult{BotURL: j.botURL, Scenario: j.scenario}
	scenario, err := Load(j.scenario)
	if err != nil {
		sr.Err = err
		return sr
	}
	res, err := RunURL(ctx, scenario, j.botURL, judge)
	sr.Result = res
	sr.Err = err
	return sr
}
