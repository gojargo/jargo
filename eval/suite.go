package eval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gojargo/jargo/event"
	"gopkg.in/yaml.v3"
)

// The multi-bot eval suite.
//
// A Manifest lists bots to spawn and the scenarios to run against each,
// scripted and simulated alike. A Suite builds each bot once, runs it on its
// own port, and drives it with the harness, several at a time. `jargo eval
// suite` is the command in front of it.
//
// Manifest format (YAML):
//
//	concurrency: 4
//	repeat: 1                     # run each (bot, scenario) N times
//	runs_dir: test-runs           # logs + recordings go to <runs_dir>/<timestamp>/
//	record: false                 # record conversation audio
//	cache_dir: null               # optional
//	scenarios_dir: scenarios      # resolved relative to this manifest file
//	bots_dir: .                   # bot paths are relative to it
//	# {bot}=the bot's built binary, {port}=assigned per run by the suite
//	spawn: "{bot} --port {port}"
//	suite:
//	  - bot: examples/voice/voicebot
//	    scenarios: [simple_math, greeting]
//	  - bot: examples/voice/voicebot-groq
//	    concurrency: 2                           # this bot's own cap
//	    scenarios: [simple_math, interruption]
//	  - bot: examples/vision/vision
//	    runner_body:
//	      path: scenarios/vision-cat.yaml        # passed to the bot as --runner-body
//	    scenarios: [vision_describe]
//	  - bot: examples/turns/filter-incomplete-turns
//	    name: openai/gpt-4o-mini                 # this entry's label, the bot path by default
//	    runner_body:
//	      data: {model: gpt-4o-mini}             # written to a file for the bot
//	    scenarios: [turn_completion]
//	  - bot: examples/flows/restaurant
//	    scenarios: [book_table]                  # a simulation: its file has a persona
//
// A bot is a Go main package directory. The suite builds each distinct bot
// once, with `go build` and the caller's GOFLAGS, into the run's bin/ folder,
// and spawns the binary with the spawn template. The bot serves the harness the
// way the development runner does: `--port` picks its port, and the harness
// connects to /ws on it.
//
// A `scenarios:` entry names a scenario file, and the file contributes one run
// per scenario it holds, scripted or a simulation as each says (see
// ScenarioFile), named `<file name>/<scenario name>`. A name resolves under
// scenarios_dir with .yaml added and may carry a folder, as
// `scripted/greeting`; a name ending in .yaml is a path relative to the
// manifest instead. A simulation runs as many times as its `runs:` says, and
// every run must pass.
//
// An optional `runner_body:` supplies runner-args data the bot would normally
// receive in a /start request body (e.g. a vision bot's image path), passed to
// it as --runner-body. It holds either `path:`, a YAML or JSON file resolved
// relative to the manifest, or `data:`, the body itself as a mapping, which the
// suite writes to a file among the run's logs. A bot given a file runs with the
// file's directory as its working directory, so relative paths inside the body
// (like an image) resolve next to the file; a body that holds such paths
// belongs in a file for that reason. A bare `runner_body: <file>` is the
// deprecated form of `runner_body: {path: <file>}`.
//
// An entry's `name:` is its label: what the display, the -p filter, the results
// records and the artifact file names use. It defaults to the `bot:` path, so
// it is only needed when several entries share a bot, as when sweeping a model
// with `runner_body:`; two entries may not run the same scenario under the same
// label.
//
// An entry may give `bot_url:` instead of `bot:`: a bot already running, which
// the suite connects to rather than builds and spawns. Its label defaults to
// the URL.
//
// Concurrency is how many runs execute at once. The suite keeps that many
// going, taking the next run from the first entry in manifest order that still
// has one, so an entry's scenarios finish together and no slot waits while any
// entry still has runs. An entry whose provider rate-limits sets its own
// `concurrency:`, and never has more than that many runs in flight. Two entries
// for the same bot stay separate: each keeps its place in the manifest order
// and its own cap.
//
// Manifest-relative paths (bot/bots_dir, scenarios_dir, runs_dir) resolve
// relative to the manifest file, so a manifest is portable; the same values
// passed as overrides resolve against the working directory.
//
// Repeat runs every (bot, scenario) pair N times, which is how a flaky behavior
// gets measured rather than sampled: a bot that passes a scenario half the time
// looks identical to a reliable one in a single pass. Attempts run
// attempt-major, every entry's first attempt before any entry's second, and
// carry a SuiteRun.Attempt number that joins their artifact file names, so no
// attempt overwrites another's logs.

const (
	// DefaultBasePort is the first port a suite assigns to its runs.
	DefaultBasePort = 7900
	// DefaultConcurrency is how many runs a suite executes at once when its
	// manifest says nothing.
	DefaultConcurrency = 4
	// BotConnectTimeout is how long to wait for a freshly spawned bot to start
	// listening. The harness retries the connect, so this doubles as readiness
	// waiting.
	BotConnectTimeout = 60 * time.Second
	// BotStopTimeout is how long to wait for a bot to exit after it is asked to
	// terminate before killing it.
	BotStopTimeout = 10 * time.Second
	// WorkerSafetyTimeout is the safety net for a hung harness. The harness's
	// own per-expectation timeouts bound a healthy run far below this; the cap
	// only catches a run that wedges, so it cannot hold a concurrency slot
	// forever.
	WorkerSafetyTimeout = 600 * time.Second
	// DefaultSpawn is the default spawn template; {bot} and {port} are
	// substituted per run.
	DefaultSpawn = "{bot} --port {port}"
	// EventUpdate fires with a *SuiteRun whenever that run changes status:
	//
	//	event.On(&suite.Registry, eval.EventUpdate, func(ctx context.Context, run *eval.SuiteRun) { … })
	//
	// Runs are mutated in place, so handlers run synchronously, on the
	// goroutine running that run, and must return promptly; they see the run
	// as the change left it. Several runs go at once, so a handler may be
	// called from several goroutines at the same time.
	EventUpdate = "on_update"
)

// The statuses of a SuiteRun.
const (
	// RunPending is a run not started yet.
	RunPending = "pending"
	// RunRunning is a run in progress.
	RunRunning = "running"
	// RunDone is a run that finished, with a result or an error.
	RunDone = "done"
)

// The harness's log categories: each tags the logs of one of the harness's
// own pipelines, independent of which service it uses. Anything untagged
// (RTVI, connection, harness internals) falls through to PipelineHarness.
const (
	// PipelineSpeech is the user's speech.
	PipelineSpeech = "speech"
	// PipelineTranscription is the transcription of the bot's speech.
	PipelineTranscription = "transcription"
	// PipelineJudge is the judge.
	PipelineJudge = "judge"
	// PipelinePersona is a simulation's persona LLM.
	PipelinePersona = "persona"
	// PipelineHarness is everything else.
	PipelineHarness = "harness"
)

// Settings and suite errors.
var (
	// errInvalidManifest is a manifest the suite cannot run.
	errInvalidManifest = errors.New("eval: invalid manifest")
	// errSpawnTemplate is a spawn template that cannot be expanded.
	errSpawnTemplate = errors.New("eval: spawn template")
	// errNothingToRun is a run whose scenario is a rubric.
	errNothingToRun = errors.New("which has nothing to run")
	// errHarnessTimeout is a run that outlived WorkerSafetyTimeout.
	errHarnessTimeout = errors.New("harness worker timed out")
)

// ScenarioSuffixes are what a scenario file may be named, wherever one is
// looked for.
//
//nolint:gochecknoglobals // fixed set, read only
var ScenarioSuffixes = []string{".yaml", ".yml"}

// PipelineLogCategories are the log categories, in the order a run's debug log
// lists them.
//
//nolint:gochecknoglobals // fixed set, read only
var PipelineLogCategories = []string{
	PipelineSpeech, PipelineTranscription, PipelineJudge, PipelinePersona, PipelineHarness,
}

// PipelineLogLabels are the human headings of each category's section in a
// run's debug log.
//
//nolint:gochecknoglobals // fixed set, read only
var PipelineLogLabels = map[string]string{
	PipelineSpeech:        "user speech logs",
	PipelineTranscription: "bot speech transcription logs",
	PipelineJudge:         "judge logs",
	PipelinePersona:       "persona LLM logs",
	PipelineHarness:       "harness logs",
}

// SuiteRun is the mutable per-(bot, scenario) state, updated in place so a live
// display can read it.
type SuiteRun struct {
	// Bot is the manifest's `bot:` path, or the bot URL when the run connects
	// to a bot already running.
	Bot string
	// Name is the manifest entry's `name:`, or "" when it has none; Label is
	// what the display, the filters and the results use.
	Name string
	// Scenario is the display name: the scenario or simulation, without .yaml.
	Scenario string
	// ScenarioPath is the scenario or simulation file.
	ScenarioPath string
	// Loaded is the scenario, when the run was built from a loaded file; Load
	// returns it, or reads the file when it is nil.
	Loaded LoadedScenario
	// BotPath is the bot to build and spawn: a Go main package directory.
	// Empty when connecting to BotURL.
	BotPath string
	// BotURL is where to connect instead of spawning, or "".
	BotURL string
	// RunnerBodyPath is the optional --runner-body file for the bot's runner
	// args.
	RunnerBodyPath string
	// RunnerBody is the bot's runner-args body given inline, written to a file
	// for the bot when it is spawned; nil when there is none or it comes from
	// RunnerBodyPath.
	RunnerBody map[string]any
	// Concurrency is how many runs of this run's entry may be in flight at
	// once, when its manifest entry says; zero is as many as the suite runs.
	Concurrency int
	// Entry is the position of the run's entry in the manifest's suite: list.
	// The suite takes runs in entry order and applies each entry's Concurrency
	// to that entry's runs only.
	Entry int
	// Kind is KindScript, played by a ScriptSession, or KindSimulation; ""
	// is KindScript.
	Kind Kind
	// Attempts is how many times this (bot, scenario) pair runs: the
	// manifest's repeat, or a simulation's own runs; zero is one. Above one,
	// each attempt's artifacts carry its number.
	Attempts int
	// Sweep is whether the attempts come from a repeat (a measurement: the
	// suite reports a rate and a failure is data) rather than from a
	// simulation's runs (a requirement: every attempt must pass).
	Sweep bool
	// Attempt is the 1-based attempt number when the suite repeats; zero is
	// one.
	Attempt int
	// Status is RunPending (or ""), RunRunning or RunDone.
	Status string
	// Stopping is true while the bot is being stopped after the run is done.
	// The run still counts as in flight, so the next one waits.
	Stopping bool
	// ScriptResult is a scripted run's outcome, once the run is done.
	ScriptResult *ScriptResult
	// SimulationResult is a simulation run's outcome, once the run is done.
	SimulationResult *SimulationResult
	// Error is the build, spawn or connection error message, if the run failed
	// before producing a result.
	Error string
	// StartedAt is when the run started, for a live elapsed counter; zero
	// before it starts.
	StartedAt time.Time
	// DurationMS is the wall-clock time the run took, in milliseconds; nil
	// before it is done.
	DurationMS *int
}

// Label is the run's label: the entry's Name, or its Bot when it has none.
func (r *SuiteRun) Label() string {
	if r.Name != "" {
		return r.Name
	}
	return r.Bot
}

// Stem is the scenario name as a file name stem: a group entry's / becomes __.
func (r *SuiteRun) Stem() string { return strings.ReplaceAll(r.Scenario, "/", "__") }

// RunKind is the run's kind, KindScript when Kind is unset.
func (r *SuiteRun) RunKind() Kind {
	if r.Kind == "" {
		return KindScript
	}
	return r.Kind
}

// Load is the run's scenario: as loaded when the run was built, else read from
// its file.
//
// It fails when the file is missing or invalid, holds no scenario of this
// name, or the scenario is a rubric, which has nothing to run.
func (r *SuiteRun) Load() (LoadedScenario, error) {
	if r.Loaded != nil {
		return r.Loaded, nil
	}
	file, err := LoadFile(r.ScenarioPath)
	if err != nil {
		return nil, err
	}
	found, err := file.Scenario(r.Scenario)
	if err != nil {
		return nil, err
	}
	if _, rubric := found.(*Rubric); rubric {
		return nil, fmt.Errorf("%s is a rubric, %w", r.Scenario, errNothingToRun)
	}
	r.Loaded = found
	return found, nil
}

// debugLog is the result's decision trace, or nil without a result.
func (r *SuiteRun) debugLog() []string {
	switch {
	case r.ScriptResult != nil:
		return r.ScriptResult.DebugLog
	case r.SimulationResult != nil:
		return r.SimulationResult.DebugLog
	}
	return nil
}

// attempts is Attempts, zero counting as one.
func (r *SuiteRun) attempts() int { return max(r.Attempts, 1) }

// attempt is Attempt, zero counting as one.
func (r *SuiteRun) attempt() int { return max(r.Attempt, 1) }

// runnerBody is a manifest entry's runner_body:: a file to pass to the bot, or
// data to write to one.
type runnerBody struct {
	path string
	data map[string]any
}

// Manifest is a parsed eval-suite manifest.
type Manifest struct {
	// Runs are the (bot, scenario) and (bot, simulation) runs to execute.
	Runs []*SuiteRun
	// Spawn is the spawn command template ({bot} and {port} substituted).
	Spawn string
	// Concurrency is how many runs execute at once.
	Concurrency int
	// Repeat is how many times to run each (bot, scenario) pair. Attempts run
	// attempt-major rather than grouped per bot, so every bot meets the same
	// machine conditions in the same stretch of the sweep and a transient
	// slowdown shows up as a band across all of them instead of a regression in
	// whichever bot happened to be running.
	Repeat int
	// BasePort is the first port to assign; each run gets BasePort + index, so
	// the reserved range widens with Repeat (bots x scenarios x repeat ports
	// from BasePort up).
	BasePort int
	// RunsDir is the base for run output (a <name>/ subdir is added), or "".
	RunsDir string
	// Record is whether to record conversation audio.
	Record bool
	// CacheDir is the directory for cached synthesized user audio, or "".
	CacheDir string
}

// ManifestOverrides are the values the command line sets over a manifest's
// own. A zero field (an empty string, a nil pointer) leaves the manifest's
// value; paths resolve against the working directory rather than the
// manifest's.
type ManifestOverrides struct {
	// BotsDir is the directory bot paths are relative to.
	BotsDir string
	// ScenariosDir is the directory bare scenario names resolve under.
	ScenariosDir string
	// RunsDir is the base for run output.
	RunsDir string
	// Spawn is the spawn command template.
	Spawn string
	// Concurrency is how many runs execute at once.
	Concurrency *int
	// Repeat is how many times each (bot, scenario) pair runs. Set here or in
	// the manifest, it also replaces each simulation's own runs, a repeat of
	// one included.
	Repeat *int
	// BasePort is the first port assigned.
	BasePort *int
	// Record is whether to record conversation audio.
	Record *bool
	// CacheDir is the synthesized-audio cache directory.
	CacheDir string
}

// manifestSettings are a manifest's settings once the overrides are applied.
type manifestSettings struct {
	botsDir      string
	scenariosDir string
	runsDir      string
	spawn        string
	concurrency  int
	repeat       int
	repeatGiven  bool
	basePort     int
	record       bool
	cacheDir     string
}

// LoadManifest parses a manifest YAML into a Manifest.
//
// A set override wins over the manifest's value, so the command line wins.
// Manifest paths resolve against the manifest's directory, overrides against
// the working directory.
func LoadManifest(path string, overrides ManifestOverrides) (*Manifest, error) {
	path, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	data, err := loadManifestData(path)
	if err != nil {
		return nil, err
	}
	base := filepath.Dir(path)
	settings, err := manifestSettingsOf(data, base, path, overrides)
	if err != nil {
		return nil, err
	}
	runs, err := manifestRuns(data, base, path, settings)
	if err != nil {
		return nil, err
	}
	return &Manifest{
		Runs:        runs,
		Spawn:       settings.spawn,
		Concurrency: settings.concurrency,
		Repeat:      settings.repeat,
		BasePort:    settings.basePort,
		RunsDir:     settings.runsDir,
		Record:      settings.record,
		CacheDir:    settings.cacheDir,
	}, nil
}

// loadManifestData reads a manifest's top-level mapping; an empty file is an
// empty mapping.
func loadManifestData(path string) (map[string]any, error) {
	raw, err := os.ReadFile(path) //nolint:gosec // manifest path is operator-supplied
	if err != nil {
		return nil, fmt.Errorf("eval: read manifest: %w", err)
	}
	var doc yaml.Node
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("eval: parse %s: %w", path, err)
	}
	switch v := nodeValue(&doc).(type) {
	case nil:
		return map[string]any{}, nil
	case map[string]any:
		return v, nil
	default:
		return nil, fmt.Errorf("%w: %s: top level must be a mapping", errInvalidManifest, path)
	}
}

// manifestSettingsOf is the manifest's settings with the overrides applied: an
// override wins, and its paths resolve against the working directory rather
// than the manifest's.
func manifestSettingsOf(data map[string]any, base, path string, o ManifestOverrides) (manifestSettings, error) {
	var (
		s   manifestSettings
		err error
	)
	if s.botsDir, err = dirValue(data, o.BotsDir, "bots_dir", ".", base); err != nil {
		return s, err
	}
	if s.scenariosDir, err = dirValue(data, o.ScenariosDir, "scenarios_dir", "scenarios", base); err != nil {
		return s, err
	}
	switch runsDir, _ := data["runs_dir"].(string); {
	case o.RunsDir != "":
		s.runsDir, err = filepath.Abs(o.RunsDir)
	case runsDir != "":
		s.runsDir, err = absAgainst(base, runsDir)
	}
	if err != nil {
		return s, err
	}
	// A repeat set anywhere, the command line or the manifest, decides every
	// run's attempts, a simulation's included, even when it is one; absent, a
	// simulation runs as many times as its file says.
	_, inManifest := data["repeat"]
	s.repeatGiven = o.Repeat != nil || inManifest
	if s.repeat, err = intSetting(data, o.Repeat, "repeat", 1, path); err != nil {
		return s, err
	}
	if s.repeat < 1 {
		return s, fmt.Errorf("%w: %s: 'repeat' must be at least 1", errInvalidManifest, path)
	}
	if s.spawn, err = spawnSetting(data, o.Spawn, path); err != nil {
		return s, err
	}
	if s.concurrency, err = intSetting(data, o.Concurrency, "concurrency", DefaultConcurrency, path); err != nil {
		return s, err
	}
	if s.basePort, err = intSetting(data, o.BasePort, "base_port", DefaultBasePort, path); err != nil {
		return s, err
	}
	if s.record, err = recordSetting(data, o.Record, path); err != nil {
		return s, err
	}
	s.cacheDir = o.CacheDir
	if s.cacheDir == "" {
		s.cacheDir, _ = data["cache_dir"].(string)
	}
	return s, nil
}

// dirValue is a directory setting: the override against the working
// directory, else the manifest's value, or def, against base.
func dirValue(data map[string]any, override, key, def, base string) (string, error) {
	if override != "" {
		return filepath.Abs(override)
	}
	value := def
	if v, ok := data[key]; ok && v != nil {
		value = pyStr(v)
	}
	return absAgainst(base, value)
}

// absAgainst is p made absolute against base, unless it is absolute
// already.
func absAgainst(base, p string) (string, error) {
	if filepath.IsAbs(p) {
		return filepath.Clean(p), nil
	}
	return filepath.Abs(filepath.Join(base, p))
}

// intSetting is an integer setting: the override, else the manifest's value,
// else def.
func intSetting(data map[string]any, override *int, key string, def int, path string) (int, error) {
	if override != nil {
		return *override, nil
	}
	v, ok := data[key]
	if !ok || v == nil {
		return def, nil
	}
	n, ok := v.(int)
	if !ok {
		return 0, fmt.Errorf("%w: %s: '%s' must be an integer, got %s", errInvalidManifest, path, key, pyRepr(v))
	}
	return n, nil
}

// recordSetting is the record setting: the override, else the manifest's,
// else off.
func recordSetting(data map[string]any, override *bool, path string) (bool, error) {
	if override != nil {
		return *override, nil
	}
	v, ok := data["record"]
	if !ok || v == nil {
		return false, nil
	}
	b, ok := v.(bool)
	if !ok {
		return false, fmt.Errorf("%w: %s: 'record' must be true or false, got %s", errInvalidManifest, path, pyRepr(v))
	}
	return b, nil
}

// spawnSetting is the spawn template, checked so a template that cannot be
// expanded is refused when the manifest loads rather than when its first bot
// starts.
func spawnSetting(data map[string]any, override, path string) (string, error) {
	if _, set := data["python"]; set {
		return "", fmt.Errorf("%w: %s: 'python:' does not apply: bots are Go packages the suite builds, "+
			"and {bot} is the built binary", errInvalidManifest, path)
	}
	spawn := override
	if spawn == "" {
		spawn = DefaultSpawn
		if v, ok := data["spawn"]; ok && v != nil {
			spawn = pyStr(v)
		}
	}
	if _, err := spawnArgv(spawn, "bot", 0, ""); err != nil {
		return "", fmt.Errorf("%s: %w", path, err)
	}
	return spawn, nil
}

// manifestRuns are the runs the suite: list describes, one per bot, scenario,
// and attempt.
//
// The scenario file says which kind it is. A simulation runs as many times as
// its file says unless a repeat makes the suite a measurement; a file that
// fails to load still gets its run, which reports the error. Attempts are
// attempt-major (bot A #1, bot B #1, ..., bot A #2), so a sweep runs each
// attempt across the bots without fast ones waiting on slow ones.
func manifestRuns(data map[string]any, base, path string, settings manifestSettings) ([]*SuiteRun, error) {
	var items []any
	switch v := data["suite"].(type) {
	case nil:
	case []any:
		items = v
	default:
		return nil, fmt.Errorf("%w: %s: 'suite:' must be a list", errInvalidManifest, path)
	}
	var runs []*SuiteRun
	for entry, raw := range items {
		item, ok := raw.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("%w: %s: suite entry %d must be a mapping", errInvalidManifest, path, entry)
		}
		entryRuns, err := entryRunsOf(item, entry, base, path, settings)
		if err != nil {
			return nil, err
		}
		runs = append(runs, entryRuns...)
	}
	if err := checkDuplicateRuns(runs, path); err != nil {
		return nil, err
	}
	return attemptMajor(runs), nil
}

// entryRunsOf are one suite entry's runs, an attempt of each.
func entryRunsOf(item map[string]any, entry int, base, path string, settings manifestSettings) ([]*SuiteRun, error) {
	template, err := entryTemplate(item, entry, base, path, settings)
	if err != nil {
		return nil, err
	}
	scenarios, _ := item["scenarios"].([]any)
	var runs []*SuiteRun
	for _, scenario := range scenarios {
		fileName, scenarioPath, err := resolveScenario(pyStr(scenario), base, settings.scenariosDir)
		if err != nil {
			return nil, err
		}
		// A file that fails to load still gets a run, under the manifest's
		// name for it, so the failure is reported.
		named := []LoadedScenario{nil}
		if file, err := LoadFile(scenarioPath); err == nil {
			named = file.Scenarios
		}
		for _, one := range named {
			run := *template
			run.Scenario, run.ScenarioPath, run.Loaded = fileName, scenarioPath, one
			run.Kind, run.Attempts = KindScript, settings.repeat
			switch s := one.(type) {
			case *Rubric:
				// A rubric judges a call that already happened; there is
				// nothing to run.
				continue
			case *SimulationScenario:
				run.Kind = KindSimulation
				if !settings.repeatGiven {
					run.Attempts = s.Runs
				}
			}
			if one != nil {
				run.Scenario = one.ScenarioName()
			}
			runs = append(runs, &run)
		}
	}
	return runs, nil
}

// entryTemplate is what every run of a suite entry shares: its bot, its label,
// its body and its cap.
func entryTemplate(item map[string]any, entry int, base, path string, settings manifestSettings) (*SuiteRun, error) {
	run := &SuiteRun{Entry: entry, Sweep: settings.repeatGiven}
	if err := entryBot(run, item, path, settings.botsDir); err != nil {
		return nil, err
	}
	where := fmt.Sprintf("%s: bot %s", path, pyRepr(run.Bot))
	if name, set := item["name"]; set && name != nil {
		s, ok := name.(string)
		if !ok || strings.TrimSpace(s) == "" {
			return nil, fmt.Errorf("%w: %s 'name:' must be a non-empty string", errInvalidManifest, where)
		}
		run.Name = s
	}
	body, err := parseRunnerBody(item["runner_body"], base, where)
	if err != nil {
		return nil, err
	}
	if run.BotURL != "" && (body.path != "" || body.data != nil) {
		return nil, fmt.Errorf("%w: %s 'runner_body:' needs a bot the suite spawns, not a 'bot_url:'",
			errInvalidManifest, where)
	}
	run.RunnerBodyPath, run.RunnerBody = body.path, body.data
	if c, set := item["concurrency"]; set && c != nil {
		n, ok := c.(int)
		if !ok || n < 1 {
			return nil, fmt.Errorf("%w: %s 'concurrency:' must be a positive integer", errInvalidManifest, where)
		}
		run.Concurrency = n
	}
	return run, nil
}

// entryBot sets the run's bot from a suite entry: the package to build and
// spawn (bot:, against botsDir), or the bot already running (bot_url:).
func entryBot(run *SuiteRun, item map[string]any, path, botsDir string) error {
	botURL, _ := item["bot_url"].(string)
	bot := item["bot"]
	switch {
	case bot != nil && botURL != "":
		return fmt.Errorf("%w: %s: suite entry %d gives both 'bot:' and 'bot_url:'", errInvalidManifest, path, run.Entry)
	case botURL != "":
		run.Bot, run.BotURL = botURL, botURL
		return nil
	case bot != nil:
		run.Bot = pyStr(bot)
		botPath, err := absAgainst(botsDir, run.Bot)
		run.BotPath = botPath
		return err
	default:
		return fmt.Errorf("%w: %s: suite entry %d has no 'bot:'", errInvalidManifest, path, run.Entry)
	}
}

// parseRunnerBody is an entry's runner_body: as the bot gets it: a file to
// pass, or data the suite writes to one.
func parseRunnerBody(spec any, base, where string) (runnerBody, error) {
	if spec == nil {
		return runnerBody{}, nil
	}
	if s, ok := spec.(string); ok {
		slog.Warn(fmt.Sprintf("eval: `runner_body: <file>` is deprecated and will be removed. "+
			"Use `runner_body: {path: <file>}` instead. Found in %s.", where))
		spec = map[string]any{"path": s}
	}
	m, ok := spec.(map[string]any)
	keys := slices.Sorted(maps.Keys(m))
	if !ok || (!slices.Equal(keys, []string{"data"}) && !slices.Equal(keys, []string{"path"})) {
		return runnerBody{}, fmt.Errorf("%w: %s 'runner_body:' must hold one of 'path:' or 'data:'",
			errInvalidManifest, where)
	}
	if p, set := m["path"]; set {
		resolved, err := absAgainst(base, pyStr(p))
		return runnerBody{path: resolved}, err
	}
	data, ok := m["data"].(map[string]any)
	if !ok {
		return runnerBody{}, fmt.Errorf("%w: %s 'runner_body:' 'data:' must be a mapping", errInvalidManifest, where)
	}
	return runnerBody{data: data}, nil
}

// resolveScenario is a manifest entry's display name and file.
//
// A name with a YAML suffix is a path relative to the manifest; any other names
// a file under defaultDir, folder allowed (scripted/greeting). The display
// name is the bare stem either way.
func resolveScenario(name, base, defaultDir string) (string, string, error) {
	if slices.ContainsFunc(ScenarioSuffixes, func(suffix string) bool { return strings.HasSuffix(name, suffix) }) {
		path, err := absAgainst(base, name)
		return strings.TrimSuffix(filepath.Base(name), filepath.Ext(name)), path, err
	}
	path, err := absAgainst(defaultDir, name+".yaml")
	return filepath.Base(name), path, err
}

// checkDuplicateRuns refuses two runs of one scenario under one label. They are
// two runs nothing can tell apart in the results, which is a manifest mistake
// rather than something to report twice.
func checkDuplicateRuns(runs []*SuiteRun, path string) error {
	seen := make(map[[2]string]bool, len(runs))
	for _, run := range runs {
		key := [2]string{run.Label(), run.Scenario}
		if seen[key] {
			return fmt.Errorf("%w: %s: %s runs %s twice; give one of the entries a 'name:'",
				errInvalidManifest, path, pyRepr(run.Label()), pyRepr(run.Scenario))
		}
		seen[key] = true
	}
	return nil
}

// attemptMajor expands each run into its attempts, every run's first attempt
// before any run's second.
func attemptMajor(runs []*SuiteRun) []*SuiteRun {
	most := 1
	for _, run := range runs {
		most = max(most, run.attempts())
	}
	if most == 1 {
		return runs
	}
	var out []*SuiteRun
	for n := 1; n <= most; n++ {
		for _, run := range runs {
			if n <= run.attempts() {
				attempt := *run
				attempt.Attempt = n
				out = append(out, &attempt)
			}
		}
	}
	return out
}

// entryQueue is one entry's runs for one attempt, in manifest order, with the
// entry's cap on runs in flight.
type entryQueue struct {
	// entry is the entry's place in the manifest.
	entry int
	// cap is the most of the entry's runs in flight at once; zero is no cap.
	cap  int
	runs []*SuiteRun
}

// entryQueues is one queue per entry and attempt, attempt-major then in
// manifest order, with the entry and its cap.
//
// A queue keeps its runs in manifest order. An entry's cap is its
// Concurrency, none when it sets none, and it covers all of its attempts.
func entryQueues(runs []*SuiteRun) []*entryQueue {
	type key struct{ attempt, entry int }
	var queues []*entryQueue
	attempts := map[*entryQueue]int{}
	byKey := map[key]*entryQueue{}
	caps := map[int]int{}
	for _, run := range runs {
		k := key{run.attempt(), run.Entry}
		q, ok := byKey[k]
		if !ok {
			q = &entryQueue{entry: run.Entry}
			byKey[k] = q
			attempts[q] = k.attempt
			queues = append(queues, q)
		}
		q.runs = append(q.runs, run)
		caps[run.Entry] = run.Concurrency
	}
	slices.SortStableFunc(queues, func(a, b *entryQueue) int { return attempts[a] - attempts[b] })
	for _, q := range queues {
		q.cap = max(caps[q.entry], 0)
	}
	return queues
}

// runQueue is the suite's runs, handed out to workers in manifest order.
//
// Each queue is one entry's runs for one attempt, in the order the entries were
// given. A worker takes the next run from the first queue whose entry is under
// its cap; when every queue with runs left is at its cap, it waits for a run to
// finish.
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

// take is the next run to make, or nil when none is left.
func (q *runQueue) take() *SuiteRun {
	q.mu.Lock()
	defer q.mu.Unlock()
	for q.anyLeft() {
		if run := q.pick(); run != nil {
			return run
		}
		q.changed.Wait()
	}
	return nil
}

// done counts the run as finished, so its entry may take another.
func (q *runQueue) done(run *SuiteRun) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.inFlight[run.Entry]--
	q.changed.Broadcast()
}

// anyLeft reports whether any queue still has a run to hand out.
func (q *runQueue) anyLeft() bool {
	for _, eq := range q.queues {
		if len(eq.runs) > 0 {
			return true
		}
	}
	return false
}

// pick takes the first queued run whose entry is under its cap.
func (q *runQueue) pick() *SuiteRun {
	for _, eq := range q.queues {
		if len(eq.runs) > 0 && (eq.cap == 0 || q.inFlight[eq.entry] < eq.cap) {
			q.inFlight[eq.entry]++
			run := eq.runs[0]
			eq.runs = eq.runs[1:]
			return run
		}
	}
	return nil
}

// SuiteRunOptions is how one Suite.Run call runs the suite's runs.
type SuiteRunOptions struct {
	// RecordDir is the directory for per-run conversation recordings, or "".
	RecordDir string
	// ResultsPath is the JSONL file to append one record per finished run to,
	// or "" to write none. Each line is written as its run completes, so an
	// interrupted sweep keeps everything already finished.
	ResultsPath string
	// Debug saves each run's combined <run>.debug.log.
	Debug bool
	// Params is how each run behaves. The suite sets what it owns on each
	// run's copy: the connect timeout, the recording, and the manifest's cache
	// directory.
	Params SessionParams
	// BinDir is where the suite builds its bots; "" is a bin/ folder beside
	// the logs directory.
	BinDir string
}

// suiteRunSetup is what one Suite.Run call sets up for every run it makes.
type suiteRunSetup struct {
	// ports is the port assigned to each run.
	ports       map[*SuiteRun]int
	logsDir     string
	recordDir   string
	resultsPath string
	debug       bool
	params      SessionParams
}

// runFiles is where one run's artifacts go, all named by the run's prefix.
type runFiles struct {
	// prefix is <bot>__<scenario>, with the entry's name when it has one and
	// the attempt number when the run repeats.
	prefix string
	// log is the bot's output.
	log string
	// trace is the harness's decision trace, <prefix>.eval.log.
	trace string
	// conversation is a simulation's conversation, <prefix>.conversation.json.
	conversation string
	// body is the bot's runner-args body, written when the manifest gives it
	// inline.
	body string
	// record is the conversation recording, or "" when not recording.
	record string
}

// runFilesFor is the files of run under logsDir and recordDir.
//
// The bot is part of the prefix because one bot can run several scenarios at
// once, and the run's name joins it when the entry has one of its own, so a
// bot's files list together and each entry's stay apart; the attempt number
// joins it when the suite repeats, so no attempt writes over another's
// artifacts.
func runFilesFor(run *SuiteRun, logsDir, recordDir string) runFiles {
	prefix := strings.ReplaceAll(run.Bot, "/", "_")
	if run.Name != "" {
		prefix += "__" + strings.ReplaceAll(run.Name, "/", "_")
	}
	prefix += "__" + run.Stem()
	if run.attempts() > 1 {
		prefix += fmt.Sprintf("__%03d", run.attempt())
	}
	files := runFiles{
		prefix:       prefix,
		log:          filepath.Join(logsDir, prefix+".log"),
		trace:        filepath.Join(logsDir, prefix+".eval.log"),
		conversation: filepath.Join(logsDir, prefix+".conversation.json"),
		body:         filepath.Join(logsDir, prefix+".body.json"),
	}
	if recordDir != "" {
		files.record = filepath.Join(recordDir, prefix+".wav")
	}
	return files
}

// Suite runs the (bot, scenario) runs of a Manifest, building and spawning
// each bot.
//
// Each bot serves the harness on its own port and is driven by the harness,
// several at a time up to the manifest's Concurrency. The runs are updated in
// place as they go, and EventUpdate reports each change, so a live display can
// follow their progress.
//
//	manifest, err := eval.LoadManifest("manifest.yaml", eval.ManifestOverrides{})
//	suite := eval.NewSuite(manifest)
//	suite.Filter("voice", "", "")
//	event.On(&suite.Registry, eval.EventUpdate, func(_ context.Context, run *eval.SuiteRun) {
//		fmt.Println(run.Scenario, run.Status)
//	})
//	err = suite.Run(ctx, "logs", eval.SuiteRunOptions{})
type Suite struct {
	event.Registry

	// Manifest is the parsed manifest.
	Manifest *Manifest
	// Runs is the suite's working set: the manifest's runs, narrowed by
	// Filter, executed by Run.
	Runs []*SuiteRun

	// bins is each built bot's binary, by its package path; buildErrors is
	// why a bot did not build.
	bins        map[string]string
	buildErrors map[string]string
	// resultsMu serializes the results.jsonl appends of runs finishing at
	// once.
	resultsMu sync.Mutex
	// The steps of one run, replaced by the tests that stand in for the bot
	// and the harness.
	missingFile func(run *SuiteRun) string
	spawnBot    func(ctx context.Context, run *SuiteRun, port int, files runFiles) (*botProcess, error)
	runHarness  func(ctx context.Context, run *SuiteRun, port int, files runFiles, debug bool, params SessionParams) error
	finish      func(ctx context.Context, run *SuiteRun, files runFiles, bot *botProcess, setup *suiteRunSetup)
}

// NewSuite is the suite of a parsed manifest; its runs become the suite's
// working set.
func NewSuite(manifest *Manifest) *Suite {
	s := &Suite{Manifest: manifest, Runs: slices.Clone(manifest.Runs)}
	s.missingFile = s.defaultMissingFile
	s.spawnBot = s.defaultSpawnBot
	s.runHarness = s.defaultRunHarness
	s.finish = s.defaultFinish
	// Synchronous: a SuiteRun is mutated in place as it executes, so a handler
	// deferred to a goroutine would read whatever the run has since become.
	s.Register(EventUpdate, true)
	return s
}

// Filter keeps only the runs matching a bot substring, a scenario name, and a
// kind, each when given.
//
// pattern keeps only runs whose label or bot path contains it. scenario keeps
// only runs of this scenario: its full <file>/<scenario> name, or either half
// of it, so a file's name selects every scenario it holds. kind keeps only
// runs of this kind. It returns the matching runs, in their original order.
func (s *Suite) Filter(pattern, scenario string, kind Kind) []*SuiteRun {
	runs := s.Runs
	if pattern != "" {
		runs = slices.DeleteFunc(slices.Clone(runs), func(r *SuiteRun) bool {
			return !strings.Contains(r.Label(), pattern) && !strings.Contains(r.Bot, pattern)
		})
	}
	if scenario != "" {
		runs = slices.DeleteFunc(slices.Clone(runs), func(r *SuiteRun) bool {
			return r.Scenario != scenario && !slices.Contains(strings.Split(r.Scenario, "/"), scenario)
		})
	}
	if kind != "" {
		runs = slices.DeleteFunc(slices.Clone(runs), func(r *SuiteRun) bool { return r.RunKind() != kind })
	}
	s.Runs = runs
	return runs
}

// Run runs all of the suite's runs, in place, with the manifest's concurrency.
//
// It first builds each distinct bot once, into opts.BinDir. Each run gets its
// own port (BasePort + index). Concurrency workers each take the next run and
// run it until none is left. Runs are taken in manifest order, every entry's
// first attempt before any entry's second, so an entry's scenarios finish
// together and no worker waits while any entry still has runs. An entry with a
// Concurrency of its own never has more than that many runs in flight, across
// its attempts; a worker that finds it full takes the next entry's run.
//
// While it runs, the default logger writes nothing, so the caller's display
// stays clean; a run's logs go to its debug log under opts.Debug. The default
// logger is restored when Run returns. A run's failure is in the run; the
// error is only a directory that could not be made.
func (s *Suite) Run(ctx context.Context, logsDir string, opts SuiteRunOptions) error {
	restore := QuietLogs()
	defer restore()
	if err := os.MkdirAll(logsDir, 0o750); err != nil {
		return err
	}
	if opts.RecordDir != "" {
		if err := os.MkdirAll(opts.RecordDir, 0o750); err != nil {
			return err
		}
	}
	if opts.ResultsPath != "" {
		if err := os.MkdirAll(filepath.Dir(opts.ResultsPath), 0o750); err != nil {
			return err
		}
	}
	binDir := opts.BinDir
	if binDir == "" {
		binDir = filepath.Join(filepath.Dir(filepath.Clean(logsDir)), "bin")
	}
	s.bins, s.buildErrors = s.buildBots(ctx, binDir)

	setup := &suiteRunSetup{
		ports:       make(map[*SuiteRun]int, len(s.Runs)),
		logsDir:     logsDir,
		recordDir:   opts.RecordDir,
		resultsPath: opts.ResultsPath,
		debug:       opts.Debug,
		params:      opts.Params,
	}
	for i, run := range s.Runs {
		setup.ports[run] = s.Manifest.BasePort + i
	}
	queue := newRunQueue(entryQueues(s.Runs))
	var wg sync.WaitGroup
	for range min(s.Manifest.Concurrency, len(s.Runs)) {
		wg.Go(func() {
			// Take runs from the queue and run them, one after another, until
			// none is left.
			for run := queue.take(); run != nil; run = queue.take() {
				s.runOne(ctx, run, setup)
				queue.done(run)
			}
		})
	}
	wg.Wait()
	return nil
}

// runOne spawns one bot, runs its scenario against it, and records the outcome
// on run.
func (s *Suite) runOne(ctx context.Context, run *SuiteRun, setup *suiteRunSetup) {
	port := setup.ports[run]
	files := runFilesFor(run, setup.logsDir, setup.recordDir)
	run.Status = RunRunning
	run.StartedAt = time.Now()
	s.Call(ctx, EventUpdate, s, run)
	var bot *botProcess
	defer func() {
		// A panic is a problem on the suite's side; the run reports it.
		if r := recover(); r != nil {
			run.Error = fmt.Sprintf("error: panic: %v", r)
			_ = os.WriteFile(files.trace, debug.Stack(), 0o600)
		}
		s.finish(ctx, run, files, bot, setup)
	}()
	if run.Error = s.missingFile(run); run.Error != "" {
		return
	}
	var err error
	if run.BotURL == "" {
		bot, err = s.spawnBot(ctx, run, port, files)
	}
	if err == nil {
		err = s.runHarness(ctx, run, port, files, setup.debug, setup.params)
	}
	if err != nil {
		// The harness reports its own failures in its result; this is a
		// problem on the suite's side (building, spawning, a service the
		// scenario names that cannot be built).
		run.Error = "error: " + err.Error()
		_ = os.WriteFile(files.trace, []byte(run.Error+"\n"), 0o600)
	}
}

// defaultMissingFile is why the run cannot start, when one of its files is
// missing or its bot did not build.
func (s *Suite) defaultMissingFile(run *SuiteRun) string {
	if run.BotURL == "" {
		if run.BotPath == "" || !exists(run.BotPath) {
			return "bot not found: " + run.BotPath
		}
	}
	if !exists(run.ScenarioPath) {
		return fmt.Sprintf("%s not found: %s", run.RunKind(), run.ScenarioPath)
	}
	if run.RunnerBodyPath != "" && !exists(run.RunnerBodyPath) {
		return "body not found: " + run.RunnerBodyPath
	}
	if reason := s.buildErrors[run.BotPath]; reason != "" && run.BotURL == "" {
		return reason
	}
	return ""
}

// exists reports whether path names a file or directory.
func exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// buildBots builds each distinct bot the runs spawn, once, into binDir. It
// returns each built binary by its bot's package path, and why each bot that
// did not build failed. The build honors the caller's GOFLAGS.
func (s *Suite) buildBots(ctx context.Context, binDir string) (map[string]string, map[string]string) {
	bins, failures := map[string]string{}, map[string]string{}
	names := map[string]bool{}
	for _, run := range s.Runs {
		if run.BotURL != "" || run.BotPath == "" || !exists(run.BotPath) {
			continue
		}
		if _, built := bins[run.BotPath]; built || failures[run.BotPath] != "" {
			continue
		}
		name := binaryName(run.Bot, names)
		out := filepath.Join(binDir, name)
		if output, err := goBuild(ctx, run.BotPath, out); err != nil {
			buildLog := out + ".build.log"
			_ = os.MkdirAll(binDir, 0o750)
			_ = os.WriteFile(buildLog, output, 0o600)
			failures[run.BotPath] = fmt.Sprintf("error: building %s failed: %v (see %s)", run.Bot, err, buildLog)
			continue
		}
		bins[run.BotPath] = out
	}
	return bins, failures
}

// binaryName is the file a bot is built to: its manifest path as one file
// name, made unique among the names taken.
func binaryName(bot string, taken map[string]bool) string {
	base := strings.NewReplacer("/", "_", `\`, "_").Replace(strings.Trim(bot, `/\.`))
	if base == "" {
		base = "bot"
	}
	name := base
	for i := 2; taken[name]; i++ {
		name = fmt.Sprintf("%s_%d", base, i)
	}
	taken[name] = true
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	return name
}

// goBuild builds the Go main package at pkg, a directory or a single file, to
// out, returning the build's output.
func goBuild(ctx context.Context, pkg, out string) ([]byte, error) {
	out, err := filepath.Abs(out)
	if err != nil {
		return nil, err
	}
	dir, target := pkg, "."
	if info, err := os.Stat(pkg); err == nil && !info.IsDir() {
		dir, target = filepath.Dir(pkg), filepath.Base(pkg)
	}
	cmd := exec.CommandContext(ctx, "go", "build", "-o", out, target) //nolint:gosec // the bot is operator-supplied
	cmd.Dir = dir
	return cmd.CombinedOutput()
}

// botProcess is a spawned bot, and whether it has exited.
type botProcess struct {
	cmd  *exec.Cmd
	done chan struct{}
}

// startBot starts cmd and watches for it to exit.
func startBot(cmd *exec.Cmd) (*botProcess, error) {
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	p := &botProcess{cmd: cmd, done: make(chan struct{})}
	go func() {
		_ = cmd.Wait()
		close(p.done)
	}()
	return p, nil
}

// exited reports whether the bot has exited.
func (p *botProcess) exited() bool {
	select {
	case <-p.done:
		return true
	default:
		return false
	}
}

// exitCode is the bot's exit code once it has exited, -1 before or when a
// signal ended it.
func (p *botProcess) exitCode() int {
	if !p.exited() {
		return -1
	}
	return p.cmd.ProcessState.ExitCode()
}

// defaultSpawnBot starts the bot serving the harness on port, its output going
// to the bot log.
//
// A body file's directory is the bot's working directory, so relative paths
// inside the body (an image) resolve next to the file. A body given inline is
// written to the run's body file first.
func (s *Suite) defaultSpawnBot(_ context.Context, run *SuiteRun, port int, files runFiles) (*botProcess, error) {
	bodyPath := run.RunnerBodyPath
	if run.RunnerBody != nil {
		body, err := json.Marshal(run.RunnerBody)
		if err != nil {
			return nil, err
		}
		bodyPath = files.body
		if err := os.WriteFile(bodyPath, body, 0o600); err != nil {
			return nil, err
		}
	}
	bot := s.bins[run.BotPath]
	if bot == "" {
		// Not built by this suite: the bot as given.
		bot = run.BotPath
	}
	argv, err := spawnArgv(s.Manifest.Spawn, bot, port, bodyPath)
	if err != nil {
		return nil, err
	}
	logf, err := os.Create(files.log)
	if err != nil {
		return nil, err
	}
	defer func() { _ = logf.Close() }()
	// The spawn template is operator-supplied, and the suite stops the bot
	// itself rather than through a context.
	cmd := exec.Command(argv[0], argv[1:]...) //nolint:gosec,noctx // see above
	cmd.Stdout, cmd.Stderr = logf, logf
	cmd.Env = s.botEnv()
	if run.RunnerBodyPath != "" {
		cmd.Dir = filepath.Dir(run.RunnerBodyPath)
	}
	return startBot(cmd)
}

// botEnv is the environment a bot is spawned with: the suite's own, with
// GOMAXPROCS bounded to the cores over the concurrency, so concurrent bots
// share the cores rather than each taking every one. An explicit GOMAXPROCS in
// the environment wins.
func (s *Suite) botEnv() []string {
	env := os.Environ()
	for _, kv := range env {
		if strings.HasPrefix(kv, "GOMAXPROCS=") {
			return env
		}
	}
	procs := max(1, runtime.NumCPU()/max(1, s.Manifest.Concurrency))
	return append(env, "GOMAXPROCS="+strconv.Itoa(procs))
}

// defaultRunHarness runs the harness for this run against the bot on port and
// records the outcome on run.
//
// A run that outlives WorkerSafetyTimeout leaves run.Error rather than a
// result.
func (s *Suite) defaultRunHarness(
	ctx context.Context, run *SuiteRun, port int, files runFiles, debugLogs bool, params SessionParams,
) error {
	params.ConnectTimeout = BotConnectTimeout
	params.RecordPath = files.record
	params.CacheDir = s.Manifest.CacheDir
	botURL := run.BotURL
	if botURL == "" {
		botURL = fmt.Sprintf("ws://localhost:%d/ws", port)
	}
	loaded, err := run.Load()
	if err != nil {
		return err
	}
	hctx, cancel := context.WithTimeoutCause(ctx, WorkerSafetyTimeout, errHarnessTimeout)
	defer cancel()
	hctx, finishLogs := CapturePipelineLogs(hctx, filepath.Dir(files.log), files.prefix, run.Scenario, debugLogs)
	defer finishLogs()
	err = runScenario(hctx, run, loaded, botURL, params)
	if errors.Is(context.Cause(hctx), errHarnessTimeout) {
		run.ScriptResult, run.SimulationResult = nil, nil
		return fmt.Errorf("%w after %.0fs", errHarnessTimeout, WorkerSafetyTimeout.Seconds())
	}
	return err
}

// runScenario builds the session of the scenario, whichever kind it is, and
// runs it against botURL, recording the result on run.
func runScenario(
	ctx context.Context, run *SuiteRun, loaded LoadedScenario, botURL string, params SessionParams,
) error {
	if _, ok := loaded.(*Rubric); ok {
		return fmt.Errorf("%s is a rubric, %w", loaded.ScenarioName(), errNothingToRun)
	}
	session, err := SessionFromScenario(loaded, botURL, params, SessionServices{})
	if err != nil {
		return err
	}
	switch s := session.(type) {
	case *ScriptSession:
		result, err := s.Run(ctx)
		run.ScriptResult = &result
		return err
	case *SimulationSession:
		result, err := s.Run(ctx)
		run.SimulationResult = result
		return err
	}
	return nil
}

// defaultFinish marks the run done, stops its bot, and keeps the artifacts.
func (s *Suite) defaultFinish(
	ctx context.Context, run *SuiteRun, files runFiles, bot *botProcess, setup *suiteRunSetup,
) {
	// The duration is measured the way the live counter ticks, and excludes
	// the teardown.
	if !run.StartedAt.IsZero() {
		ms := int(time.Since(run.StartedAt).Milliseconds())
		run.DurationMS = &ms
	}
	run.Status = RunDone
	s.Call(ctx, EventUpdate, s, run)
	run.Stopping = true
	if bot != nil {
		stopBot(bot)
	}
	run.Stopping = false
	if log := run.debugLog(); len(log) > 0 {
		_ = os.WriteFile(files.trace, []byte(strings.Join(log, "\n")+"\n"), 0o600)
	}
	if run.SimulationResult != nil {
		if data, err := json.MarshalIndent(nonNil(run.SimulationResult.Conversation), "", "  "); err == nil {
			_ = os.WriteFile(files.conversation, append(data, '\n'), 0o600)
		}
	}
	if setup.resultsPath != "" {
		s.resultsMu.Lock()
		appendResult(setup.resultsPath, run, files.prefix, setup.logsDir, setup.recordDir)
		s.resultsMu.Unlock()
	}
}

// stopBot terminates the bot, and kills it if it lingers.
//
// The bot's session ended with the harness's connection, but the runner
// serving it keeps running until it is told to stop.
func stopBot(p *botProcess) {
	if p.exited() {
		return
	}
	if err := p.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		// No terminate signal on this platform.
		_ = p.cmd.Process.Kill()
	}
	select {
	case <-p.done:
	case <-time.After(BotStopTimeout):
		_ = p.cmd.Process.Kill()
		<-p.done
	}
}

// spawnArgv is the spawn argv, with {bot} and {port} substituted per token so
// a path with spaces stays one entry, and --runner-body added when the bot has
// a body.
func spawnArgv(template, bot string, port int, runnerBodyPath string) ([]string, error) {
	tokens, err := splitCommand(template)
	if err != nil {
		return nil, fmt.Errorf("%w %q: %w", errSpawnTemplate, template, err)
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("%w %q: no command", errSpawnTemplate, template)
	}
	subs := map[string]string{"bot": bot, "port": strconv.Itoa(port)}
	argv := make([]string, 0, len(tokens)+2)
	for _, tok := range tokens {
		expanded, err := formatToken(tok, subs)
		if err != nil {
			return nil, fmt.Errorf("%w %q: %w", errSpawnTemplate, template, err)
		}
		argv = append(argv, expanded)
	}
	if runnerBodyPath != "" {
		argv = append(argv, "--runner-body", runnerBodyPath)
	}
	return argv, nil
}

// Spawn template errors.
var (
	errUnknownPlaceholder = errors.New("unknown placeholder")
	errPythonPlaceholder  = errors.New("{python} does not apply: bots are Go packages the suite builds, " +
		"and {bot} is the built binary, so a template starts with {bot}")
	errUnbalancedBrace = errors.New("unbalanced brace")
	errUnclosedQuote   = errors.New("no closing quotation")
	errTrailingEscape  = errors.New("no escaped character")
)

// formatToken substitutes the {name} placeholders of one token; {{ and }}
// stand for literal braces.
func formatToken(tok string, subs map[string]string) (string, error) {
	var b strings.Builder
	for i := 0; i < len(tok); i++ {
		c := tok[i]
		switch {
		case c == '{' && i+1 < len(tok) && tok[i+1] == '{':
			b.WriteByte('{')
			i++
		case c == '}' && i+1 < len(tok) && tok[i+1] == '}':
			b.WriteByte('}')
			i++
		case c == '{':
			end := strings.IndexByte(tok[i:], '}')
			if end < 0 {
				return "", errUnbalancedBrace
			}
			name := tok[i+1 : i+end]
			value, ok := subs[name]
			switch {
			case name == "python":
				return "", errPythonPlaceholder
			case !ok:
				return "", fmt.Errorf("%w {%s}", errUnknownPlaceholder, name)
			}
			b.WriteString(value)
			i += end
		case c == '}':
			return "", errUnbalancedBrace
		default:
			b.WriteByte(c)
		}
	}
	return b.String(), nil
}

// splitCommand splits a command line into words the way a POSIX shell does:
// whitespace separates words, single quotes keep their contents literally,
// double quotes keep theirs but for backslash escapes, and a backslash outside
// quotes escapes the next character.
func splitCommand(s string) ([]string, error) {
	var (
		words  []string
		word   strings.Builder
		inWord bool
	)
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case ' ', '\t', '\n', '\r':
			if inWord {
				words = append(words, word.String())
				word.Reset()
				inWord = false
			}
		case '\'':
			end := strings.IndexByte(s[i+1:], '\'')
			if end < 0 {
				return nil, errUnclosedQuote
			}
			word.WriteString(s[i+1 : i+1+end])
			i += end + 1
			inWord = true
		case '"':
			n, err := doubleQuoted(s[i+1:], &word)
			if err != nil {
				return nil, err
			}
			i += n + 1
			inWord = true
		case '\\':
			if i+1 >= len(s) {
				return nil, errTrailingEscape
			}
			i++
			word.WriteByte(s[i])
			inWord = true
		default:
			word.WriteByte(c)
			inWord = true
		}
	}
	if inWord {
		words = append(words, word.String())
	}
	return words, nil
}

// doubleQuoted reads a double-quoted string's contents from s, which follows
// the opening quote, into word, and returns how many bytes it took, the
// closing quote included.
func doubleQuoted(s string, word *strings.Builder) (int, error) {
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == '"':
			return i + 1, nil
		case c == '\\' && i+1 < len(s) && strings.IndexByte(`\"$`+"`", s[i+1]) >= 0:
			i++
			word.WriteByte(s[i])
		default:
			word.WriteByte(c)
		}
	}
	return 0, errUnclosedQuote
}

// nonNil is list, or an empty list in its place when it is nil, so it encodes
// as [] rather than null.
func nonNil[T any](list []T) []T {
	if list == nil {
		return []T{}
	}
	return list
}
