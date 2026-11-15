package eval

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/classifier"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/events"
)

// helperEnv makes the test binary stand in for a bot: "argv" prints its
// arguments and working directory and exits, "sleep" sleeps for a minute.
const helperEnv = "JARGO_EVAL_SUITE_HELPER"

func TestMain(m *testing.M) {
	switch os.Getenv(helperEnv) {
	case "argv":
		cwd, _ := os.Getwd()
		out, _ := json.Marshal(map[string]any{"argv": os.Args[1:], "cwd": cwd})
		_, _ = os.Stdout.Write(out)
		return
	case "sleep":
		time.Sleep(time.Minute)
		return
	}
	m.Run()
}

const suiteManifest = `
bots_dir: bots
scenarios_dir: my-scenarios
concurrency: 2
runs_dir: out
record: true
suite:
  - bot: voice/voice-a
    scenarios: [simple_math, multi_turn]
  - bot: vision/vision-b
    runner_body:
      path: bodies/cat.yaml
    scenarios: [other/special.yaml]
`

// manifestDir is a temporary directory to write a manifest into, resolved so
// the paths the manifest resolves compare equal.
func manifestDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func writeIn(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func loadManifestText(t *testing.T, dir, text string, o ManifestOverrides) (*Manifest, error) {
	t.Helper()
	return LoadManifest(writeIn(t, dir, "manifest.yaml", text), o)
}

func mustManifest(t *testing.T, dir, text string, o ManifestOverrides) *Manifest {
	t.Helper()
	m, err := loadManifestText(t, dir, text, o)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// The manifest load: TestEvalManifestLoad.

func TestManifestPathsResolveRelativeToTheManifest(t *testing.T) {
	base := manifestDir(t)
	m := mustManifest(t, base, suiteManifest, ManifestOverrides{})
	if m.Concurrency != 2 || !m.Record || m.RunsDir != filepath.Join(base, "out") {
		t.Fatalf("settings = %d %v %q", m.Concurrency, m.Record, m.RunsDir)
	}
	if len(m.Runs) != 3 {
		t.Fatalf("got %d runs, want 2 + 1", len(m.Runs))
	}
	first := m.Runs[0]
	if first.Bot != "voice/voice-a" || first.BotPath != filepath.Join(base, "bots", "voice", "voice-a") {
		t.Fatalf("bot = %q at %q", first.Bot, first.BotPath)
	}
	// Bare scenario names resolve under scenarios_dir, with .yaml appended.
	if first.Scenario != "simple_math" || first.ScenarioPath != filepath.Join(base, "my-scenarios", "simple_math.yaml") {
		t.Fatalf("scenario = %q at %q", first.Scenario, first.ScenarioPath)
	}
}

func TestManifestScenarioPathsResolveRelativeToTheManifest(t *testing.T) {
	base := manifestDir(t)
	special := mustManifest(t, base, suiteManifest, ManifestOverrides{}).Runs[2]
	// A path-like scenario bypasses scenarios_dir and resolves to the manifest.
	if special.Scenario != "special" || special.ScenarioPath != filepath.Join(base, "other", "special.yaml") {
		t.Fatalf("scenario = %q at %q", special.Scenario, special.ScenarioPath)
	}
	if special.RunnerBodyPath != filepath.Join(base, "bodies", "cat.yaml") || special.RunnerBody != nil {
		t.Fatalf("body = %q, %v", special.RunnerBodyPath, special.RunnerBody)
	}
}

func TestManifestRunnerBodyGivenInline(t *testing.T) {
	run := mustManifest(t, manifestDir(t), "suite:\n  - bot: a\n    runner_body:\n"+
		"      data: {model: gpt-4o-mini, question: hi}\n    scenarios: [x]\n", ManifestOverrides{}).Runs[0]
	want := map[string]any{"model": "gpt-4o-mini", "question": "hi"}
	if fmt.Sprint(run.RunnerBody) != fmt.Sprint(want) || run.RunnerBodyPath != "" {
		t.Fatalf("body = %v, %q", run.RunnerBody, run.RunnerBodyPath)
	}
}

// warnings records the warnings logged while fn runs.
func warnings(t *testing.T, fn func()) []string {
	t.Helper()
	var (
		mu   sync.Mutex
		seen []string
	)
	prev := slog.Default()
	slog.SetDefault(slog.New(recordingHandler{func(r slog.Record) {
		if r.Level >= slog.LevelWarn {
			mu.Lock()
			seen = append(seen, r.Message)
			mu.Unlock()
		}
	}}))
	defer slog.SetDefault(prev)
	fn()
	mu.Lock()
	defer mu.Unlock()
	return slices.Clone(seen)
}

type recordingHandler struct{ record func(slog.Record) }

func (recordingHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h recordingHandler) Handle(_ context.Context, r slog.Record) error {
	h.record(r)
	return nil
}
func (h recordingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h recordingHandler) WithGroup(string) slog.Handler      { return h }

func TestManifestBareRunnerBodyPathIsDeprecated(t *testing.T) {
	base := manifestDir(t)
	var run *SuiteRun
	seen := warnings(t, func() {
		run = mustManifest(t, base, "suite:\n  - bot: a\n    runner_body: bodies/cat.yaml\n    scenarios: [x]\n",
			ManifestOverrides{}).Runs[0]
	})
	if len(seen) != 1 || !strings.Contains(seen[0], "runner_body: {path: <file>}") {
		t.Fatalf("warnings = %q, want the one deprecation", seen)
	}
	if run.RunnerBodyPath != filepath.Join(base, "bodies", "cat.yaml") {
		t.Fatalf("body = %q", run.RunnerBodyPath)
	}
}

func TestManifestRunnerBodyMustBeAPathOrData(t *testing.T) {
	for _, bad := range []string{"{path: a.yaml, data: {}}", "{}", "{file: a.yaml}", "[a.yaml]", "{data: a.yaml}"} {
		_, err := loadManifestText(t, manifestDir(t),
			"suite:\n  - bot: a\n    runner_body: "+bad+"\n    scenarios: [x]\n", ManifestOverrides{})
		if err == nil || !strings.Contains(err.Error(), "'runner_body:'") {
			t.Errorf("%s: err = %v", bad, err)
		}
	}
}

func TestManifestDefaults(t *testing.T) {
	m := mustManifest(t, manifestDir(t), "suite: []\n", ManifestOverrides{})
	if m.Concurrency != DefaultConcurrency || m.Spawn != DefaultSpawn || m.Record || m.RunsDir != "" ||
		len(m.Runs) != 0 || m.BasePort != DefaultBasePort || m.Repeat != 1 {
		t.Fatalf("manifest = %+v", m)
	}
}

func TestManifestOverridesWin(t *testing.T) {
	record := false
	m := mustManifest(t, manifestDir(t), suiteManifest,
		ManifestOverrides{Concurrency: new(8), Record: &record, Spawn: "x {bot}"})
	if m.Concurrency != 8 || m.Record || m.Spawn != "x {bot}" {
		t.Fatalf("manifest = %+v", m)
	}
}

func TestManifestAnEntryCapsItsOwnConcurrency(t *testing.T) {
	m := mustManifest(t, manifestDir(t), "suite:\n"+
		"  - bot: a\n    concurrency: 2\n    scenarios: [x, y]\n"+
		"  - bot: b\n    scenarios: [x]\n"+
		"  - bot: a\n    scenarios: [z]\n", ManifestOverrides{Concurrency: new(8)})
	var caps, entries []int
	for _, r := range m.Runs {
		caps, entries = append(caps, r.Concurrency), append(entries, r.Entry)
	}
	if !slices.Equal(caps, []int{2, 2, 0, 0}) || !slices.Equal(entries, []int{0, 0, 1, 2}) {
		t.Fatalf("caps %v entries %v", caps, entries)
	}
	// The command line's cap is the suite's, not the entry's.
	if m.Concurrency != 8 {
		t.Fatalf("concurrency = %d", m.Concurrency)
	}
}

func TestManifestEntryConcurrencyMustBeAPositiveInteger(t *testing.T) {
	for _, bad := range []string{"0", "-1", "two", "true", "1.5"} {
		_, err := loadManifestText(t, manifestDir(t),
			"suite:\n  - bot: a\n    concurrency: "+bad+"\n    scenarios: [x]\n", ManifestOverrides{})
		if err == nil || !strings.Contains(err.Error(), "'concurrency:'") {
			t.Errorf("%s: err = %v", bad, err)
		}
	}
}

// Bots are Go packages the suite builds, so a template naming an interpreter
// is refused with a reason.
func TestManifestSpawnTemplateRefusesPython(t *testing.T) {
	_, err := loadManifestText(t, manifestDir(t), "spawn: \"{python} {bot} --port {port}\"\nsuite: []\n",
		ManifestOverrides{})
	if err == nil || !strings.Contains(err.Error(), "{python} does not apply") {
		t.Fatalf("err = %v", err)
	}
	_, err = loadManifestText(t, manifestDir(t), "python: python3\nsuite: []\n", ManifestOverrides{})
	if err == nil || !strings.Contains(err.Error(), "'python:' does not apply") {
		t.Fatalf("err = %v", err)
	}
	_, err = loadManifestText(t, manifestDir(t), "suite: []\n", ManifestOverrides{Spawn: "{bot} {nope}"})
	if err == nil || !strings.Contains(err.Error(), "unknown placeholder {nope}") {
		t.Fatalf("err = %v", err)
	}
}

func TestSpawnArgvSubstitutesPerToken(t *testing.T) {
	argv, err := spawnArgv(`{bot} --port {port} --name 'a b' "{{x}}"`, "/bin dir/bot", 7901, "/b.json")
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"/bin dir/bot", "--port", "7901", "--name", "a b", "{x}", "--runner-body", "/b.json"}
	if !slices.Equal(argv, want) {
		t.Fatalf("argv = %q, want %q", argv, want)
	}
}

// The per-run log capture: TestCapturePipelineLogs.

func captureQuietly(t *testing.T) {
	t.Helper()
	t.Cleanup(QuietLogs())
}

func TestCapturePipelineLogsWritesSectionsPerPipeline(t *testing.T) {
	captureQuietly(t)
	dir := t.TempDir()
	ctx, finish := CapturePipelineLogs(t.Context(), dir, "run1", "simple_math", true)
	slog.DebugContext(withEvalPipeline(ctx, PipelineJudge), "judge line")
	slog.DebugContext(ctx, "harness line")
	finish()
	content, err := os.ReadFile(filepath.Join(dir, "run1.debug.log")) //nolint:gosec // a test file
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"===== judge logs: simple_math =====", "judge line", "===== harness logs: simple_math =====", "harness line",
	} {
		if !strings.Contains(string(content), want) {
			t.Errorf("debug log lacks %q:\n%s", want, content)
		}
	}
}

func TestCapturePipelineLogsDisabledWritesNothing(t *testing.T) {
	captureQuietly(t)
	dir := t.TempDir()
	ctx, finish := CapturePipelineLogs(t.Context(), dir, "run1", "x", false)
	slog.DebugContext(ctx, "dropped")
	finish()
	if entries, _ := os.ReadDir(dir); len(entries) != 0 {
		t.Fatalf("wrote %v", entries)
	}
}

func TestCapturePipelineLogsConcurrentRunsDoNotMix(t *testing.T) {
	captureQuietly(t)
	dir := t.TempDir()
	// Logs emitted under a different run id must not land in run1's file.
	ctx, finish := CapturePipelineLogs(t.Context(), dir, "run1", "a", true)
	slog.DebugContext(ctx, "mine")
	slog.DebugContext(withEvalRun(ctx, "run2"), "theirs")
	finish()
	content, _ := os.ReadFile(filepath.Join(dir, "run1.debug.log")) //nolint:gosec // a test file
	if !strings.Contains(string(content), "mine") || strings.Contains(string(content), "theirs") {
		t.Fatalf("debug log:\n%s", content)
	}
}

// The update event: TestSuiteUpdateEvent.

func suiteOfRuns(runs []*SuiteRun, concurrency int) *Suite {
	return NewSuite(&Manifest{
		Runs: runs, Spawn: DefaultSpawn, Concurrency: concurrency, Repeat: 1, BasePort: DefaultBasePort,
	})
}

func TestSuiteUpdateEventReceivesRuns(t *testing.T) {
	dir := t.TempDir()
	// A bot that does not exist: the run errors out before spawning anything,
	// which is enough to drive both status changes.
	suite := suiteOfRuns([]*SuiteRun{{
		Bot: "missing", Scenario: "none", ScenarioPath: filepath.Join(dir, "none.yaml"),
		BotPath: filepath.Join(dir, "missing"),
	}}, 1)
	var sources []any
	var statuses []string
	suite.Add(EventUpdate, func(_ context.Context, source any, args ...any) {
		sources = append(sources, source)
		statuses = append(statuses, args[0].(*SuiteRun).Status) //nolint:forcetypeassert // the event's argument
	})
	if err := suite.Run(t.Context(), dir, SuiteRunOptions{}); err != nil {
		t.Fatal(err)
	}
	for _, s := range sources {
		if s != suite {
			t.Fatalf("source = %v, want the suite", s)
		}
	}
	if !slices.Equal(statuses, []string{RunRunning, RunDone}) {
		t.Fatalf("statuses = %v", statuses)
	}
	if run := suite.Runs[0]; !strings.HasPrefix(run.Error, "bot not found: ") {
		t.Fatalf("error = %q", run.Error)
	}
}

// The scheduling: TestBotConcurrency.

// countOverlap stands in for the bot and the harness, recording the most runs
// held at once per key (the bot, unless key says otherwise) and in all.
func countOverlap(s *Suite, hold time.Duration, key func(*SuiteRun) string) map[string]int {
	var mu sync.Mutex
	active, peak := map[string]int{}, map[string]int{}
	s.missingFile = func(*SuiteRun) string { return "" }
	s.spawnBot = func(_ context.Context, run *SuiteRun, _ int, _ runFiles) (*botProcess, error) {
		k := key(run)
		mu.Lock()
		active[k]++
		peak[k] = max(peak[k], active[k])
		total := 0
		for _, n := range active {
			total += n
		}
		peak["total"] = max(peak["total"], total)
		mu.Unlock()
		time.Sleep(hold)
		mu.Lock()
		active[k]--
		mu.Unlock()
		return nil, nil //nolint:nilnil // no bot to stop
	}
	s.runHarness = func(context.Context, *SuiteRun, int, runFiles, bool, SessionParams) error { return nil }
	s.finish = func(_ context.Context, run *SuiteRun, _ runFiles, _ *botProcess, _ *suiteRunSetup) {
		run.Status = RunDone
	}
	return peak
}

func byBot(r *SuiteRun) string { return r.Bot }

func overlapRuns(dir string, bots []string, counts []int, caps map[string]int) []*SuiteRun {
	var runs []*SuiteRun
	for entry, bot := range bots {
		for i := range counts[entry] {
			runs = append(runs, &SuiteRun{
				Bot: bot, Scenario: fmt.Sprintf("s%d", i), ScenarioPath: filepath.Join(dir, "s.yaml"),
				BotPath: filepath.Join(dir, bot), Concurrency: caps[bot], Entry: entry,
			})
		}
	}
	return runs
}

func allDone(t *testing.T, runs []*SuiteRun) {
	t.Helper()
	for i, r := range runs {
		if r.Status != RunDone {
			t.Fatalf("run %d is %q", i, r.Status)
		}
	}
}

func TestSuiteOneEntryFillsEverySlot(t *testing.T) {
	dir := t.TempDir()
	runs := overlapRuns(dir, []string{"a"}, []int{10}, nil)
	suite := suiteOfRuns(runs, 4)
	peak := countOverlap(suite, 50*time.Millisecond, byBot)
	if err := suite.Run(t.Context(), dir, SuiteRunOptions{}); err != nil {
		t.Fatal(err)
	}
	if peak["a"] != 4 {
		t.Fatalf("peak = %d, want 4", peak["a"])
	}
	allDone(t, runs)
}

func TestSuiteAnEntryCapLimitsItsRunsInFlight(t *testing.T) {
	dir := t.TempDir()
	runs := overlapRuns(dir, []string{"capped", "plain"}, []int{3, 3}, map[string]int{"capped": 1})
	suite := suiteOfRuns(runs, 4)
	peak := countOverlap(suite, 50*time.Millisecond, byBot)
	if err := suite.Run(t.Context(), dir, SuiteRunOptions{}); err != nil {
		t.Fatal(err)
	}
	if peak["capped"] != 1 || peak["total"] != 4 {
		t.Fatalf("peak = %v", peak)
	}
	allDone(t, runs)
}

func TestSuiteAnEntryCapHoldsAcrossAttempts(t *testing.T) {
	dir := t.TempDir()
	var runs []*SuiteRun
	for attempt := 1; attempt <= 3; attempt++ {
		for entry, bot := range []string{"capped", "plain"} {
			c := 0
			if bot == "capped" {
				c = 1
			}
			runs = append(runs, &SuiteRun{
				Bot: bot, Scenario: "s", ScenarioPath: filepath.Join(dir, "s.yaml"), BotPath: filepath.Join(dir, bot),
				Concurrency: c, Entry: entry, Attempt: attempt,
			})
		}
	}
	suite := suiteOfRuns(runs, 2)
	peak := countOverlap(suite, 50*time.Millisecond, byBot)
	if err := suite.Run(t.Context(), dir, SuiteRunOptions{}); err != nil {
		t.Fatal(err)
	}
	if peak["capped"] != 1 || peak["total"] != 2 {
		t.Fatalf("peak = %v", peak)
	}
	allDone(t, runs)
}

func TestSuiteAnEntryCapLeavesAnotherEntryForTheSameBotAlone(t *testing.T) {
	dir := t.TempDir()
	var runs []*SuiteRun
	for entry := range 2 {
		for i := range 3 {
			c := 0
			if entry == 0 {
				c = 1
			}
			runs = append(runs, &SuiteRun{
				Bot: "a", Scenario: fmt.Sprintf("s%d%d", entry, i), ScenarioPath: filepath.Join(dir, "s.yaml"),
				BotPath: filepath.Join(dir, "a"), Concurrency: c, Entry: entry,
			})
		}
	}
	suite := suiteOfRuns(runs, 4)
	peak := countOverlap(suite, 50*time.Millisecond, func(r *SuiteRun) string { return fmt.Sprint(r.Entry) })
	if err := suite.Run(t.Context(), dir, SuiteRunOptions{}); err != nil {
		t.Fatal(err)
	}
	if peak["0"] != 1 || peak["1"] != 3 || peak["total"] != 4 {
		t.Fatalf("peak = %v", peak)
	}
	allDone(t, runs)
}

func TestSuiteRunsAreTakenInManifestOrder(t *testing.T) {
	dir := t.TempDir()
	var runs []*SuiteRun
	for _, r := range []struct {
		entry         int
		bot, scenario string
	}{
		{0, "a", "s1"}, {0, "a", "s2"}, {0, "a", "s3"}, {1, "b", "s1"}, {2, "c", "s1"}, {2, "c", "s2"}, {3, "a", "s4"},
	} {
		runs = append(runs, &SuiteRun{
			Bot: r.bot, Scenario: r.scenario, ScenarioPath: filepath.Join(dir, "s.yaml"),
			BotPath: filepath.Join(dir, r.bot), Entry: r.entry,
		})
	}
	suite := suiteOfRuns(runs, 1)
	countOverlap(suite, 0, byBot)
	var started []string
	spawn := suite.spawnBot
	suite.spawnBot = func(ctx context.Context, run *SuiteRun, port int, files runFiles) (*botProcess, error) {
		started = append(started, run.Bot+" "+run.Scenario)
		return spawn(ctx, run, port, files)
	}
	if err := suite.Run(t.Context(), dir, SuiteRunOptions{}); err != nil {
		t.Fatal(err)
	}
	// One at a time, each entry's scenarios run together, in manifest order,
	// and a bot's second entry waits for its place.
	want := []string{"a s1", "a s2", "a s3", "b s1", "c s1", "c s2", "a s4"}
	if !slices.Equal(started, want) {
		t.Fatalf("started %v, want %v", started, want)
	}
}

// The artifact names: TestRunFiles.

func TestRunFilesPrefixCarriesTheBotAndTheNameWhenTheEntryHasOne(t *testing.T) {
	run := &SuiteRun{Bot: "turns/bot", Scenario: "scripted/turn"}
	if got := runFilesFor(run, "/logs", "").prefix; got != "turns_bot__scripted__turn" {
		t.Errorf("prefix = %q", got)
	}
	named := &SuiteRun{Bot: "turns/bot", Name: "groq/llama", Scenario: "scripted/turn"}
	if got := runFilesFor(named, "/logs", "").prefix; got != "turns_bot__groq_llama__scripted__turn" {
		t.Errorf("prefix = %q", got)
	}
	repeated := &SuiteRun{Bot: "turns/bot", Name: "groq/llama", Scenario: "turn", Attempts: 2, Attempt: 2}
	if got := runFilesFor(repeated, "/logs", "").prefix; got != "turns_bot__groq_llama__turn__002" {
		t.Errorf("prefix = %q", got)
	}
}

// The spawn: TestSpawnWithRunnerBody.

// spawnHelper spawns run's bot, the test binary printing what it was given,
// and returns what it printed.
func spawnHelper(t *testing.T, base string, run *SuiteRun) map[string]any {
	t.Helper()
	t.Setenv(helperEnv, "argv")
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	run.BotPath = self
	suite := suiteOfRuns([]*SuiteRun{run}, 1)
	files := runFilesFor(run, filepath.Join(base, "logs"), "")
	if err := os.MkdirAll(filepath.Dir(files.log), 0o750); err != nil {
		t.Fatal(err)
	}
	proc, err := suite.spawnBot(t.Context(), run, 7900, files)
	if err != nil {
		t.Fatal(err)
	}
	<-proc.done
	var seen map[string]any
	data, _ := os.ReadFile(files.log) //nolint:gosec // a test file
	if err := json.Unmarshal(data, &seen); err != nil {
		t.Fatalf("bot output %q: %v", data, err)
	}
	return seen
}

func argvOf(seen map[string]any) []string {
	var argv []string
	for _, a := range seen["argv"].([]any) { //nolint:forcetypeassert // the helper's output
		argv = append(argv, a.(string)) //nolint:forcetypeassert // the helper's output
	}
	return argv
}

func TestSpawnInlineBodyIsWrittenForTheBot(t *testing.T) {
	base := manifestDir(t)
	seen := spawnHelper(t, base, &SuiteRun{
		Bot: "bot", Scenario: "x", ScenarioPath: filepath.Join(base, "x.yaml"),
		RunnerBody: map[string]any{"model": "gpt-4o-mini"},
	})
	argv := argvOf(seen)
	if len(argv) != 4 || argv[0] != "--port" || argv[1] != "7900" || argv[2] != "--runner-body" {
		t.Fatalf("argv = %q", argv)
	}
	if filepath.Dir(argv[3]) != filepath.Join(base, "logs") {
		t.Fatalf("body at %q", argv[3])
	}
	body, _ := os.ReadFile(argv[3]) //nolint:gosec // a test file
	if string(body) != `{"model":"gpt-4o-mini"}` {
		t.Fatalf("body = %s", body)
	}
	// No body file to anchor it, so the bot runs where the suite does.
	if cwd, _ := os.Getwd(); seen["cwd"] != cwd {
		t.Fatalf("cwd = %v, want %s", seen["cwd"], cwd)
	}
}

func TestSpawnBodyFileIsTheBotsDirectory(t *testing.T) {
	base := manifestDir(t)
	body := writeIn(t, base, "bodies/cat.yaml", "image_path: cat.jpg\n")
	seen := spawnHelper(t, base, &SuiteRun{
		Bot: "bot", Scenario: "x", ScenarioPath: filepath.Join(base, "x.yaml"), RunnerBodyPath: body,
	})
	if argv := argvOf(seen); !slices.Equal(argv[2:], []string{"--runner-body", body}) {
		t.Fatalf("argv = %q", argv)
	}
	if seen["cwd"] != filepath.Dir(body) {
		t.Fatalf("cwd = %v", seen["cwd"])
	}
}

// Stopping the bot: TestStopBot.

func helperProcess(t *testing.T, mode string) *botProcess {
	t.Helper()
	t.Setenv(helperEnv, mode)
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	proc, err := startBot(exec.Command(self)) //nolint:gosec,noctx // the test binary itself
	if err != nil {
		t.Fatal(err)
	}
	return proc
}

func TestStopBotTerminatesARunningBotPromptly(t *testing.T) {
	proc := helperProcess(t, "sleep")
	started := time.Now()
	stopBot(proc)
	if !proc.exited() {
		t.Fatal("the bot is still running")
	}
	if elapsed := time.Since(started); elapsed > 5*time.Second {
		t.Fatalf("took %s", elapsed)
	}
}

func TestStopBotLeavesAnExitedBotAlone(t *testing.T) {
	proc := helperProcess(t, "argv")
	<-proc.done
	stopBot(proc)
	if proc.exitCode() != 0 {
		t.Fatalf("exit code = %d", proc.exitCode())
	}
}

// Simulations in a manifest: TestManifestSimulations.

const (
	simulationFile = `
name: %[1]s
simulator: {service: openai}
scenarios:
  - name: %[1]s
    persona: "A caller."
    goal: "Get it done."
    success: "it got done"
    runs: %[2]d
`
	scriptFile = "name: %[1]s\nscenarios:\n  - name: %[1]s\n    turns: []\n"
)

func simulationsDir(t *testing.T) string {
	t.Helper()
	base := manifestDir(t)
	writeIn(t, base, "scenarios/book.yaml", fmt.Sprintf(simulationFile, "book", 3))
	writeIn(t, base, "scenarios/once.yaml", fmt.Sprintf(simulationFile, "once", 1))
	writeIn(t, base, "scenarios/greet.yaml", fmt.Sprintf(scriptFile, "greet"))
	return base
}

func scenarioNames(runs []*SuiteRun) []string {
	var names []string
	for _, r := range runs {
		names = append(names, r.Scenario)
	}
	return names
}

func TestManifestTheFileSaysWhichKindAScenarioIsAndHowOftenItRuns(t *testing.T) {
	base := simulationsDir(t)
	m := mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [greet, book, once]\n", ManifestOverrides{})
	byName := map[string][]*SuiteRun{}
	for _, r := range m.Runs {
		byName[r.Scenario] = append(byName[r.Scenario], r)
	}
	attempts := func(name string) []int {
		var out []int
		for _, r := range byName[name] {
			out = append(out, r.Attempt)
		}
		return out
	}
	if !slices.Equal(attempts("book/book"), []int{1, 2, 3}) || !slices.Equal(attempts("once/once"), []int{1}) ||
		!slices.Equal(attempts("greet/greet"), []int{1}) {
		t.Fatalf("attempts book %v once %v greet %v", attempts("book/book"), attempts("once/once"), attempts("greet/greet"))
	}
	book := byName["book/book"][0]
	if book.Kind != KindSimulation || book.Attempts != 3 || book.Sweep ||
		book.ScenarioPath != filepath.Join(base, "scenarios", "book.yaml") {
		t.Fatalf("book = %+v", book)
	}
	greet := byName["greet/greet"][0]
	if greet.Kind != KindScript || greet.Attempts != 1 || greet.Sweep {
		t.Fatalf("greet = %+v", greet)
	}
	// Attempt-major: every scenario's first attempt precedes any second one.
	var order []int
	for _, r := range m.Runs {
		order = append(order, r.Attempt)
	}
	if !slices.Equal(order, []int{1, 1, 1, 2, 3}) {
		t.Fatalf("attempts in order = %v", order)
	}
}

func TestManifestARubricEntryHasNoRun(t *testing.T) {
	base := simulationsDir(t)
	writeIn(t, base, "scenarios/mixed.yaml",
		fmt.Sprintf(scriptFile, "mixed")+"  - name: production\n    success: it got done\n")
	m := mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [mixed]\n", ManifestOverrides{})
	if got := scenarioNames(m.Runs); !slices.Equal(got, []string{"mixed/mixed"}) {
		t.Fatalf("runs = %v", got)
	}
}

func TestManifestRepeatOverridesASimulationsRuns(t *testing.T) {
	m := mustManifest(t, simulationsDir(t), "suite:\n  - bot: bot\n    scenarios: [book]\n",
		ManifestOverrides{Repeat: new(2)})
	if len(m.Runs) != 2 || m.Runs[0].Attempt != 1 || m.Runs[1].Attempt != 2 || m.Runs[0].Attempts != 2 {
		t.Fatalf("runs = %+v", m.Runs)
	}
	// A repeat makes the suite a measurement.
	for _, r := range m.Runs {
		if !r.Sweep {
			t.Fatal("a repeated run is not a sweep")
		}
	}
}

// Set on the command line or in the manifest, 1 means one run, not the file's
// three.
func TestManifestARepeatOfOneIsAnOverrideToo(t *testing.T) {
	base := simulationsDir(t)
	m := mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [book]\n", ManifestOverrides{Repeat: new(1)})
	if len(m.Runs) != 1 {
		t.Fatalf("got %d runs", len(m.Runs))
	}
	m = mustManifest(t, base, "repeat: 1\nsuite:\n  - bot: bot\n    scenarios: [book]\n", ManifestOverrides{})
	if len(m.Runs) != 1 {
		t.Fatalf("got %d runs", len(m.Runs))
	}
}

func TestManifestAFolderInANameStaysUnderTheScenariosDir(t *testing.T) {
	base := simulationsDir(t)
	writeIn(t, base, "scenarios/scripted/greet.yaml", fmt.Sprintf(scriptFile, "greet"))
	run := mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [scripted/greet]\n", ManifestOverrides{}).Runs[0]
	if run.Scenario != "greet/greet" || run.ScenarioPath != filepath.Join(base, "scenarios", "scripted", "greet.yaml") {
		t.Fatalf("run = %q at %q", run.Scenario, run.ScenarioPath)
	}
}

func TestSuiteFiltersByKind(t *testing.T) {
	m := mustManifest(t, simulationsDir(t), "suite:\n  - bot: bot\n    scenarios: [greet, book]\n", ManifestOverrides{})
	runs := NewSuite(m).Filter("", "", KindSimulation)
	if len(runs) != 3 || slices.ContainsFunc(runs, func(r *SuiteRun) bool { return r.Scenario != "book/book" }) {
		t.Fatalf("runs = %v", scenarioNames(runs))
	}
}

// Its kind cannot be read, so it runs once as a scenario and reports the error.
func TestManifestAMissingScenarioStillGetsARun(t *testing.T) {
	m := mustManifest(t, simulationsDir(t), "suite:\n  - bot: bot\n    scenarios: [nope]\n", ManifestOverrides{})
	if len(m.Runs) != 1 {
		t.Fatalf("got %d runs", len(m.Runs))
	}
	if r := m.Runs[0]; r.Scenario != "nope" || r.Kind != KindScript || r.attempts() != 1 {
		t.Fatalf("run = %+v", r)
	}
}

// Each scenario runs under its own name and as its own kind.
func TestManifestAFileContributesARunPerScenario(t *testing.T) {
	base := simulationsDir(t)
	writeIn(t, base, "scenarios/mixed.yaml", "name: mixed\nscenarios:\n"+
		"  - name: hi\n    turns: []\n"+
		"  - name: call\n    persona: p\n    goal: g\n    success: s\n    runs: 2\n")
	m := mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [greet, mixed]\n", ManifestOverrides{})
	var first []*SuiteRun
	for _, r := range m.Runs {
		if r.Attempt == 1 {
			first = append(first, r)
		}
	}
	if got := scenarioNames(first); !slices.Equal(got, []string{"greet/greet", "mixed/hi", "mixed/call"}) {
		t.Fatalf("runs = %v", got)
	}
	var kinds []Kind
	var attempts []int
	for _, r := range first {
		kinds, attempts = append(kinds, r.Kind), append(attempts, r.Attempts)
	}
	if !slices.Equal(kinds, []Kind{KindScript, KindScript, KindSimulation}) || !slices.Equal(attempts, []int{1, 1, 2}) {
		t.Fatalf("kinds %v attempts %v", kinds, attempts)
	}
	for _, r := range first[1:] {
		if filepath.Base(r.ScenarioPath) != "mixed.yaml" {
			t.Fatalf("path = %q", r.ScenarioPath)
		}
	}
	// A scenario's name is a file stem without the slash.
	if first[1].Stem() != "mixed__hi" {
		t.Fatalf("stem = %q", first[1].Stem())
	}
}

func TestSuiteFiltersByScenarioNameOrEitherHalf(t *testing.T) {
	base := simulationsDir(t)
	writeIn(t, base, "scenarios/mixed.yaml",
		"name: mixed\nscenarios:\n  - name: hi\n    turns: []\n  - name: bye\n    turns: []\n")
	m := mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [greet, mixed]\n", ManifestOverrides{})
	for scenario, want := range map[string][]string{
		"mixed/hi": {"mixed/hi"}, "hi": {"mixed/hi"}, "mixed": {"mixed/hi", "mixed/bye"}, "greet": {"greet/greet"},
	} {
		if got := scenarioNames(NewSuite(m).Filter("", scenario, "")); !slices.Equal(got, want) {
			t.Errorf("filter %q = %v, want %v", scenario, got, want)
		}
	}
}

func TestSuiteRunLoadsItsOwnScenario(t *testing.T) {
	base := simulationsDir(t)
	writeIn(t, base, "scenarios/mixed.yaml",
		"name: mixed\nscenarios:\n  - name: hi\n    turns: []\n  - name: bye\n    turns: []\n")
	m := mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [mixed]\n", ManifestOverrides{})
	// Built from a loaded file, the runs hold their scenarios already.
	var names []string
	for _, r := range m.Runs {
		if r.Loaded == nil {
			t.Fatal("a run was built without its scenario")
		}
		loaded, err := r.Load()
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, loaded.ScenarioName())
	}
	if !slices.Equal(names, []string{"mixed/hi", "mixed/bye"}) {
		t.Fatalf("names = %v", names)
	}
	// A run that only knows its file and name reads the file.
	path := m.Runs[0].ScenarioPath
	loaded, err := (&SuiteRun{Bot: "b", Scenario: "mixed/bye", ScenarioPath: path}).Load()
	if err != nil || loaded.ScenarioName() != "mixed/bye" {
		t.Fatalf("Load = %v, %v", loaded, err)
	}
	_, err = (&SuiteRun{Bot: "b", Scenario: "mixed/nope", ScenarioPath: path}).Load()
	if err == nil || !strings.Contains(err.Error(), "no scenario called 'mixed/nope'") {
		t.Fatalf("err = %v", err)
	}
}

func TestManifestAnEntryNameLabelsItsRuns(t *testing.T) {
	m := mustManifest(t, simulationsDir(t), "suite:\n"+
		"  - bot: bot\n    name: openai/gpt-4o-mini\n    scenarios: [greet]\n"+
		"  - bot: bot\n    name: groq/llama\n    scenarios: [greet]\n"+
		"  - bot: other\n    scenarios: [greet]\n", ManifestOverrides{})
	var labels, names, bots []string
	for _, r := range m.Runs {
		labels, names, bots = append(labels, r.Label()), append(names, r.Name), append(bots, r.Bot)
	}
	if !slices.Equal(labels, []string{"openai/gpt-4o-mini", "groq/llama", "other"}) ||
		!slices.Equal(names, []string{"openai/gpt-4o-mini", "groq/llama", ""}) ||
		!slices.Equal(bots, []string{"bot", "bot", "other"}) {
		t.Fatalf("labels %v names %v bots %v", labels, names, bots)
	}
	// The pattern filter sees the name and the bot path alike.
	if got := NewSuite(m).Filter("groq", "", ""); len(got) != 1 || got[0].Label() != "groq/llama" {
		t.Fatalf("filter groq = %v", got)
	}
	if got := NewSuite(m).Filter("bot", "", ""); len(got) != 2 {
		t.Fatalf("filter bot = %d runs", len(got))
	}
}

func TestManifestTwoEntriesMayNotRunAScenarioUnderOneLabel(t *testing.T) {
	base := simulationsDir(t)
	_, err := loadManifestText(t, base,
		"suite:\n  - bot: bot\n    scenarios: [greet]\n  - bot: bot\n    scenarios: [greet]\n", ManifestOverrides{})
	if err == nil || !strings.Contains(err.Error(), "'bot' runs 'greet/greet' twice") {
		t.Fatalf("err = %v", err)
	}
	// The same bot on different scenarios is fine, as is a named second entry.
	mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [greet]\n  - bot: bot\n    scenarios: [book]\n",
		ManifestOverrides{})
	mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [greet]\n"+
		"  - bot: bot\n    name: again\n    scenarios: [greet]\n", ManifestOverrides{})
}

func TestManifestAnEntryNameMustBeANonEmptyString(t *testing.T) {
	for _, bad := range []string{`""`, "3", "[a]"} {
		_, err := loadManifestText(t, simulationsDir(t),
			"suite:\n  - bot: bot\n    name: "+bad+"\n    scenarios: [greet]\n", ManifestOverrides{})
		if err == nil || !strings.Contains(err.Error(), "'name:'") {
			t.Errorf("%s: err = %v", bad, err)
		}
	}
}

func TestManifestAFlatFileStillLoadsAndWarns(t *testing.T) {
	base := simulationsDir(t)
	writeIn(t, base, "scenarios/old.yaml", "name: old\nturns: []\n")
	var m *Manifest
	seen := warnings(t, func() {
		m = mustManifest(t, base, "suite:\n  - bot: bot\n    scenarios: [old]\n", ManifestOverrides{})
	})
	if len(seen) == 0 {
		t.Fatal("no deprecation warning")
	}
	if got := scenarioNames(m.Runs); !slices.Equal(got, []string{"old"}) {
		t.Fatalf("runs = %v", got)
	}
}

// yesClassifier answers every question yes, surely: the judge a suite test's
// simulation is judged by.
type yesClassifier struct{ *classifier.Base }

func (*yesClassifier) Model() string { return "" }

func (*yesClassifier) AskQuestions(
	_ context.Context, _ any, questions []classifier.Named[classifier.Question], _ []classifier.Input,
) (map[string]classifier.Result, *frames.LLMTokenUsage, error) {
	results := map[string]classifier.Result{}
	for _, q := range questions {
		switch q.Question.(type) {
		case classifier.YesNoQuestion:
			results[q.Name] = classifier.YesNoResult{Probability: 0.99}
		case classifier.ChoiceQuestion:
			results[q.Name] = classifier.ChoiceResult{
				Choice: VerdictYes, Probabilities: map[string]float64{VerdictYes: 0.99}, Confidence: 0.99,
			}
		}
	}
	return results, nil, nil
}

func init() {
	RegisterFactory("evaltest.suite_persona", func(map[string]any) (any, error) {
		persona, _ := scriptedPersona(map[string]any{
			"Hello! How can I help?": "What is the capital of Germany?",
			"Berlin.":                endCallReply{"success": true, "reason": "I learned it"},
		})
		return persona, nil
	})
	RegisterFactory("evaltest.yes_classifier", func(map[string]any) (any, error) {
		c := &yesClassifier{}
		c.Base = classifier.NewBase("YesClassifier", "", c)
		return c, nil
	})
}

// A simulation run in a suite plays the persona against the bot, records its
// result, and keeps its conversation beside its logs.
func TestSuiteASimulationRunRecordsItsResultAndItsConversation(t *testing.T) {
	bot := newFakeRTVIServer(t)
	bot.greeting = reply("Hello! How can I help?")
	bot.onText("What is the capital of Germany?", reply("Berlin.")...)
	base := manifestDir(t)
	writeIn(t, base, "scenarios/capital.yaml", `
name: capital
simulator: {factory: evaltest.suite_persona}
judge:
  eval: {factory: evaltest.yes_classifier, explainer: false}
scenarios:
  - name: capital
    persona: "A curious traveler."
    goal: "Learn the capital of Germany."
    success: "the bot named Berlin"
    max_turns: 5
    max_duration_s: 10
`)
	m := mustManifest(t, base, "suite:\n  - bot_url: "+bot.url()+"\n    scenarios: [capital]\n", ManifestOverrides{})
	suite := NewSuite(m)
	logs := filepath.Join(base, "logs")
	opts := SuiteRunOptions{ResultsPath: filepath.Join(base, "results.jsonl"), Debug: true}
	if err := suite.Run(t.Context(), logs, opts); err != nil {
		t.Fatal(err)
	}
	run := suite.Runs[0]
	if run.Status != RunDone || run.Error != "" || run.SimulationResult == nil {
		t.Fatalf("run = %+v", run)
	}
	if res := run.SimulationResult; !res.Passed() || res.EndedBy != SimulationEndedByEndCall || res.Turns != 1 {
		t.Fatalf("result = %+v\n%s", res, strings.Join(res.DebugLog, "\n"))
	}
	record := readRecord(t, filepath.Join(base, "results.jsonl"))
	if record["kind"] != "simulation" || record["passed"] != true || record["ended_by"] != "end_call" {
		t.Fatalf("record = %v", record)
	}
	artifacts, _ := record["artifacts"].(map[string]any)
	path, _ := artifacts["conversation"].(string)
	data, err := os.ReadFile(path) //nolint:gosec // a test file
	if err != nil {
		t.Fatalf("conversation file: %v (artifacts %v)", err, artifacts)
	}
	var conversation []map[string]any
	if err := json.Unmarshal(data, &conversation); err != nil {
		t.Fatal(err)
	}
	want := []map[string]any{
		{"role": "assistant", "content": "Hello! How can I help?"},
		{"role": "user", "content": "What is the capital of Germany?"},
		{"role": "assistant", "content": "Berlin."},
	}
	if !reflect.DeepEqual(conversation, want) {
		t.Fatalf("conversation = %v", conversation)
	}
	// The persona LLM's own logs are filed under their heading.
	debugLogs, _ := filepath.Glob(filepath.Join(logs, "*.debug.log"))
	if len(debugLogs) != 1 {
		t.Fatalf("debug logs = %v", debugLogs)
	}
	content, err := os.ReadFile(debugLogs[0])
	if err != nil {
		t.Fatal(err)
	}
	heading := "===== persona LLM logs: capital/capital ====="
	_, section, found := strings.Cut(string(content), heading)
	section, _, _ = strings.Cut(section, "=====")
	if !found || !strings.Contains(section, "scripted persona: answering Hello! How can I help?") {
		t.Fatalf("debug log lacks the persona's section:\n%s", content)
	}
}

// A simulation whose bot cannot be reached is a run that failed to connect,
// not an error of the suite.
func TestSuiteASimulationThatCannotConnectIsAFailedRun(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "forbidden", http.StatusForbidden)
	}))
	defer srv.Close()
	base := simulationsDir(t)
	m := mustManifest(t, base, "suite:\n  - bot_url: "+wsURL(srv.URL)+"\n    scenarios: [once]\n",
		ManifestOverrides{})
	suite := NewSuite(m)
	logs := filepath.Join(base, "logs")
	opts := SuiteRunOptions{ResultsPath: filepath.Join(base, "results.jsonl")}
	if err := suite.Run(t.Context(), logs, opts); err != nil {
		t.Fatal(err)
	}
	run := suite.Runs[0]
	if run.Status != RunDone || run.Error != "" || run.SimulationResult == nil ||
		run.SimulationResult.FailureKind != FailureConnectFailed {
		t.Fatalf("run = %+v", run)
	}
	record := readRecord(t, filepath.Join(base, "results.jsonl"))
	if record["kind"] != "simulation" || record["passed"] != false || record["ended_by"] != "error" ||
		record["failure_kind"] != FailureConnectFailed {
		t.Fatalf("record = %v", record)
	}
}

// The results records: TestScenarioRecords and TestSimulationRecords.

func readRecord(t *testing.T, path string) map[string]any {
	t.Helper()
	data, err := os.ReadFile(path) //nolint:gosec // a test file
	if err != nil {
		t.Fatal(err)
	}
	var record map[string]any
	if err := json.Unmarshal(data, &record); err != nil {
		t.Fatalf("%s: %v", data, err)
	}
	return record
}

func TestResultsRecordKeepsWhatEachExpectationMatched(t *testing.T) {
	base := t.TempDir()
	run := &SuiteRun{
		Bot: "voice/x", Name: "openai/gpt-4o-mini", Scenario: "greet", ScenarioPath: filepath.Join(base, "greet.yaml"),
		Status: RunDone,
		ScriptResult: &ScriptResult{
			ScenarioName: "greet",
			Turns: []ScriptTurnResult{{
				TurnIndex: 0, Status: TurnPassed, DurationMS: 50,
				Expectations: []ExpectationResult{
					{ExpectationIndex: 0, EventName: "llm_marker", Passed: true, Matched: "◐"},
					{ExpectationIndex: 1, EventName: "llm_response", Passed: true, Matched: "Hi"},
				},
			}},
			DurationMS: 60,
		},
	}
	appendResult(filepath.Join(base, "results.jsonl"), run, "voice_x__greet", base, "")
	record := readRecord(t, filepath.Join(base, "results.jsonl"))
	if record["passed"] != true || record["bot"] != "voice/x" || record["name"] != "openai/gpt-4o-mini" {
		t.Fatalf("record = %v", record)
	}
	turn := record["turns"].([]any)[0].(map[string]any) //nolint:forcetypeassert // the record's shape
	expectations, _ := json.Marshal(turn["expectations"])
	want := `[{"event_name":"llm_marker","expectation_index":0,"matched":"◐","passed":true},` +
		`{"event_name":"llm_response","expectation_index":1,"matched":"Hi","passed":true}]`
	if string(expectations) != want {
		t.Fatalf("expectations = %s", expectations)
	}
	if _, ok := record["events_seen"]; ok {
		t.Fatal("a passed run carries events_seen")
	}
	// Exactly the reference's keys.
	keys := slices.Sorted(func(yield func(string) bool) {
		for k := range record {
			if !yield(k) {
				return
			}
		}
	})
	wantKeys := []string{
		"artifacts", "attempt", "bot", "duration_ms", "error", "failures", "kind", "name", "passed", "scenario",
		"skipped", "turns",
	}
	if !slices.Equal(keys, wantKeys) {
		t.Fatalf("keys = %v", keys)
	}
}

func TestResultsRecordSavesAFailedSimulationsKind(t *testing.T) {
	base := t.TempDir()
	run := &SuiteRun{
		Bot: "flows/x", Scenario: "book", ScenarioPath: filepath.Join(base, "book.yaml"), Kind: KindSimulation,
		Status: RunDone,
		SimulationResult: &SimulationResult{
			SimulationName: "book", Reason: "server rejected WebSocket connection: HTTP 403",
			Error: "server rejected WebSocket connection: HTTP 403", FailureKind: FailureConnectFailed,
		},
	}
	appendResult(filepath.Join(base, "results.jsonl"), run, "flows_x__book", base, "")
	if record := readRecord(t, filepath.Join(base, "results.jsonl")); record["failure_kind"] != "connect_failed" {
		t.Fatalf("record = %v", record)
	}
}

func TestResultsRecordForASimulationRun(t *testing.T) {
	base := t.TempDir()
	duration := 1234
	run := &SuiteRun{
		Bot: "flows/x", Scenario: "book", ScenarioPath: filepath.Join(base, "book.yaml"), Kind: KindSimulation,
		Attempts: 3, Attempt: 2, Status: RunDone, DurationMS: &duration,
		SimulationResult: &SimulationResult{
			SimulationName: "book", Reason: "no table", Turns: 3, EndedBy: SimulationEndedByBot,
			EventsSeen: []map[string]any{{"type": "llm_started"}}, Success: "the bot booked a table",
			Metrics: []SimulationMetricScore{{Name: "politeness", Score: new(1.0), Passed: true, Criterion: "stays polite"}},
		},
	}
	appendResult(filepath.Join(base, "results.jsonl"), run, "flows_x__book__002", base, "")
	record := readRecord(t, filepath.Join(base, "results.jsonl"))
	for key, want := range map[string]any{
		"scenario": "book", "name": "flows/x", "kind": "simulation", "attempt": 2.0, "passed": false,
		"succeeded": false, "ended_by": "bot", "reason": "no table", "success": "the bot booked a table",
		"duration_ms": 1234.0,
	} {
		if record[key] != want {
			t.Errorf("%s = %v, want %v", key, record[key], want)
		}
	}
	metric := record["metrics"].([]any)[0].(map[string]any) //nolint:forcetypeassert // the record's shape
	if metric["criterion"] != "stays polite" {
		t.Errorf("criterion = %v", metric["criterion"])
	}
	if seen, _ := json.Marshal(record["events_seen"]); string(seen) != `[{"type":"llm_started"}]` {
		t.Errorf("events_seen = %s", seen)
	}
}

// TestSimulationConversationFile: a finished simulation writes its
// conversation, and the record points to it.
func TestSuiteAFinishedSimulationWritesItsConversation(t *testing.T) {
	base := t.TempDir()
	conversation := []map[string]any{
		{"role": "user", "content": "A table at six."},
		{
			"role": "tool_call", "name": "book", "arguments": map[string]any{"time": "6"},
			"cancelled": false, //nolint:misspell // the conversation file's key
		},
		{"role": "assistant", "content": "Booked."},
	}
	run := &SuiteRun{
		Bot: "flows/x", Scenario: "book", ScenarioPath: filepath.Join(base, "book.yaml"), Kind: KindSimulation,
		SimulationResult: &SimulationResult{SimulationName: "book", Succeeded: true, Conversation: conversation},
	}
	suite := suiteOfRuns([]*SuiteRun{run}, 1)
	files := runFilesFor(run, base, "")
	setup := &suiteRunSetup{resultsPath: filepath.Join(base, "results.jsonl"), logsDir: base}
	suite.finish(t.Context(), run, files, nil, setup)
	got, _ := os.ReadFile(files.conversation) //nolint:gosec // a test file
	want, _ := json.MarshalIndent(conversation, "", "  ")
	if string(got) != string(want)+"\n" {
		t.Fatalf("conversation = %s", got)
	}
	record := readRecord(t, filepath.Join(base, "results.jsonl"))
	artifacts := record["artifacts"].(map[string]any) //nolint:forcetypeassert // the record's shape
	if artifacts["conversation"] != files.conversation {
		t.Fatalf("artifacts = %v", artifacts)
	}
}

// The queues: TestEntryQueues.

func queueRun(label, scenario string, entry, attempt, concurrency int) *SuiteRun {
	return &SuiteRun{
		Bot: "bot", Name: label, Scenario: scenario, BotPath: "bot", ScenarioPath: scenario + ".yaml",
		Entry: entry, Attempt: attempt, Concurrency: concurrency,
	}
}

func TestEntryQueuesOnePerEntryInManifestOrder(t *testing.T) {
	queues := entryQueues([]*SuiteRun{
		queueRun("a", "s1", 0, 1, 0), queueRun("a", "s2", 0, 1, 0), queueRun("b", "s1", 1, 1, 0),
		queueRun("a", "s3", 2, 1, 0), queueRun("c", "s1", 3, 1, 0),
	})
	// A label's second entry is a queue of its own, in its manifest place.
	var got [][]string
	var entries, caps []int
	for _, q := range queues {
		var items []string
		for _, r := range q.runs {
			items = append(items, r.Label()+" "+r.Scenario)
		}
		got, entries, caps = append(got, items), append(entries, q.entry), append(caps, q.cap)
	}
	want := [][]string{{"a s1", "a s2"}, {"b s1"}, {"a s3"}, {"c s1"}}
	if !slices.EqualFunc(got, want, slices.Equal) || !slices.Equal(entries, []int{0, 1, 2, 3}) ||
		!slices.Equal(caps, []int{0, 0, 0, 0}) {
		t.Fatalf("queues %v entries %v caps %v", got, entries, caps)
	}
}

func TestEntryQueuesAttemptsFormTheirOwnQueuesAttemptMajor(t *testing.T) {
	queues := entryQueues([]*SuiteRun{
		queueRun("a", "s1", 0, 1, 0), queueRun("a", "s2", 0, 1, 0), queueRun("b", "s1", 1, 1, 0),
		queueRun("a", "s1", 0, 2, 0), queueRun("a", "s2", 0, 2, 0), queueRun("b", "s1", 1, 2, 0),
	})
	var got [][]string
	for _, q := range queues {
		var items []string
		for _, r := range q.runs {
			items = append(items, fmt.Sprintf("%s %s %d", r.Label(), r.Scenario, r.Attempt))
		}
		got = append(got, items)
	}
	want := [][]string{{"a s1 1", "a s2 1"}, {"b s1 1"}, {"a s1 2", "a s2 2"}, {"b s1 2"}}
	if !slices.EqualFunc(got, want, slices.Equal) {
		t.Fatalf("queues %v", got)
	}
}

func TestEntryQueuesEachEntryHasItsOwnCapAndNoneWithoutOne(t *testing.T) {
	var caps []int
	for _, q := range entryQueues([]*SuiteRun{
		queueRun("a", "s1", 0, 1, 3), queueRun("a", "s2", 1, 1, 2), queueRun("b", "s1", 2, 1, 0),
	}) {
		caps = append(caps, q.cap)
	}
	if !slices.Equal(caps, []int{3, 2, 0}) {
		t.Fatalf("caps %v", caps)
	}
}

// The handler a caller adds sees the suite's runs change, typed.
func TestSuiteUpdateEventIsTyped(t *testing.T) {
	dir := t.TempDir()
	suite := suiteOfRuns([]*SuiteRun{{
		Bot: "missing", Scenario: "none", ScenarioPath: filepath.Join(dir, "none.yaml"), BotPath: filepath.Join(dir, "x"),
	}}, 1)
	var statuses []string
	events.On(&suite.Registry, EventUpdate, func(_ context.Context, run *SuiteRun) {
		statuses = append(statuses, run.Status)
	})
	if err := suite.Run(t.Context(), dir, SuiteRunOptions{}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(statuses, []string{RunRunning, RunDone}) {
		t.Fatalf("statuses = %v", statuses)
	}
}
