package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/utils/events"
	"github.com/spf13/cobra"
)

// `jargo eval suite`: build and spawn the bots in a manifest and run their
// scenarios. The progress is plain text, a line per run as it finishes.

// Suite command errors.
var (
	// errNoRunsMatch is a filter that leaves no run.
	errNoRunsMatch = errors.New("no runs match")
	// errKind is a --kind of no kind.
	errKind = errors.New("--kind must be script or simulation")
)

// The display verdicts of a run.
const (
	verdictPassed  = "passed"
	verdictFailed  = "failed"
	verdictSkipped = "skipped"
	verdictError   = "error"
)

// evalGlyphs are each verdict's glyph and ANSI color.
//
//nolint:gochecknoglobals // fixed table, read only
var evalGlyphs = map[string][2]string{
	verdictPassed:  {"✓", "32"},
	verdictFailed:  {"✗", "31"},
	verdictSkipped: {"⊘", "33"},
	verdictError:   {"✗", "31"},
}

// suiteFlags are what `jargo eval suite` is given.
type suiteFlags struct {
	pattern, scenario, kind, name  string
	runsDir, botsDir, scenariosDir string
	concurrency, repeat, basePort  int
	cacheDir, spawn                string
	noCache, audio, debug          bool
	timeout                        int
}

// evalSuiteCmd is `jargo eval suite`: run a manifest's bots and scenarios.
func evalSuiteCmd() *cobra.Command {
	var f suiteFlags
	cmd := &cobra.Command{
		Use:   "suite <manifest.yaml>",
		Short: "Build and spawn the bots in a manifest and run their scenarios concurrently",
		Long: "Build and spawn the bots in a manifest and run their scenarios concurrently.\n\n" +
			"Everything except the suite: list can be set in the manifest or overridden\n" +
			"here (the command line wins), so a manifest can be just a suite: list. A\n" +
			"scenario file is scripted or a simulation, and the file says which; a\n" +
			"simulation runs as many times as its file says, and every run must pass.\n\n" +
			"Each bot is a Go main package, built once with `go build` (honoring GOFLAGS)\n" +
			"into <runs-dir>/<name>/bin, and spawned with the spawn template.",
		Args: cobra.ExactArgs(1),
	}
	fl := cmd.Flags()
	fl.StringVarP(&f.pattern, "pattern", "p", "", "only entries whose name or bot path contains this")
	fl.StringVarP(&f.scenario, "scenario", "s", "", "only this scenario name")
	fl.StringVarP(&f.kind, "kind", "k", "", "only scenarios of this kind (script or simulation)")
	fl.StringVarP(&f.name, "name", "n", "", "run subdir name under runs_dir (default a timestamp)")
	fl.StringVar(&f.runsDir, "runs-dir", "", "output base, overriding the manifest's runs_dir (a <name>/ subdir "+
		"with logs/ and recordings/ is created under it; default eval-runs)")
	fl.StringVar(&f.botsDir, "bots-dir", "", "override manifest bots_dir")
	fl.StringVar(&f.scenariosDir, "scenarios-dir", "", "override manifest scenarios_dir")
	fl.IntVarP(&f.concurrency, "concurrency", "c", 0, "override manifest concurrency")
	fl.IntVarP(&f.repeat, "repeat", "r", 0, "run each (bot, scenario) this many times, to measure flakiness, "+
		"and each simulation this many times instead of its own runs (1 included); a repeated sweep reports "+
		"rates and exits 0. Attempts run attempt-major across bots and each writes its own logs")
	fl.IntVar(&f.basePort, "base-port", 0, "override manifest base_port")
	fl.StringVar(&f.cacheDir, "cache-dir", "", "override manifest cache_dir")
	fl.BoolVar(&f.noCache, "no-cache", false,
		"disable the user-audio cache: synthesize every turn again (no reads or writes)")
	fl.IntVarP(&f.timeout, "timeout", "t", eval.DefaultEventTimeoutMS/1000,
		"default per-expectation timeout in seconds (for expectations without their own within_ms)")
	fl.StringVar(&f.spawn, "spawn", "",
		"override manifest spawn template ({bot} is the built binary, {port} the run's port)")
	fl.BoolVarP(&f.audio, "audio", "a", false, "record conversation audio")
	fl.BoolVarP(&f.debug, "debug", "d", false, "also save <run>.debug.log with the harness's full per-pipeline logs")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		return runSuite(cmd.Context(), cmd, args[0], f)
	}
	return cmd
}

// suiteOverrides are the manifest overrides the flags set.
func suiteOverrides(cmd *cobra.Command, f suiteFlags) eval.ManifestOverrides {
	o := eval.ManifestOverrides{
		BotsDir: f.botsDir, ScenariosDir: f.scenariosDir, RunsDir: f.runsDir, Spawn: f.spawn, CacheDir: f.cacheDir,
	}
	changed := cmd.Flags().Changed
	if changed("concurrency") {
		o.Concurrency = &f.concurrency
	}
	if changed("repeat") {
		o.Repeat = &f.repeat
	}
	if changed("base-port") {
		o.BasePort = &f.basePort
	}
	if f.audio {
		record := true
		o.Record = &record
	}
	return o
}

// runSuite loads the manifest, runs the selected runs, and reports them.
func runSuite(ctx context.Context, cmd *cobra.Command, manifestPath string, f suiteFlags) error {
	kind := eval.Kind(f.kind)
	if kind != "" && kind != eval.KindScript && kind != eval.KindSimulation {
		return fmt.Errorf("%w, not %q", errKind, f.kind)
	}
	manifest, err := eval.LoadManifest(manifestPath, suiteOverrides(cmd, f))
	if err != nil {
		return err
	}
	suite := eval.NewSuite(manifest)
	runs := suite.Filter(f.pattern, f.scenario, kind)
	out := cmd.OutOrStdout()
	if len(runs) == 0 {
		_, _ = fmt.Fprintln(out, "No runs match.")
		return errNoRunsMatch
	}

	// A per-run subdir named by --name (default a timestamp) holds this run's
	// logs and recordings, under the runs_dir.
	base := manifest.RunsDir
	if base == "" {
		base = "eval-runs"
	}
	name := f.name
	if name == "" {
		name = time.Now().Format("20060102_150405")
	}
	runDir := filepath.Join(base, name)
	logsDir := filepath.Join(runDir, "logs")
	var recordDir string
	if manifest.Record {
		recordDir = filepath.Join(runDir, "recordings")
	}

	d := display{out: out, color: isTerminal(out)}
	d.scenarioConfigs(runs)
	d.runSettings(runs, manifest.Concurrency, manifest.Record)

	grouped := slices.ContainsFunc(runs, func(r *eval.SuiteRun) bool { return r.Attempts > 1 })
	var mu sync.Mutex
	events.On(&suite.Registry, eval.EventUpdate, func(_ context.Context, run *eval.SuiteRun) {
		mu.Lock()
		defer mu.Unlock()
		d.evalLine(run, grouped)
	})
	useCache := !f.noCache
	started := time.Now()
	if err := suite.Run(ctx, logsDir, eval.SuiteRunOptions{
		RecordDir:   recordDir,
		ResultsPath: filepath.Join(runDir, "results.jsonl"),
		Debug:       f.debug,
		Params:      eval.SessionParams{DefaultTimeoutMS: f.timeout * 1000, UseCache: &useCache},
	}); err != nil {
		return err
	}
	if abs, err := filepath.Abs(runDir); err == nil {
		runDir = abs
	}
	return d.finalize(runs, runDir, time.Since(started))
}

// isTerminal reports whether w is a terminal, where the output is colored.
func isTerminal(w io.Writer) bool {
	f, ok := w.(*os.File)
	if !ok {
		return false
	}
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

// display prints a suite's progress and outcome as plain text.
type display struct {
	out   io.Writer
	color bool
}

func (d display) printf(format string, a ...any) { _, _ = fmt.Fprintf(d.out, format, a...) }

// paint is text in an ANSI color when the output is a terminal.
func (d display) paint(text, code string) string {
	if !d.color {
		return text
	}
	return "\033[" + code + "m" + text + "\033[0m"
}

func (d display) dim(s string) string   { return d.paint(s, "2") }
func (d display) red(s string) string   { return d.paint(s, "31") }
func (d display) green(s string) string { return d.paint(s, "32") }

// runVerdict collapses a run into a display verdict.
func runVerdict(r *eval.SuiteRun) string {
	switch {
	case r.Status != eval.RunDone:
		if r.Status == "" {
			return eval.RunPending
		}
		return r.Status
	case r.Error != "":
		return verdictError
	case r.ScriptResult != nil:
		if r.ScriptResult.Skipped != "" {
			return verdictSkipped
		}
		if r.ScriptResult.Passed() {
			return verdictPassed
		}
		return verdictFailed
	case r.SimulationResult != nil:
		if r.SimulationResult.Error != "" {
			return verdictError
		}
		if r.SimulationResult.Passed() {
			return verdictPassed
		}
		return verdictFailed
	}
	return verdictError
}

// runTally is the run's detail beside its verdict, or "": a scenario's turn
// tally (see turnTally), or how a simulation ended.
func runTally(r *eval.SuiteRun) string {
	switch {
	case r.SimulationResult != nil:
		return r.SimulationResult.EndedBy
	case r.ScriptResult != nil:
		return turnTally(*r.ScriptResult)
	}
	return ""
}

// fmtDuration is a human-friendly elapsed time, e.g. 12.3s or 2m 04s: tenths
// under a minute, then minutes and seconds, deciding on the rounded value so
// nothing ever reads 60.0s.
func fmtDuration(seconds float64) string {
	tenths := math.Round(seconds*10) / 10
	if tenths < 60 {
		return fmt.Sprintf("%.1fs", tenths)
	}
	total := int(math.Round(seconds))
	return fmt.Sprintf("%dm %02ds", total/60, total%60)
}

// passRate is "23/50 (46%)" over the attempts that finished, or "-" if none
// have.
func passRate(passed, done int) string {
	if done == 0 {
		return "-"
	}
	return fmt.Sprintf("%d/%d (%d%%)", passed, done, 100*passed/done)
}

// rateColor is the ANSI color of a pass rate: green when every finished
// attempt passed, red once one has not, dim before any finished.
func rateColor(passed, done int) string {
	switch {
	case done == 0:
		return "2"
	case passed == done:
		return "32"
	default:
		return "31"
	}
}

// evalLine prints a single result line, as each run finishes.
func (d display) evalLine(r *eval.SuiteRun, showAttempt bool) {
	if r.Status != eval.RunDone {
		return
	}
	glyph := evalGlyphs[runVerdict(r)]
	extra := r.Error
	if r.DurationMS != nil && r.Error == "" {
		extra = fmt.Sprintf("(%dms)", *r.DurationMS)
	}
	if tally := runTally(r); tally != "" {
		extra = tally + " " + extra
	}
	scenario := r.Scenario
	if showAttempt {
		scenario = fmt.Sprintf("%s #%d", r.Scenario, max(r.Attempt, 1))
	}
	d.printf("  %s %s %s %s\n", d.paint(glyph[0], glyph[1]), r.Label(), d.paint(scenario, "36"), d.dim(extra))
}

// plural is a count and its noun, pluralized with a trailing -s.
func plural(count int, noun string) string {
	if count == 1 {
		return fmt.Sprintf("%d %s", count, noun)
	}
	return fmt.Sprintf("%d %ss", count, noun)
}

// botAudio reports whether the run's scenario has the bot speak; a run whose
// scenario does not load counts as text.
func botAudio(r *eval.SuiteRun) bool {
	loaded, err := r.Load()
	if err != nil {
		return false
	}
	switch s := loaded.(type) {
	case *eval.ScriptScenario:
		return s.BotAudio
	case *eval.SimulationScenario:
		return s.BotAudio
	}
	return false
}

// recordingSetting is the recording settings value: whether the selected runs
// will produce audio. Only an audio-mode run records: a text-mode bot skips TTS
// and the user's turns go over as text, so there is nothing to record.
func (d display) recordingSetting(runs []*eval.SuiteRun, record bool) string {
	if !record {
		return "off"
	}
	audio := 0
	for _, r := range runs {
		if botAudio(r) {
			audio++
		}
	}
	switch audio {
	case len(runs):
		return "on"
	case 0:
		return "off " + d.paint("(all runs text mode)", "33")
	}
	return "on " + d.paint(fmt.Sprintf("(%d of %d runs; text mode skipped)", audio, len(runs)), "33")
}

// runSettings prints how the suite will execute: the run count, the
// concurrency, and whether audio is being recorded.
func (d display) runSettings(runs []*eval.SuiteRun, concurrency int, record bool) {
	counts := fmt.Sprint(len(runs))
	var attempts []int
	for _, r := range runs {
		if a := max(r.Attempts, 1); !slices.Contains(attempts, a) {
			attempts = append(attempts, a)
		}
	}
	slices.Sort(attempts)
	if len(attempts) > 0 && attempts[len(attempts)-1] > 1 {
		evals := len(groupRuns(runs))
		if len(attempts) == 1 {
			counts = fmt.Sprintf("%d (%s x %s)", len(runs), plural(evals, "eval"), plural(attempts[0], "attempt"))
		} else {
			counts = fmt.Sprintf("%d (%s, %d to %d attempts each)",
				len(runs), plural(evals, "eval"), attempts[0], attempts[len(attempts)-1])
		}
	}
	d.printf("Settings:\n")
	for _, kv := range [][2]string{
		{"runs", counts},
		{"concurrency", fmt.Sprint(concurrency)},
		{"recording", d.recordingSetting(runs, record)},
	} {
		d.printf("  %s -> %s\n", d.paint(fmt.Sprintf("%-11s", kv[0]), "1"), kv[1])
	}
	d.printf("\n")
}

// scenarioConfigs prints each distinct scenario's and simulation's config
// once, up front.
func (d display) scenarioConfigs(runs []*eval.SuiteRun) {
	for _, section := range []struct {
		kind    eval.Kind
		heading string
	}{{eval.KindScript, "Scripted scenarios:"}, {eval.KindSimulation, "Simulated scenarios:"}} {
		seen := map[string]bool{}
		for _, r := range runs {
			if r.RunKind() != section.kind || seen[r.Scenario] {
				continue
			}
			if len(seen) == 0 {
				d.printf("%s\n", section.heading)
			}
			seen[r.Scenario] = true
			var cfg string
			switch loaded, err := r.Load(); s := loaded.(type) {
			case *eval.ScriptScenario:
				cfg = eval.DescribeConfig(&s.Configured, d.color)
			case *eval.SimulationScenario:
				cfg = eval.DescribeSimulation(s, d.color)
			default:
				cfg = fmt.Sprintf("(failed to load: %v)", err)
			}
			d.printf("  %s\n", d.paint(r.Scenario+":", "1;36"))
			for line := range strings.Lines(cfg) {
				d.printf("    %s\n", strings.TrimRight(line, "\n"))
			}
		}
		if len(seen) > 0 {
			d.printf("\n")
		}
	}
}

// failures prints every failed run: what it was, and what went wrong.
//
// Every failure is listed, however many there are. Reading them is how a person
// tells a bot that misbehaved from one that never started, and only the
// failure's own reason carries that. Counting and grouping belong to the
// results.jsonl written alongside.
func (d display) failures(failed []*eval.SuiteRun, total int, showAttempt bool) {
	if len(failed) == 0 {
		return
	}
	d.printf("\n  %s\n\n", d.paint(fmt.Sprintf("Failures (%d of %d):", len(failed), total), "1;31"))
	for _, r := range failed {
		header := fmt.Sprintf("  %s %s %s", d.red("✗"), r.Label(), d.paint(r.Scenario, "36"))
		if showAttempt {
			header += " " + d.dim(fmt.Sprintf("#%d", max(r.Attempt, 1)))
		}
		tally := runTally(r)
		if tally != "" && r.ScriptResult != nil {
			header += " " + d.dim(tally+" passed")
		}
		switch {
		case r.Error != "":
			d.printf("%s %s\n", header, d.dim(": "+r.Error))
		case r.SimulationResult != nil && r.SimulationResult.Error != "":
			d.printf("%s %s\n", header, d.dim(": "+r.SimulationResult.Error))
		case r.SimulationResult != nil:
			d.printf("%s\n      %s %s %s\n", header, d.red("•"), r.SimulationResult.Failure(), d.dim(tally))
		case r.ScriptResult != nil:
			d.printf("%s\n", header)
			for _, f := range r.ScriptResult.Failures {
				d.printf("      %s %s\n", d.red("•"), failureLine(f))
			}
		}
	}
}

// runGroup is the attempts of one (label, scenario) pair.
type runGroup struct {
	label, scenario string
	runs            []*eval.SuiteRun
}

// groupRuns buckets runs by (label, scenario), in first-seen order.
func groupRuns(runs []*eval.SuiteRun) []*runGroup {
	var groups []*runGroup
	byKey := map[[2]string]*runGroup{}
	for _, r := range runs {
		key := [2]string{r.Label(), r.Scenario}
		g, ok := byKey[key]
		if !ok {
			g = &runGroup{label: key[0], scenario: key[1]}
			byKey[key] = g
			groups = append(groups, g)
		}
		g.runs = append(g.runs, r)
	}
	return groups
}

// outcome is a group's passed, completed and errored counts.
//
// Errored runs are left out of the completed count: a run that crashed or never
// connected says nothing about whether the bot did its job, and folding it into
// the rate would both drag the rate down unfairly and hide an infrastructure
// problem behind a behavioral one.
func (g *runGroup) outcome() (passed, completed, errored int) {
	for _, r := range g.runs {
		switch runVerdict(r) {
		case verdictPassed:
			passed++
			completed++
		case verdictFailed, verdictSkipped:
			completed++
		case verdictError:
			errored++
		}
	}
	return passed, completed, errored
}

// seconds is how long the group took: from its first attempt's start to its
// last attempt's end, or the sum of its durations when none has a start time;
// false before any attempt has started.
func (g *runGroup) seconds() (float64, bool) {
	var first, last time.Time
	total, timed := 0, false
	for _, r := range g.runs {
		if r.DurationMS != nil {
			total += *r.DurationMS
			timed = true
		}
		if r.StartedAt.IsZero() {
			continue
		}
		if first.IsZero() || r.StartedAt.Before(first) {
			first = r.StartedAt
		}
		end := time.Now()
		if r.DurationMS != nil {
			end = r.StartedAt.Add(time.Duration(*r.DurationMS) * time.Millisecond)
		}
		if end.After(last) {
			last = end
		}
	}
	if first.IsZero() {
		return float64(total) / 1000, timed
	}
	return last.Sub(first).Seconds(), true
}

// repeatSummary prints per-(label, scenario) pass rates and every failure.
//
// A repeated sweep is measuring a rate, not a verdict, so the useful output is
// how often each pair passed and which runs account for the rest. A simulation
// running its own runs is a requirement instead, marked ✓ or ✗ beside its rate:
// every run had to pass. Either way how many runs errored (those are outside
// the rate) and the time the pair took follow.
func (d display) repeatSummary(runs, failed []*eval.SuiteRun) {
	d.printf("\n  %s\n", d.paint("Pass rate:", "1"))
	groups := groupRuns(runs)
	labelW, scenarioW := 0, 0
	for _, g := range groups {
		labelW, scenarioW = max(labelW, len(g.label)), max(scenarioW, len(g.scenario))
	}
	for _, g := range groups {
		passed, completed, errored := g.outcome()
		rate := d.paint(fmt.Sprintf("%14s", passRate(passed, completed)), rateColor(passed, completed))
		var extra []string
		if errored > 0 {
			extra = append(extra, fmt.Sprintf("%d errored", errored))
		}
		if seconds, ok := g.seconds(); ok {
			extra = append(extra, fmtDuration(seconds))
		}
		mark := ""
		if !g.runs[0].Sweep {
			mark = "  " + d.red("✗")
			if passed == len(g.runs) {
				mark = "  " + d.green("✓")
			}
		}
		d.printf("  %-*s  %s  %s%s  %s\n", labelW, g.label, d.paint(fmt.Sprintf("%-*s", scenarioW, g.scenario), "36"),
			rate, mark, d.dim(strings.Join(extra, " · ")))
	}
	d.failures(failed, len(runs), true)
}

// finalize prints the failed set and the final tally, and returns the
// command's error: nil when it passed.
func (d display) finalize(runs []*eval.SuiteRun, runDir string, elapsed time.Duration) error {
	var failed []*eval.SuiteRun
	passed, skipped := 0, 0
	for _, r := range runs {
		switch runVerdict(r) {
		case verdictFailed, verdictError:
			failed = append(failed, r)
		case verdictPassed:
			passed++
		case verdictSkipped:
			skipped++
		}
	}
	color := "32"
	if len(failed) > 0 {
		color = "31"
	}
	if slices.ContainsFunc(runs, func(r *eval.SuiteRun) bool { return r.Attempts > 1 }) {
		d.repeatSummary(runs, failed)
		summary := fmt.Sprintf("%d/%d passed  ·  %s", passed, len(runs), fmtDuration(elapsed.Seconds()))
		d.printf("\n  %s\n", d.paint(summary, color))
		d.printf("  logs: %s\n  results: %s\n\n", runDir, filepath.Join(runDir, "results.jsonl"))
		// A repeated sweep measures a pass rate; what counts as acceptable is
		// the caller's policy, so failures there are data rather than a build
		// break. A simulation's own runs are a requirement, and one of them
		// failing is a break.
		if slices.ContainsFunc(failed, func(r *eval.SuiteRun) bool { return !r.Sweep }) {
			return fmt.Errorf("%w: %d of %d", errScenariosFailed, len(failed), len(runs))
		}
		return nil
	}
	d.failures(failed, len(runs), false)
	summary := fmt.Sprintf("%d/%d passed", passed, len(runs))
	if len(failed) > 0 {
		summary += fmt.Sprintf(", %d failed", len(failed))
	}
	if skipped > 0 {
		summary += fmt.Sprintf(", %d skipped", skipped)
	}
	summary += "  ·  " + fmtDuration(elapsed.Seconds())
	d.printf("\n  %s\n  logs: %s\n\n", d.paint(summary, color), runDir)
	if len(failed) > 0 {
		return fmt.Errorf("%w: %d of %d", errScenariosFailed, len(failed), len(runs))
	}
	return nil
}
