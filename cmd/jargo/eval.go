package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	llmclassifier "github.com/gojargo/jargo/classifier/llm"
	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// CLI errors.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errBotURLRequired   = errors.New("--bot-url is required")
	errScenariosFailed  = errors.New("scenarios failed")
	errNoScenariosInDir = errors.New("no .yaml or .yml scenario files found in")
)

// evalCmd is the `jargo eval` command group.
func evalCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "eval",
		Short: "Run behavioral eval scenarios against a jargo bot",
	}
	cmd.AddCommand(evalRunCmd(), evalSuiteCmd())
	return cmd
}

// evalRunCmd is `jargo eval run` — play scenarios against a running bot.
func evalRunCmd() *cobra.Command {
	var (
		botURL          string
		botReadyTimeout float64
	)
	cmd := &cobra.Command{
		Use:   "run <scenario.yaml|dir>...",
		Short: "Play one or more scenarios against a running bot's RTVI WebSocket endpoint",
		Long: "Play one or more scenarios against a running bot. A directory stands\n" +
			"for the scenarios directly in it, in name order.\n\n" +
			"The bot must expose an RTVI WebSocket endpoint (see eval.Handler). Each\n" +
			"scenario's result is printed; the command exits non-zero if any fail.\n\n" +
			"Scenarios that use judge: need an LLM judge — enable one with --judge-model.",
		Args: cobra.MinimumNArgs(1),
	}
	getJudge := addJudgeFlags(cmd.Flags())
	cmd.Flags().StringVar(&botURL, "bot-url", "", "RTVI WebSocket URL of the running bot (ws:// or wss://)")
	cmd.Flags().Float64Var(&botReadyTimeout, "bot-ready-timeout", eval.DefaultBotReadyTimeout.Seconds(),
		"seconds to wait, once connected, for the bot's RTVI bot-ready, which covers the bot's startup")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		if botURL == "" {
			return errBotURLRequired
		}
		paths, err := expandScenarioPaths(args)
		if err != nil {
			return err
		}
		out := cmd.OutOrStdout()
		failed := 0
		played := 0
		for _, path := range paths {
			file, err := eval.LoadFile(path)
			if err != nil {
				// A scenario that will not load is a failure of that scenario,
				// not of the run: the rest are still worth playing, and this one
				// shows in the tally like any other.
				_, _ = fmt.Fprintf(out, "FAIL %s (failed to load: %v)\n", scenarioName(path), err)
				failed++
				played++
				continue
			}
			for _, scenario := range file.Scenarios {
				played++
				// A fresh judge per scenario: it holds the conversation it grades
				// against, so one scenario's turns must not reach the next one's.
				res, err := eval.RunURL(cmd.Context(), scenario, botURL, eval.Options{
					Judge:           getJudge(),
					BotReadyTimeout: time.Duration(botReadyTimeout * float64(time.Second)),
				})
				if err != nil {
					return fmt.Errorf("%s: %w", scenario.Name, err)
				}
				printResult(out, res)
				if !res.Passed() {
					failed++
				}
			}
		}
		if failed > 0 {
			return fmt.Errorf("%w: %d of %d", errScenariosFailed, failed, played)
		}
		return nil
	}
	return cmd
}

// scenarioSuffixes are the file extensions a scenario is written under, which
// are the ones a manifest resolves a bare scenario name to.
//
//nolint:gochecknoglobals // fixed set, read only
var scenarioSuffixes = []string{".yaml", ".yml"}

// expandScenarioPaths turns each argument into the scenarios to play. A file
// stands for itself; a directory stands for the scenarios directly in it, in
// name order, so a whole suite can be played by naming the folder holding it.
// The fragments its scenarios include, which have no `name:`, are left out of
// a directory; a file given explicitly is always taken.
//
// A directory holding no scenario is refused rather than quietly contributing
// nothing, since a run that plays nothing at all looks like a run that passed.
func expandScenarioPaths(args []string) ([]string, error) {
	var paths []string
	for _, arg := range args {
		info, err := os.Stat(arg)
		if err != nil || !info.IsDir() {
			// A path that does not exist is left alone, so loading it reports
			// the failure against that scenario rather than ending the run.
			paths = append(paths, arg)
			continue
		}
		entries, err := os.ReadDir(arg)
		if err != nil {
			return nil, err
		}
		found := len(paths)
		for _, e := range entries {
			path := filepath.Join(arg, e.Name())
			if !e.IsDir() && slices.Contains(scenarioSuffixes, filepath.Ext(e.Name())) && eval.IsScenarioFile(path) {
				paths = append(paths, path)
			}
		}
		if len(paths) == found {
			return nil, fmt.Errorf("%w: %s", errNoScenariosInDir, arg)
		}
	}
	return paths, nil
}

// scenarioName is what a scenario that could not be loaded is called in the
// output: its file name without the suffix, since its declared name is part of
// what could not be read.
func scenarioName(path string) string {
	return strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
}

// evalSuiteCmd is `jargo eval suite` — run a manifest of scenarios concurrently.
func evalSuiteCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "suite <manifest.yaml>",
		Short: "Run a manifest of scenarios against one or more bots, concurrently",
		Args:  cobra.ExactArgs(1),
	}
	getJudge := addJudgeFlags(cmd.Flags())

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		m, err := eval.LoadManifest(args[0])
		if err != nil {
			return err
		}
		results := eval.RunSuite(cmd.Context(), m, getJudge)
		out := cmd.OutOrStdout()
		p := func(format string, a ...any) { _, _ = fmt.Fprintf(out, format, a...) }

		failed := 0
		for _, r := range results {
			line := fmt.Sprintf("%-5s %s (%s)", suiteStatus(r), filepath.Base(r.Scenario), r.BotURL)
			if tally := turnTally(r.Result); tally != "" {
				line += " " + tally
			}
			p("%s\n", line)
			if r.Err != nil {
				p("  %v\n", r.Err)
			}
			for _, f := range r.Result.Failures {
				p("  - %s\n", failureLine(f))
			}
			if !r.Passed() {
				failed++
			}
		}
		p("\n%d/%d scenarios passed\n", len(results)-failed, len(results))
		if failed > 0 {
			return fmt.Errorf("%w: %d of %d", errScenariosFailed, failed, len(results))
		}
		return nil
	}
	return cmd
}

// printResult prints a scenario's result: its verdict and, for a run that
// failed, how many turns passed and every failure.
func printResult(out io.Writer, res eval.Result) {
	if res.Passed() {
		_, _ = fmt.Fprintf(out, "PASS %s\n", res.Scenario)
		return
	}
	line := "FAIL " + res.Scenario
	if tally := turnTally(res); tally != "" {
		line += " " + tally
	}
	_, _ = fmt.Fprintln(out, line)
	for _, f := range res.Failures {
		_, _ = fmt.Fprintf(out, "  - %s\n", failureLine(f))
	}
}

// failureLine is one failure as the run reports it: where, on what, its kind,
// and its reason. Reading the reasons is how a person tells a bot that
// misbehaved from one that never started; the kind says which assertion gave
// way.
func failureLine(f eval.Failure) string {
	where := fmt.Sprintf("turn %d", f.Turn)
	if f.Turn < 1 {
		where = "run"
	}
	return fmt.Sprintf("%s %s %s: %s", where, f.Event, f.Kind, f.Reason)
}

// turnTally is the run's detail beside its verdict, or "": "7/10 turns" for a
// run that drove every turn and failed some.
//
// A rate needs every turn scored. A run that stopped at its first failure left
// the rest undriven, and a fraction over those would read as turns that failed
// when they were never attempted; what it stopped on is in the failure listing
// instead. A run that passed already says so.
func turnTally(res eval.Result) string {
	if len(res.Turns) == 0 {
		return ""
	}
	failed := 0
	for _, t := range res.Turns {
		switch t.Status {
		case eval.TurnNotRun:
			return ""
		case eval.TurnFailed:
			failed++
		}
	}
	if failed == 0 {
		return ""
	}
	return fmt.Sprintf("%d/%d turns", len(res.Turns)-failed, len(res.Turns))
}

// suiteStatus is the one-word status for a suite result.
func suiteStatus(r eval.SuiteResult) string {
	switch {
	case r.Err != nil:
		return "ERROR"
	case !r.Result.Passed():
		return "FAIL"
	default:
		return "PASS"
	}
}

// addJudgeFlags registers the --judge-* flags on f and returns a getter that
// builds the configured judge (nil when no --judge-model is set).
func addJudgeFlags(f *pflag.FlagSet) func() eval.Judge {
	var opts judgeOptions
	f.StringVar(&opts.model, "judge-model", "", "enable the LLM judge with this model id (for scenarios using judge:)")
	f.StringVar(&opts.baseURL, "judge-url", "", "OpenAI-compatible base URL for the judge (e.g. a local Ollama)")
	f.StringVar(&opts.key, "judge-key", "", "API key for the judge endpoint (falls back to $OPENAI_API_KEY)")
	f.BoolVar(&opts.explainer, "judge-explainer", true,
		"ask the judge's model for the reason behind a no or an unsure verdict")
	f.Float64Var(&opts.explainBelow, "judge-explain-below", 0.75,
		"explain a yes less sure than this too (0 explains the no verdicts only)")
	f.BoolVar(&opts.allowContinue, "judge-allow-continue", true,
		"let a reply be judged continue while the bot is still working toward its answer")
	return func() eval.Judge { return buildJudge(opts) }
}

// judgeOptions is what the --judge-* flags configure.
type judgeOptions struct {
	model, baseURL, key string
	explainer           bool
	explainBelow        float64
	allowContinue       bool
}

// buildJudge constructs a judge from the flags, or nil when no judge model is
// set. It classifies with a model behind any OpenAI-compatible endpoint (OpenAI,
// a local Ollama via --judge-url, etc.), and the same model gives the reasons.
func buildJudge(opts judgeOptions) eval.Judge {
	if opts.model == "" {
		return nil
	}
	key := opts.key
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	service := chat.NewLLM(chat.LLMConfig{
		BaseURL: opts.baseURL,
		Model:   opts.model,
		APIKey:  key,
	})
	c, err := llmclassifier.New(llmclassifier.Config{LLM: service})
	if err != nil {
		return nil
	}
	cfg := eval.EvalJudgeConfig{
		Classifier:    c,
		ExplainBelow:  &opts.explainBelow,
		AllowContinue: &opts.allowContinue,
	}
	if opts.explainer {
		cfg.Explainer = service
	}
	judge, err := eval.NewEvalJudge(cfg)
	if err != nil {
		return nil
	}
	return judge
}
