package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

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
	var botURL string
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
		for _, path := range paths {
			scenario, err := eval.Load(path)
			if err != nil {
				// A scenario that will not load is a failure of that scenario,
				// not of the run: the rest are still worth playing, and this one
				// shows in the tally like any other.
				_, _ = fmt.Fprintf(out, "FAIL %s (failed to load: %v)\n", scenarioName(path), err)
				failed++
				continue
			}
			// A fresh judge per scenario: it holds the conversation it grades
			// against, so one scenario's turns must not reach the next one's.
			res, err := eval.RunURL(cmd.Context(), scenario, botURL, getJudge())
			if err != nil {
				return fmt.Errorf("%s: %w", scenario.Name, err)
			}
			_, _ = fmt.Fprintln(out, res.String())
			if !res.Passed() {
				failed++
			}
		}
		if failed > 0 {
			return fmt.Errorf("%w: %d of %d", errScenariosFailed, failed, len(paths))
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
			if !e.IsDir() && slices.Contains(scenarioSuffixes, filepath.Ext(e.Name())) {
				paths = append(paths, filepath.Join(arg, e.Name()))
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
			p("%-5s %s (%s)\n", suiteStatus(r), filepath.Base(r.Scenario), r.BotURL)
			if r.Err != nil {
				p("  %v\n", r.Err)
			}
			for _, f := range r.Result.Failures {
				p("  - %s\n", f)
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
	var model, baseURL, key string
	f.StringVar(&model, "judge-model", "", "enable the LLM judge with this model id (for scenarios using judge:)")
	f.StringVar(&baseURL, "judge-url", "", "OpenAI-compatible base URL for the judge (e.g. a local Ollama)")
	f.StringVar(&key, "judge-key", "", "API key for the judge endpoint (falls back to $OPENAI_API_KEY)")
	return func() eval.Judge { return buildJudge(model, baseURL, key) }
}

// buildJudge constructs an LLM judge from the flags, or nil when no judge model
// is set. It targets any OpenAI-compatible endpoint (OpenAI, a local Ollama via
// --judge-url, etc.).
func buildJudge(model, baseURL, key string) eval.Judge {
	if model == "" {
		return nil
	}
	if key == "" {
		key = os.Getenv("OPENAI_API_KEY")
	}
	return eval.NewLLMJudge(chat.NewLLM(chat.LLMConfig{
		BaseURL: baseURL,
		Model:   model,
		APIKey:  key,
	}))
}
