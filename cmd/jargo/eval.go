package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

// CLI errors.
//
//nolint:gochecknoglobals // sentinel errors
var (
	errBotURLRequired  = errors.New("--bot-url is required")
	errScenariosFailed = errors.New("scenarios failed")
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
		Use:   "run <scenario.yaml>...",
		Short: "Play one or more scenarios against a running bot's RTVI WebSocket endpoint",
		Long: "Play one or more scenarios against a running bot.\n\n" +
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
		judge := getJudge()
		out := cmd.OutOrStdout()
		failed := 0
		for _, path := range args {
			scenario, err := eval.Load(path)
			if err != nil {
				return err
			}
			res, err := eval.RunURL(cmd.Context(), scenario, botURL, judge)
			if err != nil {
				return fmt.Errorf("%s: %w", scenario.Name, err)
			}
			_, _ = fmt.Fprintln(out, res.String())
			if !res.Passed() {
				failed++
			}
		}
		if failed > 0 {
			return fmt.Errorf("%w: %d of %d", errScenariosFailed, failed, len(args))
		}
		return nil
	}
	return cmd
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
		results := eval.RunSuite(cmd.Context(), m, getJudge())
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
	return eval.NewLLMJudge(openai.NewLLM(openai.LLMConfig{
		BaseURL: baseURL,
		Model:   model,
		APIKey:  key,
	}))
}
