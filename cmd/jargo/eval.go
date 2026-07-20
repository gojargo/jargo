package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/provider/openai"
	"github.com/spf13/cobra"
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
	cmd.AddCommand(evalRunCmd())
	return cmd
}

// evalRunCmd is `jargo eval run` — play scenarios against a running bot.
func evalRunCmd() *cobra.Command {
	var (
		botURL     string
		judgeModel string
		judgeURL   string
		judgeKey   string
	)
	cmd := &cobra.Command{
		Use:   "run <scenario.yaml>...",
		Short: "Play one or more scenarios against a running bot's RTVI WebSocket endpoint",
		Long: "Play one or more scenarios against a running bot.\n\n" +
			"The bot must expose an RTVI WebSocket endpoint (see eval.Handler). Each\n" +
			"scenario's result is printed; the command exits non-zero if any fail.\n\n" +
			"Scenarios that use judge: need an LLM judge — enable one with --judge-model.",
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if botURL == "" {
				return errBotURLRequired
			}
			judge := buildJudge(judgeModel, judgeURL, judgeKey)
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
		},
	}
	f := cmd.Flags()
	f.StringVar(&botURL, "bot-url", "", "RTVI WebSocket URL of the running bot (ws:// or wss://)")
	f.StringVar(&judgeModel, "judge-model", "", "enable the LLM judge with this model id (for scenarios using judge:)")
	f.StringVar(&judgeURL, "judge-url", "", "OpenAI-compatible base URL for the judge (e.g. a local Ollama)")
	f.StringVar(&judgeKey, "judge-key", "", "API key for the judge endpoint (falls back to $OPENAI_API_KEY)")
	return cmd
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
