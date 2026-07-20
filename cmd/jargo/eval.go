package main

import (
	"errors"
	"fmt"

	"github.com/gojargo/jargo/eval"
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
	var botURL string
	cmd := &cobra.Command{
		Use:   "run <scenario.yaml>...",
		Short: "Play one or more scenarios against a running bot's RTVI WebSocket endpoint",
		Long: "Play one or more scenarios against a running bot.\n\n" +
			"The bot must expose an RTVI WebSocket endpoint (see eval.Handler). Each\n" +
			"scenario's result is printed; the command exits non-zero if any fail.",
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if botURL == "" {
				return errBotURLRequired
			}
			out := cmd.OutOrStdout()
			failed := 0
			for _, path := range args {
				scenario, err := eval.Load(path)
				if err != nil {
					return err
				}
				res, err := eval.RunURL(cmd.Context(), scenario, botURL, nil)
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
	cmd.Flags().StringVar(&botURL, "bot-url", "", "RTVI WebSocket URL of the running bot (ws:// or wss://)")
	return cmd
}
