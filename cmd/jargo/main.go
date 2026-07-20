// Command jargo is the command-line entry point for jargo tooling. Today it
// hosts the behavioral eval runner (jargo eval run); the subcommand tree leaves
// room for more as the toolchain grows.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"

	"github.com/spf13/cobra"
)

func main() {
	os.Exit(run())
}

// run executes the CLI and returns a process exit code, so main's deferred
// cleanup (signal reset) runs before the process exits.
func run() int {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if err := rootCmd().ExecuteContext(ctx); err != nil {
		_, _ = fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	return 0
}

// rootCmd builds the top-level jargo command.
func rootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:          "jargo",
		Short:        "Tooling for building and testing jargo voice agents",
		SilenceUsage: true,
	}
	root.AddCommand(evalCmd())
	return root
}
