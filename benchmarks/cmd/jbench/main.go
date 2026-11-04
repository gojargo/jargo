// Command jbench runs a scenario file against jargo and prints the results in
// the Go benchmark format, so benchstat can summarize them and set them beside
// the Python side's.
//
//	jbench run [-count N] [-run regexp] [-framework-version V] scenarios/quick.json
//
// A deployment benchmark starts jbench serve processes of its own, which hold
// the sessions and are driven over their standard input and output.
//
// The output opens with config lines (framework, framework_version, runtime,
// the machine) and then holds one line per sample:
//
//	BenchmarkPlumbingLatency/depth=4	1000	51234.00 ns/frame	12808.50 ns/hop	...
//
// A sample that runs past its budget prints a --- TIMEOUT line, which benchstat
// ignores, and the larger sizes of that benchmark are skipped.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"regexp"
	"strings"

	"github.com/gojargo/jargo/benchmarks/internal/run"
	"github.com/gojargo/jargo/benchmarks/internal/scenario"
)

// frameworkVersion is the jargo revision being measured. It is set at link time
// where the build has no VCS information to read, as in the Docker image:
//
//	go build -ldflags "-X main.frameworkVersion=$(git describe --always --dirty)"
//
//nolint:gochecknoglobals // set by the linker
var frameworkVersion string

var (
	// errUsage is returned for a command line jbench cannot make sense of.
	errUsage = errors.New("usage: jbench run [-count N] [-run regexp] [-framework-version V] scenario.json | jbench serve")
	// errFailed is returned when a sample failed for a reason other than its
	// budget, once every other sample has been taken.
	errFailed = errors.New("some benchmarks failed")
	// errNoMatch is returned when -run selects nothing.
	errNoMatch = errors.New("no sub-benchmark matches")
)

func main() {
	if err := runMain(context.Background(), os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "jbench:", err)
		os.Exit(1)
	}
}

// options are what the run subcommand was asked for.
type options struct {
	count   int
	filter  *regexp.Regexp
	version string
	path    string
}

func parseArgs(args []string) (options, error) {
	if len(args) == 0 || args[0] != "run" {
		return options{}, errUsage
	}
	fs := flag.NewFlagSet("jbench run", flag.ContinueOnError)
	count := fs.Int("count", 1, "how many samples of each sub-benchmark to take")
	filter := fs.String("run", "", "only run sub-benchmarks whose full name matches this regexp")
	version := fs.String("framework-version", "", "the jargo revision to report; read from the build when empty")
	if err := fs.Parse(args[1:]); err != nil {
		return options{}, err
	}
	if fs.NArg() != 1 || *count < 1 {
		return options{}, errUsage
	}
	re, err := regexp.Compile(*filter)
	if err != nil {
		return options{}, fmt.Errorf("-run: %w", err)
	}
	return options{count: *count, filter: re, version: *version, path: fs.Arg(0)}, nil
}

func runMain(ctx context.Context, args []string, out io.Writer) error {
	if len(args) == 1 && args[0] == "serve" {
		slog.SetDefault(slog.New(slog.DiscardHandler))
		return run.Serve(ctx, os.Stdin, out)
	}
	opts, err := parseArgs(args)
	if err != nil {
		return err
	}
	sc, err := scenario.Load(opts.path)
	if err != nil {
		return err
	}

	// The pipeline logs at debug level as frames go through it. Discarding the
	// records keeps the logger off the measured path; the Python side removes
	// its log handlers for the same reason.
	slog.SetDefault(slog.New(slog.DiscardHandler))

	version := opts.version
	if version == "" {
		version = buildVersion()
	}
	w := &lineWriter{w: out}
	for _, kv := range configLines(sc.Name, version) {
		w.printf("%s: %s\n", kv[0], kv[1])
	}
	runErr := execute(ctx, sc, opts, w)
	return errors.Join(runErr, w.err)
}

// lineWriter writes the output, keeping the first write error so a failed
// write is reported once, at the end, rather than checked line by line.
type lineWriter struct {
	w   io.Writer
	err error
}

func (l *lineWriter) printf(format string, args ...any) {
	if l.err != nil {
		return
	}
	_, l.err = fmt.Fprintf(l.w, format, args...)
}

// unit is one sub-benchmark: a benchmark at one of its sizes.
type unit struct {
	bench scenario.Benchmark
	// group and index place the unit among the sizes of its benchmark, so a
	// timeout can skip the larger ones.
	group, index int
	name         string
	size         int
}

// units lists the sub-benchmarks a scenario runs, in order.
func units(sc *scenario.Scenario, filter *regexp.Regexp) []unit {
	var list []unit
	for g, b := range sc.Benchmarks {
		label, sizes := "depth", b.Depths
		if b.Kind == scenario.KindSessions || b.Kind == scenario.KindDeployment {
			label, sizes = "sessions", b.Sessions
		}
		for i, n := range sizes {
			name := fmt.Sprintf("%s/%s=%d", b.Name, label, n)
			if filter.MatchString(name) {
				list = append(list, unit{bench: b, group: g, index: i, name: name, size: n})
			}
		}
	}
	return list
}

// execute takes count samples of every sub-benchmark. The samples are
// interleaved, every sub-benchmark once per pass, so a slow drift in the
// machine spreads across all of them instead of landing on one.
func execute(ctx context.Context, sc *scenario.Scenario, opts options, out *lineWriter) error {
	list := units(sc, opts.filter)
	if len(list) == 0 {
		return fmt.Errorf("%w -run %q", errNoMatch, opts.filter)
	}
	// skipFrom maps a benchmark to the first of its sizes no longer run.
	skipFrom := make(map[int]int)
	var failed bool
	for range opts.count {
		for _, u := range list {
			if from, ok := skipFrom[u.group]; ok && u.index >= from {
				continue
			}
			res, err := sample(ctx, u)
			switch {
			case errors.Is(err, run.ErrTimeout):
				out.printf("--- TIMEOUT: Benchmark%s exceeded its %gs budget; larger sizes skipped\n",
					u.name, u.bench.BudgetSeconds)
				skipFrom[u.group] = u.index
			case err != nil:
				out.printf("--- FAIL: Benchmark%s: %v\n", u.name, err)
				skipFrom[u.group] = u.index
				failed = true
			default:
				out.printf("%s\n", formatResult(u.name, res))
			}
		}
	}
	if failed {
		return errFailed
	}
	return nil
}

// sample takes one sample of a sub-benchmark within its budget.
func sample(ctx context.Context, u unit) (run.Result, error) {
	ctx, cancel := context.WithTimeout(ctx, u.bench.Budget())
	defer cancel()
	b := u.bench
	switch b.Kind {
	case scenario.KindLatency:
		return run.Latency(ctx, u.size, b.Warmup, b.Frames)
	case scenario.KindThroughput:
		return run.Throughput(ctx, u.size, b.Warmup, b.Frames)
	case scenario.KindStartup:
		return run.Startup(ctx, u.size, b.Repeats)
	case scenario.KindSessions:
		return run.Sessions(ctx, b.Depth, u.size, b.Warmup, b.Rounds)
	case scenario.KindDeployment:
		exe, err := os.Executable()
		if err != nil {
			return run.Result{}, err
		}
		// One process serves every session: a Go program uses every CPU it is
		// given from one process.
		return run.Deployment(ctx, []string{exe, "serve"}, 1, b.Depth, u.size, b.Warmup, b.Rounds)
	default:
		return run.Result{}, fmt.Errorf("%w: unknown kind %q", scenario.ErrInvalid, b.Kind)
	}
}

// formatResult writes a sample as a benchfmt line.
func formatResult(name string, res run.Result) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "Benchmark%s\t%d", name, res.Iterations)
	for _, m := range res.Metrics {
		fmt.Fprintf(&sb, "\t%.2f %s", m.Value, m.Unit)
	}
	return sb.String()
}
