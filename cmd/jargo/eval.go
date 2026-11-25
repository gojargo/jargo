package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/gojargo/jargo/classifier/llmclassifier"
	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/event"
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
	cmd.AddCommand(evalRunCmd(), evalJudgeCmd(), evalSuiteCmd())
	return cmd
}

// evalRunCmd is `jargo eval run`: play scenarios against a running bot.
func evalRunCmd() *cobra.Command {
	var (
		botURL          string
		botReadyTimeout float64
		timeout         int
		verbose         bool
		cacheDir        string
		noCache         bool
		logsDir         string
		debugLogs       bool
		audio           bool
		recordDir       string
	)
	cmd := &cobra.Command{
		Use:   "run <scenario.yaml|dir>...",
		Short: "Play one or more scenarios against a running bot's RTVI WebSocket endpoint",
		Long: "Play one or more scenarios, scripted or simulations, against a running bot.\n" +
			"A directory stands for the scenarios directly in it, in name order. A\n" +
			"simulation runs once here, and its conversation is saved as\n" +
			"<logs-dir>/<scenario>.conversation.json; a rubric has nothing to run and is\n" +
			"judged with `jargo eval judge`.\n\n" +
			"The bot must expose an RTVI WebSocket endpoint (see eval.Handler). Each\n" +
			"scenario's result is printed; the command exits non-zero if any fail.\n\n" +
			"A scenario using eval: is judged by the judge its judge: block configures,\n" +
			"or by the LLM judge --judge-model enables.",
		Args: cobra.MinimumNArgs(1),
	}
	getJudge := addJudgeFlags(cmd.Flags())
	cmd.Flags().StringVar(&botURL, "bot-url", "", "RTVI WebSocket URL of the running bot (ws:// or wss://)")
	cmd.Flags().Float64Var(&botReadyTimeout, "bot-ready-timeout", eval.DefaultBotReadyTimeout.Seconds(),
		"seconds to wait, once connected, for the bot's RTVI bot-ready, which covers the bot's startup")
	cmd.Flags().IntVarP(&timeout, "timeout", "t", eval.DefaultEventTimeoutMS/1000,
		"default per-expectation timeout in seconds (for expectations without their own within_ms)")
	cmd.Flags().BoolVarP(&verbose, "verbose", "v", false,
		"print a line for each turn and expectation as it resolves; for a simulation, "+
			"the conversation as it happens and the judge's reasons at the end")
	cmd.Flags().StringVar(&cacheDir, "cache-dir", "",
		"directory for cached synthesized user audio (default <user-cache-dir>/jargo/eval/tts)")
	cmd.Flags().BoolVar(&noCache, "no-cache", false,
		"disable the user-audio cache: synthesize every turn again (no reads or writes)")
	cmd.Flags().StringVar(&logsDir, "logs-dir", ".",
		"directory for each scenario's logs: <logs-dir>/<scenario>.eval.log (+ .debug.log)")
	cmd.Flags().BoolVarP(&debugLogs, "debug", "d", false,
		"also save <scenario>.debug.log with the harness's full per-pipeline logs")
	cmd.Flags().BoolVarP(&audio, "audio", "a", false,
		"record each scenario's conversation audio (audio-mode scenarios)")
	cmd.Flags().StringVar(&recordDir, "record-dir", "recordings",
		"directory for --audio recordings: <record-dir>/<scenario>.wav")

	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		if botURL == "" {
			return errBotURLRequired
		}
		// The logs go to <scenario>.debug.log under --debug; either way, keep
		// them off the console so they cannot corrupt the output.
		restore := eval.QuietLogs()
		defer restore()
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
			for _, loaded := range file.Scenarios {
				// A fresh judge per scenario: it holds the conversation it grades
				// against, so one scenario's turns must not reach the next one's.
				useCache := !noCache
				params := eval.SessionParams{
					BotReadyTimeout:  time.Duration(botReadyTimeout * float64(time.Second)),
					DefaultTimeoutMS: timeout * 1000,
					CacheDir:         cacheDir,
					UseCache:         &useCache,
				}
				if audio {
					params.RecordPath = recordPath(recordDir, loaded.ScenarioName())
				}
				var passed bool
				switch scenario := loaded.(type) {
				case *eval.ScriptScenario:
					passed, err = playScript(cmd.Context(), out, scenario, botURL, getJudge(), params, runLogs{
						dir: logsDir, debug: debugLogs, verbose: verbose,
					})
				case *eval.SimulationScenario:
					passed, err = playSimulation(cmd.Context(), out, scenario, botURL, getJudge(), params, runLogs{
						dir: logsDir, debug: debugLogs, verbose: verbose,
					})
				default:
					// A rubric judges a call that already happened; there is
					// nothing to run (see `jargo eval judge`).
					continue
				}
				if err != nil {
					return err
				}
				played++
				if !passed {
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

// writeTrace saves a run's decision trace as <logs-dir>/<stem>.eval.log. A
// trace that cannot be written is dropped: it is a by-product of the run.
func writeTrace(logsDir, stem string, trace []string) {
	if len(trace) == 0 || os.MkdirAll(logsDir, 0o750) != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(logsDir, stem+".eval.log"), []byte(strings.Join(trace, "\n")+"\n"), 0o600)
}

// recordPath is where a scenario's recording goes under recordDir, or "" when
// recording is off. A group entry's "/" becomes "__" in the file name.
func recordPath(recordDir, scenarioName string) string {
	if recordDir == "" {
		return ""
	}
	return filepath.Join(recordDir, strings.ReplaceAll(scenarioName, "/", "__")+".wav")
}

// runLogs is where a scenario's logs go, and what is printed as it runs.
type runLogs struct {
	dir     string
	debug   bool
	verbose bool
}

// playScript plays a scripted scenario against the bot at botURL, prints its
// result, and reports whether it passed.
func playScript(
	ctx context.Context, out io.Writer, scenario *eval.ScriptScenario, botURL string, judge eval.Judge,
	params eval.SessionParams, logs runLogs,
) (bool, error) {
	opts := eval.Options{Judge: judge, Params: params}
	if logs.verbose {
		opts.OnProgress = func(p eval.ScriptTurnProgress) { printProgress(out, p) }
	}
	stem := strings.ReplaceAll(scenario.Name, "/", "__")
	ctx, finishLogs := eval.CapturePipelineLogs(ctx, logs.dir, stem, scenario.Name, logs.debug)
	res, err := eval.RunURL(ctx, scenario, botURL, opts)
	finishLogs()
	if err != nil {
		return false, fmt.Errorf("%s: %w", scenario.Name, err)
	}
	writeTrace(logs.dir, stem, res.DebugLog)
	printResult(out, res)
	return res.Passed(), nil
}

// playSimulation runs a simulation once against the bot at botURL, writes its
// trace and its <scenario>.conversation.json, prints its result, and reports
// whether it passed. Verbose, the conversation is printed as it happens and the
// judge's reasons at the end.
//
// A judge from the --judge-* flags judges it when it can judge a whole run;
// otherwise the simulation's judge: block builds one.
func playSimulation(
	ctx context.Context, out io.Writer, scenario *eval.SimulationScenario, botURL string, judge eval.Judge,
	params eval.SessionParams, logs runLogs,
) (bool, error) {
	runJudge, _ := judge.(eval.RunJudge)
	session, err := eval.SimulationSessionFromScenario(scenario, botURL, params, nil, runJudge, nil, nil)
	if err != nil {
		return false, fmt.Errorf("%s: %w", scenario.Name, err)
	}
	d := display{out: out, color: isTerminal(out)}
	if logs.verbose {
		d.printf("    %s\n", d.paint("conversation:", "1"))
		event.On(&session.Registry, eval.EventProgress, func(_ context.Context, p eval.SimulationProgress) {
			d.simulationProgress(p)
		})
	}
	stem := strings.ReplaceAll(scenario.Name, "/", "__")
	ctx, finishLogs := eval.CapturePipelineLogs(ctx, logs.dir, stem, scenario.Name, logs.debug)
	res, err := session.Run(ctx)
	finishLogs()
	if err != nil {
		return false, fmt.Errorf("%s: %w", scenario.Name, err)
	}
	writeTrace(logs.dir, stem, res.DebugLog)
	writeConversation(logs.dir, stem, res.Conversation)
	if logs.verbose {
		// The detail closes the conversation above it; the verdict line comes
		// last, as a scripted scenario's does after its turns.
		d.simulationDetail(res)
	}
	printSimulationResult(out, res)
	return res.Passed(), nil
}

// endingColors are the colors of how a simulation ended: the persona hanging
// up is green, the bot yellow; a cap cutting it short is red.
//
//nolint:gochecknoglobals // fixed table, read only
var endingColors = map[string]string{eval.SimulationEndedByEndCall: "32", eval.SimulationEndedByBot: "33"}

// simulationProgress prints a simulation's progress record as it arrives
// (verbose mode): the conversation as it is spoken, and how it ended.
func (d display) simulationProgress(p eval.SimulationProgress) {
	if p.Status == "ended" {
		color, ok := endingColors[p.Text]
		if !ok {
			color = "31"
		}
		d.printf("\n      %s %s %s\n", d.dim("ended by"), d.paint(p.Text, color),
			d.dim(fmt.Sprintf("after %d persona turn(s)", p.Turn)))
		return
	}
	color := "36"
	if p.Status == "bot" {
		color = "32"
	}
	d.printf("      %s %s\n", d.paint(p.Status+":", color), p.Text)
}

// printSimulationResult prints a simulation's result: its verdict and, for a
// run that did not pass, how it ended and why.
func printSimulationResult(out io.Writer, res *eval.SimulationResult) {
	if res.Passed() {
		_, _ = fmt.Fprintf(out, "PASS %s\n", res.SimulationName)
		return
	}
	_, _ = fmt.Fprintf(out, "FAIL %s (ended by %s)\n", res.SimulationName, res.EndedBy)
	line := res.Failure()
	if res.FailureKind != "" {
		line = res.FailureKind + ": " + line
	}
	_, _ = fmt.Fprintf(out, "  - %s\n", line)
}

// writeConversation saves a simulation's conversation as
// <logs-dir>/<stem>.conversation.json. A file that cannot be written is
// dropped: it is a by-product of the run.
func writeConversation(logsDir, stem string, conversation []map[string]any) {
	if conversation == nil {
		conversation = []map[string]any{}
	}
	data, err := json.MarshalIndent(conversation, "", "  ")
	if err != nil || os.MkdirAll(logsDir, 0o750) != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(logsDir, stem+".conversation.json"), append(data, '\n'), 0o600)
}

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
			if !e.IsDir() && slices.Contains(eval.ScenarioSuffixes, filepath.Ext(e.Name())) && eval.IsScenarioFile(path) {
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

// printProgress prints a progress record as it arrives (verbose mode): a line
// for each turn as it is sent, and for each expectation as it resolves.
func printProgress(out io.Writer, p eval.ScriptTurnProgress) {
	if p.Status == eval.StatusTurn {
		label := "(observe)"
		if p.EventName != "" {
			label = fmt.Sprintf("%q", p.EventName)
		}
		_, _ = fmt.Fprintf(out, "      turn %d → %s\n", p.TurnIndex, label)
		return
	}
	badge := "✗"
	if p.Status == eval.StatusMatched {
		badge = "✓"
	}
	line := fmt.Sprintf("        %s %s", badge, p.EventName)
	if p.Detail != "" {
		line += ": " + strings.Join(strings.Fields(p.Detail), " ")
	}
	_, _ = fmt.Fprintln(out, line)
}

// printResult prints a scenario's result: its verdict and, for a run that
// failed, how many turns passed and every failure.
func printResult(out io.Writer, res eval.ScriptResult) {
	if res.Passed() {
		_, _ = fmt.Fprintf(out, "PASS %s\n", res.ScenarioName)
		return
	}
	line := "FAIL " + res.ScenarioName
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
func failureLine(f eval.AssertionFailure) string {
	where := fmt.Sprintf("turn %d", f.TurnIndex)
	if f.TurnIndex < 0 {
		where = "run"
	}
	return fmt.Sprintf("%s %s %s: %s", where, f.EventName, f.Kind, f.Reason)
}

// turnTally is the run's detail beside its verdict, or "": "7/10 turns" for a
// run that drove every turn and failed some.
//
// A rate needs every turn scored. A run that stopped at its first failure left
// the rest undriven, and a fraction over those would read as turns that failed
// when they were never attempted; what it stopped on is in the failure listing
// instead. A run that passed already says so.
func turnTally(res eval.ScriptResult) string {
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
