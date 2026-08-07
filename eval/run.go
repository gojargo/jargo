package eval

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
)

// Bot builds the bot's pipeline task around the harness-provided transport
// endpoints. The pipeline must include an rtvi.Processor so the harness can
// drive it (send-text) and observe its events. The builder is transport-
// agnostic: the harness supplies a WebSocket transport's input and output.
type Bot func(in, out processor.Processor) *pipeline.Task

// Options configures a scenario run.
type Options struct {
	// Judge grades `judge:` assertions; nil is fine when no scenario uses one.
	// It holds the conversation it grades against, so give each scenario its own
	// rather than reusing one across scenarios.
	Judge Judge
	// UserTTS selects audio mode. When set, each user turn is synthesized with
	// this TTS service and streamed to the bot as microphone audio (so the bot's
	// real VAD, turn detection and STT run), instead of the text-mode send-text.
	// Any jargo TTS service works, e.g. cartesia.NewTTS(...).
	UserTTS *tts.Base
}

// Run plays the scenario at path against buildBot, hosted in-process over a
// loopback WebSocket, and reports every unmet expectation through t. Text mode:
// each user turn is injected as RTVI send-text, so audio processors sit idle.
// For an LLM judge or audio mode, use RunWith.
func Run(t *testing.T, path string, buildBot Bot) {
	t.Helper()
	RunWith(t, path, buildBot, Options{})
}

// RunWithJudge is Run with an LLM judge for the scenario's `judge:` assertions
// (see NewLLMJudge). Pass nil when no scenario uses `judge:`.
func RunWithJudge(t *testing.T, path string, buildBot Bot, judge Judge) {
	t.Helper()
	RunWith(t, path, buildBot, Options{Judge: judge})
}

// RunWith is Run with explicit Options (a judge, audio mode, or both).
func RunWith(t *testing.T, path string, buildBot Bot, opts Options) {
	t.Helper()
	scenario, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Host(context.Background(), scenario, buildBot, opts)
	if err != nil {
		t.Fatalf("eval: %v", err)
	}
	for _, f := range res.Failures {
		t.Errorf("%s", f)
	}
}

// Host stands up buildBot on an in-process loopback WebSocket and plays scenario
// against it, returning the result. Run wraps this for the go-test path; callers
// that want the Result directly (custom reporting, benchmarks) can use it.
func Host(ctx context.Context, scenario *Scenario, buildBot Bot, opts Options) (Result, error) {
	srv := httptest.NewServer(Handler(buildBot))
	defer srv.Close()

	return runAgainst(ctx, scenario, wsURL(srv.URL), opts)
}

// Handler serves buildBot over RTVI on a plain WebSocket: it accepts one client
// per connection, wires the transport into the bot's pipeline, and runs it until
// the client disconnects. Mount it in a bot's own HTTP server to expose an eval
// endpoint that `jargo eval run`, or any RTVI WebSocket client, can drive. The
// in-process Host uses it too.
//
// The endpoint speaks RTVI through the eval serializer, which additionally
// understands the harness's own control messages. Serving the bot's production
// RTVI endpoint to the harness instead works, but a scenario asserting on tool
// arguments will not see them: raising the report level is deliberately
// something only this serializer allows.
func Handler(buildBot Bot) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tr, err := wsserver.Accept(w, r, newSerializer(), transport.DefaultParams())
		if err != nil {
			return
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		go func() {
			<-tr.Done() // the client disconnected or the socket closed
			cancel()
		}()
		_ = buildBot(tr.Input(), tr.Output()).Run(ctx)
	})
}

// RunURL plays scenario against a bot already listening at botURL (a ws:// or
// wss:// RTVI endpoint). It backs the command-line runner.
func RunURL(ctx context.Context, scenario *Scenario, botURL string, judge Judge) (Result, error) {
	return runAgainst(ctx, scenario, botURL, Options{Judge: judge})
}

// runAgainst connects to a bot and plays the scenario.
func runAgainst(ctx context.Context, scenario *Scenario, url string, opts Options) (Result, error) {
	c, err := dial(ctx, url)
	if err != nil {
		return Result{Scenario: scenario.Name}, err
	}
	defer c.close()

	sess := &session{client: c, scenario: scenario, judge: opts.Judge, userTTS: opts.UserTTS}
	return sess.run(ctx)
}

// wsURL rewrites an http(s) base URL to its ws(s) form.
func wsURL(httpURL string) string {
	return "ws" + strings.TrimPrefix(httpURL, "http")
}
