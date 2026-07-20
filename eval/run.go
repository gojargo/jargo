package eval

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/rtviws"
)

// Bot builds the bot's pipeline task around the harness-provided transport
// endpoints. The pipeline must include an rtvi.Processor so the harness can
// drive it (send-text) and observe its events. The builder is transport-
// agnostic: the harness supplies a WebSocket transport's input and output.
type Bot func(in, out processor.Processor) *pipeline.Task

// Run plays the scenario at path against buildBot, hosted in-process over a
// loopback WebSocket, and reports every unmet expectation through t. The bot's
// real LLM (and STT/TTS, if wired) run; each user turn is injected as RTVI
// send-text, so audio processors sit idle.
func Run(t *testing.T, path string, buildBot Bot) {
	t.Helper()
	scenario, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Host(context.Background(), scenario, buildBot, nil)
	if err != nil {
		t.Fatalf("eval: %v", err)
	}
	for _, f := range res.Failures {
		t.Errorf("%s", f)
	}
}

// Host stands up buildBot on an in-process loopback WebSocket and plays scenario
// against it, returning the result. judge may be nil when the scenario has no
// `judge:` assertions. Run wraps this for the go-test path; callers that want
// the Result directly (custom reporting, benchmarks) can use it.
func Host(ctx context.Context, scenario *Scenario, buildBot Bot, judge Judge) (Result, error) {
	srv := httptest.NewServer(Handler(buildBot))
	defer srv.Close()

	return runAgainst(ctx, scenario, wsURL(srv.URL), judge)
}

// Handler serves buildBot over RTVI on a plain WebSocket: it accepts one client
// per connection, wires the transport into the bot's pipeline, and runs it until
// the client disconnects. Mount it in a bot's own HTTP server to expose an eval
// endpoint that `jargo eval run` — or any RTVI WebSocket client — can drive. The
// in-process Host uses it too.
func Handler(buildBot Bot) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		tr, err := wsserver.Accept(w, r, rtviws.New(), transport.DefaultParams())
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
	return runAgainst(ctx, scenario, botURL, judge)
}

// runAgainst connects to a bot and plays the scenario.
func runAgainst(ctx context.Context, scenario *Scenario, url string, judge Judge) (Result, error) {
	c, err := dial(ctx, url)
	if err != nil {
		return Result{Scenario: scenario.Name}, err
	}
	defer c.close()

	sess := &session{client: c, scenario: scenario, judge: judge}
	return sess.run(ctx)
}

// wsURL rewrites an http(s) base URL to its ws(s) form.
func wsURL(httpURL string) string {
	return "ws" + strings.TrimPrefix(httpURL, "http")
}
