package eval

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// Bot builds the bot's pipeline task around the harness-provided transport
// endpoints. The pipeline must include an rtvi.Processor so the harness can
// drive it (send-text) and observe its events. The builder is transport-
// agnostic: the harness supplies a WebSocket transport's input and output.
type Bot func(in, out processor.Processor) *pipeline.Worker

// Options configures a scenario run: the services a ScriptSession is given, and
// its SessionParams.
type Options struct {
	// Judge grades `eval:` assertions. Nil builds one from the scenario's
	// judge: block when it has any. It holds the conversation it grades
	// against, so give each scenario its own rather than reusing one across
	// scenarios.
	Judge Judge
	// UserTTS speaks the user's turns. When set, each user turn is synthesized
	// with this TTS service's synthesizer and streamed to the bot as audio (so
	// the bot's real VAD, turn detection and STT run), instead of the text-mode
	// send-text. Nil builds one from the scenario's user: block's speech, if
	// any. A local or HTTP service works, e.g. a Kokoro or an OpenAI TTS; a
	// WebSocket-streaming one does not.
	UserTTS TTSService
	// BotSTT transcribes the bot's audio for a scenario asserting response in
	// audio mode. Nil builds one from the scenario's judge: block's
	// transcription.
	BotSTT SegmentedSTT
	// OnProgress, when set, is called as each turn and expectation resolves, in
	// order, for a caller reporting a long run as it happens rather than at the
	// end (see EventProgress). Keep it quick: the records queue behind it.
	OnProgress func(ScriptTurnProgress)
	// Params is how the run behaves: its timeouts, its cache, its audio
	// support check.
	Params SessionParams
	// BotReadyTimeout is how long the bot has, once connected, to answer the
	// RTVI handshake with bot-ready, which covers the bot's startup; zero uses
	// Params. A bot that closes the connection first fails the run at once.
	BotReadyTimeout time.Duration
	// ConnectTimeout is how long the bot has to accept connections at all
	// before the run fails to connect; zero uses Params.
	ConnectTimeout time.Duration
}

// Run plays the scenario at path against buildBot, hosted in-process over a
// loopback WebSocket, and reports every unmet expectation through t. Text mode:
// each user turn is injected as RTVI send-text, so audio processors sit idle.
// For an LLM judge or audio mode, use RunWith.
func Run(t *testing.T, path string, buildBot Bot) {
	t.Helper()
	RunWith(t, path, buildBot, Options{})
}

// RunWithJudge is Run with a judge for the scenario's `judge:` assertions (see
// NewClassifierJudge). Pass nil when no scenario uses `judge:`.
func RunWithJudge(t *testing.T, path string, buildBot Bot, judge Judge) {
	t.Helper()
	RunWith(t, path, buildBot, Options{Judge: judge})
}

// RunWith is Run with explicit Options (a judge, audio mode, or both).
func RunWith(t *testing.T, path string, buildBot Bot, opts Options) {
	t.Helper()
	scenario, err := loadScript(path)
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
//
// What goes wrong with the run itself, a bot that never becomes ready or a
// harness that fails mid-run, is a failure in the Result, and the error is what
// RunURL returns.
func Host(ctx context.Context, scenario *ScriptScenario, buildBot Bot, opts Options) (ScriptResult, error) {
	srv := httptest.NewServer(Handler(buildBot))
	defer srv.Close()

	return runAgainst(ctx, scenario, wsURL(srv.URL), opts)
}

// Handler serves buildBot over RTVI on a plain WebSocket: it accepts one client
// per connection, wires the eval transport into the bot's pipeline, and runs it
// until the client disconnects. Mount it in a bot's own HTTP server to expose an
// eval endpoint that `jargo eval run`, or any RTVI WebSocket client, can drive.
// The in-process Host uses it too.
//
// The endpoint speaks RTVI through the eval serializer, which additionally
// understands the harness's own control messages. Serving the bot's production
// RTVI endpoint to the harness instead works, but a scenario asserting on tool
// arguments will not see them: raising the report level is deliberately
// something only this serializer allows.
//
// The harness sets each connection's flags on its URL (see
// evalws.SkipTTSQueryParam and the parameters beside it), and they apply to that
// connection alone. A bot run under the development runner needs none of this:
// the harness connects to the runner's /ws, and runner.CreateTransport builds
// the same transport from its eval-connect message.
func Handler(buildBot Bot) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		params := evalws.TransportParams{Params: wsserver.DefaultParams(), Serializer: evalws.NewSerializer()}
		conn, err := wsserver.AcceptConn(w, r, params.Params)
		if err != nil {
			return
		}
		tr := evalws.NewTransport(conn, params, evalws.ConnectionFlags{
			SkipTTS:          evalws.QueryFlag(r, evalws.SkipTTSQueryParam),
			CaptureBotAudio:  evalws.QueryFlag(r, evalws.CaptureAudioQueryParam),
			CaptureBotImages: evalws.QueryFlag(r, evalws.CaptureImagesQueryParam),
		})
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
//
// A bot that never accepts the connection, refuses it, or never becomes ready
// is a failure of the Result (a <connect> or <bot-ready> one), and so is the
// harness failing mid-run (an <error> or <harness> one), with every turn not
// reached left TurnNotRun. The error is ctx ending, or a service the scenario
// names that cannot be built.
func RunURL(ctx context.Context, scenario *ScriptScenario, botURL string, opts Options) (ScriptResult, error) {
	return runAgainst(ctx, scenario, botURL, opts)
}

// runAgainst plays the scenario against the bot at url, in a ScriptSession.
func runAgainst(ctx context.Context, scenario *ScriptScenario, url string, opts Options) (ScriptResult, error) {
	params := opts.Params
	if opts.BotReadyTimeout > 0 {
		params.BotReadyTimeout = opts.BotReadyTimeout
	}
	if opts.ConnectTimeout > 0 {
		params.ConnectTimeout = opts.ConnectTimeout
	}
	var userTTS *CachingTTSService
	if opts.UserTTS != nil {
		syn := opts.UserTTS.Synthesizer()
		var err error
		noCache := false
		userTTS, err = NewCachingTTSService(syn, CachingTTSConfig{UseCache: &noCache, SampleRate: syn.SampleRate()})
		if err != nil {
			closeJudge(ctx, opts.Judge)
			return ScriptResult{ScenarioName: scenario.Name}, err
		}
	}
	session, err := ScriptSessionFromScenario(scenario, url, params, opts.Judge, userTTS, opts.BotSTT)
	if err != nil {
		// The session closes its judge however its run ends; one that was never
		// built closes the caller's here.
		closeJudge(ctx, opts.Judge)
		return ScriptResult{ScenarioName: scenario.Name}, err
	}
	if opts.OnProgress != nil {
		event.On(&session.Registry, EventProgress, func(_ context.Context, p ScriptTurnProgress) { opts.OnProgress(p) })
	}
	return session.Run(ctx)
}

// wsURL rewrites an http(s) base URL to its ws(s) form.
func wsURL(httpURL string) string {
	return "ws" + strings.TrimPrefix(httpURL, "http")
}
