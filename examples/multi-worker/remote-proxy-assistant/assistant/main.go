// Command assistant is the remote assistant LLM server of the
// remote-proxy-assistant example.
//
// It runs an HTTP server that accepts WebSocket connections from the main
// command of the example. Each connection starts a server proxy that bridges
// the socket to a local worker runner, and an LLM worker that handles the
// conversation.
//
//	OPENAI_API_KEY=… go run ./examples/multi-worker/remote-proxy-assistant/assistant
//	OPENAI_API_KEY=… go run ./examples/multi-worker/remote-proxy-assistant/assistant -port 9000
//
// Then start the main command, which connects to ws://localhost:8765/ws by
// default.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/service/wsutil"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
	"github.com/gojargo/jargo/workers/proxy"
	"github.com/joho/godotenv"
)

// readLimit bounds a single message off the socket. A frame carrying audio is
// the large case.
const readLimit = 8 << 20

// acmeAssistant handles greetings, product questions, and the end of the
// conversation.
type acmeAssistant struct {
	*llmworker.Worker
}

func newAcmeAssistant() *acmeAssistant {
	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction("You are a friendly assistant for Acme Corp. You know about three " +
		"products: Acme Rocket Boots (jet-powered boots, $299, run up to " +
		"60 mph), Acme Invisible Paint (makes anything invisible for 24 hours, " +
		"$49 per can), and Acme Tornado Kit (portable tornado generator, $199, " +
		"batteries included). Greet the user, help them with product questions, " +
		"and call end_conversation when the user says goodbye. " +
		"Keep responses brief, this is a voice conversation.")

	a := &acmeAssistant{}
	a.Worker = llmworker.New(llmworker.Config{
		Name: "assistant",
		LLM:  model,
		Tools: []frames.Tool{{
			Name:        "end_conversation",
			Description: "End the conversation when the user says goodbye.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"reason": {
						"type": "string",
						"description": "Why the conversation is ending."
					}
				},
				"required": ["reason"]
			}`),
			Handler: llm.FunctionCallHandler(a.endConversation),
		}},
		Bridged: []string{},
	})
	return a
}

// endConversation ends the conversation when the user says goodbye.
func (a *acmeAssistant) endConversation(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Info(fmt.Sprintf("Worker '%s': ending conversation (%s)", a.Name(), args.Reason))
	if err := params.Result(ctx, args.Reason, nil); err != nil {
		return err
	}
	a.End(ctx, args.Reason)
	return nil
}

// websocketEndpoint handles a WebSocket connection from the main bot's proxy.
func websocketEndpoint(w http.ResponseWriter, r *http.Request) {
	raw, err := websocket.Accept(w, r, nil)
	if err != nil {
		slog.Error("could not accept the WebSocket connection", "err", err)
		return
	}
	ctx := r.Context()

	gateway := proxy.NewServer(proxy.ServerConfig{
		Name:            "gateway",
		Conn:            wsutil.Adopt(raw, readLimit),
		LocalWorker:     "assistant",
		RemoteWorker:    "acme",
		ForwardMessages: []bus.Message{&bus.FrameMessage{}},
	})

	assistant := newAcmeAssistant()

	runner := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: new(false)})

	runner.AddWorkers(ctx, gateway, assistant)

	events.OnSignal(gateway.Events(), proxy.EventClientConnected, func(context.Context) {
		slog.Info("WebSocket client connected")
	})

	events.OnSignal(gateway.Events(), proxy.EventClientDisconnected, func(ctx context.Context) {
		slog.Info("WebSocket client disconnected")
		runner.Cancel(ctx, "WebSocket client disconnected")
	})

	slog.Info("Assistant server ready, waiting for activation")
	if err := runner.Run(ctx, workers.RunOptions{}); err != nil {
		slog.Error("Assistant server session failed", "err", err)
	}
	slog.Info("Assistant server session ended")
}

func main() {
	host := flag.String("host", "0.0.0.0", "Host to bind to")
	port := flag.Int("port", 8765, "Port to listen on")
	flag.Parse()

	_ = godotenv.Overload()

	mux := http.NewServeMux()
	mux.HandleFunc("/ws", websocketEndpoint)
	srv := &http.Server{
		Addr:              net.JoinHostPort(*host, strconv.Itoa(*port)),
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	slog.Info("Remote assistant LLM server listening", "addr", srv.Addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		slog.Error("server stopped", "err", err)
		os.Exit(1) //nolint:forbidigo // the program's entry point
	}
}
