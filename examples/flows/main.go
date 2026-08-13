// Command flows is a headless jargo voice backend driven by a conversation flow.
//
// It runs a short, structured coffee order: the assistant greets the caller,
// moves on once they are ready, asks which drink and what size, records the
// order with a tool call, confirms it and stops. The conversation graph lives in
// the flows package; the transitions are driven by the tools the assistant
// calls, not by branching prompt text.
//
// Like the other voice examples this is a server only: it exposes the WebRTC
// signaling endpoint POST /offer and no web UI. Point a browser client at it
// (the nextjs-voicebot example in gojargo/jargo-client-react, with
// NEXT_PUBLIC_JARGO_URL=http://localhost:8080).
//
//	DEEPGRAM_API_KEY=… ANTHROPIC_API_KEY=… ELEVENLABS_API_KEY=… go run ./examples/flows
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/processor/vadproc"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/rtc"
	"github.com/pion/webrtc/v4"
)

func main() {
	http.HandleFunc("/offer", withCORS(handleOffer))
	slog.Info("jargo flows backend listening", "url", "http://localhost:8080", "signaling", "POST /offer")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handleOffer(w http.ResponseWriter, r *http.Request) {
	var offer webrtc.SessionDescription
	if err := json.NewDecoder(r.Body).Decode(&offer); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	conn, err := rtc.NewConnection()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	answer, err := conn.Answer(offer)
	if err != nil {
		_ = conn.Close()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	go runBot(conn)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(answer); err != nil {
		slog.Error("write answer", "err", err)
	}
}

// runBot builds and runs the STT -> LLM -> TTS pipeline for one connection and
// drives it with a FlowManager entered at the opening node.
func runBot(conn *rtc.Connection) {
	defer func() { _ = conn.Close() }()

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY"), SampleRate: opus.SampleRate})
	llmSvc := anthropic.NewLLM(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
	tts := elevenlabs.NewTTS(elevenlabs.Config{APIKey: os.Getenv("ELEVENLABS_API_KEY")})

	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	t := rtc.NewTransport(conn, params)

	convo := frames.NewLLMContext("") // the flow's opening node sets the persona.

	vadProc, turnsCfg := buildTurnStack()
	rtviProc := rtvi.NewProcessor()
	procs := []processor.Processor{rtviProc, t.Input()}
	if vadProc != nil {
		procs = append(procs, vadProc)
	}
	procs = append(procs, stt)
	var aggOpts []aggregators.Option
	if turnsCfg != nil {
		// The turn strategies run inside the user aggregator, so a turn that
		// ends on a transcript ends with that transcript in the message.
		aggOpts = append(aggOpts, aggregators.WithTurns(*turnsCfg))
	}
	agg := aggregators.New(convo, aggOpts...)
	procs = append(procs, agg.User(), llmSvc, tts, t.Output(), agg.Assistant())

	task := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{
		// The observer reports pipeline events; the processor carries them.
		Observers: []pipeline.Observer{rtvi.NewObserver(rtviProc)},
		Params: pipeline.Params{
			AudioInSampleRate:  opus.SampleRate,
			AudioOutSampleRate: opus.SampleRate,
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
	})

	fm, err := flows.New(flows.Config{
		Enqueuer:    task,
		Watcher:     task,
		Aggregators: agg,
		LLM:         llmSvc,
	})
	if err != nil {
		slog.Error("build flow manager", "err", err)
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-conn.Done()
		cancel()
	}()

	// Enter the flow; the opening node greets the caller as soon as they connect.
	if err := fm.Initialize(ctx, startNode()); err != nil {
		slog.Error("initialize flow", "err", err)
		return
	}

	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("pipeline ended", "err", err)
	}
}

// startNode greets the caller and waits for them to be ready before taking the
// order. It sets the assistant's persona for the whole session.
func startNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "start",
		RoleMessage: "You are a friendly barista taking a coffee order by voice. " +
			"Keep your replies short and warm, one or two sentences.",
		TaskMessages: []frames.Message{{
			Role: frames.RoleUser,
			Text: "Greet the customer in one sentence and ask whether they are ready to order.",
		}},
		Functions: []flows.NodeFunction{{
			Name:        "start_order",
			Description: "The customer is ready to place their order.",
			Handler:     startOrder,
		}},
	}
}

// orderNode collects the drink and size and records them.
func orderNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "order",
		TaskMessages: []frames.Message{{
			Role: frames.RoleUser,
			Text: "Ask which drink the customer would like, then what size. Ask one question at a " +
				"time. When you have both, call the record_order tool.",
		}},
		Functions: []flows.NodeFunction{{
			Name:        "record_order",
			Description: "Record the order once the drink and size are known.",
			Properties: map[string]any{
				"drink": map[string]any{"type": "string", "description": "The drink the customer ordered."},
				"size":  map[string]any{"type": "string", "description": "The size of the drink."},
			},
			Required: []string{"drink", "size"},
			Handler:  recordOrder,
		}},
	}
}

// doneNode confirms the order and ends the conversation. It has no functions, so
// once it has spoken the conversation simply waits.
func doneNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "done",
		TaskMessages: []frames.Message{{
			Role: frames.RoleUser,
			Text: "Confirm the order back to the customer and let them know it will be ready shortly.",
		}},
	}
}

// startOrder moves the flow from the greeting to the order node.
func startOrder(_ context.Context, _ json.RawMessage, _ *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", orderNode(), nil
}

// recordOrder logs the collected order and moves the flow to the closing node.
func recordOrder(_ context.Context, args json.RawMessage, _ *flows.FlowManager) (string, *flows.NodeConfig, error) {
	var in struct {
		Drink string `json:"drink"`
		Size  string `json:"size"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, fmt.Errorf("decode record_order args: %w", err)
	}
	slog.Info("order recorded", "drink", in.Drink, "size", in.Size)
	return `{"status": "recorded"}`, doneNode(), nil
}

// buildTurnStack builds the turn-taking stack (Silero VAD + Smart Turn v3). If
// the ONNX runtime or models cannot be loaded it logs a warning and returns
// nil, nil, so the bot runs without turn taking (and without barge-in).
func buildTurnStack() (*vadproc.Processor, *turns.Config) {
	vd, err := vad.NewSilero()
	if err != nil {
		slog.Warn("turn taking disabled: Silero VAD unavailable (set JARGO_ONNXRUNTIME_LIB)", "err", err)
		return nil, nil
	}
	tr, err := turn.NewSmartTurnV3()
	if err != nil {
		slog.Warn("turn taking disabled: Smart Turn unavailable", "err", err)
		_ = vd.Close()
		return nil, nil
	}
	vp := vadproc.New(vadproc.Config{VAD: vd})
	tp := &turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: turns.DefaultStartStrategies(),
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: tr})},
		},
	}
	return vp, tp
}

// withCORS allows a browser client served from another origin (e.g. the Next.js
// dev server on :3000) to POST offers to this backend.
func withCORS(h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		if r.Method == http.MethodOptions {
			w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
			w.Header().Set("Access-Control-Max-Age", "86400")
			w.WriteHeader(http.StatusNoContent)
			return
		}
		h(w, r)
	}
}
