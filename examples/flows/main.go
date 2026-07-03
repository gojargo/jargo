// Command flows is a headless jargo voice backend driven by a conversation flow.
//
// It runs a short, structured check-in: the assistant greets the caller, moves
// on once they agree to start, asks how they slept and whether they took their
// medication, records the answers with a tool call, thanks them and stops. The
// conversation graph lives in the flows package; the transitions are driven by
// the tools the assistant calls, not by branching prompt text.
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

	"github.com/gojargo/jargo/aggregators"
	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/turn"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/audio/vadproc"
	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/anthropic"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/gojargo/jargo/turns"
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
	conn, err := pionrtc.NewConnection()
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
func runBot(conn *pionrtc.Connection) {
	defer func() { _ = conn.Close() }()

	stt := deepgram.NewSTT(deepgram.Config{
		APIKey:     os.Getenv("DEEPGRAM_API_KEY"),
		SampleRate: opus.SampleRate,
		Language:   language.French,
	})
	llmSvc := anthropic.NewLLM(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
	tts := elevenlabs.NewTTS(elevenlabs.Config{
		APIKey:   os.Getenv("ELEVENLABS_API_KEY"),
		Language: language.French,
	})

	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate
	t := pionrtc.NewTransport(conn, params)

	convo := frames.NewLLMContext("") // the flow's opening node sets the persona.

	vadProc, turnsProc := buildTurnStack()
	procs := []processor.Processor{t.Input()}
	if vadProc != nil {
		procs = append(procs, vadProc)
	}
	procs = append(procs, stt)
	var aggOpts []aggregators.Option
	if turnsProc != nil {
		procs = append(procs, turnsProc)
		aggOpts = append(aggOpts, aggregators.WithTurnTaking())
	}
	agg := aggregators.New(convo, aggOpts...)
	procs = append(procs, agg.User(), llmSvc, tts, rtvi.NewProcessor(), t.Output(), agg.Assistant())

	task := pipeline.NewTask(pipeline.New(procs...), pipeline.TaskParams{
		AudioInSampleRate:  opus.SampleRate,
		AudioOutSampleRate: opus.SampleRate,
		EnableMetrics:      true,
		EnableUsageMetrics: true,
	})

	fm, err := flows.New(flows.Config{LLM: llmSvc, Context: convo, Enqueuer: task})
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

// startNode greets the caller and waits for them to agree before starting the
// check-in. It sets the assistant's persona for the whole session.
func startNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "start",
		RoleMessage: "Tu es Domio, un assistant vocal chaleureux qui accompagne une personne âgée. " +
			"Parle en français, avec des phrases courtes et bienveillantes.",
		TaskMessages: []frames.Message{{
			Role: frames.RoleUser,
			Text: "Salue la personne en une phrase et demande-lui si elle est prête à faire son point quotidien.",
		}},
		Functions: []flows.NodeFunction{{
			Name:        "begin_checkin",
			Description: "La personne est prête à commencer son point quotidien.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{}}`),
			Handler:     beginCheckin,
		}},
	}
}

// questionsNode collects the two check-in answers and records them.
func questionsNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "questions",
		TaskMessages: []frames.Message{{
			Role: frames.RoleUser,
			Text: "Demande à la personne comment elle a dormi, puis si elle a bien pris ses médicaments. " +
				"Pose une seule question à la fois. Quand tu as les deux réponses, appelle l'outil record_checkin.",
		}},
		Functions: []flows.NodeFunction{{
			Name:        "record_checkin",
			Description: "Enregistre le point quotidien une fois le sommeil et la prise de médicaments connus.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"sleep": {"type": "string", "description": "Comment la personne a dormi."},
					"medication": {"type": "string", "description": "Si la personne a pris ses médicaments."}
				},
				"required": ["sleep", "medication"]
			}`),
			Handler: recordCheckin,
		}},
	}
}

// doneNode thanks the caller and ends the check-in. It has no functions, so once
// it has spoken the conversation simply waits.
func doneNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "done",
		TaskMessages: []frames.Message{{
			Role: frames.RoleUser,
			Text: "Remercie chaleureusement la personne pour son point quotidien et souhaite-lui une belle journée.",
		}},
	}
}

// beginCheckin moves the flow from the greeting to the questions node.
func beginCheckin(_ context.Context, _ json.RawMessage, _ *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", questionsNode(), nil
}

// recordCheckin logs the collected answers and moves the flow to the closing
// node.
func recordCheckin(_ context.Context, args json.RawMessage, _ *flows.FlowManager) (string, *flows.NodeConfig, error) {
	var in struct {
		Sleep      string `json:"sleep"`
		Medication string `json:"medication"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, fmt.Errorf("decode record_checkin args: %w", err)
	}
	slog.Info("check-in recorded", "sleep", in.Sleep, "medication", in.Medication)
	return `{"status": "recorded"}`, doneNode(), nil
}

// buildTurnStack builds the turn-taking stack (Silero VAD + Smart Turn v3). If
// the ONNX runtime or models cannot be loaded it logs a warning and returns
// nil, nil, so the bot runs without turn taking (and without barge-in).
func buildTurnStack() (*vadproc.Processor, *turns.UserTurnProcessor) {
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
	tp := turns.NewUserTurnProcessor(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: turns.DefaultStartStrategies(),
			Stop:  []turns.StopStrategy{turns.NewTurnAnalyzerStop(turns.TurnAnalyzerConfig{Analyzer: tr})},
		},
	})
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
