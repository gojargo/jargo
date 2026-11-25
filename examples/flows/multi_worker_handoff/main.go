// Command multi_worker_handoff composes conversation flows with multiple
// workers: a free-form LLM router and a structured flow worker.
//
// Three workers share a single bus:
//
//   - A main worker owns the transport (STT, TTS) and the shared conversation
//     context. It does not run an LLM itself; instead it bridges user and
//     assistant frames onto the bus so other workers can take turns speaking
//     to the user.
//   - A router worker (a plain LLM worker) handles open-ended chit-chat and
//     general questions about the restaurant. When the user wants to book a
//     table it hands off to the reservation worker.
//   - A reservation worker (buildReservationWorker) drives a structured flow:
//     party size, then time, then an availability check, then confirmation.
//     When it is done, or if the user changes their mind, it hands control
//     back to the router.
//
// Only one worker is active at a time. Hand-offs are seamless: the user never
// hears that they've been transferred.
//
// The reservation worker is built as a plain pipeline worker, the same way the
// sensor-controller example builds its worker. A FlowManager is wired onto the
// worker and the flow is (re)initialized from the worker's on_activated event
// handler each time control is handed to it. The shared aggregators are owned
// by the main worker, so every worker speaks into the same conversation
// history.
//
// LLM_PROVIDER picks the LLM: openai_responses (the default), openai,
// anthropic, google or aws, each with its own key (see the utils package).
//
//	CARTESIA_API_KEY=… DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/flows/multi_worker_handoff
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"slices"
	"strings"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/examples/flows/utils"
	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
	"github.com/gojargo/jargo/workers/llmworker"
)

const (
	mainName        = "restaurant"
	routerName      = "router"
	reservationName = "reservation"
)

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

// toJSON renders a function result the way it is written to the conversation.
func toJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// developerMessages is the activation messages for a worker: one developer
// message carrying content.
func developerMessages(content string) []map[string]any {
	return []map[string]any{{"role": "developer", "content": content}}
}

// =============================================================================
// Mock reservation backend.
// =============================================================================

// mockReservationSystem simulates a restaurant reservation API.
type mockReservationSystem struct {
	bookedTimes []string
}

func newMockReservationSystem() *mockReservationSystem {
	return &mockReservationSystem{bookedTimes: []string{"7:00 PM", "8:00 PM"}}
}

// checkAvailability returns whether a time is open and, if not, some
// alternatives.
func (m *mockReservationSystem) checkAvailability(ctx context.Context, _ int, t string) (bool, []string, error) {
	// Simulate a network call.
	select {
	case <-time.After(500 * time.Millisecond):
	case <-ctx.Done():
		return false, nil, ctx.Err()
	}
	isAvailable := !slices.Contains(m.bookedTimes, t)
	var alternatives []string
	if !isAvailable {
		allTimes := []string{"5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM"}
		for _, at := range allTimes {
			if !slices.Contains(m.bookedTimes, at) {
				alternatives = append(alternatives, at)
			}
		}
	}
	return isAvailable, alternatives, nil
}

// =============================================================================
// Reservation worker: a structured conversation flow.
// =============================================================================

// partySizeResult is the result of recording the party size.
type partySizeResult struct {
	Size int `json:"size"`
}

// availabilityResult is the result of an availability check.
type availabilityResult struct {
	Time      string `json:"time"`
	Available bool   `json:"available"`
}

// reservationFlow holds what the reservation worker's nodes and functions
// reach.
type reservationFlow struct {
	worker            *pipeline.Worker
	reservationSystem *mockReservationSystem
}

// buildReservationWorker builds the reservation worker: a conversation flow
// as a pipeline worker.
//
// The worker's pipeline is just the LLM. A non-nil Bridged wraps it with bus
// edge processors so user frames arrive from the main worker and generated
// frames are sent back the same way. A FlowManager drives the conversation;
// it shares the main worker's aggregators so the whole session uses a single
// conversation history.
//
// The worker starts inactive and stays quiet until the router hands it
// control. The on_activated event handler initializes the flow the first time
// and resumes it on subsequent hand-offs.
func buildReservationWorker(
	model utils.LLM, agg *aggregators.Pair, reservationSystem *mockReservationSystem,
) (*pipeline.Worker, error) {
	worker := pipeline.NewWorker(pipeline.New(model), pipeline.WorkerConfig{
		Name:                    reservationName,
		Active:                  new(false),
		Bridged:                 []string{},
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	flowManager, err := flows.New(flows.Config{
		Enqueuer:    worker,
		Watcher:     worker,
		Aggregators: agg,
		LLM:         model,
	})
	if err != nil {
		return nil, err
	}

	rf := &reservationFlow{worker: worker, reservationSystem: reservationSystem}

	// --- Activation: start or resume the flow ------------------------------

	endConversationAction := func(ctx context.Context, action flows.ActionConfig, _ *flows.FlowManager) error {
		reason, _ := action.Params["reason"].(string)
		worker.End(ctx, reason)
		return nil
	}

	if err := flowManager.RegisterAction("end_conversation", endConversationAction); err != nil {
		return nil, err
	}

	var initialized atomic.Bool

	event.OnSignal(worker.Events(), workers.EventActivated, func(ctx context.Context) {
		if !initialized.Swap(true) {
			if err := flowManager.Initialize(ctx, rf.partySizeNode()); err != nil {
				slog.Error("initialize flow", "err", err)
			}
		} else {
			// Control was handed back to us; restart the reservation flow.
			if err := flowManager.SetNode(ctx, rf.partySizeNode()); err != nil {
				slog.Error("set node", "err", err)
			}
		}
	})

	return worker, nil
}

// --- Nodes -------------------------------------------------------------

func (rf *reservationFlow) partySizeNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "party_size",
		RoleMessage: "You are a reservation assistant for La Maison, an upscale French " +
			"restaurant. Your only job is collecting reservation details. When the " +
			"user gives a detail like party size or time, always call the tool that " +
			"records it to move the booking forward, even if that detail already " +
			"appears earlier in the conversation; never advance based on history " +
			"alone. If the user asks about anything else (weather, menu, general " +
			"questions), call the transfer_to_router tool instead of answering " +
			"yourself. Be casual and friendly. This is a voice conversation, so " +
			"avoid special characters and emojis.",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Ask how many people are in their party. When they answer, " +
				"always call collect_party_size to record it and continue, " +
				"even if a party size came up earlier in the conversation.",
		}},
		Functions: []flows.NodeFunction{rf.collectPartySizeFunc(), rf.transferToRouterFunc()},
	}
}

func (rf *reservationFlow) timeSelectionNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "get_time",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Ask what time they would like to dine. The restaurant is " +
				"open from 5 PM to 10 PM. When they give a time, always call " +
				"check_availability to verify it and continue; do not assume " +
				"availability from earlier in the conversation.",
		}},
		Functions: []flows.NodeFunction{rf.checkAvailabilityFunc(), rf.transferToRouterFunc()},
	}
}

func (rf *reservationFlow) confirmationNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "confirm",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Confirm the reservation details and ask if there is anything else.",
		}},
		Functions: []flows.NodeFunction{rf.endReservationFunc(), rf.transferToRouterFunc()},
	}
}

func endNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "end",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Thank them for their reservation and say goodbye.",
		}},
		PostActions: []flows.ActionConfig{{Type: "end_conversation"}},
	}
}

// --- Flow functions ----------------------------------------------------

// collectPartySizeFunc records the party size and advances the reservation to
// choosing a time.
func (rf *reservationFlow) collectPartySizeFunc() flows.NodeFunction {
	return flows.NodeFunction{
		Name: "collect_party_size",
		Description: "Record the party size and advance the reservation to choosing a time.\n\n" +
			"Call this whenever the user gives a party size. Always call it to move\n" +
			"forward; do not skip it just because a size already appears earlier in\n" +
			"the conversation.",
		Properties: map[string]any{
			"size": json.RawMessage(`{"type": "integer", ` +
				`"description": "Number of people in the party. Must be between 1 and 12."}`),
		},
		Required: []string{"size"},
		Handler:  rf.collectPartySize,
	}
}

func (rf *reservationFlow) collectPartySize(
	_ context.Context, args json.RawMessage, fm *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Size int `json:"size"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	fm.State().Set("party_size", in.Size)
	result, err := toJSON(partySizeResult{Size: in.Size})
	if err != nil {
		return "", nil, err
	}
	return result, rf.timeSelectionNode(), nil
}

// checkAvailabilityFunc checks availability for the requested time and
// advances the reservation.
func (rf *reservationFlow) checkAvailabilityFunc() flows.NodeFunction {
	return flows.NodeFunction{
		Name: "check_availability",
		Description: "Check availability for the requested time and advance the reservation.\n\n" +
			"Call this whenever the user gives a time. Availability can change, so\n" +
			"always check it fresh rather than assuming it from earlier in the\n" +
			"conversation, and call it every time to move the booking forward.",
		Properties: map[string]any{
			"time": json.RawMessage(`{"type": "string", "description": "Reservation time (e.g., '6:00 PM')."}`),
		},
		Required: []string{"time"},
		Handler:  rf.checkAvailability,
	}
}

func (rf *reservationFlow) checkAvailability(
	ctx context.Context, args json.RawMessage, fm *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Time string `json:"time"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}

	partySize := 2
	if v, ok := fm.State().Get("party_size"); ok {
		if n, ok := v.(int); ok {
			partySize = n
		}
	}
	isAvailable, alternatives, err := rf.reservationSystem.checkAvailability(ctx, partySize, in.Time)
	if err != nil {
		return "", nil, err
	}

	if isAvailable {
		fm.State().Set("time", in.Time)
		var result string
		if result, err = toJSON(availabilityResult{Time: in.Time, Available: true}); err != nil {
			return "", nil, err
		}
		return result, rf.confirmationNode(), nil
	}

	timesList := strings.Join(alternatives, ", ")
	noAvailability := &flows.NodeConfig{
		Name: "no_availability",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: fmt.Sprintf("Apologize that %s is not available. "+
				"Suggest these alternative times: %s.", in.Time, timesList),
		}},
		Functions: []flows.NodeFunction{rf.checkAvailabilityFunc(), rf.transferToRouterFunc()},
	}
	result, err := toJSON(availabilityResult{Time: in.Time, Available: false})
	if err != nil {
		return "", nil, err
	}
	return result, noAvailability, nil
}

// endReservationFunc finalizes the reservation.
func (rf *reservationFlow) endReservationFunc() flows.NodeFunction {
	return flows.NodeFunction{
		Name:        "end_reservation",
		Description: "Finalize the reservation. Call this once the user has confirmed the details.",
		Handler: func(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
			return "", endNode(), nil
		},
	}
}

// transferToRouterFunc hands the conversation back to the general assistant.
func (rf *reservationFlow) transferToRouterFunc() flows.NodeFunction {
	return flows.NodeFunction{
		Name: "transfer_to_router",
		Description: "Hand the conversation back to the general assistant.\n\n" +
			"Call this when the user no longer wants to make a reservation, or asks\n" +
			"a general question unrelated to booking a table.",
		Properties: map[string]any{
			"reason": json.RawMessage(`{"type": "string", "description": "Why control is being handed back ` +
				`(e.g. 'user changed\ntheir mind', 'user asked about the menu')."}`),
		},
		Required: []string{"reason"},
		Handler:  rf.transferToRouter,
	}
}

func (rf *reservationFlow) transferToRouter(
	ctx context.Context, args json.RawMessage, _ *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	slog.Info(fmt.Sprintf("Worker '%s': handing back to '%s' (%s)", reservationName, routerName, in.Reason))
	rf.worker.ActivateWorker(ctx, routerName, workers.ActivateOptions{
		Args: llmworker.ActivationArgs{
			Messages: developerMessages(in.Reason + ". Address this, then offer to continue. If the " +
				"user returns to their reservation (for example by asking " +
				"to continue booking or answering a booking question), " +
				"call the transfer_to_reservation tool instead of " +
				"collecting reservation details yourself."),
		},
		DeactivateSelf: true,
	})
	// The router is now responsible for the next turn, so hand off with
	// NoResponse to avoid running the LLM. Note that we don't need to
	// transition to any next node: on_activated re-seeds partySizeNode when
	// control returns to the reservation worker.
	return `{"status": "transferred"}`, flows.NoResponse, nil
}

// =============================================================================
// Router worker: free-form LLM that routes to the reservation flow.
// =============================================================================

// routerWorker is an open-ended assistant that transfers to the reservation
// worker.
type routerWorker struct {
	*llmworker.Worker
}

// buildRouter builds the free-form router worker.
func buildRouter(model utils.LLM) *routerWorker {
	w := &routerWorker{}
	w.Worker = llmworker.New(llmworker.Config{
		Name:    routerName,
		LLM:     model,
		Tools:   w.tools(),
		Bridged: []string{},
	})
	return w
}

// tools are the router's tools, each carrying its handler.
func (w *routerWorker) tools() []frames.Tool {
	return []frames.Tool{
		{
			Name: "transfer_to_reservation",
			Description: "Transfer the user to the reservation assistant.\n\n" +
				"Call this as soon as the user wants to book, change, or ask about\n" +
				"making a table reservation.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"reason": {
						"type": "string",
						"description": "Why the user is being transferred."
					}
				},
				"required": ["reason"]
			}`),
			Handler:              llm.FunctionCallHandler(w.transferToReservation),
			CancelOnInterruption: new(false),
		},
		{
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
			Handler: llm.FunctionCallHandler(w.endConversation),
		},
	}
}

// transferToReservation transfers the user to the reservation assistant.
func (w *routerWorker) transferToReservation(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Info(fmt.Sprintf("Worker '%s': transferring to '%s' (%s)", w.Name(), reservationName, args.Reason))
	// The reservation worker speaks on activation, so this call has no reply.
	if err := params.Result(ctx, "", &frames.FunctionCallResultProperties{RunLLM: new(false)}); err != nil {
		return err
	}
	w.ActivateWorker(ctx, reservationName, workers.ActivateOptions{
		Args: llmworker.ActivationArgs{
			Messages: developerMessages(args.Reason),
		},
		DeactivateSelf: true,
	})
	return nil
}

// endConversation ends the conversation when the user says goodbye.
func (w *routerWorker) endConversation(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Info(fmt.Sprintf("Worker '%s': ending conversation (%s)", w.Name(), args.Reason))
	// The result makes the LLM say goodbye, and that is spoken before the end.
	result, err := toJSON(args.Reason)
	if err != nil {
		return err
	}
	if err := params.Result(ctx, result, nil); err != nil {
		return err
	}
	w.End(ctx, args.Reason)
	return nil
}

// =============================================================================
// Bot setup.
// =============================================================================

// runBot wires up the transport, the shared context, and the three workers.
func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting multi-worker handoff bot")

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", // Jacqueline
	})

	// The shared conversation context lives in the main worker. Both the
	// router and the reservation worker speak into this same history via the
	// bus.
	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	// The main bridge sends user-side context to the active worker and brings
	// its generated frames back so the TTS can speak them.
	bridge := bus.NewBridgeProcessor(bus.BridgeConfig{
		Bus:        r.Bus(),
		WorkerName: mainName,
	})

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		bridge,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Name: mainName,
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	// Each LLM worker gets its own LLM service instance.
	routerLLM, err := utils.CreateLLM(ctx, "", "")
	if err != nil {
		return err
	}
	router := buildRouter(routerLLM)
	reservationLLM, err := utils.CreateLLM(ctx, "", "")
	if err != nil {
		return err
	}
	reservation, err := buildReservationWorker(reservationLLM, agg, newMockReservationSystem())
	if err != nil {
		return err
	}

	r.AddWorkers(ctx, router, reservation, worker)

	event.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		// Start the conversation with the router.
		worker.ActivateWorker(ctx, routerName, workers.ActivateOptions{
			Args: llmworker.ActivationArgs{
				Messages: developerMessages("You are a friendly assistant for La Maison restaurant. Greet the " +
					"user, mention you can answer questions or book a table, and ask how " +
					"you can help. You can answer questions about the restaurant only; " +
					"you have no live information such as the weather or the news, so " +
					"if asked, say you cannot help with that and offer to help with the " +
					"restaurant instead. When the user wants to make a reservation, call " +
					"the transfer_to_reservation tool. If the user says goodbye, call the " +
					"end_conversation tool. Do not mention transferring, just do it " +
					"seamlessly. Keep responses brief, this is a voice conversation."),
			},
		})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	return r.Run(ctx, workers.RunOptions{})
}

func bot(ctx context.Context, args runner.Arguments) error {
	t, err := runner.CreateTransport(ctx, args, transportParams)
	if err != nil {
		return err
	}
	return runBot(ctx, t, args)
}

func main() {
	runner.Main(bot)
}
