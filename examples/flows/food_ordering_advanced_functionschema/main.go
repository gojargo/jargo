// Command food_ordering_advanced_functionschema is the food ordering flow with
// each node function's schema written out by hand.
//
// It is the counterpart to the food_ordering example, whose function schemas
// are derived from a description and typed arguments. Write a schema out when
// it needs control a derived one cannot give: for example a strict enum
// constraint or a numeric minimum and maximum (both used below, on the pizza
// size and type and the sushi count and type), which a derived schema can only
// hint at in prose in its descriptions.
//
// The flow handles:
//
//  1. An initial greeting and food type selection (pizza or sushi).
//  2. Order details collection based on the food type.
//  3. Order confirmation and revision.
//  4. Order completion.
//
// LLM_PROVIDER picks the LLM: openai_responses (the default), openai,
// anthropic, google or aws, each with its own key (see the utils package).
//
//	CARTESIA_API_KEY=… DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/flows/food_ordering_advanced_functionschema
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/examples/flows/utils"
	"github.com/gojargo/jargo/flows"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// typeArg is the argument both order functions take for the kind of food.
const typeArg = "type"

// errUnknownSize reports a pizza size the menu has no price for.
var errUnknownSize = errors.New("unknown pizza size")

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

// pizzaOrderResult is what select_pizza_order reports.
type pizzaOrderResult struct {
	Size  string  `json:"size"`
	Type  string  `json:"type"`
	Price float64 `json:"price"`
}

// sushiOrderResult is what select_sushi_order reports.
type sushiOrderResult struct {
	Count int     `json:"count"`
	Type  string  `json:"type"`
	Price float64 `json:"price"`
}

// pizzaOrder is the pizza order kept in the flow state.
type pizzaOrder struct {
	Type      string  `json:"type"`
	Size      string  `json:"size"`
	PizzaType string  `json:"pizza_type"`
	Price     float64 `json:"price"`
}

// sushiOrder is the sushi order kept in the flow state.
type sushiOrder struct {
	Type     string  `json:"type"`
	Count    int     `json:"count"`
	RollType string  `json:"roll_type"`
	Price    float64 `json:"price"`
}

// deliveryEstimateResult is what get_delivery_estimate reports.
type deliveryEstimateResult struct {
	Time string `json:"time"`
}

// toJSON renders a function result the way it is written to the conversation.
func toJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// formatDateTime renders a time as a date and a time of day with
// microseconds, leaving the microseconds out when there are none.
func formatDateTime(t time.Time) string {
	if t.Nanosecond()/int(time.Microsecond) == 0 {
		return t.Format("2006-01-02 15:04:05")
	}
	return t.Format("2006-01-02 15:04:05.000000")
}

// Pre-action handlers

// checkKitchenStatus checks if the kitchen is open and logs the status.
func checkKitchenStatus(context.Context, flows.ActionConfig, *flows.FlowManager) error {
	slog.Info("Checking kitchen status")
	return nil
}

// Node creation functions

// choosePizza transitions to pizza order selection.
func choosePizza(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createPizzaNode(), nil
}

// chooseSushi transitions to sushi order selection.
func chooseSushi(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createSushiNode(), nil
}

// createInitialNode creates the initial node for food type selection.
func createInitialNode() *flows.NodeConfig {
	choosePizzaFunc := flows.NodeFunction{
		Name:        "choose_pizza",
		Handler:     choosePizza,
		Description: "User wants to order pizza. Let's get that order started.",
		Properties:  map[string]any{},
		Required:    []string{},
	}

	chooseSushiFunc := flows.NodeFunction{
		Name:        "choose_sushi",
		Handler:     chooseSushi,
		Description: "User wants to order sushi. Let's get that order started.",
		Properties:  map[string]any{},
		Required:    []string{},
	}

	return &flows.NodeConfig{
		Name: "initial",
		RoleMessage: "You are an order-taking assistant. You must ALWAYS use the available functions to " +
			"progress the conversation. This is a phone conversation and your responses will be converted " +
			"to audio. Keep the conversation friendly, casual, and polite. Avoid outputting special " +
			"characters and emojis.",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "For this step, ask the user if they want pizza or sushi, and wait for them to use a " +
				"function to choose. Start off by greeting them. Be friendly and casual; you're taking an " +
				"order for food over the phone.",
		}},
		PreActions: []flows.ActionConfig{{
			Type:    "function",
			Handler: checkKitchenStatus,
		}},
		Functions: []flows.NodeFunction{choosePizzaFunc, chooseSushiFunc},
	}
}

// selectPizzaOrder handles pizza size and type selection.
func selectPizzaOrder(
	_ context.Context, args json.RawMessage, fm *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Size string `json:"size"`
		Type string `json:"type"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	size := in.Size
	pizzaType := in.Type

	// Simple pricing
	basePrice := map[string]float64{"small": 10.00, "medium": 15.00, "large": 20.00}
	price, ok := basePrice[size]
	if !ok {
		return "", nil, fmt.Errorf("%w: %q", errUnknownSize, size)
	}

	result, err := toJSON(pizzaOrderResult{Size: size, Type: pizzaType, Price: price})
	if err != nil {
		return "", nil, err
	}

	// Store the order details in the flow state.
	fm.State().Set("order", pizzaOrder{Type: "pizza", Size: size, PizzaType: pizzaType, Price: price})

	return result, createConfirmationNode(), nil
}

// createPizzaNode creates the pizza ordering node.
func createPizzaNode() *flows.NodeConfig {
	// Spelling the schema out gives precise control over the parameters (here,
	// strict enum constraints on size and type, and in the sushi node a numeric
	// minimum and maximum on the roll count) that a derived schema could only
	// describe in prose.
	selectPizzaFunc := flows.NodeFunction{
		Name:        "select_pizza_order",
		Handler:     selectPizzaOrder,
		Description: "Record the pizza order details",
		Properties: map[string]any{
			"size": json.RawMessage(`{
				"type": "string",
				"enum": ["small", "medium", "large"],
				"description": "Size of the pizza"
			}`),
			typeArg: json.RawMessage(`{
				"type": "string",
				"enum": ["pepperoni", "cheese", "supreme", "vegetarian"],
				"description": "Type of pizza"
			}`),
		},
		Required: []string{"size", typeArg},
	}

	return &flows.NodeConfig{
		Name: "choose_pizza",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: `You are handling a pizza order.

As soon as the user has given both a size AND a type, immediately call
select_pizza_order to record it. Do not acknowledge the order conversationally,
ask whether they want anything else, or wait for further confirmation first: the
confirmation step handles all of that. If the size or the type is still missing,
ask only for the missing detail.

Pricing:
- Small: $10
- Medium: $15
- Large: $20

Remember to be friendly and casual.`,
		}},
		Functions: []flows.NodeFunction{selectPizzaFunc},
	}
}

// selectSushiOrder handles sushi roll count and type selection.
func selectSushiOrder(
	_ context.Context, args json.RawMessage, fm *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Count int    `json:"count"`
		Type  string `json:"type"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	count := in.Count
	rollType := in.Type

	// Simple pricing: $8 per roll
	price := float64(count) * 8.00

	result, err := toJSON(sushiOrderResult{Count: count, Type: rollType, Price: price})
	if err != nil {
		return "", nil, err
	}

	// Store the order details in the flow state.
	fm.State().Set("order", sushiOrder{Type: "sushi", Count: count, RollType: rollType, Price: price})

	return result, createConfirmationNode(), nil
}

// createSushiNode creates the sushi ordering node.
func createSushiNode() *flows.NodeConfig {
	selectSushiFunc := flows.NodeFunction{
		Name:        "select_sushi_order",
		Handler:     selectSushiOrder,
		Description: "Record the sushi order details",
		Properties: map[string]any{
			"count": json.RawMessage(`{
				"type": "integer",
				"minimum": 1,
				"maximum": 10,
				"description": "Number of rolls to order"
			}`),
			typeArg: json.RawMessage(`{
				"type": "string",
				"enum": ["california", "spicy tuna", "rainbow", "dragon"],
				"description": "Type of sushi roll"
			}`),
		},
		Required: []string{"count", typeArg},
	}

	return &flows.NodeConfig{
		Name: "choose_sushi",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: `You are handling a sushi order.

As soon as the user has given both a roll count AND a roll type, immediately call
select_sushi_order to record it. Do not acknowledge the order conversationally,
ask whether they want anything else, or wait for further confirmation first: the
confirmation step handles all of that. If the count or the type is still missing,
ask only for the missing detail.

Pricing:
- $8 per roll

Remember to be friendly and casual.`,
		}},
		Functions: []flows.NodeFunction{selectSushiFunc},
	}
}

// completeOrder transitions to the end state.
func completeOrder(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createEndNode(), nil
}

// reviseOrder transitions to the start for order revision.
func reviseOrder(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createInitialNode(), nil
}

// createConfirmationNode creates the order confirmation node.
func createConfirmationNode() *flows.NodeConfig {
	completeOrderFunc := flows.NodeFunction{
		Name:        "complete_order",
		Handler:     completeOrder,
		Description: "User confirms the order is correct",
		Properties:  map[string]any{},
		Required:    []string{},
	}

	reviseOrderFunc := flows.NodeFunction{
		Name:        "revise_order",
		Handler:     reviseOrder,
		Description: "User wants to make changes to their order",
		Properties:  map[string]any{},
		Required:    []string{},
	}

	return &flows.NodeConfig{
		Name: "confirm",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: `Read back the complete order details to the user and ask if they want anything else or ` +
				`if they want to make changes. Use the available functions:
- Use complete_order when the user confirms that the order is correct and no changes are needed
- Use revise_order if they want to change something

Be friendly and clear when reading back the order details.`,
		}},
		Functions: []flows.NodeFunction{completeOrderFunc, reviseOrderFunc},
	}
}

// createEndNode creates the final node.
func createEndNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "end",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Thank the user for their order and end the conversation politely and concisely.",
		}},
		PostActions: []flows.ActionConfig{{Type: "end_conversation"}},
	}
}

// getDeliveryEstimate provides delivery estimate information. It is
// available at every node.
func getDeliveryEstimate(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	deliveryTime := time.Now().Add(30 * time.Minute)
	result, err := toJSON(deliveryEstimateResult{Time: formatDateTime(deliveryTime)})
	if err != nil {
		return "", nil, err
	}
	return result, nil, nil
}

// runBot runs the food ordering bot.
func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "820a3788-2b37-4d21-847a-b65d8a68c99a", // Salesman
	})
	// The LLM service is built by utils.CreateLLM. The default is OpenAI
	// Responses; set LLM_PROVIDER to change it.
	model, err := utils.CreateLLM(ctx, "", "")
	if err != nil {
		return err
	}

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo,
		aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}),
		aggregators.WithTurns(turns.Config{
			Strategies: turns.FilterIncompleteUserTurnStrategies(nil, llm.UserTurnCompletionConfig{}),
		}),
	)

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		model,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	// The "global" function available at every node.
	getDeliveryEstimateFunc := flows.NodeFunction{
		Name:        "get_delivery_estimate",
		Handler:     getDeliveryEstimate,
		Description: "Get a delivery estimate for the current order",
		Properties:  map[string]any{},
		Required:    []string{},
	}

	// Initialize the flow manager.
	flowManager, err := flows.New(flows.Config{
		Enqueuer:        worker,
		Watcher:         worker,
		Aggregators:     agg,
		LLM:             model,
		GlobalFunctions: []flows.NodeFunction{getDeliveryEstimateFunc},
	})
	if err != nil {
		return err
	}

	event.On(t.Events(), transport.EventClientConnected, func(ctx context.Context, _ any) {
		slog.Info("Client connected")
		// Kick off the conversation with the initial node.
		if err := flowManager.Initialize(ctx, createInitialNode()); err != nil {
			slog.Error("initialize flow", "err", err)
		}
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
