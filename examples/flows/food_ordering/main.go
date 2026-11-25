// Command food_ordering is a food ordering flow whose conversation paths are
// decided at run time.
//
// The flow handles:
//
//  1. An initial greeting and food type selection (pizza or sushi).
//  2. Order details collection based on the food type.
//  3. Order confirmation and revision.
//  4. Order completion.
//
// The node functions derive their schemas the way plain functions do: a
// description, and each argument as a typed property with its own description.
// The food_ordering_advanced_functionschema example spells the schemas out.
//
// LLM_PROVIDER picks the LLM: openai_responses (the default), openai,
// anthropic, google or aws, each with its own key (see the utils package).
//
//	CARTESIA_API_KEY=… DEEPGRAM_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/flows/food_ordering
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

// Pre-action handlers

// checkKitchenStatus checks if the kitchen is open and logs the status.
func checkKitchenStatus(context.Context, flows.ActionConfig, *flows.FlowManager) error {
	slog.Info("Checking kitchen status")
	return nil
}

// Functions for the initial node

// choosePizza: the user wants to order pizza. Let's get that order started.
func choosePizza(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createPizzaNode(), nil
}

// chooseSushi: the user wants to order sushi. Let's get that order started.
func chooseSushi(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createSushiNode(), nil
}

// Functions for the pizza node

// selectPizzaOrder records the pizza order details.
func selectPizzaOrder(
	_ context.Context, args json.RawMessage, fm *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Size      string `json:"size"`
		PizzaType string `json:"pizza_type"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}

	// Simple pricing
	basePrice := map[string]float64{"small": 10.00, "medium": 15.00, "large": 20.00}
	price, ok := basePrice[in.Size]
	if !ok {
		return "", nil, fmt.Errorf("%w: %q", errUnknownSize, in.Size)
	}

	result, err := toJSON(pizzaOrderResult{Size: in.Size, Type: in.PizzaType, Price: price})
	if err != nil {
		return "", nil, err
	}

	// Store the order details in the flow state.
	fm.State().Set("order", pizzaOrder{Type: "pizza", Size: in.Size, PizzaType: in.PizzaType, Price: price})

	return result, createConfirmationNode(), nil
}

// Functions for the sushi node

// selectSushiOrder records the sushi order details.
func selectSushiOrder(
	_ context.Context, args json.RawMessage, fm *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Count    int    `json:"count"`
		RollType string `json:"roll_type"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}

	// Simple pricing: $8 per roll
	price := float64(in.Count) * 8.00

	result, err := toJSON(sushiOrderResult{Count: in.Count, Type: in.RollType, Price: price})
	if err != nil {
		return "", nil, err
	}

	// Store the order details in the flow state.
	fm.State().Set("order", sushiOrder{Type: "sushi", Count: in.Count, RollType: in.RollType, Price: price})

	return result, createConfirmationNode(), nil
}

// Functions for the confirmation node

// completeOrder: the user confirms the order is correct.
func completeOrder(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createEndNode(), nil
}

// reviseOrder: the user wants to make changes to their order.
func reviseOrder(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	return "", createInitialNode(), nil
}

// Global functions

// getDeliveryEstimate provides delivery estimate information.
func getDeliveryEstimate(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	deliveryTime := time.Now().Add(30 * time.Minute)
	result, err := toJSON(deliveryEstimateResult{
		Time: "about 30 minutes, around " + deliveryTime.Format("3:04 PM"),
	})
	if err != nil {
		return "", nil, err
	}
	return result, nil, nil
}

// Node creation functions

// createInitialNode creates the initial node for food type selection.
func createInitialNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "initial",
		RoleMessage: "You are an order-taking assistant. You must ALWAYS use the available functions to " +
			"progress the conversation. This is a phone conversation and your responses will be converted " +
			"to audio. Keep the conversation friendly, casual, and polite. Keep every reply to one or two " +
			"short sentences, the way a person on the phone talks, and don't repeat the order unless the " +
			"caller asks. Only give a delivery time that came from get_delivery_estimate, and say that " +
			"time, never just 'soon'. Avoid outputting special characters and emojis.",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Greet the caller briefly and ask whether they'd like pizza or sushi, then wait for them " +
				"to use a function to choose. Be friendly and casual; you're taking an order for food over " +
				"the phone.",
		}},
		PreActions: []flows.ActionConfig{{
			Type:    "function",
			Handler: checkKitchenStatus,
		}},
		Functions: []flows.NodeFunction{
			{
				Name:        "choose_pizza",
				Description: "User wants to order pizza. Let's get that order started.",
				Handler:     choosePizza,
			},
			{
				Name:        "choose_sushi",
				Description: "User wants to order sushi. Let's get that order started.",
				Handler:     chooseSushi,
			},
		},
	}
}

// createPizzaNode creates the pizza ordering node.
func createPizzaNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "choose_pizza",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: `Your only job on this step is to get the pizza's size and type and call select_pizza_order ` +
				`with them. If the caller has already given both, call it immediately; if one is missing, ask ` +
				`for it in one short sentence. Don't quote a price or say the order is in: the function works ` +
				`out the price and the next step reads the order back. Pizza is the only thing on this menu; ` +
				`don't offer drinks or sides.

Pricing, for questions only:
- Small: $10
- Medium: $15
- Large: $20`,
		}},
		Functions: []flows.NodeFunction{{
			Name:        "select_pizza_order",
			Description: "Record the pizza order details.",
			Properties: map[string]any{
				"size": json.RawMessage(`{"type": "string", ` +
					`"description": "Size of the pizza. Must be one of \"small\", \"medium\", or \"large\"."}`),
				"pizza_type": json.RawMessage(`{"type": "string", ` +
					`"description": "Type of pizza. Must be one of \"pepperoni\", \"cheese\", \"supreme\", ` +
					`or \"vegetarian\"."}`),
			},
			Required: []string{"size", "pizza_type"},
			Handler:  selectPizzaOrder,
		}},
	}
}

// createSushiNode creates the sushi ordering node.
func createSushiNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "choose_sushi",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: `Your only job on this step is to get the roll count and type and call select_sushi_order ` +
				`with them. If the caller has already given both, call it immediately; if one is missing, ask ` +
				`for it in one short sentence. Don't quote a price or say the order is in: the function works ` +
				`out the price and the next step reads the order back. Sushi is the only thing on this menu; ` +
				`don't offer drinks or sides.

Pricing, for questions only:
- $8 per roll`,
		}},
		Functions: []flows.NodeFunction{{
			Name:        "select_sushi_order",
			Description: "Record the sushi order details.",
			Properties: map[string]any{
				"count": json.RawMessage(`{"type": "integer", ` +
					`"description": "Number of sushi rolls to order. Must be between 1 and 10."}`),
				"roll_type": json.RawMessage(`{"type": "string", ` +
					`"description": "Type of sushi roll. Must be one of \"california\", \"spicy tuna\", ` +
					`\"rainbow\", or \"dragon\"."}`),
			},
			Required: []string{"count", "roll_type"},
			Handler:  selectSushiOrder,
		}},
	}
}

// createConfirmationNode creates the order confirmation node.
func createConfirmationNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "confirm",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: `Say the order back once with the total and ask if that's right, for example "So that's ` +
				`one large pepperoni, twenty dollars. Sound good?" Nothing is ordered until the caller ` +
				`confirms, so don't say the order is placed or on its way before then. If they ask a ` +
				`question, answer it in a sentence without repeating the order, and only then complete the ` +
				`order: if they ask how long delivery takes, call get_delivery_estimate and tell them the ` +
				`time it returns, and call complete_order in a later turn, never in the same turn. Use the ` +
				`available functions:
- Use complete_order when the user confirms the order is correct and has no open question
- Use revise_order if they want to change something`,
		}},
		Functions: []flows.NodeFunction{
			{
				Name:        "complete_order",
				Description: "User confirms the order is correct.",
				Handler:     completeOrder,
			},
			{
				Name:        "revise_order",
				Description: "User wants to make changes to their order.",
				Handler:     reviseOrder,
			},
		},
	}
}

// createEndNode creates the final node.
func createEndNode() *flows.NodeConfig {
	return &flows.NodeConfig{
		Name: "end",
		TaskMessages: []frames.Message{{
			Role: frames.RoleDeveloper,
			Text: "Thank the caller for the order, say when it will arrive if get_delivery_estimate gave a " +
				"time, and say goodbye, in one or two sentences.",
		}},
		PostActions: []flows.ActionConfig{{Type: "end_conversation"}},
	}
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

	// Initialize the flow manager, with the functions available at every node.
	flowManager, err := flows.New(flows.Config{
		Enqueuer:    worker,
		Watcher:     worker,
		Aggregators: agg,
		LLM:         model,
		GlobalFunctions: []flows.NodeFunction{{
			Name:        "get_delivery_estimate",
			Description: "Provide delivery estimate information.",
			Handler:     getDeliveryEstimate,
		}},
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
