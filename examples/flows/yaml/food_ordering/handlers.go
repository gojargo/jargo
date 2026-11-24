package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/gojargo/jargo/flows"
)

// The order fields, named in each tool's parameters and in the order a tool
// records.
const (
	fieldSize      = "size"
	fieldPizzaType = "pizza_type"
	fieldCount     = "count"
	fieldRollType  = "roll_type"
)

// errUnknownSize reports a pizza size the menu has no price for.
var errUnknownSize = errors.New("unknown pizza size")

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

// deliveryEstimateResult is what get_delivery_estimate reports.
type deliveryEstimateResult struct {
	Time string `json:"time"`
}

// handlers is the code behind the food-ordering flow defined in flow.yaml.
//
// Each tool is a NodeFunction: its description and parameters are written
// here, and its handler does the work. None of them chooses the next node.
// They return flows.TransitionInYAML in the next-node slot and the flow config
// decides where each one leads.
func handlers() flows.Handlers {
	return flows.Handlers{
		// Pre-action handler
		"check_kitchen_status": flows.ActionHandler(checkKitchenStatus),

		// Tools that do work
		"select_pizza_order": flows.NodeFunction{
			Name:        "select_pizza_order",
			Description: "Record the pizza order details.",
			Properties: map[string]any{
				fieldSize: json.RawMessage(`{"type": "string", ` +
					`"description": "Size of the pizza. Must be one of \"small\", \"medium\", or \"large\"."}`),
				fieldPizzaType: json.RawMessage(`{"type": "string", ` +
					`"description": "Type of pizza. Must be one of \"pepperoni\", \"cheese\", \"supreme\", ` +
					`or \"vegetarian\"."}`),
			},
			Required: []string{fieldSize, fieldPizzaType},
			Handler:  selectPizzaOrder,
		},
		"select_sushi_order": flows.NodeFunction{
			Name:        "select_sushi_order",
			Description: "Record the sushi order details.",
			Properties: map[string]any{
				fieldCount: json.RawMessage(`{"type": "integer", ` +
					`"description": "Number of sushi rolls to order. Must be between 1 and 10."}`),
				fieldRollType: json.RawMessage(`{"type": "string", ` +
					`"description": "Type of sushi roll. Must be one of \"california\", \"spicy tuna\", ` +
					`\"rainbow\", or \"dragon\"."}`),
			},
			Required: []string{fieldCount, fieldRollType},
			Handler:  selectSushiOrder,
		},
		"get_delivery_estimate": flows.NodeFunction{
			Name:        "get_delivery_estimate",
			Description: "Provide delivery estimate information.",
			Handler:     getDeliveryEstimate,
		},
	}
}

// toJSON renders a function result the way it is written to the conversation.
func toJSON(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// checkKitchenStatus checks if the kitchen is open and logs the status.
func checkKitchenStatus(context.Context, flows.ActionConfig, *flows.FlowManager) error {
	slog.Info("Checking kitchen status")
	return nil
}

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

	basePrice := map[string]float64{"small": 10.00, "medium": 15.00, "large": 20.00}
	price, ok := basePrice[in.Size]
	if !ok {
		return "", nil, fmt.Errorf("%w: %q", errUnknownSize, in.Size)
	}

	result, err := toJSON(pizzaOrderResult{Size: in.Size, Type: in.PizzaType, Price: price})
	if err != nil {
		return "", nil, err
	}

	fm.State().Set("order", map[string]any{
		"type":         "pizza",
		fieldSize:      in.Size,
		fieldPizzaType: in.PizzaType,
		"price":        price,
	})

	return result, flows.TransitionInYAML, nil
}

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

	price := float64(in.Count) * 8.00

	result, err := toJSON(sushiOrderResult{Count: in.Count, Type: in.RollType, Price: price})
	if err != nil {
		return "", nil, err
	}

	fm.State().Set("order", map[string]any{
		"type":        "sushi",
		fieldCount:    in.Count,
		fieldRollType: in.RollType,
		"price":       price,
	})

	return result, flows.TransitionInYAML, nil
}

// getDeliveryEstimate provides delivery estimate information.
func getDeliveryEstimate(context.Context, json.RawMessage, *flows.FlowManager) (string, *flows.NodeConfig, error) {
	deliveryTime := time.Now().Add(30 * time.Minute)
	result, err := toJSON(deliveryEstimateResult{
		Time: "about 30 minutes, around " + deliveryTime.Format("3:04 PM"),
	})
	if err != nil {
		return "", nil, err
	}
	return result, flows.TransitionInYAML, nil
}
