package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/gojargo/jargo/flows"
)

// handlers is the one tool for flow.yaml.
//
// A tool is a NodeFunction: its description and parameters are written here,
// beside the code. It returns flows.TransitionInYAML in its next-node slot;
// flow.yaml decides that the conversation moves to the end node afterwards.
func handlers() flows.Handlers {
	return flows.Handlers{
		"record_favorite_color": flows.NodeFunction{
			Name: "record_favorite_color",
			Description: "Record the color the user said is their favorite.\n\n" +
				"Here \"record\" means print to the console, but any logic could go here:\n" +
				"write to a database, make an API call, etc.",
			Properties: map[string]any{
				"color": map[string]any{"type": "string", "description": "The user's favorite color."},
			},
			Required: []string{"color"},
			Handler:  recordFavoriteColor,
		},
	}
}

// recordFavoriteColor records the color the user said is their favorite.
//
// Here "record" means print to the console, but any logic could go here:
// write to a database, make an API call, etc.
func recordFavoriteColor(
	_ context.Context, args json.RawMessage, _ *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Color string `json:"color"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	fmt.Println("Your favorite color is: " + in.Color)
	result, err := json.Marshal(in.Color)
	if err != nil {
		return "", nil, err
	}
	return string(result), flows.TransitionInYAML, nil
}
