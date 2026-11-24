package main

import (
	"context"
	"encoding/json"

	"github.com/gojargo/jargo/flows"
)

// proceedToTopicResult is what proceed_to_topic reports.
type proceedToTopicResult struct {
	GuestSummary string `json:"guest_summary"`
}

// startInterviewResult is what start_interview reports.
type startInterviewResult struct {
	Topic string `json:"topic"`
}

// handlers is the code behind the podcast interview flow defined in flow.yaml.
//
// Each tool is a NodeFunction that carries its own description and parameters.
// None of them chooses the next node. They return flows.TransitionInYAML in the
// next-node slot and the flow config decides where each one leads, including
// the interview node's transition back to itself.
func handlers() flows.Handlers {
	return flows.Handlers{
		"proceed_to_topic": flows.NodeFunction{
			Name:        "proceed_to_topic",
			Description: "Use after the guest has introduced themselves.",
			Properties: map[string]any{
				"guest_summary": map[string]any{
					"type":        "string",
					"description": "A quick summary of who the guest is (name, role, area of expertise, etc.).",
				},
			},
			Required: []string{"guest_summary"},
			Handler:  proceedToTopic,
		},
		"start_interview": flows.NodeFunction{
			Name:        "start_interview",
			Description: "Use this when the guest has shared a clear topic they want to explore.",
			Properties: map[string]any{
				"topic": map[string]any{
					"type":        "string",
					"description": "The topic the guest wants to discuss.",
				},
			},
			Required: []string{"topic"},
			Handler:  startInterview,
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

// proceedToTopic is used after the guest has introduced themselves.
func proceedToTopic(
	_ context.Context, args json.RawMessage, _ *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		GuestSummary string `json:"guest_summary"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	result, err := toJSON(proceedToTopicResult(in))
	if err != nil {
		return "", nil, err
	}
	return result, flows.TransitionInYAML, nil
}

// startInterview is used when the guest has shared a clear topic they want to
// explore.
func startInterview(
	_ context.Context, args json.RawMessage, _ *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Topic string `json:"topic"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	result, err := toJSON(startInterviewResult(in))
	if err != nil {
		return "", nil, err
	}
	return result, flows.TransitionInYAML, nil
}
