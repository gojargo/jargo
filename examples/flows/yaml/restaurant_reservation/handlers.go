package main

import (
	"context"
	"encoding/json"
	"slices"
	"time"

	"github.com/gojargo/jargo/flows"
)

// This file has two layers, and the split is the point.
//
// The business logic, mockReservationSystem, knows nothing about the flow. It
// answers questions about tables in whatever shape suits it.
//
// The tools are thin shims over that logic. Each is a NodeFunction that
// carries its own description and parameters. A tool calls the business logic,
// then reports the outcome as a field of its result, and returns
// flows.TransitionInYAML in the next-node slot. It never chooses the next node.
// check_availability reports status as available or unavailable, and flow.yaml
// branches on that field. Any transition that depends on logic takes this
// shape: compute the decision in the tool, report it as a field, route on it in
// the config.

// Business logic: knows nothing about the flow

// mockReservationSystem simulates a restaurant reservation system API.
type mockReservationSystem struct {
	// bookedTimes are the times that are fully booked.
	bookedTimes []string
}

// newMockReservationSystem builds the system with its booked times.
func newMockReservationSystem() *mockReservationSystem {
	return &mockReservationSystem{bookedTimes: []string{"7:00 PM", "8:00 PM"}}
}

// checkAvailability checks if a table is available for the given party size
// and time. It returns the alternative times when it is not.
func (s *mockReservationSystem) checkAvailability(
	ctx context.Context, _ int, requestedTime string,
) (bool, []string, error) {
	// Simulate API call delay
	select {
	case <-ctx.Done():
		return false, nil, ctx.Err()
	case <-time.After(500 * time.Millisecond):
	}

	isAvailable := !slices.Contains(s.bookedTimes, requestedTime)

	alternatives := []string{}
	if !isAvailable {
		baseTimes := []string{"5:00 PM", "6:00 PM", "7:00 PM", "8:00 PM", "9:00 PM", "10:00 PM"}
		for _, t := range baseTimes {
			if !slices.Contains(s.bookedTimes, t) {
				alternatives = append(alternatives, t)
			}
		}
	}

	return isAvailable, alternatives, nil
}

// reservationSystem is the reservation system the tools ask.
//
//nolint:gochecknoglobals // the example's one reservation system
var reservationSystem = newMockReservationSystem()

// Flow tools: thin shims that report outcomes for the config to route on

// partySizeResult is what collect_party_size reports.
type partySizeResult struct {
	Size   int    `json:"size"`
	Status string `json:"status"`
}

// timeResult is what check_availability reports. Status is "available" or
// "unavailable".
type timeResult struct {
	Status           string   `json:"status"`
	Time             string   `json:"time"`
	AlternativeTimes []string `json:"alternative_times"`
}

// handlers is the code behind the restaurant reservation flow defined in
// flow.yaml.
func handlers() flows.Handlers {
	return flows.Handlers{
		"collect_party_size": flows.NodeFunction{
			Name:        "collect_party_size",
			Description: "Record the number of people in the party.",
			Properties: map[string]any{
				"size": json.RawMessage(`{"type": "integer", ` +
					`"description": "Number of people in the party. Must be between 1 and 12."}`),
			},
			Required: []string{"size"},
			Handler:  collectPartySize,
		},
		"check_availability": flows.NodeFunction{
			Name:        "check_availability",
			Description: "Check availability for requested time.",
			Properties: map[string]any{
				"time": json.RawMessage(`{"type": "string", ` +
					`"description": "Requested reservation time in \"HH:MM AM/PM\" format. ` +
					`Must be between 5 PM and 10 PM."}`),
				"party_size": json.RawMessage(`{"type": "integer", ` +
					`"description": "Number of people in the party."}`),
			},
			Required: []string{"time", "party_size"},
			Handler:  checkAvailability,
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

// collectPartySize records the number of people in the party.
func collectPartySize(
	_ context.Context, args json.RawMessage, fm *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Size int `json:"size"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	fm.State().Set("party_size", in.Size)
	result, err := toJSON(partySizeResult{Size: in.Size, Status: "success"})
	if err != nil {
		return "", nil, err
	}
	return result, flows.TransitionInYAML, nil
}

// checkAvailability checks availability for the requested time.
func checkAvailability(
	ctx context.Context, args json.RawMessage, _ *flows.FlowManager,
) (string, *flows.NodeConfig, error) {
	var in struct {
		Time      string `json:"time"`
		PartySize int    `json:"party_size"`
	}
	if err := json.Unmarshal(args, &in); err != nil {
		return "", nil, err
	}
	isAvailable, alternativeTimes, err := reservationSystem.checkAvailability(ctx, in.PartySize, in.Time)
	if err != nil {
		return "", nil, err
	}

	// The business logic answered with a bool. Report it as a named status so
	// the config can branch on it; the config, not this tool, picks the node.
	status := "unavailable"
	if isAvailable {
		status = "available"
	}
	result, err := toJSON(timeResult{Status: status, Time: in.Time, AlternativeTimes: alternativeTimes})
	if err != nil {
		return "", nil, err
	}
	return result, flows.TransitionInYAML, nil
}
