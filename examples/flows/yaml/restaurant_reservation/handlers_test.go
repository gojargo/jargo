package main

import (
	"testing"

	"github.com/gojargo/jargo/flows"
)

// TestExampleBuilds checks that flow.yaml loads and joins to the handlers.
func TestExampleBuilds(t *testing.T) {
	config, err := flows.FlowConfigFromFile(flowConfigPath())
	if err != nil {
		t.Fatalf("FlowConfigFromFile: %v", err)
	}
	flow, err := flows.NewFlow(config, handlers())
	if err != nil {
		t.Fatalf("NewFlow: %v", err)
	}
	if len(config.Flows) == 0 {
		for name := range config.Nodes {
			node, err := flow.Node(name)
			if err != nil {
				t.Fatalf("Node(%q): %v", name, err)
			}
			if node.Name != name {
				t.Errorf("Node(%q) is built as %q", name, node.Name)
			}
		}
	}
	if flow.InitialNode().TaskMessages == nil {
		t.Error("the initial node should have task messages")
	}
}
