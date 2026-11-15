package main

import (
	"slices"
	"testing"
)

// TestDictionaryLoads checks the embedded dictionary has the Cartesia entry,
// with its hinted words in the file's order.
func TestDictionaryLoads(t *testing.T) {
	d, err := loadDictionary(pronunciations, "cartesia")
	if err != nil {
		t.Fatal(err)
	}
	if d.Model == "" {
		t.Fatal("no model")
	}
	if len(d.hinted) != len(d.IPA) {
		t.Fatalf("hinted %d words, IPA has %d", len(d.hinted), len(d.IPA))
	}
	if !slices.Equal(d.hinted[:2], []string{"Adalimumab", "Alfuzosin"}) {
		t.Fatalf("hinted out of order: %v", d.hinted)
	}
}
