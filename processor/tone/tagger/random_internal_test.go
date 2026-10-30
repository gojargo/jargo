package tagger

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"strconv"
	"testing"
)

// TestBreathKindsAreDrawnAsPythonsRandomDraws them: random.Random(seed) for
// fifty seeds (empty, leading NULs, multi-line, non-ASCII, long), its first
// random() values, and twenty-five choices(population, weights)[0] in a row for
// each of seven weightings.
func TestBreathKindsAreDrawnAsPythonsRandomDraws(t *testing.T) {
	raw, err := os.ReadFile("testdata/random.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Weights [][2]json.RawMessage `json:"weights"`
		Cases   []struct {
			Seed   string    `json:"seed"`
			Random []float64 `json:"random"`
			Draws  []string  `json:"draws"`
		} `json:"cases"`
		Errors []struct {
			Weights []float64 `json:"weights"`
			Error   string    `json:"error"`
		} `json:"errors"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	type weighting struct {
		population []string
		weights    []float64
	}
	weightings := make([]weighting, len(fixture.Weights))
	for i, w := range fixture.Weights {
		if err := json.Unmarshal(w[0], &weightings[i].population); err != nil {
			t.Fatal(err)
		}
		if err := json.Unmarshal(w[1], &weightings[i].weights); err != nil {
			t.Fatal(err)
		}
	}
	for _, c := range fixture.Cases {
		d := newDraw(c.Seed)
		for i, want := range c.Random {
			if got := d.random(); got != want {
				t.Errorf("%.20q: random() %d = %v, want %v", c.Seed, i, got, want)
			}
		}
		for i, w := range weightings {
			d := newDraw(c.Seed)
			var got []byte
			for range len(c.Draws[i]) {
				kind, err := d.choice(w.population, w.weights)
				if err != nil {
					t.Fatal(err)
				}
				for j, p := range w.population {
					if p == kind {
						got = strconv.AppendInt(got, int64(j), 10)
					}
				}
			}
			if string(got) != c.Draws[i] {
				t.Errorf("%.20q, weights %v: drew %s, want %s", c.Seed, w.weights, got, c.Draws[i])
			}
		}
	}
	for _, e := range fixture.Errors {
		if _, err := newDraw("x").choice([]string{"a", "b"}, e.Weights); !errors.Is(err, errNonPositive) {
			t.Errorf("weights %v: err = %v, want %q", e.Weights, err, e.Error)
		}
	}
	// Python: "Total of weights must be finite".
	if _, err := newDraw("x").choice([]string{"a", "b"}, []float64{math.Inf(1), 1}); !errors.Is(err, errInfiniteWeight) {
		t.Errorf("err = %v", err)
	}
}
