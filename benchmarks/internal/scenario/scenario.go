// Package scenario reads the JSON files that say what to measure. The same files
// drive both sides of the comparison, the Go one here and the Python one under
// pipecat/, so a scenario is defined once and both measure exactly it.
package scenario

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
	"time"
)

// Kind is what a benchmark measures, and so how it is run.
type Kind string

// The kinds of benchmark a scenario can hold.
const (
	// KindLatency is closed-loop latency: one frame is queued, and the next only
	// once it has reached the end of the pipeline. Warm-up frames are excluded,
	// and so is starting the pipeline.
	KindLatency Kind = "latency"
	// KindThroughput is open-loop throughput: every frame is queued at once, and
	// the time runs from the first being queued to the last arriving. Starting
	// the pipeline is excluded.
	KindThroughput Kind = "throughput"
	// KindStartup is the time from asking a pipeline to run to its StartFrame
	// having reached the end, which is when it is ready for frames.
	KindStartup Kind = "startup"
	// KindSessions runs many pipelines at once and reports what each one costs:
	// starting it, the memory it holds, and the frames all of them move.
	KindSessions Kind = "sessions"
)

// Scenario is one file of benchmarks, run in order.
type Scenario struct {
	// Name identifies the scenario in the results, as the scenario config line.
	Name string `json:"name"`
	// Description says what the scenario is for. It is not read by the runners.
	Description string `json:"description"`
	// Benchmarks are the benchmarks the scenario runs.
	Benchmarks []Benchmark `json:"benchmarks"`
}

// Benchmark is one benchmark: a kind, and the sizes it is run at. Each size is
// reported as a sub-benchmark, BenchmarkName/depth=4 or
// BenchmarkName/sessions=100.
type Benchmark struct {
	// Name is the benchmark's name without the Benchmark prefix.
	Name string `json:"name"`
	// Kind is what it measures.
	Kind Kind `json:"kind"`
	// Depths are the numbers of pass-through processors in the chain, one
	// sub-benchmark each. Used by latency, throughput and startup.
	Depths []int `json:"depths,omitempty"`
	// Depth is the chain depth of every pipeline. Used by sessions.
	Depth int `json:"depth,omitempty"`
	// Sessions are the numbers of pipelines run at once, one sub-benchmark
	// each. Used by sessions.
	Sessions []int `json:"sessions,omitempty"`
	// Warmup is how many frames (latency, throughput) or rounds (sessions) go
	// through before measuring starts.
	Warmup int `json:"warmup,omitempty"`
	// Frames is how many frames one sample measures. Used by latency and
	// throughput.
	Frames int `json:"frames,omitempty"`
	// Rounds is how many rounds one sample measures, a round being one frame
	// into every pipeline and back out. Used by sessions.
	Rounds int `json:"rounds,omitempty"`
	// Repeats is how many pipelines one sample starts. Used by startup.
	Repeats int `json:"repeats,omitempty"`
	// BudgetSeconds bounds one sample. A sample that runs past it is reported
	// as timed out, and the larger sizes of the same benchmark are skipped.
	BudgetSeconds float64 `json:"budget_seconds"`
}

// Budget is BudgetSeconds as a duration.
func (b Benchmark) Budget() time.Duration {
	return time.Duration(b.BudgetSeconds * float64(time.Second))
}

// ErrInvalid is wrapped by every error Validate returns.
var ErrInvalid = errors.New("invalid scenario")

// namePattern is what a benchmark name may hold: it becomes part of a benchfmt
// line, where a space or a slash would change what it means.
var namePattern = regexp.MustCompile(`^[A-Z][A-Za-z0-9]*$`)

// Load reads and validates the scenario at path. Unknown fields are refused, so
// a misspelled one is reported rather than silently left at its zero value.
func Load(path string) (*Scenario, error) {
	data, err := os.ReadFile(path) //nolint:gosec // the path is the operator's own argument
	if err != nil {
		return nil, fmt.Errorf("read scenario: %w", err)
	}
	return Parse(data)
}

// Parse decodes and validates a scenario.
func Parse(data []byte) (*Scenario, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	var s Scenario
	if err := dec.Decode(&s); err != nil {
		return nil, fmt.Errorf("decode scenario: %w", err)
	}
	if err := s.Validate(); err != nil {
		return nil, err
	}
	return &s, nil
}

// Validate reports the first thing wrong with the scenario.
func (s *Scenario) Validate() error {
	if s.Name == "" {
		return fmt.Errorf("%w: no name", ErrInvalid)
	}
	if len(s.Benchmarks) == 0 {
		return fmt.Errorf("%w: no benchmarks", ErrInvalid)
	}
	seen := make(map[string]bool, len(s.Benchmarks))
	for i := range s.Benchmarks {
		b := &s.Benchmarks[i]
		if seen[b.Name] {
			return fmt.Errorf("%w: benchmark %q appears twice", ErrInvalid, b.Name)
		}
		seen[b.Name] = true
		if err := b.validate(); err != nil {
			return fmt.Errorf("%w: benchmark %q: %w", ErrInvalid, b.Name, err)
		}
	}
	return nil
}

// errField is what a benchmark's own checks wrap.
var errField = errors.New("bad field")

func (b *Benchmark) validate() error {
	if !namePattern.MatchString(b.Name) {
		return fmt.Errorf("%w: name must match %s", errField, namePattern)
	}
	if b.BudgetSeconds <= 0 {
		return fmt.Errorf("%w: budget_seconds must be positive", errField)
	}
	if b.Warmup < 0 {
		return fmt.Errorf("%w: warmup must not be negative", errField)
	}
	switch b.Kind {
	case KindLatency, KindThroughput:
		if err := positive("depths", b.Depths); err != nil {
			return err
		}
		if b.Frames < 1 {
			return fmt.Errorf("%w: frames must be at least 1", errField)
		}
	case KindStartup:
		if err := positive("depths", b.Depths); err != nil {
			return err
		}
		if b.Repeats < 1 {
			return fmt.Errorf("%w: repeats must be at least 1", errField)
		}
	case KindSessions:
		if b.Depth < 1 {
			return fmt.Errorf("%w: depth must be at least 1", errField)
		}
		if err := positive("sessions", b.Sessions); err != nil {
			return err
		}
		if b.Rounds < 1 {
			return fmt.Errorf("%w: rounds must be at least 1", errField)
		}
	default:
		return fmt.Errorf("%w: unknown kind %q", errField, b.Kind)
	}
	return nil
}

// positive checks a list of sizes is not empty and holds only positive values.
func positive(field string, sizes []int) error {
	if len(sizes) == 0 {
		return fmt.Errorf("%w: %s must not be empty", errField, field)
	}
	for _, n := range sizes {
		if n < 1 {
			return fmt.Errorf("%w: %s must all be at least 1", errField, field)
		}
	}
	return nil
}
