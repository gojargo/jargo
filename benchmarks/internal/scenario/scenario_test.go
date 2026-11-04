package scenario_test

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gojargo/jargo/benchmarks/internal/scenario"
)

// The scenario files shipped in the tree are the ones CI and the comparison
// run, so each has to load.
func TestShippedScenariosLoad(t *testing.T) {
	paths, err := filepath.Glob("../../scenarios/*.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) == 0 {
		t.Fatal("no scenario files found")
	}
	for _, p := range paths {
		if _, err := scenario.Load(p); err != nil {
			t.Errorf("%s: %v", p, err)
		}
	}
}

// doc wraps benchmark entries in a scenario.
func doc(benchmarks ...string) []byte {
	return []byte(`{"name":"x","benchmarks":[` + strings.Join(benchmarks, ",") + `]}`)
}

func TestParseRejects(t *testing.T) {
	const startup = `{"name":"A","kind":"startup","depths":[1],"repeats":1,"budget_seconds":1`
	cases := map[string][]byte{
		"unknown field": doc(startup + `,"typo":1}`),
		"no benchmarks": doc(),
		"unknown kind":  doc(`{"name":"A","kind":"nope","budget_seconds":1}`),
		"bad name":      doc(`{"name":"a b","kind":"startup","depths":[1],"repeats":1,"budget_seconds":1}`),
		"no budget":     doc(`{"name":"A","kind":"startup","depths":[1],"repeats":1}`),
		"zero depth":    doc(`{"name":"A","kind":"latency","depths":[0],"frames":1,"budget_seconds":1}`),
		"no frames":     doc(`{"name":"A","kind":"throughput","depths":[1],"budget_seconds":1}`),
		"no sessions":   doc(`{"name":"A","kind":"sessions","depth":4,"rounds":1,"budget_seconds":1}`),
		"duplicate":     doc(startup+`}`, startup+`}`),
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := scenario.Parse(data); err == nil {
				t.Fatal("parsed, want an error")
			}
		})
	}
}

func TestParseValidatesKinds(t *testing.T) {
	_, err := scenario.Parse(doc(`{"name":"A","kind":"sessions","depth":4,"sessions":[1],"rounds":0,"budget_seconds":1}`))
	if !errors.Is(err, scenario.ErrInvalid) {
		t.Fatalf("err = %v, want ErrInvalid", err)
	}
}

// A deployment is checked like sessions: a depth, session counts and rounds.
func TestParseValidatesDeployment(t *testing.T) {
	const ok = `{"name":"A","kind":"deployment","depth":4,"sessions":[10],"rounds":1,"budget_seconds":1}`
	if _, err := scenario.Parse(doc(ok)); err != nil {
		t.Fatalf("valid deployment rejected: %v", err)
	}
	for name, b := range map[string]string{
		"no depth":    `{"name":"A","kind":"deployment","sessions":[10],"rounds":1,"budget_seconds":1}`,
		"no sessions": `{"name":"A","kind":"deployment","depth":4,"rounds":1,"budget_seconds":1}`,
		"no rounds":   `{"name":"A","kind":"deployment","depth":4,"sessions":[10],"budget_seconds":1}`,
	} {
		if _, err := scenario.Parse(doc(b)); !errors.Is(err, scenario.ErrInvalid) {
			t.Errorf("%s: err = %v, want ErrInvalid", name, err)
		}
	}
}
