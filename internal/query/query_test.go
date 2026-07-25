package query_test

import (
	"net/url"
	"testing"

	"github.com/gojargo/jargo/internal/query"
)

// TestSetBoolTrue covers the "on unless told otherwise" setter: a nil pointer
// means the caller did not configure the field, and the parameter is still sent
// as true because that is the provider default these fields describe.
func TestSetBoolTrue(t *testing.T) {
	tests := []struct {
		name string
		v    *bool
		want string
	}{
		{"nil defaults to true", nil, "true"},
		{"explicit true", new(true), "true"},
		{"explicit false", new(false), "false"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := url.Values{}
			query.SetBoolTrue(q, "punctuate", tt.v)
			if got := q.Get("punctuate"); got != tt.want {
				t.Errorf("punctuate = %q, want %q", got, tt.want)
			}
		})
	}
}

// TestOptionalSettersOmitUnset checks the whole point of these helpers: an unset
// field must not appear in the query at all, so the provider applies its own
// default rather than receiving a zero value.
func TestOptionalSettersOmitUnset(t *testing.T) {
	q := url.Values{}
	query.SetBoolOpt(q, "b", nil)
	query.SetIntOpt(q, "i", nil)
	query.SetFloatOpt(q, "f", nil)
	query.SetStrOpt(q, "s", "")

	if len(q) != 0 {
		t.Errorf("query = %v, want every unset field omitted", q)
	}
}

func TestOptionalSettersFormatting(t *testing.T) {
	tests := []struct {
		name string
		set  func(url.Values)
		key  string
		want string
	}{
		{"bool true", func(q url.Values) { query.SetBoolOpt(q, "b", new(true)) }, "b", "true"},
		{"bool false", func(q url.Values) { query.SetBoolOpt(q, "b", new(false)) }, "b", "false"},
		{"int", func(q url.Values) { query.SetIntOpt(q, "i", new(42)) }, "i", "42"},
		// A zero value is still sent once the caller has set it explicitly.
		{"int zero", func(q url.Values) { query.SetIntOpt(q, "i", new(0)) }, "i", "0"},
		{"negative int", func(q url.Values) { query.SetIntOpt(q, "i", new(-5)) }, "i", "-5"},
		{"float", func(q url.Values) { query.SetFloatOpt(q, "f", new(0.75)) }, "f", "0.75"},
		// 'g' with -1 precision keeps the shortest exact representation rather
		// than padding to a fixed number of decimals.
		{"float whole number", func(q url.Values) { query.SetFloatOpt(q, "f", new(2.0)) }, "f", "2"},
		{"string", func(q url.Values) { query.SetStrOpt(q, "s", "en-US") }, "s", "en-US"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			q := url.Values{}
			tt.set(q)
			if got := q.Get(tt.key); got != tt.want {
				t.Errorf("%s = %q, want %q", tt.key, got, tt.want)
			}
		})
	}
}

// TestAddAll checks repeated parameters are preserved rather than collapsed,
// which is how providers take lists such as keyword boosts.
func TestAddAll(t *testing.T) {
	q := url.Values{}
	query.AddAll(q, "keywords", []string{"jargo", "pipeline", "jargo"})

	got := q["keywords"]
	want := []string{"jargo", "pipeline", "jargo"}
	if len(got) != len(want) {
		t.Fatalf("keywords = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("keywords[%d] = %q, want %q", i, got[i], want[i])
		}
	}

	// An empty list adds nothing rather than an empty parameter.
	q2 := url.Values{}
	query.AddAll(q2, "keywords", nil)
	if len(q2) != 0 {
		t.Errorf("query = %v, want an empty list to add nothing", q2)
	}
}
