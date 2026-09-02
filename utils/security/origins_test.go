package security_test

import (
	"slices"
	"testing"

	"github.com/gojargo/jargo/utils/security"
)

func TestIsOriginAllowed(t *testing.T) {
	cases := []struct {
		name    string
		origin  string
		allowed []string
		want    bool
	}{
		{"no list allows any origin", "https://evil.example", nil, true},
		{"no list allows a missing origin", "", nil, true},
		{"empty list allows any origin", "https://evil.example", []string{}, true},
		{"a listed origin is allowed", "https://app.example", []string{"https://app.example"}, true},
		{
			"one of several listed origins is allowed",
			"https://b.example",
			[]string{"https://a.example", "https://b.example"},
			true,
		},
		{"case is not significant", "HTTPS://App.Example", []string{"https://app.example"}, true},
		{"an unlisted origin is refused", "https://evil.example", []string{"https://app.example"}, false},
		{"a missing origin is refused once a list is set", "", []string{"https://app.example"}, false},
		{
			"a listed origin is matched whole, not as a prefix",
			"https://app.example.evil.test",
			[]string{"https://app.example"},
			false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := security.IsOriginAllowed(c.origin, c.allowed); got != c.want {
				t.Errorf("IsOriginAllowed(%q, %v) = %v, want %v", c.origin, c.allowed, got, c.want)
			}
		})
	}
}

// TestDefaultAllowedOrigins covers the deployment-wide policy: the variable is
// where an operator sets the origins once, and an unset one has to keep meaning
// "no restriction" so a telephony endpoint is not broken by a default.
func TestDefaultAllowedOrigins(t *testing.T) {
	cases := []struct {
		name string
		env  string
		want []string
	}{
		{"unset allows every origin", "", nil},
		{"one origin", "https://app.example", []string{"https://app.example"}},
		{
			"several, with the spacing an operator writes",
			"https://a.example, https://b.example",
			[]string{"https://a.example", "https://b.example"},
		},
		{"a trailing comma is not an origin", "https://a.example,", []string{"https://a.example"}},
		{"blanks alone allow every origin", " , ", nil},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			t.Setenv(security.AllowedOriginsEnv, c.env)
			got := security.DefaultAllowedOrigins()
			if !slices.Equal(got, c.want) {
				t.Errorf("DefaultAllowedOrigins() = %q, want %q", got, c.want)
			}
		})
	}
}
