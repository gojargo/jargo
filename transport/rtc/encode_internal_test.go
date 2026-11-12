package rtc

import "testing"

// Every application message reaches the data channel as JSON, a string as a
// JSON string literal, since the client parses every message it receives.
func TestEncodeMessageIsAlwaysJSON(t *testing.T) {
	cases := []struct {
		message any
		want    string
	}{
		{"hello", `"hello"`},
		{`{"type":"x"}`, `"{\"type\":\"x\"}"`},
		{map[string]any{"type": "x"}, `{"type":"x"}`},
	}
	for _, tc := range cases {
		got, err := encodeMessage(tc.message)
		if err != nil {
			t.Fatalf("encode %v: %v", tc.message, err)
		}
		if string(got) != tc.want {
			t.Errorf("encode %v = %s, want %s", tc.message, got, tc.want)
		}
	}
}
