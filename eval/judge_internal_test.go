package eval

import (
	"strings"
	"testing"
)

// TestParseVerdict covers the verdict read out of each shape of explainer
// answer: the JSON it is asked for, JSON wrapped in prose or fences, and the
// keyword fallback for a model that ignores the instruction entirely.
func TestParseVerdict(t *testing.T) {
	cases := map[string]struct {
		out         string
		wantVerdict string
		wantReason  string
	}{
		"clean json yes": {
			out:         `{"verdict": "yes", "reason": "greets the user warmly"}`,
			wantVerdict: VerdictYes, wantReason: "greets the user warmly",
		},
		"clean json no": {
			out:         `{"verdict": "no", "reason": "too terse"}`,
			wantVerdict: VerdictNo, wantReason: "too terse",
		},
		"clean json continue": {
			out:         `{"verdict": "continue", "reason": "only filler so far"}`,
			wantVerdict: VerdictContinue, wantReason: "only filler so far",
		},
		"unknown verdict fails closed": {
			out:         `{"verdict": "maybe", "reason": "unsure"}`,
			wantVerdict: VerdictNo, wantReason: "unsure",
		},
		"fenced json": {
			out:         "```json\n{\"verdict\": \"yes\", \"reason\": \"fine\"}\n```",
			wantVerdict: VerdictYes, wantReason: "fine",
		},
		"fenced json without lang": {
			out:         "```\n{\"verdict\": \"no\", \"reason\": \"wrong\"}\n```",
			wantVerdict: VerdictNo, wantReason: "wrong",
		},
		"trailing prose after json": {
			out: `{"verdict": "yes", "reason": "answers it"} ` +
				"Let me know if you'd like to evaluate further turns!",
			wantVerdict: VerdictYes, wantReason: "answers it",
		},
		"leading prose before json": {
			out:         `Sure! {"verdict": "no", "reason": "off topic"}`,
			wantVerdict: VerdictNo, wantReason: "off topic",
		},
		"missing reason": {
			out:         `{"verdict": "yes"}`,
			wantVerdict: VerdictYes, wantReason: "(no reason given)",
		},
		"extra whitespace": {
			out:         "\n\n   {\"verdict\": \"yes\", \"reason\": \"ok\"}   \n",
			wantVerdict: VerdictYes, wantReason: "ok",
		},
		"unstructured yes": {
			out:         "Yes, the reply satisfies it.",
			wantVerdict: VerdictYes, wantReason: "(unstructured yes)",
		},
		"unstructured no": {
			out:         "That does not satisfy the criterion.",
			wantVerdict: VerdictNo, wantReason: "(unstructured no)",
		},
		"unstructured continue": {
			out:         "I would continue and wait for more.",
			wantVerdict: VerdictContinue, wantReason: "(unstructured continue)",
		},
		"ambiguous response fails closed": {
			out:         "yes and no",
			wantVerdict: VerdictNo, wantReason: "could not parse explainer response",
		},
		"garbage response": {
			out:         "The reply seems fine.",
			wantVerdict: VerdictNo, wantReason: "could not parse explainer response",
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			v := parseVerdict(tc.out)
			if v.Verdict != tc.wantVerdict {
				t.Fatalf("verdict = %q, want %q (reason %q)", v.Verdict, tc.wantVerdict, v.Reason)
			}
			if !strings.Contains(v.Reason, tc.wantReason) {
				t.Fatalf("reason %q does not contain %q", v.Reason, tc.wantReason)
			}
		})
	}
}
