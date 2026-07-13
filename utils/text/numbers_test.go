package text

import "testing"

// The expected spellings are the verified num2words(lang="en") outputs.
func TestCardinal(t *testing.T) {
	cases := map[int64]string{
		0:       "zero",
		1:       "one",
		15:      "fifteen",
		20:      "twenty",
		21:      "twenty-one",
		42:      "forty-two",
		70:      "seventy",
		99:      "ninety-nine",
		100:     "one hundred",
		101:     "one hundred and one",
		123:     "one hundred and twenty-three",
		200:     "two hundred",
		999:     "nine hundred and ninety-nine",
		1000:    "one thousand",
		1005:    "one thousand and five",
		1023:    "one thousand and twenty-three",
		1234:    "one thousand, two hundred and thirty-four",
		2000:    "two thousand",
		2023:    "two thousand and twenty-three",
		2100:    "two thousand, one hundred",
		2123:    "two thousand, one hundred and twenty-three",
		1000000: "one million",
		1000023: "one million and twenty-three",
		1234567: "one million, two hundred and thirty-four thousand, five hundred and sixty-seven",
	}
	for n, want := range cases {
		if got := cardinal(n); got != want {
			t.Errorf("cardinal(%d) = %q, want %q", n, got, want)
		}
	}
	if got := cardinal(-42); got != "minus forty-two" {
		t.Errorf("cardinal(-42) = %q, want %q", got, "minus forty-two")
	}
}

func TestFloatToWords(t *testing.T) {
	cases := map[string]string{
		"50":    "fifty",
		"42.5":  "forty-two point five",
		"42.50": "forty-two point five",
		"0.25":  "zero point two five",
		"3.14":  "three point one four",
		"42.05": "forty-two point zero five",
		"42.00": "forty-two",
	}
	for in, want := range cases {
		if got := floatToWords(in); got != want {
			t.Errorf("floatToWords(%q) = %q, want %q", in, got, want)
		}
	}
}
