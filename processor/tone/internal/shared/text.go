package shared

import (
	"math"
	"strconv"
	"strings"
	"unicode"
)

// The models were trained on text written by Python, and their inputs and
// offsets are worked out by its string rules, which differ from Go's at the
// edges. These follow Python's.

// IsSpace reports whether r is whitespace to Python's str.isspace: Unicode's
// White_Space, and the four separators U+001C to U+001F, which Go leaves out.
func IsSpace(r rune) bool {
	return unicode.IsSpace(r) || (r >= 0x1c && r <= 0x1f)
}

// IsAlnum reports whether r is a letter or a number to Python's str.isalnum.
func IsAlnum(r rune) bool {
	return unicode.IsLetter(r) || unicode.IsNumber(r)
}

// Strip is Python's str.strip with no argument.
func Strip(s string) string {
	return strings.TrimFunc(s, IsSpace)
}

// Fields is Python's str.split with no argument.
func Fields(s string) []string {
	return strings.FieldsFunc(s, IsSpace)
}

// Round is Python's round(x, digits): the decimal nearest x, with a tie, which
// only an exact binary value can be, going to the even digit.
func Round(x float64, digits int) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return x
	}
	r, err := strconv.ParseFloat(strconv.FormatFloat(x, 'f', digits, 64), 64)
	if err != nil {
		return x
	}
	return r
}
