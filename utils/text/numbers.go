package text

import (
	"slices"
	"strconv"
	"strings"
)

// English cardinal-number spelling, matching the num2words "en" conventions the
// upstream transforms rely on: hyphenated tens ("forty-two"), an "and" before a
// sub-hundred remainder ("one hundred and one"), and magnitude groups joined by
// ", " with a final sub-hundred group joined by " and " ("one thousand and
// five", "one million, two hundred and thirty-four thousand, five hundred and
// sixty-seven").

//nolint:gochecknoglobals // immutable spelling tables
var (
	onesWords = []string{
		"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
		"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
		"sixteen", "seventeen", "eighteen", "nineteen",
	}
	tensWords = []string{
		"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
		"eighty", "ninety",
	}
	scaleWords = []string{
		"", "thousand", "million", "billion", "trillion", "quadrillion",
		"quintillion",
	}
)

// cardinal spells a non-negative integer in English. Values beyond the scale
// table (>= 10^21) fall back to their decimal digits.
func cardinal(n int64) string {
	if n == 0 {
		return onesWords[0]
	}
	neg := n < 0
	if neg {
		n = -n
	}

	// Split into groups of three digits, least significant first.
	var groups []int64
	for n > 0 {
		groups = append(groups, n%1000)
		n /= 1000
	}
	if len(groups) > len(scaleWords) {
		return decimalDigits(strconv.FormatInt(orig(neg, groups), 10))
	}

	type part struct {
		text  string
		scale int
		val   int64
	}
	var parts []part
	for i, g := range slices.Backward(groups) {
		if g == 0 {
			continue
		}
		t := threeDigitWords(g)
		if i > 0 {
			t += " " + scaleWords[i]
		}
		parts = append(parts, part{t, i, g})
	}

	var b strings.Builder
	b.WriteString(parts[0].text)
	for k := 1; k < len(parts); k++ {
		sep := ", "
		// num2words joins a final sub-hundred units group with " and " rather
		// than a comma ("one thousand and five").
		if k == len(parts)-1 && parts[k].scale == 0 && parts[k].val < 100 {
			sep = " and "
		}
		b.WriteString(sep)
		b.WriteString(parts[k].text)
	}
	result := b.String()
	if neg {
		return "minus " + result
	}
	return result
}

// orig reassembles the signed integer from its digit groups, used only for the
// out-of-range fallback.
func orig(neg bool, groups []int64) int64 {
	var n int64
	for _, g := range slices.Backward(groups) {
		n = n*1000 + g
	}
	if neg {
		return -n
	}
	return n
}

// threeDigitWords spells 1..999.
func threeDigitWords(n int64) string {
	h := n / 100
	r := n % 100
	if h > 0 {
		s := onesWords[h] + " hundred"
		if r > 0 {
			s += " and " + twoDigitWords(r)
		}
		return s
	}
	return twoDigitWords(r)
}

// twoDigitWords spells 0..99.
func twoDigitWords(n int64) string {
	if n < 20 {
		return onesWords[n]
	}
	w := tensWords[n/10]
	if n%10 > 0 {
		w += "-" + onesWords[n%10]
	}
	return w
}

// floatToWords spells a decimal string the way num2words spells a float: the
// integer part as a cardinal, then " point " and the remaining fractional digits
// read individually, with trailing zeros dropped ("42.50" → "forty-two point
// five", "42.0" → "forty-two", "0.25" → "zero point two five").
func floatToWords(s string) string {
	intPart, fracPart, hasDot := strings.Cut(s, ".")
	if intPart == "" {
		intPart = "0"
	}
	n, err := strconv.ParseInt(intPart, 10, 64)
	if err != nil {
		return s
	}
	words := cardinal(n)
	if !hasDot {
		return words
	}
	frac := strings.TrimRight(fracPart, "0")
	if frac == "" {
		return words
	}
	digits := make([]string, len(frac))
	for i := range frac {
		digits[i] = onesWords[frac[i]-'0']
	}
	return words + " point " + strings.Join(digits, " ")
}

// decimalDigits spaces out the characters of a digit string ("2026" → "2 0 2
// 6") so a synthesizer reads them one by one.
func decimalDigits(s string) string {
	return strings.Join(strings.Split(s, ""), " ")
}
