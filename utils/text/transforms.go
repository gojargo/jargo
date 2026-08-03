package text

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// replaceAllSubmatch rewrites every match of re, passing the replacer the full
// match (groups[0]) and its capture groups (groups[1:]); a group that did not
// participate is the empty string. It is the group-aware counterpart to
// regexp.ReplaceAllStringFunc.
func replaceAllSubmatch(re *regexp.Regexp, text string, repl func(groups []string) string) string {
	matches := re.FindAllStringSubmatchIndex(text, -1)
	if matches == nil {
		return text
	}
	var b strings.Builder
	last := 0
	for _, idx := range matches {
		b.WriteString(text[last:idx[0]])
		groups := make([]string, len(idx)/2)
		for g := range groups {
			if idx[2*g] >= 0 {
				groups[g] = text[idx[2*g]:idx[2*g+1]]
			}
		}
		b.WriteString(repl(groups))
		last = idx[1]
	}
	b.WriteString(text[last:])
	return b.String()
}

//
// Markdown
//

//nolint:gochecknoglobals // compiled once, immutable
var (
	mdFence1     = regexp.MustCompile("```[\\s\\S]*?```")
	mdFence2     = regexp.MustCompile(`~~~[\s\S]*?~~~`)
	mdInlineCode = regexp.MustCompile("`([^`]+)`")
	mdBoldItalic = regexp.MustCompile(`\*{3}(.+?)\*{3}`)
	mdBoldItalU  = regexp.MustCompile(`_{3}(.+?)_{3}`)
	mdBold       = regexp.MustCompile(`\*{2}(.+?)\*{2}`)
	mdBoldU      = regexp.MustCompile(`_{2}(.+?)_{2}`)
	mdItalic     = regexp.MustCompile(`\*(.+?)\*`)
	mdItalicU    = regexp.MustCompile(`\b_(.+?)_\b`)
	mdHeader     = regexp.MustCompile(`(?m)^#{1,6}\s+`)
	mdQuote      = regexp.MustCompile(`(?m)^>\s?`)
	mdRule       = regexp.MustCompile(`(?m)^\s*[-*_]{3,}\s*$`)
)

// StripMarkdown removes Markdown formatting symbols that have no spoken
// equivalent: fenced and inline code, bold/italic markers, ATX headers,
// blockquote markers and horizontal rules. Link and image syntax is left
// intact, since the label text is meaningful.
func StripMarkdown(text string) string {
	text = mdFence1.ReplaceAllString(text, "")
	text = mdFence2.ReplaceAllString(text, "")
	text = mdInlineCode.ReplaceAllString(text, "$1")
	text = mdBoldItalic.ReplaceAllString(text, "$1")
	text = mdBoldItalU.ReplaceAllString(text, "$1")
	text = mdBold.ReplaceAllString(text, "$1")
	text = mdBoldU.ReplaceAllString(text, "$1")
	text = mdItalic.ReplaceAllString(text, "$1")
	text = mdItalicU.ReplaceAllString(text, "$1")
	text = mdHeader.ReplaceAllString(text, "")
	text = mdQuote.ReplaceAllString(text, "")
	text = mdRule.ReplaceAllString(text, "")
	return text
}

//
// Repeated punctuation
//

// DefaultPunctuationMarks are the marks CollapseRepeatedPunctuation collapses
// when none are configured: the two terminators a model repeats for emphasis.
// The full stop is left out on purpose, since a run of them is an ellipsis and
// synthesizers read it as a pause rather than as shouting.
const DefaultPunctuationMarks = "!?"

// RepeatedPunctuationOptions configures CollapseRepeatedPunctuation. The zero
// value collapses runs of "!" and "?" of two or more down to a single mark.
type RepeatedPunctuationOptions struct {
	// Marks are the characters whose runs collapse, given as a plain string of
	// them (for example "!?" or "!?."). Empty uses DefaultPunctuationMarks. Each
	// character is counted on its own, so a mixed run such as "?!?!" is left
	// alone; only a repeat of one and the same mark is a run.
	Marks string
	// Keep is how many marks a collapsed run is reduced to. Zero or less keeps
	// one.
	Keep int
	// MinRun is the run length at which collapsing starts, letting a deliberate
	// "!!" through while still catching longer runs. Zero or less collapses
	// anything longer than Keep, and a value that would leave nothing to collapse
	// is raised to Keep+1.
	MinRun int
}

// CollapseRepeatedPunctuation returns a transform that shortens a run of the
// same punctuation mark down to Keep of them, so "C'est super !!!!!" is spoken
// as "C'est super !".
//
// Synthesizers read punctuation as delivery, so a model writing emphasis as
// "!!!!!" or "????" has the voice shout the sentence rather than say it. That is
// the model's formatting choice leaking into how the bot sounds, not anything
// the words themselves carry.
func CollapseRepeatedPunctuation(opts RepeatedPunctuationOptions) Transform {
	marks := opts.Marks
	if marks == "" {
		marks = DefaultPunctuationMarks
	}
	keep := max(opts.Keep, 1)
	minRun := max(opts.MinRun, keep+1)

	// One alternative per mark rather than a single character class, so a class
	// does not match across two different marks and read "?!?!" as one run.
	// RE2 has no backreferences, so the repeat cannot be expressed generically.
	seen := map[rune]bool{}
	var alts []string
	for _, r := range marks {
		if seen[r] {
			continue
		}
		seen[r] = true
		alts = append(alts, regexp.QuoteMeta(string(r))+"{"+strconv.Itoa(minRun)+",}")
	}
	if len(alts) == 0 {
		return func(text string) string { return text }
	}
	re := regexp.MustCompile(strings.Join(alts, "|"))

	return func(text string) string {
		return re.ReplaceAllStringFunc(text, func(run string) string {
			return strings.Repeat(string([]rune(run)[0]), keep)
		})
	}
}

//
// Email
//

//nolint:gochecknoglobals // compiled once, immutable
var emailRE = regexp.MustCompile(`[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}`)

// EmailToSpeech rewrites email addresses into a spoken form, e.g.
// "user@example.com" → "user at example dot com".
func EmailToSpeech(text string) string {
	return emailRE.ReplaceAllStringFunc(text, func(email string) string {
		local, domain, _ := strings.Cut(email, "@")
		local = strings.NewReplacer(
			".", " dot ", "_", " underscore ", "-", " dash ", "+", " plus ",
		).Replace(local)
		domain = strings.NewReplacer(".", " dot ", "-", " dash ").Replace(domain)
		return local + " at " + domain
	})
}

//
// Phone numbers
//

//nolint:gochecknoglobals // compiled once, immutable
var (
	phoneRE    = regexp.MustCompile(`(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)(\d{3}[\s.\-]?)(\d{4})`)
	nonDigitRE = regexp.MustCompile(`\D`)
)

// ExpandPhoneNumbers spaces out phone-number digits so a synthesizer reads them
// one by one, e.g. "123-456-7890" → "1 2 3 4 5 6 7 8 9 0". A run bordered by
// another digit is left alone, so digits inside a longer number are not split.
func ExpandPhoneNumbers(text string) string {
	matches := phoneRE.FindAllStringIndex(text, -1)
	if matches == nil {
		return text
	}
	var b strings.Builder
	last := 0
	for _, m := range matches {
		start, end := m[0], m[1]
		// Reproduce the upstream (?<!\d) / (?!\d) guards: skip a match that
		// abuts another digit, so it is not a fragment of a longer number.
		if (start > 0 && isDigit(text[start-1])) || (end < len(text) && isDigit(text[end])) {
			continue
		}
		b.WriteString(text[last:start])
		digits := nonDigitRE.ReplaceAllString(text[start:end], "")
		b.WriteString(strings.Join(strings.Split(digits, ""), " "))
		last = end
	}
	b.WriteString(text[last:])
	return b.String()
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

//
// Acronyms
//

// Two or more consecutive uppercase letters standing as a whole word. The
// trailing \b excludes the uppercase prefix of a CamelCase word ("IPhone") and a
// trailing lowercase plural ("APIs"), matching the upstream (?![a-z]) guard.
//
//nolint:gochecknoglobals // compiled once, immutable
var acronymRE = regexp.MustCompile(`\b[A-Z]{2,}\b`)

// NormalizeAcronyms spaces out the letters of an all-caps acronym so each is
// pronounced individually, e.g. "API" → "A P I".
func NormalizeAcronyms(text string) string {
	return acronymRE.ReplaceAllStringFunc(text, func(a string) string {
		return strings.Join(strings.Split(a, ""), " ")
	})
}

//
// Dates
//

//nolint:gochecknoglobals // compiled once, immutable
var (
	isoDateRE  = regexp.MustCompile(`\b(\d{4})-(\d{2})-(\d{2})\b`)
	usDateRE   = regexp.MustCompile(`\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b`)
	monthNames = []string{
		"January", "February", "March", "April", "May", "June", "July",
		"August", "September", "October", "November", "December",
	}
)

// NormalizeDates expands ISO (YYYY-MM-DD) and US (MM/DD/YYYY or MM-DD-YYYY)
// dates into spoken form, e.g. "2023-05-10" → "May 10th, two thousand and
// twenty-three". A string that parses to no valid calendar date is left unchanged.
func NormalizeDates(text string) string {
	text = replaceAllSubmatch(isoDateRE, text, func(g []string) string {
		return spokenDate(atoi(g[1]), atoi(g[2]), atoi(g[3]), g[0])
	})
	text = replaceAllSubmatch(usDateRE, text, func(g []string) string {
		return spokenDate(atoi(g[3]), atoi(g[1]), atoi(g[2]), g[0])
	})
	return text
}

func spokenDate(year, month, day int, original string) string {
	if month < 1 || month > 12 || day < 1 {
		return original
	}
	t := time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC)
	if t.Year() != year || int(t.Month()) != month || t.Day() != day {
		return original
	}
	return monthNames[month-1] + " " + ordinal(day) + ", " + cardinal(int64(year))
}

func ordinal(n int) string {
	suffix := "th"
	if r := n % 100; r < 11 || r > 13 {
		switch n % 10 {
		case 1:
			suffix = "st"
		case 2:
			suffix = "nd"
		case 3:
			suffix = "rd"
		}
	}
	return strconv.Itoa(n) + suffix
}

//
// Currency
//

//nolint:gochecknoglobals // compiled once, immutable
var (
	// The fractional part takes every digit, not just the two that are spoken:
	// stopping at two leaves the rest behind as bare digits after the words.
	currencyRE = regexp.MustCompile(`([€£¥₹$])\s*(\d{1,3}(?:,\d{3})*|\d+)\b(?:\.(\d+))?`)

	currencyMap = map[string]struct {
		singular, plural, centSingular, centPlural string
	}{
		"$": {"dollar", "dollars", "cent", "cents"},
		"€": {"euro", "euros", "cent", "cents"},
		"£": {"pound", "pounds", "penny", "pence"},
		"¥": {"yen", "yen", "", ""},
		"₹": {"rupee", "rupees", "paisa", "paise"},
	}
)

// ExpandCurrency expands currency amounts into spoken form, e.g. "$42.50" →
// "forty-two dollars and fifty cents".
func ExpandCurrency(text string) string {
	return replaceAllSubmatch(currencyRE, text, func(g []string) string {
		cur, ok := currencyMap[g[1]]
		if !ok {
			return g[0]
		}
		whole, _ := strconv.ParseInt(strings.ReplaceAll(g[2], ",", ""), 10, 64)
		result := amountWords(whole, cur.singular, cur.plural)
		if g[3] != "" && cur.centSingular != "" {
			// Only the first two digits are subunits; anything past them is
			// below what the currency can express.
			cents, _ := strconv.ParseInt(padRight(truncate(g[3], 2), 2, '0'), 10, 64)
			if cents > 0 {
				result += " and " + amountWords(cents, cur.centSingular, cur.centPlural)
			}
		}
		return result
	})
}

func amountWords(n int64, singular, plural string) string {
	unit := plural
	if n == 1 {
		unit = singular
	}
	return cardinal(n) + " " + unit
}

// truncate returns at most the first n bytes of s. It is used on digits, which
// are one byte each.
func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

func padRight(s string, width int, pad byte) string {
	for len(s) < width {
		s += string(pad)
	}
	return s
}

//
// Percentages
//

//nolint:gochecknoglobals // compiled once, immutable
var percentRE = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*%`)

// ExpandPercentages expands percentage expressions into spoken form, e.g. "50%"
// → "fifty percent".
func ExpandPercentages(text string) string {
	return replaceAllSubmatch(percentRE, text, func(g []string) string {
		return floatToWords(g[1]) + " percent"
	})
}

//
// Numbers
//

//nolint:gochecknoglobals // compiled once, immutable
var numberRE = regexp.MustCompile(`\b(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\b`)

// ExpandNumbers returns a transform that expands numbers into spoken form. A
// number whose whole part exceeds digitCutoff is read digit-by-digit ("2026" →
// "2 0 2 6"); at or below the cutoff it is spelled as a quantity ("42" →
// "forty-two"). A digitCutoff of zero or less disables the cutoff, so every
// number is spelled as a quantity.
func ExpandNumbers(digitCutoff int) Transform {
	return func(text string) string {
		return replaceAllSubmatch(numberRE, text, func(g []string) string {
			wholeStr := strings.ReplaceAll(g[1], ",", "")
			frac := g[2]
			whole, err := strconv.ParseInt(wholeStr, 10, 64)
			if err != nil {
				return g[0]
			}
			if digitCutoff > 0 && whole > int64(digitCutoff) {
				out := decimalDigits(wholeStr)
				if frac != "" {
					out += " point " + decimalDigits(frac)
				}
				return out
			}
			if frac != "" {
				return floatToWords(wholeStr + "." + frac)
			}
			return cardinal(whole)
		})
	}
}

//
// Units
//

//nolint:gochecknoglobals // compiled once, immutable
var (
	unitMap = map[string]string{
		"km": "kilometers", "m": "meters", "cm": "centimeters", "mm": "millimeters",
		"mi": "miles", "ft": "feet", "in": "inches", "yd": "yards",
		"kg": "kilograms", "g": "grams", "mg": "milligrams", "lb": "pounds", "oz": "ounces",
		"l": "liters", "ml": "milliliters",
		"mph": "miles per hour", "kph": "kilometers per hour", "kmh": "kilometers per hour",
		"gb": "gigabytes", "mb": "megabytes", "kb": "kilobytes", "tb": "terabytes",
		"hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
	}
	// Single-letter units that are also common English words: expand only when
	// they immediately follow a digit with no space ("5m", not "1 m people").
	ambiguousUnits = map[string]bool{"in": true, "m": true, "g": true, "l": true}

	// Unambiguous units allow optional whitespace; ambiguous ones do not.
	unitRE          = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(` + unitAlternation(false) + `)\b`)
	ambiguousUnitRE = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)(` + unitAlternation(true) + `)\b`)
)

// unitAlternation builds a regex alternation of the ambiguous or unambiguous
// unit abbreviations, longest first so a longer unit wins over a prefix of it
// ("mph" before "mi").
func unitAlternation(ambiguous bool) string {
	var units []string
	for u := range unitMap {
		if ambiguousUnits[u] == ambiguous {
			units = append(units, u)
		}
	}
	sort.Slice(units, func(i, j int) bool {
		if len(units[i]) != len(units[j]) {
			return len(units[i]) > len(units[j])
		}
		return units[i] < units[j]
	})
	return strings.Join(units, "|")
}

// ExpandUnits expands unit abbreviations after a number into their spoken form,
// e.g. "5km" → "5 kilometers", "100kph" → "100 kilometers per hour".
func ExpandUnits(text string) string {
	expand := func(g []string) string {
		return g[1] + " " + unitMap[strings.ToLower(g[2])]
	}
	text = replaceAllSubmatch(unitRE, text, expand)
	return replaceAllSubmatch(ambiguousUnitRE, text, expand)
}

//
// Custom replacements
//

// ReplaceText returns a transform that applies a list of {pattern, replacement}
// rules in order. Patterns are regular expressions compiled up front, so an
// invalid pattern returns an error here rather than failing during synthesis.
func ReplaceText(rules [][2]string) (Transform, error) {
	type rule struct {
		re   *regexp.Regexp
		with string
	}
	compiled := make([]rule, len(rules))
	for i, r := range rules {
		re, err := regexp.Compile(r[0])
		if err != nil {
			return nil, err
		}
		compiled[i] = rule{re, r[1]}
	}
	return func(text string) string {
		for _, r := range compiled {
			text = r.re.ReplaceAllString(text, r.with)
		}
		return text
	}, nil
}

func atoi(s string) int {
	n, _ := strconv.Atoi(s)
	return n
}
