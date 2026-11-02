package text

import (
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
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
	mdLink       = regexp.MustCompile(`!?\[([^\]]*)\]\([^)]*\)`)
	mdInlineCode = regexp.MustCompile("`([^`]+)`")
	mdRule       = regexp.MustCompile(`(?m)^[ \t]*(?:-(?:[ \t]*-){2,}|\*(?:[ \t]*\*){2,}|_(?:[ \t]*_){2,})[ \t]*$\n?`)
	mdBullet     = regexp.MustCompile(`(?m)^([ \t]*)[-+*][ \t]+(\S)`)
	mdBoldItalic = regexp.MustCompile(`\*{3}(.+?)\*{3}`)
	mdBoldItalU  = regexp.MustCompile(`_{3}(.+?)_{3}`)
	mdBold       = regexp.MustCompile(`\*{2}(.+?)\*{2}`)
	mdBoldU      = regexp.MustCompile(`_{2}(.+?)_{2}`)
	mdItalicU    = regexp.MustCompile(`\b_(.+?)_\b`)
	mdHeader     = regexp.MustCompile(`(?m)^#{1,6}\s+`)
	mdQuote      = regexp.MustCompile(`(?m)^>\s?`)
)

// mdStar stands in for an asterisk that is content rather than emphasis while
// the emphasis rules run. It is a private-use character, so it cannot occur in
// the text.
const mdStar = "\uE000"

// StripMarkdown removes Markdown formatting symbols that have no spoken
// equivalent: bold/italic markers, backtick code spans, fenced code blocks, ATX
// headers, list bullets, blockquote markers and horizontal rules. Struck-through
// text (~~like this~~) is removed along with its markers, since it is text the
// reply takes back. A link or image is reduced to its label or alt text, so the
// URL is not read out; NewStripMarkdown builds one that keeps them.
//
// An asterisk that is content stays: one between digits (2*3), one escaped as
// \*, or one standing alone between spaces ("Press * then #").
//
// Outside of fenced code, struck-through text and link URLs, it preserves the
// text's letters and digits: the normalized alphanumeric sequence of the output
// is the input's, so word tracking needs no segment map.
func StripMarkdown(text string) string {
	return stripMarkdown(text, true)
}

// NewStripMarkdown returns StripMarkdown with the given URL handling. With
// stripURLs off, [label](url) and ![alt](url) are left as written.
func NewStripMarkdown(stripURLs bool) Transform {
	return func(text string) string { return stripMarkdown(text, stripURLs) }
}

func stripMarkdown(text string, stripURLs bool) string {
	text = mdFence1.ReplaceAllString(text, "")
	text = mdFence2.ReplaceAllString(text, "")
	// Images and links, keeping the alt text or label.
	if stripURLs {
		text = mdLink.ReplaceAllString(text, "$1")
	}
	text = mdInlineCode.ReplaceAllString(text, "$1")
	text = stripStrikethrough(text)
	// Horizontal rules (---, ***, * * *) go before list bullets, so a spaced
	// one is not taken for an item.
	text = mdRule.ReplaceAllString(text, "")
	// List bullets (-, +, *) at any indent, since nested items are indented.
	text = mdBullet.ReplaceAllString(text, "$1$2")
	text = protectContentAsterisks(text)
	text = mdBoldItalic.ReplaceAllString(text, "$1")
	text = mdBoldItalU.ReplaceAllString(text, "$1")
	text = mdBold.ReplaceAllString(text, "$1")
	text = mdBoldU.ReplaceAllString(text, "$1")
	text = stripItalic(text)
	text = mdItalicU.ReplaceAllString(text, "$1")
	text = mdHeader.ReplaceAllString(text, "")
	text = mdQuote.ReplaceAllString(text, "")
	return strings.ReplaceAll(text, mdStar, "*")
}

// protectContentAsterisks replaces each asterisk that is content with mdStar:
// one escaped as \*, which loses its backslash, and each in a run between two
// digits (2*3, 2**3).
func protectContentAsterisks(text string) string {
	rs := []rune(text)
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		if rs[i] == '\\' && i+1 < len(rs) && rs[i+1] == '*' {
			b.WriteString(mdStar)
			i++
			continue
		}
		if rs[i] == '*' && i > 0 && unicode.IsDigit(rs[i-1]) {
			j := i
			for j < len(rs) && rs[j] == '*' {
				j++
			}
			if j < len(rs) && unicode.IsDigit(rs[j]) {
				b.WriteString(strings.Repeat(mdStar, j-i))
				i = j - 1
				continue
			}
		}
		b.WriteRune(rs[i])
	}
	return b.String()
}

// stripItalic unwraps *text* whose markers hug it: the opening asterisk is
// followed by a non-space and the closing one preceded by one, so a standalone
// asterisk ("Press * then #") opens and closes nothing. The text is one line,
// and the closing asterisk is the nearest one that qualifies.
func stripItalic(text string) string {
	rs := []rune(text)
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		if rs[i] == '*' {
			if end := hugged(rs, i+1, "*"); end >= 0 {
				b.WriteString(string(rs[i+1 : end]))
				i = end
				continue
			}
		}
		b.WriteRune(rs[i])
	}
	return b.String()
}

// stripStrikethrough removes ~~struck~~ text whose markers hug it, along with
// one space or tab before it so the words around it still join. The text is
// one line, and the closing marker is the nearest one that qualifies.
func stripStrikethrough(text string) string {
	rs := []rune(text)
	var b strings.Builder
	for i := 0; i < len(rs); i++ {
		at := i
		if (rs[i] == ' ' || rs[i] == '\t') && hasPrefixAt(rs, i+1, "~~") {
			at = i + 1
		}
		if hasPrefixAt(rs, at, "~~") {
			if end := hugged(rs, at+2, "~~"); end >= 0 {
				i = end + 1
				continue
			}
		}
		b.WriteRune(rs[i])
	}
	return b.String()
}

// hugged finds the closing marker for content starting at from. The content is
// at least one character, starts with a non-space, holds no newline, and ends
// with a non-space right before the marker. It returns where the nearest such
// marker starts, or -1 when there is none.
func hugged(rs []rune, from int, marker string) int {
	if from >= len(rs) || unicode.IsSpace(rs[from]) {
		return -1
	}
	for j := from + 1; j < len(rs); j++ {
		if rs[j] == '\n' {
			return -1
		}
		if hasPrefixAt(rs, j, marker) && !unicode.IsSpace(rs[j-1]) {
			return j
		}
	}
	return -1
}

// hasPrefixAt reports whether rs holds prefix starting at i.
func hasPrefixAt(rs []rune, i int, prefix string) bool {
	for _, r := range prefix {
		if i >= len(rs) || rs[i] != r {
			return false
		}
		i++
	}
	return true
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
//
// Decimals are read as the whole part followed by their fractional digits, one
// at a time, so every written digit is spoken ("1.0" → "one point zero").
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
				// Reading the fraction one digit at a time keeps trailing zeros
				// audible, which also keeps a decimal agreeing with the plural
				// unit ExpandUnits picks for it ("1.0 kilometers").
				digits := make([]string, len(frac))
				for i := range frac {
					digits[i] = onesWords[frac[i]-'0']
				}
				return cardinal(whole) + " point " + strings.Join(digits, " ")
			}
			return cardinal(whole)
		})
	}
}

//
// Units
//

// unitForms is the singular and plural spoken form of one unit abbreviation.
// They are the same word for a unit that does not inflect, "hertz" say.
type unitForms struct{ singular, plural string }

//nolint:gochecknoglobals // compiled once, immutable
var (
	// unitMap maps a unit abbreviation to its singular and plural spoken forms.
	unitMap = map[string]unitForms{
		"km": {"kilometer", "kilometers"},
		"m":  {"meter", "meters"},
		"cm": {"centimeter", "centimeters"},
		"mm": {"millimeter", "millimeters"},
		"mi": {"mile", "miles"},
		"ft": {"foot", "feet"},
		"in": {"inch", "inches"},
		"yd": {"yard", "yards"},
		"kg": {"kilogram", "kilograms"},
		"g":  {"gram", "grams"},
		"mg": {"milligram", "milligrams"},
		"lb": {"pound", "pounds"},
		"oz": {"ounce", "ounces"},
		"l":  {"liter", "liters"},
		"ml": {"milliliter", "milliliters"},

		"mph": {"mile per hour", "miles per hour"},
		"kph": {"kilometer per hour", "kilometers per hour"},
		"kmh": {"kilometer per hour", "kilometers per hour"},

		"gb": {"gigabyte", "gigabytes"},
		"mb": {"megabyte", "megabytes"},
		"kb": {"kilobyte", "kilobytes"},
		"tb": {"terabyte", "terabytes"},

		"hz":  {"hertz", "hertz"},
		"khz": {"kilohertz", "kilohertz"},
		"mhz": {"megahertz", "megahertz"},
		"ghz": {"gigahertz", "gigahertz"},
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
// e.g. "5km" → "5 kilometers", "100kph" → "100 kilometers per hour". A quantity
// of exactly one takes the singular form, "1km" → "1 kilometer".
func ExpandUnits(text string) string {
	expand := func(g []string) string {
		forms := unitMap[strings.ToLower(g[2])]
		// Only a bare "1" takes the singular; a decimal such as "1.0" reads as
		// plural in speech, and ExpandNumbers spells it out that way.
		if g[1] == "1" {
			return g[1] + " " + forms.singular
		}
		return g[1] + " " + forms.plural
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
