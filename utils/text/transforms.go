package text

import (
	"regexp"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
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
	mdFence1 = regexp.MustCompile("```[\\s\\S]*?```")
	mdFence2 = regexp.MustCompile(`~~~[\s\S]*?~~~`)
	// A link or image. The label may hold one level of brackets and the URL
	// one of parentheses (wiki/Mercury_(planet)); a URL with an unbalanced
	// parenthesis ends at the first ")".
	mdLink       = regexp.MustCompile(`!?\[((?:[^\[\]]|\[[^\[\]]*\])*)\](?:\((?:[^()]|\([^()]*\))*\)|\([^)]*\))`)
	mdCodeRef    = regexp.MustCompile(`\x{E001}(\d+)\x{E002}`)
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

// mdCodeOpen and mdCodeClose bracket the index of an inline code span held
// aside while the other rules run, so Markdown-like text inside backticks
// (“__init__“) is kept as written. Both are private-use characters.
const (
	mdCodeOpen  = "\uE001"
	mdCodeClose = "\uE002"
)

// StripMarkdown removes Markdown formatting symbols that have no spoken
// equivalent: bold/italic markers, backtick code spans, fenced code blocks, ATX
// headers, list bullets, blockquote markers and horizontal rules. The content
// of an inline code span is kept as written, Markdown-like text included. Struck-through
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
	// Inline code: hold the payload aside so later rules cannot rewrite it.
	text, codes := holdInlineCode(text)
	// Images and links, keeping the alt text or label.
	if stripURLs {
		text = mdLink.ReplaceAllString(text, "$1")
	}
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
	text = strings.ReplaceAll(text, mdStar, "*")
	return replaceAllSubmatch(mdCodeRef, text, func(g []string) string {
		if i, err := strconv.Atoi(g[1]); err == nil && i < len(codes) {
			return codes[i]
		}
		return g[0]
	})
}

// holdInlineCode replaces each inline code span with a placeholder holding its
// index in codes, the span contents. A span opens on a whole run of one or two
// backticks and closes on the nearest later run of the same length; a run of
// another length inside it is code content. A span holds at least one
// character and may cross lines. A run with no closer is left as written.
func holdInlineCode(text string) (string, []string) {
	var codes []string
	var b strings.Builder
	for i := 0; i < len(text); {
		if text[i] != '`' {
			b.WriteByte(text[i])
			i++
			continue
		}
		n := backtickRun(text, i)
		if n <= 2 {
			if end := inlineCodeCloser(text, i+n, n); end >= 0 {
				codes = append(codes, text[i+n:end])
				b.WriteString(mdCodeOpen + strconv.Itoa(len(codes)-1) + mdCodeClose)
				i = end + n
				continue
			}
		}
		b.WriteString(text[i : i+n])
		i += n
	}
	return b.String(), codes
}

// inlineCodeCloser returns where the run of exactly n backticks closing a span
// whose content starts at from begins, or -1 when there is none. The content
// starts with a character that is not a backtick, since the opening run ended
// there.
func inlineCodeCloser(text string, from, n int) int {
	if from >= len(text) {
		return -1
	}
	for j := from + 1; j < len(text); j++ {
		if text[j] != '`' {
			continue
		}
		m := backtickRun(text, j)
		if m == n {
			return j
		}
		j += m - 1
	}
	return -1
}

// backtickRun returns the length of the run of backticks starting at i.
func backtickRun(text string, i int) int {
	j := i
	for j < len(text) && text[j] == '`' {
		j++
	}
	return j - i
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

// phoneStep is one part of a phone number. Given where it starts, it lists
// where it may end, in the order the readings are tried.
type phoneStep func(text string, p int) []int

// phoneSteps are the parts of a phone number in order: an optional country
// code ("+1", "1", with an optional separator), the area code (optionally in
// parentheses), the exchange and the subscriber number, each optionally
// followed by a separator (a space, a dot or a dash).
//
//nolint:gochecknoglobals // immutable pattern
var phoneSteps = []phoneStep{
	phoneCountryCode,
	optionalByte(func(c byte) bool { return c == '(' }),
	phoneDigits(3),
	optionalByte(func(c byte) bool { return c == ')' }),
	optionalByte(isPhoneSeparator),
	phoneDigits(3),
	optionalByte(isPhoneSeparator),
	phoneDigits(4),
}

// ExpandPhoneNumbers spaces out phone-number digits so a synthesizer reads them
// one by one, e.g. "123-456-7890" → "1 2 3 4 5 6 7 8 9 0". A number must not
// start or end next to another digit, so digits inside a longer number are not
// split; a number that does is looked for again from the next character.
func ExpandPhoneNumbers(text string) string {
	var b strings.Builder
	last := 0
	for i := 0; i < len(text); {
		end, ok := 0, false
		if i == 0 || !isDigit(text[i-1]) {
			end, ok = matchPhone(text, i)
		}
		if !ok {
			_, size := utf8.DecodeRuneInString(text[i:])
			i += size
			continue
		}
		b.WriteString(text[last:i])
		digits := make([]string, 0, end-i)
		for j := i; j < end; j++ {
			if isDigit(text[j]) {
				digits = append(digits, text[j:j+1])
			}
		}
		b.WriteString(strings.Join(digits, " "))
		i, last = end, end
	}
	if last == 0 {
		return text
	}
	b.WriteString(text[last:])
	return b.String()
}

// matchPhone matches a phone number starting at byte offset p and returns
// where it ends. Where its parts can be read more than one way, the readings
// are tried in a fixed order (a part that is present before one that is not)
// and the first one not followed by a digit wins.
func matchPhone(text string, p int) (int, bool) {
	var try func(step, pos int) (int, bool)
	try = func(step, pos int) (int, bool) {
		if step == len(phoneSteps) {
			return pos, pos >= len(text) || !isDigit(text[pos])
		}
		for _, next := range phoneSteps[step](text, pos) {
			if end, ok := try(step+1, next); ok {
				return end, true
			}
		}
		return 0, false
	}
	return try(0, p)
}

// phoneCountryCode reads an optional "1" or "+1" with an optional separator
// after it, trying the longest reading first and no country code last.
func phoneCountryCode(text string, p int) []int {
	var q int
	switch {
	case strings.HasPrefix(text[p:], "+1"):
		q = p + 2
	case strings.HasPrefix(text[p:], "1"):
		q = p + 1
	default:
		return []int{p}
	}
	return append(optionalByte(isPhoneSeparator)(text, q), p)
}

// optionalByte reads one byte that satisfies ok, or nothing: with it first,
// then without.
func optionalByte(ok func(byte) bool) phoneStep {
	return func(text string, p int) []int {
		if p < len(text) && ok(text[p]) {
			return []int{p + 1, p}
		}
		return []int{p}
	}
}

// phoneDigits reads exactly n digits.
func phoneDigits(n int) phoneStep {
	return func(text string, p int) []int {
		if p+n > len(text) || digitRunEnd(text, p) < p+n {
			return nil
		}
		return []int{p + n}
	}
}

// isPhoneSeparator reports whether c may separate the parts of a phone
// number: whitespace, a dot or a dash.
func isPhoneSeparator(c byte) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '\f', '\v', '.', '-':
		return true
	}
	return false
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
// dates into spoken form, e.g. "2023-05-10" → "May 10th, twenty twenty-three".
// The year is read the way a date is said aloud ("nineteen ninety-nine"), not
// as a quantity. A string that parses to no valid calendar date is left
// unchanged.
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
	// The calendar starts at year 1: there is no year 0 to read.
	if year < 1 || month < 1 || month > 12 || day < 1 {
		return original
	}
	t := time.Date(year, time.Month(month), day, 0, 0, 0, 0, time.UTC)
	if t.Year() != year || int(t.Month()) != month || t.Day() != day {
		return original
	}
	return monthNames[month-1] + " " + ordinal(day) + ", " + spokenYear(year)
}

// spokenYear reads a year the way it is said aloud: in two halves
// ("nineteen ninety-nine", "twenty twenty-three"), with "oh" before a single
// digit ("nineteen oh-five") and "hundred" for a round century ("nineteen
// hundred"). A year below 100, one beyond 9999, and one whose first half ends
// in a zero while its second half is below ten ("two thousand and five") are
// read as a quantity.
func spokenYear(year int) string {
	high, low := year/100, year%100
	if high == 0 || (high%10 == 0 && low < 10) || high >= 100 {
		return cardinal(int64(year))
	}
	var lowText string
	switch {
	case low == 0:
		lowText = "hundred"
	case low < 10:
		lowText = "oh-" + cardinal(int64(low))
	default:
		lowText = cardinal(int64(low))
	}
	return cardinal(int64(high)) + " " + lowText
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
	currencyMap = map[rune]struct {
		singular, plural, centSingular, centPlural string
	}{
		'$': {"dollar", "dollars", "cent", "cents"},
		'€': {"euro", "euros", "cent", "cents"},
		'£': {"pound", "pounds", "penny", "pence"},
		'¥': {"yen", "yen", "", ""},
		'₹': {"rupee", "rupees", "paisa", "paise"},
	}

	// Scale words that may follow an amount after a space ("$2.5 billion"):
	// thousand, million, billion and trillion.
	currencyScaleWords = scaleWords[1:5]

	// Abbreviated scales written directly after an amount ("$5k", "$2.5m",
	// "$1.5bn"), in the order they are tried, each with the index of its word
	// in scaleWords.
	currencyScaleAbbreviations = []struct {
		abbr  string
		scale int
	}{
		{"k", 1},
		{"mn", 2},
		{"m", 2},
		{"bn", 3},
		{"b", 3},
		{"tn", 4},
		{"t", 4},
	}
)

// currencyMatch is one currency amount found in the text: the symbol, the
// whole and fractional digits as written, the scale word it is multiplied by
// (empty when none), and the byte offset just past it.
type currencyMatch struct {
	symbol      rune
	whole, frac string
	scale       string
	end         int
}

// ExpandCurrency expands currency amounts into spoken form, e.g. "$42.50" →
// "forty-two dollars and fifty cents".
//
// An amount may be followed by a scale word ("$2.5 billion") or its
// abbreviation ("$2.5bn"), which multiplies the whole amount, so the fraction is
// part of the number rather than a subunit: "two point five billion dollars".
// The amount must end the word, so one with an unrecognized suffix ("$1.5x") is
// left alone rather than read as cents. A currency code written right before
// the symbol ("US$5") is kept apart from the spoken amount.
func ExpandCurrency(text string) string {
	var b strings.Builder
	last := 0
	for i := 0; i < len(text); {
		m, ok := matchCurrency(text, i)
		if !ok {
			_, size := utf8.DecodeRuneInString(text[i:])
			i += size
			continue
		}
		b.WriteString(text[last:i])
		if i > 0 {
			if r, _ := utf8.DecodeLastRuneInString(text[:i]); unicode.IsLetter(r) {
				b.WriteByte(' ')
			}
		}
		b.WriteString(spokenCurrency(m, text[i:m.end]))
		i, last = m.end, m.end
	}
	if last == 0 {
		return text
	}
	b.WriteString(text[last:])
	return b.String()
}

// spokenCurrency reads one matched amount aloud. original is the matched text,
// returned as is when the whole part does not fit an int64.
func spokenCurrency(m currencyMatch, original string) string {
	cur := currencyMap[m.symbol]
	whole, err := strconv.ParseInt(strings.ReplaceAll(m.whole, ",", ""), 10, 64)
	if err != nil {
		return original
	}
	if m.scale != "" {
		// The fraction belongs to the scaled number, and the unit is always
		// plural. "$2.50 million" is two and a half million, so a trailing
		// zero in the written fraction is not a digit to read out.
		number := cardinal(whole)
		if fraction := strings.TrimRight(m.frac, "0"); fraction != "" {
			digits := make([]string, len(fraction))
			for i := range fraction {
				digits[i] = onesWords[fraction[i]-'0']
			}
			number += " point " + strings.Join(digits, " ")
		}
		return number + " " + m.scale + " " + cur.plural
	}
	result := amountWords(whole, cur.singular, cur.plural)
	if m.frac != "" && cur.centSingular != "" {
		// Only the first two digits are subunits; anything past them is below
		// what the currency can express.
		cents, _ := strconv.ParseInt(padRight(truncate(m.frac, 2), 2, '0'), 10, 64)
		if cents > 0 {
			result += " and " + amountWords(cents, cur.centSingular, cur.centPlural)
		}
	}
	return result
}

// matchCurrency matches a currency amount starting at byte offset i: a symbol,
// optional whitespace, the whole part (digits grouped by commas, or a plain
// run of digits), an optional fraction, then an optional scale word or
// abbreviation, ending the word. Where the amount can be read more than one
// way, the readings are tried in a fixed order and the first that ends the
// word wins: longer before shorter, a fraction before none, a scale before
// none.
func matchCurrency(text string, i int) (currencyMatch, bool) {
	sym, size := utf8.DecodeRuneInString(text[i:])
	if _, ok := currencyMap[sym]; !ok {
		return currencyMatch{}, false
	}
	p := skipSpaces(text, i+size)
	for _, wholeEnd := range currencyWholeEnds(text, p) {
		if wholeEnd < len(text) && isDigit(text[wholeEnd]) {
			continue
		}
		m := currencyMatch{symbol: sym, whole: text[p:wholeEnd]}
		// With a fraction, every digit of it: a shorter one leaves a digit
		// behind, which never ends the word.
		if wholeEnd+1 < len(text) && text[wholeEnd] == '.' && isDigit(text[wholeEnd+1]) {
			fracEnd := digitRunEnd(text, wholeEnd+1)
			if !dotDigitAt(text, fracEnd) {
				if scale, end, ok := matchCurrencyScale(text, fracEnd); ok {
					m.frac, m.scale, m.end = text[wholeEnd+1:fracEnd], scale, end
					return m, true
				}
			}
		}
		if dotDigitAt(text, wholeEnd) {
			continue
		}
		if scale, end, ok := matchCurrencyScale(text, wholeEnd); ok {
			m.scale, m.end = scale, end
			return m, true
		}
	}
	return currencyMatch{}, false
}

// currencyWholeEnds lists, in the order they are tried, the offsets where the
// whole part of an amount starting at p may end: digits grouped by commas
// (1,000), longest reading first, then a plain run of digits.
func currencyWholeEnds(text string, p int) []int {
	run := digitRunEnd(text, p) - p
	if run == 0 {
		return nil
	}
	var ends []int
	for lead := min(run, 3); lead >= 1; lead-- {
		groups := []int{p + lead}
		for q := p + lead; q+4 <= len(text) && text[q] == ','; q += 4 {
			if digitRunEnd(text, q+1) < q+4 {
				break
			}
			groups = append(groups, q+4)
		}
		for _, end := range slices.Backward(groups) {
			ends = append(ends, end)
		}
	}
	for n := run; n >= 1; n-- {
		ends = append(ends, p+n)
	}
	return ends
}

// matchCurrencyScale matches what may follow an amount ending at p: a space
// and a scale word, an abbreviated scale, or nothing; whichever it is, it must
// end the word. It returns the scale word (empty for none) and where the match
// ends.
func matchCurrencyScale(text string, p int) (string, int, bool) {
	if q := skipSpaces(text, p); q > p {
		for _, w := range currencyScaleWords {
			if hasFoldPrefix(text[q:], w) && !wordRuneAt(text, q+len(w)) {
				return w, q + len(w), true
			}
		}
	}
	for _, a := range currencyScaleAbbreviations {
		if hasFoldPrefix(text[p:], a.abbr) && !wordRuneAt(text, p+len(a.abbr)) {
			return scaleWords[a.scale], p + len(a.abbr), true
		}
	}
	if !wordRuneAt(text, p) {
		return "", p, true
	}
	return "", 0, false
}

// hasFoldPrefix reports whether s starts with the lower-case ASCII word w,
// ignoring case.
func hasFoldPrefix(s, w string) bool {
	return len(s) >= len(w) && strings.EqualFold(s[:len(w)], w)
}

// wordRuneAt reports whether a word character (a letter, a number or an
// underscore) starts at byte offset i.
func wordRuneAt(text string, i int) bool {
	if i >= len(text) {
		return false
	}
	r, _ := utf8.DecodeRuneInString(text[i:])
	return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}

// dotDigitAt reports whether a "." followed by a digit starts at byte offset i.
func dotDigitAt(text string, i int) bool {
	return i+1 < len(text) && text[i] == '.' && isDigit(text[i+1])
}

// digitRunEnd returns the offset just past the run of ASCII digits starting
// at i.
func digitRunEnd(text string, i int) int {
	for i < len(text) && isDigit(text[i]) {
		i++
	}
	return i
}

// skipSpaces returns the offset just past the run of whitespace starting at i.
func skipSpaces(text string, i int) int {
	for i < len(text) {
		r, size := utf8.DecodeRuneInString(text[i:])
		if !unicode.IsSpace(r) {
			break
		}
		i += size
	}
	return i
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
var percentRE = regexp.MustCompile(`(\d{1,3}(?:,\d{3})*|\d+)(?:\.(\d+))?\s*%`)

// ExpandPercentages expands percentage expressions into spoken form, e.g. "50%"
// → "fifty percent". A thousands separator is part of the number, so "1,500%"
// is "one thousand, five hundred percent".
func ExpandPercentages(text string) string {
	return replaceAllSubmatch(percentRE, text, func(g []string) string {
		value := strings.ReplaceAll(g[1], ",", "")
		if g[2] != "" {
			value += "." + g[2]
		}
		return floatToWords(value) + " percent"
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

	// Unambiguous units allow optional whitespace between the number and the
	// unit, and match in any case.
	unitRE = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)\s*(` + unitAlternation(false) + `)\b`)
	// Ambiguous units must follow the digit with no space, and match case
	// sensitively: "5M" (million) and "5G" (the network) are not meters or
	// grams. Liters are also written with a capital "L".
	ambiguousUnitRE = regexp.MustCompile(`(\d+(?:\.\d+)?)(` + unitAlternation(true, "L") + `)\b`)
)

// unitAlternation builds a regex alternation of the ambiguous or unambiguous
// unit abbreviations, plus any extra spellings of them, longest first so a
// longer unit wins over a prefix of it ("mph" before "mi").
func unitAlternation(ambiguous bool, extra ...string) string {
	units := slices.Clone(extra)
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
