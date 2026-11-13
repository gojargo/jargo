package text

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
)

func TestStripMarkdown(t *testing.T) {
	cases := map[string]string{
		"**Hello** and _world_":    "Hello and world",
		"**Hello** world":          "Hello world",
		"*italic*":                 "italic",
		"_italic_":                 "italic",
		"# Title\n\nBody":          "Title\n\nBody",
		"***bold italic***":        "bold italic",
		"`code` span":              "code span",
		"# Heading":                "Heading",
		"> quoted":                 "quoted",
		"plain text":               "plain text",
		"see [the docs](x) please": "see the docs please",
	}
	for in, want := range cases {
		if got := StripMarkdown(in); got != want {
			t.Errorf("StripMarkdown(%q) = %q, want %q", in, got, want)
		}
	}
}

func assertStripMarkdown(t *testing.T, strip Transform, cases map[string]string) {
	t.Helper()
	for in, want := range cases {
		if got := strip(in); got != want {
			t.Errorf("StripMarkdown(%q) = %q, want %q", in, got, want)
		}
	}
}

// An asterisk that is content rather than emphasis is kept.
func TestStripMarkdownKeepsContentAsterisks(t *testing.T) {
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		"Press * then #":              "Press * then #",
		"5 * 3 = 15":                  "5 * 3 = 15",
		"2 * 3 * 4 = 24":              "2 * 3 * 4 = 24",
		"Use the * key, then * again": "Use the * key, then * again",
		"2*3*4 is 24":                 "2*3*4 is 24",
		"2**3 is 8":                   "2**3 is 8",
		`Type \*literal\* asterisks`:  "Type *literal* asterisks",
	})
}

// List bullets are stripped at any indent, and numbered items are left alone.
func TestStripMarkdownStripsListBullets(t *testing.T) {
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		"\n- Coffee for three dollars.": "\nCoffee for three dollars.",
		"- Step one.\n    - Sub step.":  "Step one.\n    Sub step.",
		"* item one\n* item two":        "item one\nitem two",
		"+ plus item":                   "plus item",
		"1. First\n2. Second":           "1. First\n2. Second",
		"a - b":                         "a - b",
	})
}

// Struck-through text is text the reply takes back, so it is dropped along
// with its markers.
func TestStripMarkdownDropsStrikethrough(t *testing.T) {
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		"The price was ~~ten~~ eight dollars.": "The price was eight dollars.",
		"It is ~~not ready~~ ready now.":       "It is ready now.",
		"~~Old.~~ New.":                        " New.",
		"About ~~ 5 minutes":                   "About ~~ 5 minutes",
		"Roughly ~5 minutes":                   "Roughly ~5 minutes",
	})
}

// A link or image is reduced to its label or alt text, so the URL is not read
// out.
func TestStripMarkdownReducesLinksToTheirLabel(t *testing.T) {
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		"See [the docs](https://example.com/docs) here.": "See the docs here.",
		"See ![a cat](cat.png) here.":                    "See a cat here.",
		"Go to [example.com](https://example.com).":      "Go to example.com.",
		"Visit https://example.com today":                "Visit https://example.com today",
		"[**bold** link](x)":                             "bold link",
		// The URL may hold one level of parentheses, an image a title, and the
		// label one level of brackets.
		"See [Mercury (planet)](https://en.wikipedia.org/wiki/Mercury_(planet)) now.": "See Mercury (planet) now.",
		`![chart](c.png "Q3 (draft)")`:                        "chart",
		"Read [the [beta] guide](https://example.com/guide).": "Read the [beta] guide.",
		// A URL with an unbalanced parenthesis ends at the first ")".
		"Open [the page](https://example.com/a_(b) now.": "Open the page now.",
	})
}

// An inline code span opens on a run of one or two backticks and closes on the
// next run of the same length; a run of another length inside it is content.
func TestStripMarkdownMatchesInlineCodeDelimitersByLength(t *testing.T) {
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		"Use ``__init__`` here.":    "Use __init__ here.",
		"Use ``a ` b`` here.":       "Use a ` b here.",
		"Use `a `` b` here.":        "Use a `` b here.",
		"``one ` two`` and `three`": "one ` two and three",
		"**bold ``a ` b``** end":    "bold a ` b end",
		"``first\nsecond``":         "first\nsecond",
	})
}

// Backticks with no matching closer are left as written.
func TestStripMarkdownLeavesUnmatchedCodeDelimiters(t *testing.T) {
	for _, in := range []string{"Use ``unclosed` here.", "Use `unclosed`` here.", "Use ``unclosed here."} {
		if got := StripMarkdown(in); got != in {
			t.Errorf("StripMarkdown(%q) = %q, want it unchanged", in, got)
		}
	}
}

// A fenced code block is removed whole, backticks inside it included.
func TestStripMarkdownRemovesFencedCodeHoldingBackticks(t *testing.T) {
	for _, fence := range []string{"```", "~~~"} {
		in := "Before " + fence + "go\nvalue = `code`\n" + fence + " after ``shown``."
		if got, want := StripMarkdown(in), "Before  after shown."; got != want {
			t.Errorf("StripMarkdown(%q) = %q, want %q", in, got, want)
		}
	}
}

// What is inside an inline code span is kept as written, even when it looks
// like Markdown.
func TestStripMarkdownKeepsInlineCodeContent(t *testing.T) {
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		"call `__init__` please": "call __init__ please",
		"`*args`":                "*args",
		"`# comment`":            "# comment",
		"`> quoted`":             "> quoted",
		"`- item`":               "- item",
		"**bold `__x__`** end":   "bold __x__ end",
		"- `*x*` item":           "*x* item",
		"`a_b` and `c*d`":        "a_b and c*d",
	})
}

// Link markup inside inline code is code, so it is kept whether or not URLs are
// stripped.
func TestStripMarkdownKeepsLinkMarkupInInlineCode(t *testing.T) {
	cases := map[string]string{
		"Use `[label](https://example.com)` here.":          "Use [label](https://example.com) here.",
		"Use `![alt](image.png)` here.":                     "Use ![alt](image.png) here.",
		"`[Mercury](https://example.com/Mercury_(planet))`": "[Mercury](https://example.com/Mercury_(planet))",
		"``[label](url)``":                                  "[label](url)",
		"``![alt](image.png)``":                             "![alt](image.png)",
	}
	for _, stripURLs := range []bool{true, false} {
		assertStripMarkdown(t, NewStripMarkdown(stripURLs), cases)
	}
}

// Inline code and real links in the same text: the links are reduced to their
// label, code inside a label is kept, and link markup in code stays.
func TestStripMarkdownHandlesInlineCodeAndLinksTogether(t *testing.T) {
	const linkAndCode = "[Guide](https://example.com) shows `[label](url)` and `![alt](image.png)`."
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		linkAndCode: "Guide shows [label](url) and ![alt](image.png).",
		"Read [`__init__`](https://example.com).":      "Read __init__.",
		"[Link](`url`) followed by `[label](target)`.": "Link followed by [label](target).",
	})
}

// With URL stripping off, a link is left as written.
func TestStripMarkdownCanKeepLinks(t *testing.T) {
	text := "See [the docs](https://example.com/docs) here."
	if got := NewStripMarkdown(false)(text); got != text {
		t.Errorf("NewStripMarkdown(false)(%q) = %q, want it unchanged", text, got)
	}
}

// Horizontal rules are removed, spaced ones included.
func TestStripMarkdownRemovesHorizontalRules(t *testing.T) {
	assertStripMarkdown(t, StripMarkdown, map[string]string{
		"First part.\n\n---\n\nSecond part.": "First part.\n\n\nSecond part.",
		"* * *\ntext":                        "text",
		"___\ntext":                          "text",
	})
}

func TestEmailToSpeech(t *testing.T) {
	got := EmailToSpeech("Contact user_1@example.com today")
	want := "Contact user underscore 1 at example dot com today"
	if got != want {
		t.Errorf("EmailToSpeech = %q, want %q", got, want)
	}
	assertTransform(t, "EmailToSpeech", EmailToSpeech, map[string]string{
		"user@example.com":             "user at example dot com",
		"Email user@example.com today": "Email user at example dot com today",
	})
}

func TestExpandPhoneNumbers(t *testing.T) {
	got := ExpandPhoneNumbers("Call 123-456-7890 now")
	want := "Call 1 2 3 4 5 6 7 8 9 0 now"
	if got != want {
		t.Errorf("ExpandPhoneNumbers = %q, want %q", got, want)
	}
	// A 14-digit blob is not a phone number: bordered by digits, it is skipped.
	if got := ExpandPhoneNumbers("12345678901234"); got != "12345678901234" {
		t.Errorf("long digit run should be untouched, got %q", got)
	}
}

// A number written with dots is spaced out like one written with dashes.
func TestExpandPhoneNumbersReadsDottedNumbers(t *testing.T) {
	assertTransform(t, "ExpandPhoneNumbers", ExpandPhoneNumbers, map[string]string{
		"123.456.7890": "1 2 3 4 5 6 7 8 9 0",
	})
}

// A number that cannot start where a candidate starts is looked for again from
// the next character, so one that follows other digits after a space is still
// found, and a country code is read with the number.
func TestExpandPhoneNumbersLooksAgainFromTheNextCharacter(t *testing.T) {
	assertTransform(t, "ExpandPhoneNumbers", ExpandPhoneNumbers, map[string]string{
		"Hello world":                        "Hello world",
		"91 123-456-7890":                    "91 1 2 3 4 5 6 7 8 9 0",
		"Call +1 (555) 123-4567 now":         "Call 1 5 5 5 1 2 3 4 5 6 7 now",
		"12345678901":                        "1 2 3 4 5 6 7 8 9 0 1",
		"Card 4111111111111111 is invalid":   "Card 4111111111111111 is invalid",
		"Ref 123-456-78901 and 555.123.4567": "Ref 123-456-78901 and 5 5 5 1 2 3 4 5 6 7",
	})
}

func TestNormalizeAcronyms(t *testing.T) {
	got := NormalizeAcronyms("Use the API or HTTP endpoint")
	want := "Use the A P I or H T T P endpoint"
	if got != want {
		t.Errorf("NormalizeAcronyms = %q, want %q", got, want)
	}
	// CamelCase and plural acronyms are left alone.
	if got := NormalizeAcronyms("IPhone APIs"); got != "IPhone APIs" {
		t.Errorf("NormalizeAcronyms(camel/plural) = %q, want unchanged", got)
	}
	assertTransform(t, "NormalizeAcronyms", NormalizeAcronyms, map[string]string{
		"NASA launched": "N A S A launched",
		"iPhone":        "iPhone",
	})
}

func TestNormalizeDates(t *testing.T) {
	cases := map[string]string{
		"Meeting on 2023-05-10": "Meeting on May 10th, twenty twenty-three",
		"Due 05/10/2023":        "Due May 10th, twenty twenty-three",
		"Meeting on 05-10-2023": "Meeting on May 10th, twenty twenty-three",
		"Not 2023-13-40":        "Not 2023-13-40", // invalid date left as-is
		"Due 13/45/2023 now":    "Due 13/45/2023 now",
	}
	assertTransform(t, "NormalizeDates", NormalizeDates, cases)
}

func assertTransform(t *testing.T, name string, transform Transform, cases map[string]string) {
	t.Helper()
	for in, want := range cases {
		if got := transform(in); got != want {
			t.Errorf("%s(%q) = %q, want %q", name, in, got, want)
		}
	}
}

func assertTransformContains(t *testing.T, name string, transform Transform, in, want string) {
	t.Helper()
	if got := transform(in); !strings.Contains(got, want) {
		t.Errorf("%s(%q) = %q, want it to contain %q", name, in, got, want)
	}
}

// The day takes its ordinal suffix, with the teens taking "th".
func TestNormalizeDatesUsesOrdinalSuffixes(t *testing.T) {
	for in, want := range map[string]string{
		"2023-05-01": "1st",
		"2023-05-02": "2nd",
		"2023-05-03": "3rd",
		"2023-05-11": "11th",
		"2023-05-12": "12th",
		"2023-05-13": "13th",
		"2023-05-21": "21st",
	} {
		assertTransformContains(t, "NormalizeDates", NormalizeDates, in, want)
	}
}

// A year is spoken the way a date is said aloud, not as a quantity ("one
// thousand, nine hundred and ninety-nine").
func TestNormalizeDatesReadsTheYearAsAYear(t *testing.T) {
	assertTransform(t, "NormalizeDates", NormalizeDates, map[string]string{
		"Born on 1999-03-05":    "Born on March 5th, nineteen ninety-nine",
		"Due 07/04/1905":        "Due July 4th, nineteen oh-five",
		"Meeting on 2023-05-10": "Meeting on May 10th, twenty twenty-three",
		"Launch on 2005-01-02":  "Launch on January 2nd, two thousand and five",
		"Built 1900-06-01":      "Built June 1st, nineteen hundred",
		"Since 2000-01-01":      "Since January 1st, two thousand",
		"Year 0000-01-01":       "Year 0000-01-01",
	})
}

func TestExpandCurrency(t *testing.T) {
	cases := map[string]string{
		"Your balance is $42.50": "Your balance is forty-two dollars and fifty cents",
		"It costs $1":            "It costs one dollar",
		"Paid £3.01":             "Paid three pounds and one penny",
		"£1.01":                  "one pound and one penny",
		"Total ¥100":             "Total one hundred yen",
		// A fraction with more digits than the currency can express is read to
		// subunit precision, with nothing left over to say after the words.
		"The item costs $5.500 today": "The item costs five dollars and fifty cents today",
		"$3.567":                      "three dollars and fifty-six cents",
		"£1.999":                      "one pound and ninety-nine pence",
	}
	for in, want := range cases {
		if got := ExpandCurrency(in); got != want {
			t.Errorf("ExpandCurrency(%q) = %q, want %q", in, got, want)
		}
	}
}

// A ".00" fraction does not add "and zero cents".
func TestExpandCurrencyOmitsZeroCents(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"$5.00": "five dollars",
	})
}

// A currency with no subunit drops the fractional part.
func TestExpandCurrencyDropsTheFractionOfACurrencyWithNoSubunit(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"¥500.50": "five hundred yen",
	})
}

// A thousands separator is part of the amount, and an amount of four digits or
// more is read whole without one.
func TestExpandCurrencyReadsTheWholeAmount(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"$1,000":   "one thousand dollars",
		"$1000":    "one thousand dollars",
		"$1000.50": "one thousand dollars and fifty cents",
	})
}

// A scale word after the amount multiplies all of it, so the fraction is part
// of the number rather than cents, and the unit is plural.
func TestExpandCurrencyReadsAScaleWordAfterTheAmount(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"Revenue hit $2.5 billion last year": "Revenue hit two point five billion dollars last year",
		"We raised $5 million":               "We raised five million dollars",
		"$1 million":                         "one million dollars",
		"€3.25 Trillion":                     "three point two five trillion euros",
		"¥40 thousand":                       "forty thousand yen",
	})
}

// A trailing zero in the fraction of a scaled amount is not read out, since
// "$2.50 million" is two and a half million; a zero inside it is.
func TestExpandCurrencyDropsTrailingZerosOfAScaledAmount(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"We raised $2.50 million": "We raised two point five million dollars",
		"$1.50 billion":           "one point five billion dollars",
		"$1.0 million":            "one million dollars",
		"$2.05 million":           "two point zero five million dollars",
	})
}

// An abbreviated scale written right after the amount reads as its word.
func TestExpandCurrencyReadsAnAbbreviatedScale(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"They raised $2.5m": "They raised two point five million dollars",
		"a $1.5bn deal":     "a one point five billion dollars deal",
		"$5m":               "five million dollars",
		"$500k":             "five hundred thousand dollars",
		"$3B":               "three billion dollars",
		"£3.5mn":            "three point five million pounds",
		"$1T":               "one trillion dollars",
	})
	f, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	if got, want := format(t, f, "raised $5m last year"), "raised five million dollars last year"; got != want {
		t.Errorf("VoiceFormatter(%q) = %q, want %q", "raised $5m last year", got, want)
	}
}

// An amount followed by letters that are not a scale is left alone, rather than
// read with its fraction as cents.
func TestExpandCurrencyLeavesAnAmountFollowedByOtherLettersAlone(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"$1.5x": "$1.5x",
		"$5mo":  "$5mo",
	})
}

// A currency code written right before the symbol stays apart from the spoken
// amount.
func TestExpandCurrencyKeepsACurrencyCodeApartFromTheAmount(t *testing.T) {
	assertTransform(t, "ExpandCurrency", ExpandCurrency, map[string]string{
		"US$5":     "US five dollars",
		"CAD$10":   "CAD ten dollars",
		" pay $5":  " pay five dollars",
		"Pay: $5.": "Pay: five dollars.",
	})
}

func TestExpandPercentages(t *testing.T) {
	cases := map[string]string{
		"50% off":    "fifty percent off",
		"12.5% rate": "twelve point five percent rate",
		"3.5% rate":  "three point five percent rate",
	}
	for in, want := range cases {
		if got := ExpandPercentages(in); got != want {
			t.Errorf("ExpandPercentages(%q) = %q, want %q", in, got, want)
		}
	}
}

// A thousands separator is part of the number, not a prefix left in front of
// it, and a long number without one is still read whole.
func TestExpandPercentagesReadsThousandsSeparators(t *testing.T) {
	assertTransform(t, "ExpandPercentages", ExpandPercentages, map[string]string{
		"1,500%":     "one thousand, five hundred percent",
		"12,500%":    "twelve thousand, five hundred percent",
		"1,000,000%": "one million percent",
		"1,234.5%":   "one thousand, two hundred and thirty-four point five percent",
		"1000%":      "one thousand percent",
	})
}

func TestExpandUnits(t *testing.T) {
	cases := map[string]string{
		"Run 5km at 100kph": "Run 5 kilometers at 100 kilometers per hour",
		"Weighs 5kg":        "Weighs 5 kilograms",
		"5m tall":           "5 meters tall", // ambiguous unit, no space → expand
		"1 m people":        "1 m people",    // ambiguous unit with space → keep
		"1 m people voted":  "1 m people voted",
		"60mph speed":       "60 miles per hour speed",
		"2GB file":          "2 gigabytes file",
		"Hello world":       "Hello world",
	}
	for in, want := range cases {
		if got := ExpandUnits(in); got != want {
			t.Errorf("ExpandUnits(%q) = %q, want %q", in, got, want)
		}
	}
}

// A unit that is also an English word is not expanded when it reads as the
// word: "in" as a preposition or in a ratio, a "g" standing apart from the
// number, and an uppercase "M" or "G" (a million, the network).
func TestExpandUnitsLeavesAmbiguousUnitsAlone(t *testing.T) {
	assertTransform(t, "ExpandUnits", ExpandUnits, map[string]string{
		"1 in 5 people":                 "1 in 5 people",
		"9 in 10 dentists recommend it": "9 in 10 dentists recommend it",
		"1 g of something":              "1 g of something",
		"We raised $5M":                 "We raised $5M",
		"10M downloads":                 "10M downloads",
		"Works on 5G networks":          "Works on 5G networks",
	})
}

// A lowercase ambiguous unit right after the number is expanded, and so is the
// capital "L" for liters.
func TestExpandUnitsExpandsLowercaseUnitsAndTheLiterSymbol(t *testing.T) {
	assertTransform(t, "ExpandUnits", ExpandUnits, map[string]string{
		"Run 5m":            "Run 5 meters",
		"Add 200g of flour": "Add 200 grams of flour",
		"Buy 2L of milk":    "Buy 2 liters of milk",
	})
}

// A unit that cannot be mistaken for a word is expanded with a space before it.
func TestExpandUnitsAllowsASpaceBeforeAnUnambiguousUnit(t *testing.T) {
	assertTransform(t, "ExpandUnits", ExpandUnits, map[string]string{
		"5 km away":          "5 kilometers away",
		"60 mph speed limit": "60 miles per hour speed limit",
		"3 GHz processor":    "3 gigahertz processor",
	})
}

func TestExpandNumbers(t *testing.T) {
	expand := ExpandNumbers(2025)
	cases := map[string]string{
		"Room 42":       "Room forty-two",
		"opens in 2026": "opens in 2 0 2 6", // above cutoff → digit-by-digit
		"pi is 3.14":    "pi is three point one four",
	}
	for in, want := range cases {
		if got := expand(in); got != want {
			t.Errorf("ExpandNumbers(%q) = %q, want %q", in, got, want)
		}
	}
	// No cutoff spells everything.
	if got := ExpandNumbers(0)("year 2026"); got != "year two thousand and twenty-six" {
		t.Errorf("ExpandNumbers(0) = %q", got)
	}
	assertTransform(t, "ExpandNumbers(0)", ExpandNumbers(0), map[string]string{
		"9999": "nine thousand, nine hundred and ninety-nine",
		"3.5":  "three point five",
	})
}

// A number equal to the cutoff is still spelled as a quantity.
func TestExpandNumbersCutoffIsInclusive(t *testing.T) {
	assertTransform(t, "ExpandNumbers", ExpandNumbers(2025), map[string]string{
		"2025": "two thousand and twenty-five",
	})
}

func TestReplaceText(t *testing.T) {
	tr, err := ReplaceText([][2]string{{`\bDr\.`, "Doctor"}, {`\bvs\b`, "versus"}})
	if err != nil {
		t.Fatalf("ReplaceText: %v", err)
	}
	if got := tr("Dr. Smith vs Jones"); got != "Doctor Smith versus Jones" {
		t.Errorf("ReplaceText = %q", got)
	}
	if _, err := ReplaceText([][2]string{{`(`, "x"}}); err == nil {
		t.Error("ReplaceText should reject an invalid pattern")
	}
}

func replaceText(t *testing.T, rules [][2]string) Transform {
	t.Helper()
	tr, err := ReplaceText(rules)
	if err != nil {
		t.Fatalf("ReplaceText: %v", err)
	}
	return tr
}

// Text that no rule matches comes back unchanged.
func TestReplaceTextLeavesUnmatchedTextUnchanged(t *testing.T) {
	assertTransform(t, "ReplaceText", replaceText(t, [][2]string{{"xyz", "abc"}}), map[string]string{
		"Hello world": "Hello world",
	})
}

// A replacement may turn one word into several.
func TestReplaceTextCanSplitAWord(t *testing.T) {
	tr := replaceText(t, [][2]string{{`\bBODYPUMP\b`, "body pump"}})
	assertTransform(t, "ReplaceText", tr, map[string]string{
		"Try BODYPUMP on Monday morning.": "Try body pump on Monday morning.",
	})
}

// A replacement may change only the case of a word.
func TestReplaceTextCanChangeOnlyTheCase(t *testing.T) {
	tr := replaceText(t, [][2]string{{`\bSQL\b`, "sql"}})
	assertTransform(t, "ReplaceText", tr, map[string]string{
		"Contact SQL support today.": "Contact sql support today.",
	})
}

// A replacement may write a word as an inline IPA tag, or wrap it in an SSML
// phoneme tag instead of respelling it.
func TestReplaceTextCanInsertPronunciationTags(t *testing.T) {
	ipa := replaceText(t, [][2]string{{`\bleisure\b`, "<<l|ɛ|ʒ|ə|r>>"}})
	assertTransformContains(t, "ReplaceText", ipa, "The leisure center opens at six.", "<<l|ɛ|ʒ|ə|r>>")

	tag := `<phoneme alphabet="ipa" ph="ʃəˈvɔːn">Siobhan</phoneme>`
	ssml := replaceText(t, [][2]string{{`(?i)\bSiobhan\b`, tag}})
	assertTransformContains(t, "ReplaceText", ssml, "My name is Siobhan.", tag)
}

func TestVoiceFormatter(t *testing.T) {
	f, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	got := format(t, f, "**Total:** $42.50 for the API, 50% done")
	want := "Total: forty-two dollars and fifty cents for the A P I, fifty percent done"
	if got != want {
		t.Errorf("VoiceFormatter.Filter = %q, want %q", got, want)
	}
	if got, want := format(t, f, "**Hello** world"), "Hello world"; got != want {
		t.Errorf("VoiceFormatter(%q) = %q, want %q", "**Hello** world", got, want)
	}
	// An all-caps address is spoken as an address before the acronym
	// spacing could break it up.
	got = format(t, f, "Contact USER@EXAMPLE.COM today")
	if strings.Contains(got, "@") || !strings.Contains(got, " at ") {
		t.Errorf("VoiceFormatter(all-caps email) = %q, want the address spoken", got)
	}
}

// newFormatter builds a VoiceFormatter from opts, failing the test on an error.
func newFormatter(t *testing.T, opts FormatterOptions) *VoiceFormatter {
	t.Helper()
	f, err := NewVoiceFormatter(opts)
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	return f
}

// A transform switched off is not applied.
func TestVoiceFormatterSkipsADisabledTransform(t *testing.T) {
	opts := DefaultFormatterOptions()
	opts.StripMarkdown = false
	if got := format(t, newFormatter(t, opts), "**Hello**"); !strings.Contains(got, "**") {
		t.Errorf("VoiceFormatter without StripMarkdown = %q, want the markers kept", got)
	}
}

// With every transform switched off, text comes back unchanged.
func TestVoiceFormatterWithNothingEnabledLeavesTextAlone(t *testing.T) {
	if got := format(t, newFormatter(t, FormatterOptions{}), "Hello world"); got != "Hello world" {
		t.Errorf("VoiceFormatter with nothing enabled = %q, want it unchanged", got)
	}
}

// Custom replacements are applied, even with every other transform off.
func TestVoiceFormatterAppliesCustomReplacements(t *testing.T) {
	f := newFormatter(t, FormatterOptions{CustomReplacements: [][2]string{{`\bDr\.`, "Doctor"}}})
	if got := format(t, f, "Dr. Smith"); got != "Doctor Smith" {
		t.Errorf("VoiceFormatter = %q, want %q", got, "Doctor Smith")
	}
}

// Number expansion, off by default, spells numbers out when switched on.
func TestVoiceFormatterExpandsNumbersWhenEnabled(t *testing.T) {
	opts := DefaultFormatterOptions()
	opts.ExpandNumbers = true
	opts.NumberDigitCutoff = 2025
	if got := format(t, newFormatter(t, opts), "Room 42"); got != "Room forty-two" {
		t.Errorf("VoiceFormatter = %q, want %q", got, "Room forty-two")
	}
}

// Some transforms split a word into separate tokens (acronym letter-spacing
// turns "API" into "A P I") and others recase words (a unit abbreviation is
// matched case-insensitively and rewritten as a lower-case word). If the
// transforms ran in the wrong order, a later one could re-match or re-case an
// earlier one's output and mangle it. The deliberate ordering in
// NewVoiceFormatter (units before acronyms, email before phone and acronyms)
// keeps each transform from corrupting another's result. These cases lock that
// in; they fail if the pipeline is reordered.
// format runs a formatter over text the way the TTS base does, on a unit of no
// particular aggregation type.
func format(t *testing.T, f *VoiceFormatter, text string) string {
	t.Helper()
	got, err := f.Transform(t.Context(), text, frames.AnyTextType)
	if err != nil {
		t.Fatalf("Transform(%q): %v", text, err)
	}
	return got
}

func TestVoiceFormatterOrderingPreventsCorruption(t *testing.T) {
	// Units expand before acronyms letter-space them, so "MB" becomes
	// "megabytes" rather than the unreadable "M B" (which the unit transform
	// could then no longer expand).
	f, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	if got, want := format(t, f, "Copy 100 MB to the SSD"), "Copy 100 megabytes to the S S D"; got != want {
		t.Errorf("units-before-acronyms: Filter = %q, want %q", got, want)
	}

	// Email is spoken before phone and acronyms run, so its structure survives:
	// the all-caps local part is letter-spaced only after "@" has been turned
	// into "at". If acronyms ran first, letter-spacing "SALES" would split the
	// address and the email pattern would match only a truncated remainder.
	if got, want := format(t, f, "Email SALES@example.com please"),
		"Email S A L E S at example dot com please"; got != want {
		t.Errorf("email-before-acronyms: Filter = %q, want %q", got, want)
	}

	// A split acronym and number expansion coexist without one corrupting the
	// other: the letter-spaced "A P I" carries no digits for the number
	// transform to touch, and "42" carries no letters for the acronym transform.
	opts := DefaultFormatterOptions()
	opts.ExpandNumbers = true
	opts.NumberDigitCutoff = 2025
	fn, err := NewVoiceFormatter(opts)
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	if got, want := format(t, fn, "The API returns 42"), "The A P I returns forty-two"; got != want {
		t.Errorf("acronym-split with numbers: Filter = %q, want %q", got, want)
	}

	// Direct demonstration of the corruption the ordering avoids: running the
	// acronym split before the unit expansion mangles "5 GB" into "5 G B", which
	// the unit transform can no longer recognize. The formatter never does this.
	if mangled := ExpandUnits(NormalizeAcronyms("Send 5 GB now")); mangled != "Send 5 G B now" {
		t.Errorf("expected wrong-order to mangle, got %q", mangled)
	}
	if ok := NormalizeAcronyms(ExpandUnits("Send 5 GB now")); ok != "Send 5 gigabytes now" {
		t.Errorf("expected right-order to expand, got %q", ok)
	}
}

// Ported from upstream. A quantity of exactly one takes the singular form of
// the unit, so a bot does not say "one kilometers".
func TestExpandUnitsSingular(t *testing.T) {
	cases := map[string]string{
		"Only 1km left":  "Only 1 kilometer left",
		"Only 1mi left":  "Only 1 mile left",
		"Only 1ft left":  "Only 1 foot left",
		"Only 1in left":  "Only 1 inch left",
		"Only 1lb left":  "Only 1 pound left",
		"Only 1gb left":  "Only 1 gigabyte left",
		"Only 1mph left": "Only 1 mile per hour left",
		"1 km away":      "1 kilometer away",
		// A decimal such as "1.0" reads as plural in speech, unlike a bare "1".
		"1.0km away": "1.0 kilometers away",
		// Plurals are untouched.
		"5km away":       "5 kilometers away",
		"0.5 mi away":    "0.5 miles away",
		"21 lb of flour": "21 pounds of flour",
		// Units whose singular and plural forms match expand identically.
		"1hz tone":        "1 hertz tone",
		"3 GHz processor": "3 gigahertz processor",
	}
	for in, want := range cases {
		if got := ExpandUnits(in); got != want {
			t.Errorf("ExpandUnits(%q) = %q, want %q", in, got, want)
		}
	}
}

// Ported from upstream. Every written digit is spoken, trailing zeros included,
// so a decimal keeps agreeing with the plural unit ExpandUnits picked for it.
func TestExpandNumbersSpeaksEveryFractionalDigit(t *testing.T) {
	expand := ExpandNumbers(2025)
	cases := map[string]string{
		"1.0":    "one point zero",
		"1.00":   "one point zero zero",
		"2.0":    "two point zero",
		"0.50":   "zero point five zero",
		"1.10":   "one point one zero",
		"10.0":   "ten point zero",
		"3.5":    "three point five",
		"1.05":   "one point zero five",
		"1.25":   "one point two five",
		"12.345": "twelve point three four five",
		"5.0001": "five point zero zero zero one",
	}
	for in, want := range cases {
		if got := expand(in); got != want {
			t.Errorf("ExpandNumbers(%q) = %q, want %q", in, got, want)
		}
	}
	// The above-cutoff branch reads the whole part digit by digit and the
	// fraction the same way, rather than dropping it.
	if got := expand("3000.5 units"); got != "3 0 0 0 point 5 units" {
		t.Errorf("ExpandNumbers(%q) = %q, want %q", "3000.5 units", got, "3 0 0 0 point 5 units")
	}
	if got := expand("2500.75"); got != "2 5 0 0 point 7 5" {
		t.Errorf("ExpandNumbers(%q) = %q, want %q", "2500.75", got, "2 5 0 0 point 7 5")
	}
}

// Ported from upstream. ExpandUnits keeps the plural for a decimal such as
// "1.0" because it reads as "one point zero" in speech, and ExpandNumbers runs
// after it. Both have to spell the decimal out the same way, or the composed
// output is ungrammatical: "one kilometers".
func TestUnitsAndNumbersComposition(t *testing.T) {
	alone, err := NewVoiceFormatter(FormatterOptions{
		ExpandUnits:       true,
		ExpandNumbers:     true,
		NumberDigitCutoff: 2025,
	})
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	// And within the full default formatter, every number spelled out.
	opts := DefaultFormatterOptions()
	opts.ExpandNumbers = true
	full, err := NewVoiceFormatter(opts)
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	cases := map[string]string{
		"1.0km left":     "one point zero kilometers left",
		"1.00km left":    "one point zero zero kilometers left",
		"1.0kg of flour": "one point zero kilograms of flour",
		"1.0 mi away":    "one point zero miles away",
		"1km left":       "one kilometer left",
		"2.0km left":     "two point zero kilometers left",
		"1.5km left":     "one point five kilometers left",
		"1.0hz tone":     "one point zero hertz tone",
	}
	for _, f := range []*VoiceFormatter{alone, full} {
		for in, want := range cases {
			if got := format(t, f, in); got != want {
				t.Errorf("VoiceFormatter(%q) = %q, want %q", in, got, want)
			}
		}
	}
}
