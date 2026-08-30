package text

import "testing"

func TestStripMarkdown(t *testing.T) {
	cases := map[string]string{
		"**Hello** and _world_":    "Hello and world",
		"***bold italic***":        "bold italic",
		"`code` span":              "code span",
		"# Heading":                "Heading",
		"> quoted":                 "quoted",
		"plain text":               "plain text",
		"see [the docs](x) please": "see [the docs](x) please",
	}
	for in, want := range cases {
		if got := StripMarkdown(in); got != want {
			t.Errorf("StripMarkdown(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestEmailToSpeech(t *testing.T) {
	got := EmailToSpeech("Contact user_1@example.com today")
	want := "Contact user underscore 1 at example dot com today"
	if got != want {
		t.Errorf("EmailToSpeech = %q, want %q", got, want)
	}
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
}

func TestNormalizeDates(t *testing.T) {
	cases := map[string]string{
		"Meeting on 2023-05-10": "Meeting on May 10th, two thousand and twenty-three",
		"Due 05/10/2023":        "Due May 10th, two thousand and twenty-three",
		"Not 2023-13-40":        "Not 2023-13-40", // invalid date left as-is
	}
	for in, want := range cases {
		if got := NormalizeDates(in); got != want {
			t.Errorf("NormalizeDates(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestExpandCurrency(t *testing.T) {
	cases := map[string]string{
		"Your balance is $42.50": "Your balance is forty-two dollars and fifty cents",
		"It costs $1":            "It costs one dollar",
		"Paid £3.01":             "Paid three pounds and one penny",
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

func TestExpandPercentages(t *testing.T) {
	cases := map[string]string{
		"50% off":    "fifty percent off",
		"12.5% rate": "twelve point five percent rate",
	}
	for in, want := range cases {
		if got := ExpandPercentages(in); got != want {
			t.Errorf("ExpandPercentages(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestExpandUnits(t *testing.T) {
	cases := map[string]string{
		"Run 5km at 100kph": "Run 5 kilometers at 100 kilometers per hour",
		"Weighs 5kg":        "Weighs 5 kilograms",
		"5m tall":           "5 meters tall", // ambiguous unit, no space → expand
		"1 m people":        "1 m people",    // ambiguous unit with space → keep
	}
	for in, want := range cases {
		if got := ExpandUnits(in); got != want {
			t.Errorf("ExpandUnits(%q) = %q, want %q", in, got, want)
		}
	}
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

func TestVoiceFormatter(t *testing.T) {
	f, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	got := f.Filter("**Total:** $42.50 for the API, 50% done")
	want := "Total: forty-two dollars and fifty cents for the A P I, fifty percent done"
	if got != want {
		t.Errorf("VoiceFormatter.Filter = %q, want %q", got, want)
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
func TestVoiceFormatterOrderingPreventsCorruption(t *testing.T) {
	// Units expand before acronyms letter-space them, so "MB" becomes
	// "megabytes" rather than the unreadable "M B" (which the unit transform
	// could then no longer expand).
	f, err := NewVoiceFormatter(DefaultFormatterOptions())
	if err != nil {
		t.Fatalf("NewVoiceFormatter: %v", err)
	}
	if got, want := f.Filter("Copy 100 MB to the SSD"), "Copy 100 megabytes to the S S D"; got != want {
		t.Errorf("units-before-acronyms: Filter = %q, want %q", got, want)
	}

	// Email is spoken before phone and acronyms run, so its structure survives:
	// the all-caps local part is letter-spaced only after "@" has been turned
	// into "at". If acronyms ran first, letter-spacing "SALES" would split the
	// address and the email pattern would match only a truncated remainder.
	if got, want := f.Filter("Email SALES@example.com please"),
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
	if got, want := fn.Filter("The API returns 42"), "The A P I returns forty-two"; got != want {
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
	f, err := NewVoiceFormatter(FormatterOptions{
		ExpandUnits:       true,
		ExpandNumbers:     true,
		NumberDigitCutoff: 2025,
	})
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
	for in, want := range cases {
		if got := f.Filter(in); got != want {
			t.Errorf("VoiceFormatter(%q) = %q, want %q", in, got, want)
		}
	}
}
