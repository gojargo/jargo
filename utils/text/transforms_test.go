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

func TestCollapseRepeatedPunctuation(t *testing.T) {
	collapse := CollapseRepeatedPunctuation(RepeatedPunctuationOptions{})
	cases := map[string]string{
		"C'est super !!!!!":  "C'est super !",
		"Vraiment ????":      "Vraiment ?",
		"Wait!! Really??":    "Wait! Really?",
		"One! Two? Three.":   "One! Two? Three.",
		"Hold on...":         "Hold on...",
		"Mixed ?!?! run":     "Mixed ?!?! run",
		"Ends both ?!!!":     "Ends both ?!",
		"nothing to do here": "nothing to do here",
	}
	for in, want := range cases {
		if got := collapse(in); got != want {
			t.Errorf("CollapseRepeatedPunctuation(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestCollapseRepeatedPunctuationOptions(t *testing.T) {
	// Keep more than one mark, and only act on longer runs.
	lenient := CollapseRepeatedPunctuation(RepeatedPunctuationOptions{Keep: 2, MinRun: 4})
	if got, want := lenient("Really!!! Truly!!!!!"), "Really!!! Truly!!"; got != want {
		t.Errorf("lenient = %q, want %q", got, want)
	}

	// A custom mark set brings the full stop in, collapsing ellipses too.
	withStops := CollapseRepeatedPunctuation(RepeatedPunctuationOptions{Marks: "!?."})
	if got, want := withStops("Well... maybe!!"), "Well. maybe!"; got != want {
		t.Errorf("withStops = %q, want %q", got, want)
	}

	// Keep cannot swallow the mark entirely, and MinRun cannot sit below it.
	degenerate := CollapseRepeatedPunctuation(RepeatedPunctuationOptions{Keep: -3, MinRun: -1})
	if got, want := degenerate("what???"), "what?"; got != want {
		t.Errorf("degenerate = %q, want %q", got, want)
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
// NewVoiceFormatter — units before acronyms, email before phone and acronyms —
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
