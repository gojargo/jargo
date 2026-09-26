package sentencex

import (
	"embed"
	"regexp"
	"strings"
	"sync"
)

// data holds the bundled word lists: abbreviations, sentence starters,
// fronting words and trailing markers.
//
//go:embed data
var data embed.FS //nolint:gochecknoglobals // embedded word lists

// dataFile returns the bundled word list at name, relative to data/.
func dataFile(name string) string {
	b, err := data.ReadFile("data/" + name)
	if err != nil {
		panic("sentencex: missing bundled word list " + name) //nolint:forbidigo // the list is embedded at build time
	}
	return string(b)
}

// parseWordList parses one or more bundled word lists into a set. Lines are
// trimmed, and blank lines and // comments are dropped.
func parseWordList(sources ...string) map[string]struct{} {
	set := make(map[string]struct{})
	for _, source := range sources {
		for _, line := range lines(source) {
			line = strings.TrimSpace(line)
			if line == "" || strings.HasPrefix(line, "//") {
				continue
			}
			set[line] = struct{}{}
		}
	}
	return set
}

// parseLowercaseWordList is parseWordList with every word lowercased.
func parseLowercaseWordList(sources ...string) map[string]struct{} {
	set := make(map[string]struct{})
	for word := range parseWordList(sources...) {
		set[toLowercase(word)] = struct{}{}
	}
	return set
}

// parseMarkersList parses a bundled trailing-marker list.
func parseMarkersList(source string) []markerDef {
	var out []markerDef
	for _, raw := range lines(source) {
		if def, ok := parseMarkerLine(raw); ok {
			out = append(out, def)
		}
	}
	return out
}

// parseMarkerLine parses one line of a trailing-marker list:
// suffix | case | digit_breaks [| uppercase_breaks [| digit-only]].
// It returns false for a blank or comment line. A malformed line is a mistake
// in the bundled list and panics.
func parseMarkerLine(raw string) (markerDef, bool) {
	line := strings.TrimSpace(raw)
	if line == "" || strings.HasPrefix(line, "//") {
		return markerDef{}, false
	}

	parts := strings.Split(line, "|")
	for i := range parts {
		parts[i] = strings.TrimSpace(parts[i])
	}
	field := func(i int) (string, bool) {
		if i < len(parts) {
			return parts[i], true
		}
		return "", false
	}

	suffix, _ := field(0)
	caseField, ok := field(1)
	if !ok {
		markerLinePanic("missing `case` field", raw)
	}
	digitField, ok := field(2)
	if !ok {
		markerLinePanic("missing `digit_breaks` field", raw)
	}
	uppercaseField, hasUppercase := field(3)
	flagField, hasFlag := field(4)
	if len(parts) > 5 {
		markerLinePanic("has trailing fields after flag", raw)
	}

	var ignoreCase bool
	switch caseField {
	case "i":
		ignoreCase = true
	case "s":
		ignoreCase = false
	default:
		markerLinePanic("has unknown case `"+caseField+"`", raw)
	}

	digitOnly := false
	if hasFlag {
		if flagField != "digit-only" {
			markerLinePanic("has unknown flag `"+flagField+"`", raw)
		}
		digitOnly = true
	}

	uppercaseBreaks := true
	if hasUppercase {
		uppercaseBreaks = parseBreakFlag(uppercaseField, "uppercase_breaks", raw)
	}

	return markerDef{
		matcher: suffixMatcher{suffix: suffix, ignoreCase: ignoreCase, digitOnly: digitOnly},
		policy: markerPolicy{
			digitBreaks:     parseBreakFlag(digitField, "digit_breaks", raw),
			uppercaseBreaks: uppercaseBreaks,
		},
	}, true
}

func parseBreakFlag(field, axis, raw string) bool {
	switch field {
	case "break":
		return true
	case "cont":
		return false
	default:
		markerLinePanic("has unknown "+axis+" `"+field+"`", raw)
		return false
	}
}

func markerLinePanic(problem, raw string) {
	panic("sentencex: marker line " + problem + ": " + raw) //nolint:forbidigo // the list is embedded at build time
}

// sentenceBreakRegex builds a language's terminator regular expression: a run
// of the global sentence terminators, filtered by keep, and the extra
// characters.
func sentenceBreakRegex(keep func(rune) bool, extra string) *regexp.Regexp {
	var b strings.Builder
	for _, r := range globalSentenceTerminators {
		if keep(r) {
			b.WriteRune(r)
		}
	}
	return regexp.MustCompile("[" + b.String() + extra + "]+")
}

func keepAll(rune) bool { return true }

// withRomanNumerals returns abbreviations extended with the Roman numerals in
// lowercase and uppercase.
func withRomanNumerals(abbreviations map[string]struct{}) map[string]struct{} {
	for _, s := range romanNumerals {
		abbreviations[s] = struct{}{}
		abbreviations[strings.ToUpper(s)] = struct{}{}
	}
	return abbreviations
}

// The abbreviation sets that more than one language uses.
//
//nolint:gochecknoglobals // built once
var (
	englishAbbreviations = sync.OnceValue(func() map[string]struct{} {
		return parseLowercaseWordList(dataFile("abbrev/en.txt"))
	})
	hindiAbbreviations = sync.OnceValue(func() map[string]struct{} {
		return parseLowercaseWordList(dataFile("abbrev/hi.txt"), dataFile("abbrev/en.txt"))
	})
)

// withEnglish returns the abbreviations of a language whose list extends the
// English one.
func withEnglish(code string) map[string]struct{} {
	return parseLowercaseWordList(dataFile("abbrev/"+code+".txt"), dataFile("abbrev/en.txt"))
}

// ownAbbreviations returns the abbreviations of a language's own list.
func ownAbbreviations(code string) map[string]struct{} {
	return parseLowercaseWordList(dataFile("abbrev/" + code + ".txt"))
}

// languages maps each language code with its own rules to its lazily built
// rules.
//
//nolint:gochecknoglobals // fixed lookup table
var languages = map[string]func() *language{
	"am": sync.OnceValue(newAmharic),
	"ar": sync.OnceValue(newArabic),
	"bg": sync.OnceValue(newBulgarian),
	"bn": sync.OnceValue(newBengali),
	"ca": sync.OnceValue(newCatalan),
	"da": sync.OnceValue(newDanish),
	"de": sync.OnceValue(newGerman),
	"el": sync.OnceValue(newGreek),
	"en": sync.OnceValue(newEnglish),
	"es": sync.OnceValue(newSpanish),
	"fi": sync.OnceValue(newFinnish),
	"fr": sync.OnceValue(newFrench),
	"gu": sync.OnceValue(newGujarati),
	"hi": sync.OnceValue(newHindi),
	"hy": sync.OnceValue(newArmenian),
	"it": sync.OnceValue(newItalian),
	"ja": sync.OnceValue(newJapanese),
	"kk": sync.OnceValue(newKazakh),
	"kn": sync.OnceValue(newKannada),
	"ml": sync.OnceValue(newMalayalam),
	"mr": sync.OnceValue(newMarathi),
	"my": sync.OnceValue(newBurmese),
	"nl": sync.OnceValue(newDutch),
	"pa": sync.OnceValue(newPunjabi),
	"pl": sync.OnceValue(newPolish),
	"pt": sync.OnceValue(newPortuguese),
	"ru": sync.OnceValue(newRussian),
	"sk": sync.OnceValue(newSlovak),
	"ta": sync.OnceValue(newTamil),
	"te": sync.OnceValue(newTelugu),
	"uk": sync.OnceValue(newUkrainian),
}

// Amharic.
func newAmharic() *language {
	return &language{abbreviations: ownAbbreviations("am")}
}

// Arabic.
func newArabic() *language {
	return &language{abbreviations: withEnglish("ar")}
}

// Bulgarian.
func newBulgarian() *language {
	return &language{abbreviations: ownAbbreviations("bg")}
}

// Bengali.
func newBengali() *language {
	return &language{abbreviations: withEnglish("bn")}
}

// Catalan, which uses the Spanish abbreviations.
func newCatalan() *language {
	return &language{
		abbreviations:      ownAbbreviations("es"),
		continueInNextWord: continueAfterNonwordRegex.MatchString,
	}
}

// Danish.
func newDanish() *language {
	return &language{
		abbreviations:      ownAbbreviations("da"),
		continueInNextWord: continueAfterNonwordRegex.MatchString,
	}
}

// German.
func newGerman() *language {
	months := []string{
		"Januar", "Februar", "März", "April", "Mai", "Juni",
		"Juli", "August", "September", "Oktober", "November", "Dezember",
	}
	return &language{
		abbreviations: withEnglish("de"),
		continueInNextWord: func(text string) bool {
			return continuesAfterBoundary(text, months)
		},
	}
}

// Greek, where ; is the question mark.
func newGreek() *language {
	return &language{
		abbreviations:      withEnglish("el"),
		sentenceBreakRegex: sentenceBreakRegex(keepAll, ";"),
	}
}

// English.
func newEnglish() *language {
	// English I is the one capital that case cannot tell from a sentence
	// start, so after an ellipsis run "... I'm" reads as continuation.
	ellipsisI := regexp.MustCompile(`^[` + whitespaceClass + `]+I(?:[` + whitespaceClass + `'\x{2019}]|$)`)

	return &language{
		abbreviations:    englishAbbreviations(),
		sentenceStarters: parseWordList(dataFile("starters/en.txt")),
		frontingWords:    parseLowercaseWordList(dataFile("fronting/en.txt")),
		trailingMarkers:  buildMarkerTable(parseMarkersList(dataFile("trailing_markers/en.txt"))),
		ellipsisContinuation: func(textAfterRun string) bool {
			return ellipsisContinueRegex.MatchString(textAfterRun) || ellipsisI.MatchString(textAfterRun)
		},
	}
}

// Spanish.
func newSpanish() *language {
	return &language{abbreviations: ownAbbreviations("es")}
}

// Finnish.
func newFinnish() *language {
	months := []string{
		"tammikuu", "helmikuu", "maaliskuu", "huhtikuu", "toukokuu", "kesäkuu",
		"heinäkuu", "elokuu", "syyskuu", "lokakuu", "marraskuu", "joulukuu",
	}
	return &language{
		abbreviations: ownAbbreviations("fi"),
		continueInNextWord: func(text string) bool {
			return continuesAfterBoundary(text, months)
		},
	}
}

// French.
func newFrench() *language {
	return &language{abbreviations: ownAbbreviations("fr")}
}

// Gujarati.
func newGujarati() *language {
	return &language{abbreviations: withEnglish("gu")}
}

// Hindi.
func newHindi() *language {
	return &language{abbreviations: hindiAbbreviations()}
}

// Armenian, which uses the English abbreviations and ends sentences with its
// own marks rather than the period.
func newArmenian() *language {
	return &language{
		abbreviations:      englishAbbreviations(),
		sentenceBreakRegex: sentenceBreakRegex(func(r rune) bool { return r != '.' }, ";։՜:"),
	}
}

// Italian, which splits an elided article (l') off the last word.
func newItalian() *language {
	continueRegex := regexp.MustCompile(`^[0-9a-z]`)

	return &language{
		abbreviations: ownAbbreviations("it"),
		lastWord: func(text string) string {
			lastWord := lastPieceAfter(text, func(c rune) bool { return isWhitespace(c) || c == '.' })
			parts := strings.Split(lastWord, "l'")
			return parts[len(parts)-1]
		},
		continueInNextWord: continueRegex.MatchString,
	}
}

// Japanese, with no abbreviations.
func newJapanese() *language {
	return &language{abbreviations: map[string]struct{}{}}
}

// Kazakh, which extends the continuation rule with the Cyrillic lowercase
// range а-я.
func newKazakh() *language {
	continueRegex := regexp.MustCompile(`^[^` + perlWordClass + `]*[0-9a-zа-я]`)

	return &language{
		abbreviations: ownAbbreviations("kk"),
		lastWord: func(text string) string {
			words := strings.FieldsFunc(text, func(c rune) bool { return isWhitespace(c) || c == '.' })
			if len(words) == 0 {
				return ""
			}
			return words[len(words)-1]
		},
		continueInNextWord: continueRegex.MatchString,
	}
}

// Kannada.
func newKannada() *language {
	return &language{abbreviations: withEnglish("kn")}
}

// Malayalam.
func newMalayalam() *language {
	return &language{abbreviations: withEnglish("ml")}
}

// Marathi, which uses the Hindi abbreviations.
func newMarathi() *language {
	return &language{abbreviations: hindiAbbreviations()}
}

// Burmese, which uses the English abbreviations and also ends sentences with
// ၏.
func newBurmese() *language {
	return &language{
		abbreviations:      englishAbbreviations(),
		sentenceBreakRegex: sentenceBreakRegex(keepAll, "၏"),
	}
}

// Dutch.
func newDutch() *language {
	return &language{abbreviations: ownAbbreviations("nl")}
}

// Punjabi.
func newPunjabi() *language {
	return &language{abbreviations: withEnglish("pa")}
}

// Polish.
func newPolish() *language {
	return &language{abbreviations: ownAbbreviations("pl")}
}

// Portuguese, whose abbreviations include the Roman numerals.
func newPortuguese() *language {
	return &language{abbreviations: withRomanNumerals(ownAbbreviations("pt"))}
}

// Russian, which continues a sentence before a Cyrillic lowercase letter.
func newRussian() *language {
	continueRegex := regexp.MustCompile(`^[0-9a-zа-я]`)

	return &language{
		abbreviations:      ownAbbreviations("ru"),
		continueInNextWord: continueRegex.MatchString,
	}
}

// Slovak, whose abbreviations include the Roman numerals.
func newSlovak() *language {
	months := []string{
		"Január", "Február", "Marec", "Apríl", "Máj", "Jún",
		"Júl", "August", "September", "Október", "November", "December",
		"Januára", "Februára", "Marca", "Apríla", "Mája", "Júna",
		"Júla", "Augusta", "Septembra", "Októbra", "Novembra", "Decembra",
	}
	return &language{
		abbreviations: withRomanNumerals(ownAbbreviations("sk")),
		continueInNextWord: func(text string) bool {
			return continuesAfterBoundary(text, months)
		},
	}
}

// Tamil, whose abbreviations are its list and the English one as written (not
// lowercased), plus every vowel, consonant, and consonant with a vowel sign.
func newTamil() *language {
	// The vowel signs are kept exactly as the upstream table lists them,
	// including the entries from the Bengali block.
	vowelSigns := []string{
		"\u0BBE", "\u0BBF", "\u0BC0", "\u09C1", "\u09C2", "\u09C7",
		"\u09C7", "\u09C8", "\u0993", "\u09CB", "\u09CC",
	}
	vowels := []string{"அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"}
	consonants := []string{
		"க", "ங", "ச", "ஞ", "ட", "ண", "த", "ந", "ப", "ம", "ய", "ர", "ல", "வ", "ழ", "ள", "ற", "ன",
	}

	abbreviations := parseWordList(dataFile("abbrev/ta.txt"), dataFile("abbrev/en.txt"))
	for _, v := range vowels {
		abbreviations[v] = struct{}{}
	}
	for _, c := range consonants {
		abbreviations[c] = struct{}{}
		for _, s := range vowelSigns {
			abbreviations[c+s] = struct{}{}
		}
	}

	return &language{abbreviations: abbreviations}
}

// Telugu.
func newTelugu() *language {
	return &language{abbreviations: withEnglish("te")}
}

// Ukrainian, which continues a sentence before a Ukrainian lowercase letter.
func newUkrainian() *language {
	continueRegex := regexp.MustCompile(`^[0-9a-zа-яіїєґ]`)

	return &language{
		abbreviations:      ownAbbreviations("uk"),
		continueInNextWord: continueRegex.MatchString,
	}
}
