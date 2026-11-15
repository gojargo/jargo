package kokoro

import (
	"fmt"
	"log/slog"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// espeakStress is the stress marks espeak writes, removed when phonemizing
// without stress.
const espeakStress = "ˈˌ'-"

// espeakFlagsRe finds the language switch flags in espeak output, such as the
// "(fr)" and "(en)" of "something (fr)quelque chose(en) another thing".
var espeakFlagsRe = regexp.MustCompile(`\(.+?\)`)

// espeakUnderscoresRe finds repeated phone separators in espeak output.
var espeakUnderscoresRe = regexp.MustCompile(`_+`)

// phonemizerOptions is how the espeak backend phonemizes.
type phonemizerOptions struct {
	// preservePunctuation keeps the punctuation in the phonemes rather than
	// removing it.
	preservePunctuation bool
	// withStress keeps the stress marks on the phonemes.
	withStress bool
}

// espeakBackend phonemizes text in one language with espeak. Punctuation is
// hidden from espeak and put back in the phonemes, the words around quotes
// and brackets are phonemized in context, and a switch to another language
// (espeak reads some words as English, say) keeps its flags.
type espeakBackend struct {
	lib         *espeakLibrary
	language    string
	opts        phonemizerOptions
	punctuation *punctuation
}

// newEspeakBackend is a backend for language, which espeak must have a voice
// for. The language is compared ignoring case.
func newEspeakBackend(lib *espeakLibrary, language string, opts phonemizerOptions) (*espeakBackend, error) {
	code := lib.supportedLanguage(language)
	if code == "" {
		return nil, fmt.Errorf("%w: %q", errUnsupportedLanguage, language)
	}
	return &espeakBackend{
		lib:         lib,
		language:    code,
		opts:        opts,
		punctuation: newPunctuation(defaultPunctuationMarks),
	}, nil
}

// phonemize phonemizes text, which may hold several lines. Empty lines are
// dropped, and the phonemized lines are joined with a newline.
func (b *espeakBackend) phonemize(text string, sep separator, strip bool) (string, error) {
	var lines []string
	for line := range strings.SplitSeq(strings.Trim(text, "\r\n"), "\n") {
		line = strings.Trim(strings.TrimRight(line, "\r"), "\r\n")
		if trimSpace(line) != "" {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return "", nil
	}
	out, err := b.phonemizeLines(lines, sep, strip)
	if err != nil {
		return "", err
	}
	return strings.Join(out, "\n"), nil
}

// phonemizeLines phonemizes each line as an utterance.
func (b *espeakBackend) phonemizeLines(text []string, sep separator, strip bool) ([]string, error) {
	// When preserving the punctuation, the lines are split in chunks at each
	// mark. Words are counted on the input lines, not on the chunks, so that
	// a mismatch is reported on the lines phonemized.
	var (
		chunks    []string
		lineParts [][]linePart
		chunkLine []int
	)
	if b.opts.preservePunctuation {
		chunks, lineParts = b.punctuation.preserve(text)
		for num, parts := range lineParts {
			for _, part := range parts {
				if !part.isMark {
					chunkLine = append(chunkLine, num)
				}
			}
		}
	} else {
		chunks = b.punctuation.remove(text)
	}
	mismatch := newWordsMismatch(chunks, chunkLine, len(text))

	// When preserving the punctuation, the chunks separated by inline marks
	// (quotes or brackets) are phonemized as a whole as well, appended to the
	// chunks, so that their words are pronounced in context.
	var groups [][]int
	if b.opts.preservePunctuation && canAlign(sep) {
		groups = inlineGroups(lineParts)
	}
	all := append([]string(nil), chunks...)
	for _, group := range groups {
		words := make([]string, len(group))
		for i, chunk := range group {
			words[i] = chunks[chunk]
		}
		all = append(all, strings.Join(words, " "))
	}

	phonemized, switches, err := b.phonemizeAux(all, sep, strip)
	if err != nil {
		return nil, err
	}
	// The groups' switches are dropped: they are found in their chunks
	// phonemized alone as well.
	nchunks := len(phonemized) - len(groups)
	var chunkSwitches []int
	for _, num := range switches {
		if num <= nchunks {
			chunkSwitches = append(chunkSwitches, num)
		}
	}
	context := make([]contextGroup, len(groups))
	for i, group := range groups {
		context[i] = contextGroup{chunks: group, phonemized: phonemized[nchunks+i]}
	}
	phonemized = phonemized[:nchunks]

	mismatch.countPhonemized(phonemized, sep)
	b.warnLanguageSwitches(chunkSwitches)
	out := phonemized
	if b.opts.preservePunctuation {
		out = restore(phonemized, lineParts, sep, strip, context)
	}
	mismatch.warn(len(out))
	return out, nil
}

// phonemizeAux phonemizes each line with espeak and post-processes it,
// returning the lines with a language switch, numbered from 1.
func (b *espeakBackend) phonemizeAux(text []string, sep separator, strip bool) ([]string, []int, error) {
	b.lib.mu.Lock()
	defer b.lib.mu.Unlock()
	if err := b.lib.setVoice(b.language); err != nil {
		return nil, nil, err
	}
	out := make([]string, 0, len(text))
	var switches []int
	for num, line := range text {
		processed, hasSwitch := b.postprocessLine(b.lib.textToPhonemes(line), sep, strip)
		out = append(out, processed)
		if hasSwitch {
			switches = append(switches, num+1)
		}
	}
	return out, switches, nil
}

// postprocessLine formats a line as espeak phonemized it: one line, its
// words separated by the word separator and their phonemes by the phone one.
// It reports whether espeak switched language in it, the flags being kept.
func (b *espeakBackend) postprocessLine(line string, sep separator, strip bool) (string, bool) {
	// espeak can split an utterance into several lines because of
	// punctuation; they are merged into one.
	line = strings.ReplaceAll(strings.ReplaceAll(trimSpace(line), "\n", " "), "  ", " ")
	// A bug in espeak-ng can add separators at the end of a word.
	if strings.Contains(line, "__") {
		line = espeakUnderscoresRe.ReplaceAllString(line, "_")
	}
	line = strings.ReplaceAll(line, "_ ", " ")
	hasSwitch := espeakFlagsRe.MatchString(line)
	if line == "" {
		return "", hasSwitch
	}

	words := strings.Split(line, " ")
	for i, word := range words {
		word = trimSpace(word)
		if !b.opts.withStress {
			word = strings.Map(func(r rune) rune {
				if strings.ContainsRune(espeakStress, r) {
					return -1
				}
				return r
			}, word)
		}
		if !strip {
			word += "_"
		}
		words[i] = strings.ReplaceAll(word, "_", sep.phone)
	}
	out := strings.Join(words, sep.word)
	if !strip {
		// The last word is followed by a word separator as well.
		out += sep.word
	}
	return out, hasSwitch
}

// warnLanguageSwitches reports the lines espeak read partly in another
// language, whose flags were kept.
func (b *espeakBackend) warnLanguageSwitches(switches []int) {
	if len(switches) == 0 {
		return
	}
	lines := make([]string, len(switches))
	for i, n := range switches {
		lines[i] = strconv.Itoa(n)
	}
	slog.Warn(fmt.Sprintf("%d utterances containing language switches on lines %s",
		len(switches), strings.Join(lines, ", ")))
	slog.Warn(fmt.Sprintf("extra phones may appear in the %q phoneset", b.language))
	slog.Warn(`language switch flags have been kept (applying "keep-flags" policy)`)
}

// wordsMismatch counts the words of each input line and of each phonemized
// one: espeak can join two words, or drop some, and a mismatch is reported.
type wordsMismatch struct {
	lines    []int
	nlines   int
	countTxt []int
	countPhn []int
}

// newWordsMismatch counts the words of the chunks phonemized; chunkLine is the
// input line of each chunk, nil when each chunk is a line.
func newWordsMismatch(chunks []string, chunkLine []int, nlines int) *wordsMismatch {
	m := &wordsMismatch{lines: chunkLine, nlines: nlines}
	counts := make([]int, len(chunks))
	for i, c := range chunks {
		counts[i] = len(fields(c))
	}
	m.countTxt = m.perLine(counts)
	return m
}

// countPhonemized counts the words of each phonemized chunk.
func (m *wordsMismatch) countPhonemized(text []string, sep separator) {
	counts := make([]int, len(text))
	for i, line := range text {
		counts[i] = len(separatorWords(trimSpace(line), sep))
	}
	m.countPhn = m.perLine(counts)
}

// perLine sums the counts of the chunks of each input line.
func (m *wordsMismatch) perLine(counts []int) []int {
	if m.lines == nil {
		return counts
	}
	per := make([]int, m.nlines)
	for i, line := range m.lines {
		if i < len(counts) {
			per[line] += counts[i]
		}
	}
	return per
}

// warn reports the lines whose words count changed in phonemization.
func (m *wordsMismatch) warn(nlines int) {
	mismatched := 0
	for i := range min(len(m.countTxt), len(m.countPhn)) {
		if m.countTxt[i] != m.countPhn[i] {
			mismatched++
		}
	}
	if mismatched == 0 || nlines == 0 {
		return
	}
	percent := math.Round(float64(mismatched)/float64(nlines)*100) / 100 * 100
	slog.Warn(fmt.Sprintf("words count mismatch on %s%% of the lines (%d/%d)",
		strconv.FormatFloat(percent, 'f', -1, 64), mismatched, nlines))
}
