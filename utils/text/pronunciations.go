package text

import (
	"cmp"
	"log/slog"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"
)

// PronunciationFormatter turns a word and its pronunciation in IPA into the text
// a TTS service reads in place of the word, reporting false when the service
// cannot use the pronunciation.
type PronunciationFormatter func(word, ipa string) (string, bool)

// PronunciationTransform replaces words with a TTS service's pronunciation
// markup. A TTS service built with one among its text transforms skips it while
// it cannot read pronunciation markup, so the words are spoken as written.
//
// Build one through the service's PronunciationTransformIPA, which supplies the
// service's own formatter.
type PronunciationTransform struct {
	// words are the words to replace, lowercased and longest first.
	words []string
	// phonemes maps each word, lowercased, to its IPA.
	phonemes map[string]string
	format   PronunciationFormatter
}

// NewPronunciationTransform returns a transform that replaces each word with its
// formatted pronunciation.
//
// Words match whole and case-insensitively, longest first, and never as part of
// a longer or hyphenated word. The matched text is passed to the formatter, so
// markup that keeps the word (such as an SSML <phoneme> tag) keeps it as
// written.
//
// Every pronunciation is formatted once here, so one the service cannot use is
// reported when the transform is built rather than while speaking. Those words
// are left out and spoken as written. serviceName names the service in that
// warning.
func NewPronunciationTransform(
	pronunciations map[string]string, format PronunciationFormatter, serviceName string,
) *PronunciationTransform {
	if serviceName == "" {
		serviceName = "TTS service"
	}
	keys := make([]string, 0, len(pronunciations))
	for word := range pronunciations {
		keys = append(keys, word)
	}
	slices.Sort(keys)

	t := &PronunciationTransform{phonemes: map[string]string{}, format: format}
	var unusable []string
	for _, key := range keys {
		word := strings.TrimSpace(key)
		if word == "" {
			continue
		}
		ipa := pronunciations[key]
		if _, ok := format(word, ipa); !ok {
			unusable = append(unusable, word)
			continue
		}
		t.phonemes[strings.ToLower(word)] = ipa
	}
	if len(unusable) > 0 {
		slog.Warn(serviceName+" cannot use these IPA pronunciations, so they will be spoken as written: "+
			strings.Join(unusable, ", "), "service", serviceName)
	}

	for word := range t.phonemes {
		t.words = append(t.words, word)
	}
	slices.SortFunc(t.words, func(a, b string) int {
		if c := cmp.Compare(utf8.RuneCountInString(b), utf8.RuneCountInString(a)); c != 0 {
			return c
		}
		return cmp.Compare(a, b)
	})
	return t
}

// Apply replaces every matched word in text.
func (t *PronunciationTransform) Apply(text string) string {
	if len(t.words) == 0 {
		return text
	}
	var (
		out  strings.Builder
		prev rune = -1
	)
	for i := 0; i < len(text); {
		if !isWordOrHyphen(prev) {
			if matched, word, ok := t.matchAt(text[i:]); ok {
				replacement, ok := t.format(matched, t.phonemes[word])
				if !ok {
					replacement = matched
				}
				out.WriteString(replacement)
				i += len(matched)
				prev, _ = utf8.DecodeLastRuneInString(matched)
				continue
			}
		}
		r, size := utf8.DecodeRuneInString(text[i:])
		out.WriteString(text[i : i+size])
		i += size
		prev = r
	}
	return out.String()
}

// matchAt finds the longest word that starts rest whole, and reports the text it
// matched and the word it matched as.
func (t *PronunciationTransform) matchAt(rest string) (matched, word string, ok bool) {
	for _, w := range t.words {
		n := utf8.RuneCountInString(w)
		end := 0
		for range n {
			if end >= len(rest) {
				end = -1
				break
			}
			_, size := utf8.DecodeRuneInString(rest[end:])
			end += size
		}
		if end < 0 || !strings.EqualFold(rest[:end], w) {
			continue
		}
		next, _ := utf8.DecodeRuneInString(rest[end:])
		if end < len(rest) && isWordOrHyphen(next) {
			continue
		}
		return rest[:end], w, true
	}
	return "", "", false
}

// isWordOrHyphen reports whether r continues a word: a letter, a digit, an
// underscore or a hyphen.
func isWordOrHyphen(r rune) bool {
	return r == '_' || r == '-' || unicode.IsLetter(r) || unicode.IsNumber(r)
}
