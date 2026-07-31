package text

import (
	"strings"

	"github.com/neurosnap/sentences"
	"github.com/neurosnap/sentences/english"
)

// sentenceEndingPunctuation is every mark that can end a sentence, across the
// scripts a voice agent is likely to speak.
//
//nolint:gochecknoglobals // read-only lookup table
var sentenceEndingPunctuation = map[rune]bool{
	// Latin script (most European languages, Filipino).
	'.': true, '!': true, '?': true, ';': true, '…': true,
	// East Asian (Chinese, Japanese, Korean).
	'。': true, '？': true, '！': true, '；': true, '．': true, '｡': true,
	// Indic scripts.
	'।': true, '॥': true,
	// Arabic script (Arabic, Persian, Urdu, Pashto).
	'؟': true, '؛': true, '۔': true, '؏': true,
	// Myanmar.
	'၊': true, '။': true,
	// Khmer.
	'។': true, '៕': true,
	// Lao and Tibetan.
	'໌': true, '༎': true, '།': true,
	// Armenian.
	'։': true, '՜': true, '՞': true,
	// Ethiopic (Amharic).
	'።': true, '፧': true, '፨': true,
}

// latinSentenceEndingPunctuation is the punctuation the trained model
// disambiguates, since a period also appears in abbreviations and decimals.
//
//nolint:gochecknoglobals // read-only lookup table
var latinSentenceEndingPunctuation = map[rune]bool{
	'.': true, '!': true, '?': true, ';': true, '…': true,
}

// IsSentenceEnding reports whether r can end a sentence.
func IsSentenceEnding(r rune) bool { return sentenceEndingPunctuation[r] }

// isUnambiguousEnding reports whether r ends a sentence without needing the
// model's judgement, which is what makes it usable for a script the model was
// not trained on.
func isUnambiguousEnding(r rune) bool {
	return sentenceEndingPunctuation[r] && !latinSentenceEndingPunctuation[r]
}

// SentenceTokenizer finds sentence boundaries in a stream of text.
//
// This interface has no counterpart upstream, where the boundary function is
// called directly. It exists so a caller can supply a different segmenter, or
// avoid carrying a trained model, without the rest of the pipeline changing.
// The default is Punkt, which is what the behavior is defined against.
type SentenceTokenizer interface {
	// MatchEndOfSentence returns the byte offset just past the end of the first
	// complete sentence in text, or 0 when text does not yet hold one.
	MatchEndOfSentence(text string) int
}

// PunktTokenizer finds sentence boundaries with the Punkt algorithm, an
// unsupervised model trained per language. Punkt is what tells a sentence
// ending from an abbreviation: it keeps "Dr. Smith is here." and "M. Dupont est
// la." whole rather than breaking after the title.
type PunktTokenizer struct {
	tok *sentences.DefaultSentenceTokenizer
}

// NewPunktEnglish builds a tokenizer over the embedded English model.
func NewPunktEnglish() (*PunktTokenizer, error) {
	tok, err := english.NewSentenceTokenizer(nil)
	if err != nil {
		return nil, err
	}
	return &PunktTokenizer{tok: tok}, nil
}

// NewPunkt builds a tokenizer over a model trained for another language. The
// training data is the JSON the Punkt project publishes per language; embed the
// one the bot speaks and pass its bytes.
func NewPunkt(training []byte) (*PunktTokenizer, error) {
	t, err := sentences.LoadTraining(training)
	if err != nil {
		return nil, err
	}
	return &PunktTokenizer{tok: sentences.NewSentenceTokenizer(t)}, nil
}

// MatchEndOfSentence implements SentenceTokenizer.
//
// The model's answer is verified before it is trusted: asked about a single
// word it returns that word as a whole sentence, and LLM output arrives a token
// at a time, so a boundary is only reported when the text actually ends on
// sentence-ending punctuation.
func (p *PunktTokenizer) MatchEndOfSentence(text string) int {
	text = strings.TrimRight(text, " \t\n\r")
	if text == "" {
		return 0
	}
	found := p.tok.Tokenize(text)
	if len(found) == 0 {
		return 0
	}
	first := found[0].Text

	// More than one sentence means a boundary was found, so the first is
	// complete by definition.
	if len(found) > 1 {
		return len(first)
	}

	if first == text {
		if last, ok := lastRune(text); ok && sentenceEndingPunctuation[last] {
			return len(text)
		}
		// The model has no training for this script, so it returned the text
		// whole. Punctuation that needs no disambiguation still marks the end.
		for i, r := range text {
			if isUnambiguousEnding(r) {
				return i + len(string(r))
			}
		}
		return 0
	}

	// One sentence shorter than the text means it is not complete yet.
	return 0
}

// lastRune returns the final rune of text.
func lastRune(text string) (rune, bool) {
	for _, r := range reverseRunes(text) {
		return r, true
	}
	return 0, false
}

// reverseRunes returns text's runes last first.
func reverseRunes(text string) []rune {
	rs := []rune(text)
	for i, j := 0, len(rs)-1; i < j; i, j = i+1, j-1 {
		rs[i], rs[j] = rs[j], rs[i]
	}
	return rs
}

var _ SentenceTokenizer = (*PunktTokenizer)(nil)
