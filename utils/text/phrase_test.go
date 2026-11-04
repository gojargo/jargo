package text_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

func assertPhraseMatches(t *testing.T, phrase string, found, notFound []string) {
	t.Helper()
	p := text.CompilePhrasePattern(phrase)
	for _, s := range found {
		if !p.MatchString(s) {
			t.Errorf("%q not found in %q", phrase, s)
		}
	}
	for _, s := range notFound {
		if p.MatchString(s) {
			t.Errorf("%q found in %q", phrase, s)
		}
	}
}

func TestPhrasePatternSpacedPhraseNeedsWordBoundaries(t *testing.T) {
	assertPhraseMatches(t, "hey jargo",
		[]string{"well Hey   Jargo what's up", "heyjargo"},
		[]string{"they jargos"})
}

func TestPhrasePatternChinesePhraseInsideUnspacedSentence(t *testing.T) {
	assertPhraseMatches(t, "小助手",
		[]string{"小助手今天天气怎么样", "你好小助手", " 你好小助手今天天气怎么样"},
		[]string{"你好小明"})
}

func TestPhrasePatternJapanesePhraseInsideUnspacedSentence(t *testing.T) {
	assertPhraseMatches(t, "ねえアシスタント", []string{"ねえアシスタント今日の天気は"}, nil)
}

func TestPhrasePatternMixedPhraseKeepsBoundaryOnLatinSide(t *testing.T) {
	assertPhraseMatches(t, "hey 小助手", []string{"hey 小助手今天天气怎么样"}, []string{"they 小助手"})
}

// An empty phrase matches where a word boundary is, which is anywhere text
// holds a word character.
func TestPhrasePatternEmptyPhraseMatchesAWordBoundary(t *testing.T) {
	assertPhraseMatches(t, "", []string{"hello", "  a  "}, []string{"", "   ", "!?"})
}

// The boundaries are Unicode ones, as in the transcripts being searched: an
// accented letter is a word character, and so is a letter of any script.
func TestPhrasePatternBoundariesAreUnicode(t *testing.T) {
	assertPhraseMatches(t, "café",
		[]string{"un café s'il vous plaît", "café", "Café!"},
		[]string{"cafés", "décafé"})
	assertPhraseMatches(t, "привет бот", []string{"Привет бот, как дела"}, []string{"привет ботик"})
}

// Words may be separated by Unicode whitespace, such as the ideographic space
// Chinese and Japanese transcripts use.
func TestPhrasePatternWordsSeparatedByUnicodeWhitespace(t *testing.T) {
	assertPhraseMatches(t, "hey jargo", []string{"hey　jargo", "hey jargo"}, nil)
}

// A phrase starting or ending with a non-word character needs a word character
// beside it, where the boundary is.
func TestPhrasePatternNonWordEdgesNeedAWordCharacterBeside(t *testing.T) {
	assertPhraseMatches(t, "jargo!", []string{"jargo!yes"}, []string{"jargo! yes", "jargo!"})
}
