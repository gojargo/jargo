package text_test

import (
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

func assertSpokenWords(t *testing.T, s string, want int) {
	t.Helper()
	if got := text.CountSpokenWords(s); got != want {
		t.Errorf("CountSpokenWords(%q) = %d, want %d", s, got, want)
	}
}

func TestCountSpokenWordsWhitespaceSeparatedTokens(t *testing.T) {
	assertSpokenWords(t, "Hello there!", 2)
	assertSpokenWords(t, "don't stop", 2)
	assertSpokenWords(t, "", 0)
	assertSpokenWords(t, "   ", 0)
}

func TestCountSpokenWordsEachCJKCharacterCounts(t *testing.T) {
	assertSpokenWords(t, "等一下，这辆车的价格是多少？", 12)
	assertSpokenWords(t, "こんにちは", 5)
	assertSpokenWords(t, "Hello 你好", 3)
	assertSpokenWords(t, "100元", 2)
}

func TestCountSpokenWordsHangulStaysOneToken(t *testing.T) {
	assertSpokenWords(t, "안녕하세요", 1)
	assertSpokenWords(t, "안녕 하세요", 2)
}

// TestSupplementaryCJKCharactersCountIndividually checks an ideograph outside
// the basic plane counts as a word of its own, as the everyday ones do.
func TestSupplementaryCJKCharactersCountIndividually(t *testing.T) {
	for _, r := range []rune{0x20000, 0x2A700, 0x2B740, 0x2B820, 0x2CEB0, 0x2EBF0, 0x2F800, 0x30000, 0x31350, 0x323B0} {
		if got := text.CountSpokenWords(string([]rune{r, r})); got != 2 {
			t.Errorf("U+%X twice: %d words, want 2", r, got)
		}
	}
	if got := text.CountSpokenWords("Hello𠮷𠮷100"); got != 4 {
		t.Errorf("Hello𠮷𠮷100: %d words, want 4", got)
	}
	if got := text.CountSpokenWords("𠮷、𠮷。"); got != 2 {
		t.Errorf("𠮷、𠮷。: %d words, want 2", got)
	}
}
