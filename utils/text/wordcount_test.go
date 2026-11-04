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
