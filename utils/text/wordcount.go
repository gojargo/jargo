package text

import (
	"strings"
	"unicode"
)

// cjkWordRanges are the everyday Chinese and Japanese letters. Hangul is left
// out: Korean separates words with spaces, so a Hangul token stays one word.
//
//nolint:gochecknoglobals // a constant table
var cjkWordRanges = [...][2]rune{
	{0x3400, 0x4DBF}, // CJK Unified Ideographs Extension A
	{0x4E00, 0x9FFF}, // CJK Unified Ideographs
	{0xF900, 0xFAFF}, // CJK Compatibility Ideographs
	{0x3040, 0x30FF}, // Hiragana and Katakana
	{0x31F0, 0x31FF}, // Katakana Phonetic Extensions
	{0xFF66, 0xFF9D}, // Halfwidth Katakana
}

// isCJKWordChar reports whether r is a Chinese or Japanese letter, rather than
// punctuation in those blocks.
func isCJKWordChar(r rune) bool {
	for _, rg := range cjkWordRanges {
		if r >= rg[0] && r <= rg[1] {
			return !unicode.IsPunct(r)
		}
	}
	return false
}

// CountSpokenWords counts spoken words, treating each Chinese or Japanese
// character as a word.
//
// Whitespace-separated tokens count as one word each. Chinese and Japanese are
// written without spaces between words, so each ideograph, hiragana or katakana
// character inside a token counts as one word. Punctuation in that token does
// not. A run of other characters in the same token, such as a Latin word or a
// number, still counts as one word.
func CountSpokenWords(s string) int {
	total := 0
	for _, token := range strings.FieldsFunc(s, isPhraseSpace) {
		if !strings.ContainsFunc(token, isCJKWordChar) {
			total++
			continue
		}
		inRun := false
		for _, r := range token {
			switch {
			case isCJKWordChar(r):
				total++
				inRun = false
			case unicode.IsPunct(r):
				inRun = false
			case !inRun:
				total++
				inRun = true
			}
		}
	}
	return total
}
