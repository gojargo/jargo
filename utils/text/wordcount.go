package text

import (
	"strings"
	"unicode"
)

// cjkWordRanges are the Chinese and Japanese letters, supplementary
// ideographs included. Hangul is left out: Korean separates words with spaces,
// so a Hangul token stays one word.
//
//nolint:gochecknoglobals // a constant table
var cjkWordRanges = [...][2]rune{
	{0x3400, 0x4DBF},   // CJK Unified Ideographs Extension A
	{0x4E00, 0x9FFF},   // CJK Unified Ideographs
	{0xF900, 0xFAFF},   // CJK Compatibility Ideographs
	{0x3040, 0x30FF},   // Hiragana and Katakana
	{0x31F0, 0x31FF},   // Katakana Phonetic Extensions
	{0xFF66, 0xFF9D},   // Halfwidth Katakana
	{0x20000, 0x2A6DF}, // CJK Unified Ideographs Extension B
	{0x2A700, 0x2B73F}, // CJK Unified Ideographs Extension C
	{0x2B740, 0x2B81F}, // CJK Unified Ideographs Extension D
	{0x2B820, 0x2CEAF}, // CJK Unified Ideographs Extension E
	{0x2CEB0, 0x2EBEF}, // CJK Unified Ideographs Extension F
	{0x2EBF0, 0x2EE5F}, // CJK Unified Ideographs Extension I
	{0x2F800, 0x2FA1F}, // CJK Compatibility Ideographs Supplement
	{0x30000, 0x3134F}, // CJK Unified Ideographs Extension G
	{0x31350, 0x323AF}, // CJK Unified Ideographs Extension H
	{0x323B0, 0x3347F}, // CJK Unified Ideographs Extension J
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
