package text

import (
	"slices"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Pronunciation parsing and normalization shared by the TTS services.
//
// A pronunciation reaches a TTS service as an IPA string. Services each want
// their own markup for it, but they all start from the same steps: tidy the
// notation, split it into phones, and find which vowel each stress mark belongs
// to. Those steps live here, so a service's formatter only has to render the
// result.
//
// Normalization is notation-only. It unifies ways of writing the same symbol
// ("ʧ" and "tʃ", "g" and "ɡ", "'" and "ˈ") and never changes which sound is
// written, so it is safe for any language.

// The stress marks and the tie bar of IPA.
const (
	PrimaryStress   = "ˈ"
	SecondaryStress = "ˌ"
	TieBar          = "͡"
)

// ipaNotation holds the ligatures and ASCII look-alikes of IPA symbols, replaced
// in order. "tʃ" and "dʒ" are always read as one phone, so they are written
// untied; other affricates keep the tie bar that tells them apart from a
// cluster.
//
//nolint:gochecknoglobals // the notation table, fixed
var ipaNotation = [][2]string{
	{"͜", TieBar}, // tie bar below
	{"ʧ", "tʃ"},
	{"ʤ", "dʒ"},
	{"t" + TieBar + "ʃ", "tʃ"},
	{"d" + TieBar + "ʒ", "dʒ"},
	{"ʦ", "t" + TieBar + "s"},
	{"ʣ", "d" + TieBar + "z"},
	{"g", "ɡ"},
	{"'", PrimaryStress},
	{":", "ː"},
	{".", ""}, // syllable break
}

// ipaMulti holds the sequences read as a single phone even without a tie bar:
// affricates written this way in nearly every dictionary, and the English
// diphthongs. Anything else ("ts" in "cats" against Italian "pizza") is one
// phone only when tied.
//
//nolint:gochecknoglobals // the phone table, fixed
var ipaMulti = []string{"tʃ", "dʒ", "aɪ", "aʊ", "ɔɪ", "oʊ", "eɪ"}

// ipaVowels are the IPA vowel symbols.
const ipaVowels = "aeiouyæɑɒɔəɚɛɜɝɪʊʌɐᵻɨʉɯɤøœɶɵɘɞ"

// ipaModifiers belong to the phone before them rather than starting a new one.
const ipaModifiers = "ːˑ"

// NormalizeIPA unifies the notation of an IPA string without changing what it
// says.
//
// It strips surrounding /…/ or […], composes Unicode, replaces ligatures and
// ASCII look-alikes with their IPA symbols, writes tʃ and dʒ untied and other
// affricates tied (ʦ becomes t͡s), drops syllable breaks, and collapses
// whitespace:
//
//	NormalizeIPA("/ˈʧɪ.kən/") // "ˈtʃɪkən"
func NormalizeIPA(ipa string) string {
	ipa = norm.NFC.String(strings.TrimSpace(ipa))
	if r := []rune(ipa); len(r) >= 2 {
		first, last := r[0], r[len(r)-1]
		if (first == '/' && last == '/') || (first == '[' && last == ']') {
			ipa = string(r[1 : len(r)-1])
		}
	}
	for _, sub := range ipaNotation {
		ipa = strings.ReplaceAll(ipa, sub[0], sub[1])
	}
	return strings.Join(strings.Fields(ipa), " ")
}

// IPAPhones splits one IPA word into phones, in written order.
//
// Stress marks are returned as tokens of their own, where they were written.
// Tied sequences (t͡s), tʃ, dʒ and the English diphthongs are one phone each,
// returned without the tie bar; length marks and combining diacritics stay with
// the phone they modify:
//
//	IPAPhones("mɛtˈfɔɹmɪn") // ["m" "ɛ" "t" "ˈ" "f" "ɔ" "ɹ" "m" "ɪ" "n"]
func IPAPhones(ipa string) []string {
	r := []rune(strings.ReplaceAll(NormalizeIPA(ipa), " ", ""))
	tie := []rune(TieBar)[0]
	var phones []string
	for i := 0; i < len(r); {
		char := string(r[i])
		if isStressMark(char) {
			phones = append(phones, char)
			i++
			continue
		}
		var phone string
		if i+1 < len(r) && r[i+1] == tie && i+2 < len(r) {
			phone = char + string(r[i+2])
			i += 3
		} else {
			phone = char
			for _, m := range ipaMulti {
				mr := []rune(m)
				if i+len(mr) <= len(r) && slices.Equal(r[i:i+len(mr)], mr) {
					phone = m
					break
				}
			}
			i += len([]rune(phone))
		}
		for i < len(r) && isIPAModifier(r[i]) {
			phone += string(r[i])
			i++
		}
		if n := len(phones); n > 0 && !isStressMark(phones[n-1]) && isModifierOnly(phone) {
			phones[n-1] += phone
		} else {
			phones = append(phones, phone)
		}
	}
	return phones
}

// IsIPAVowel reports whether an IPA phone, as IPAPhones returns it, is a vowel.
func IsIPAVowel(phone string) bool {
	for _, r := range phone {
		return strings.ContainsRune(ipaVowels, r)
	}
	return false
}

// StressBeforeVowels moves each stress mark from where it was written to just
// before its vowel.
//
// Transcriptions usually put stress at the start of the syllable (mɛtˈfɔɹmɪn);
// some services want it on the vowel itself (mɛtfˈɔɹmɪn). A stress mark with no
// vowel after it is dropped.
func StressBeforeVowels(phones []string) []string {
	var (
		result  []string
		pending string
	)
	for _, phone := range phones {
		if isStressMark(phone) {
			pending = phone
			continue
		}
		if pending != "" && IsIPAVowel(phone) {
			result = append(result, pending)
			pending = ""
		}
		result = append(result, phone)
	}
	return result
}

func isStressMark(s string) bool { return s == PrimaryStress || s == SecondaryStress }

// isIPAModifier reports whether r modifies the phone before it: a length mark or
// a combining diacritic.
func isIPAModifier(r rune) bool {
	return strings.ContainsRune(ipaModifiers, r) || unicode.Is(unicode.Mn, r)
}

func isModifierOnly(phone string) bool {
	for _, r := range phone {
		if !isIPAModifier(r) {
			return false
		}
	}
	return true
}
