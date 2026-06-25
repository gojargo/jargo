// Package language defines a canonical Language type and helpers so an
// application can name a language once and let each STT/TTS service map it to
// that provider's own code. It mirrors Pipecat's Language enum: the value of
// each Language is a BCP-47 code (an ISO-639 base, optionally with a region).
//
// Each service that takes a language provides its own mapper (for example
// deepgram and elevenlabs in their packages), because providers disagree on the
// exact codes — Deepgram wants BCP-47 ("fr-CA"), ElevenLabs wants the base code
// ("fr").
package language

import "strings"

// Language is a BCP-47 language code (ISO-639 base plus optional region).
type Language string

// Canonical languages. The constant value is the BCP-47 code.
const (
	English   Language = "en"
	EnglishUS Language = "en-US"
	EnglishGB Language = "en-GB"
	EnglishAU Language = "en-AU"
	EnglishCA Language = "en-CA"
	EnglishIN Language = "en-IN"

	French   Language = "fr"
	FrenchCA Language = "fr-CA"
	FrenchBE Language = "fr-BE"
	FrenchCH Language = "fr-CH"

	Spanish   Language = "es"
	SpanishES Language = "es-ES"
	SpanishMX Language = "es-MX"
	SpanishUS Language = "es-US"

	German    Language = "de"
	Italian   Language = "it"
	Dutch     Language = "nl"
	DutchBE   Language = "nl-BE"
	Polish    Language = "pl"
	Russian   Language = "ru"
	Ukrainian Language = "uk"
	Romanian  Language = "ro"
	Hungarian Language = "hu"
	Bulgarian Language = "bg"
	Czech     Language = "cs"
	Slovak    Language = "sk"
	Croatian  Language = "hr"
	Greek     Language = "el"
	Swedish   Language = "sv"
	Danish    Language = "da"
	Norwegian Language = "no"
	Finnish   Language = "fi"

	Portuguese   Language = "pt"
	PortugueseBR Language = "pt-BR"

	Arabic     Language = "ar"
	Hebrew     Language = "he"
	Hindi      Language = "hi"
	Tamil      Language = "ta"
	Turkish    Language = "tr"
	Indonesian Language = "id"
	Malay      Language = "ms"
	Filipino   Language = "fil"
	Vietnamese Language = "vi"
	Thai       Language = "th"
	Japanese   Language = "ja"
	Korean     Language = "ko"

	Chinese   Language = "zh"
	ChineseCN Language = "zh-CN"
	ChineseTW Language = "zh-TW"
	ChineseHK Language = "zh-HK"
)

// Code returns the BCP-47 code (the constant's value).
func (l Language) Code() string { return string(l) }

// BaseCode returns the language without any region subtag (for example "fr" for
// FrenchCA, "zh" for ChineseCN).
func (l Language) BaseCode() string {
	base, _, _ := strings.Cut(string(l), "-")
	return base
}
