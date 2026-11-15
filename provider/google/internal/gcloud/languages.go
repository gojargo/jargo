package gcloud

import "github.com/gojargo/jargo/language"

// ttsLanguages are the languages Cloud Text-to-Speech has been verified
// against, each under the code it takes for it.
//
//nolint:gochecknoglobals,goconst // read-only lookup table, where codes recur
var ttsLanguages = map[language.Language]string{
	// Arabic
	language.Arabic: "ar-XA",
	// Bengali
	language.Bengali:   "bn-IN",
	language.BengaliIN: "bn-IN",
	// Bulgarian
	language.Bulgarian:   "bg-BG",
	language.BulgarianBG: "bg-BG",
	// Croatian
	language.Croatian:   "hr-HR",
	language.CroatianHR: "hr-HR",
	// Czech
	language.Czech:   "cs-CZ",
	language.CzechCZ: "cs-CZ",
	// Danish
	language.Danish:   "da-DK",
	language.DanishDK: "da-DK",
	// Dutch
	language.Dutch:   "nl-NL",
	language.DutchBE: "nl-BE",
	language.DutchNL: "nl-NL",
	// English
	language.English:   "en-US",
	language.EnglishUS: "en-US",
	language.EnglishAU: "en-AU",
	language.EnglishGB: "en-GB",
	language.EnglishIN: "en-IN",
	// Estonian
	language.Estonian:   "et-EE",
	language.EstonianEE: "et-EE",
	// Finnish
	language.Finnish:   "fi-FI",
	language.FinnishFI: "fi-FI",
	// French
	language.French:   "fr-FR",
	language.FrenchCA: "fr-CA",
	language.FrenchFR: "fr-FR",
	// German
	language.German:   "de-DE",
	language.GermanDE: "de-DE",
	// Greek
	language.Greek:   "el-GR",
	language.GreekGR: "el-GR",
	// Gujarati
	language.Gujarati:   "gu-IN",
	language.GujaratiIN: "gu-IN",
	// Hebrew
	language.Hebrew:   "he-IL",
	language.HebrewIL: "he-IL",
	// Hindi
	language.Hindi:   "hi-IN",
	language.HindiIN: "hi-IN",
	// Hungarian
	language.Hungarian:   "hu-HU",
	language.HungarianHU: "hu-HU",
	// Indonesian
	language.Indonesian:   "id-ID",
	language.IndonesianID: "id-ID",
	// Italian
	language.Italian:   "it-IT",
	language.ItalianIT: "it-IT",
	// Japanese
	language.Japanese:   "ja-JP",
	language.JapaneseJP: "ja-JP",
	// Kannada
	language.Kannada:   "kn-IN",
	language.KannadaIN: "kn-IN",
	// Korean
	language.Korean:   "ko-KR",
	language.KoreanKR: "ko-KR",
	// Latvian
	language.Latvian:   "lv-LV",
	language.LatvianLV: "lv-LV",
	// Lithuanian
	language.Lithuanian:   "lt-LT",
	language.LithuanianLT: "lt-LT",
	// Malayalam
	language.Malayalam:   "ml-IN",
	language.MalayalamIN: "ml-IN",
	// Chinese (Mandarin)
	language.Chinese:   "cmn-CN",
	language.ChineseCN: "cmn-CN",
	// Marathi
	language.Marathi:   "mr-IN",
	language.MarathiIN: "mr-IN",
	// Norwegian
	language.Norwegian:         "nb-NO",
	language.NorwegianBokmal:   "nb-NO",
	language.NorwegianBokmalNO: "nb-NO",
	// Polish
	language.Polish:   "pl-PL",
	language.PolishPL: "pl-PL",
	// Portuguese
	language.Portuguese:   "pt-BR",
	language.PortugueseBR: "pt-BR",
	// Romanian
	language.Romanian:   "ro-RO",
	language.RomanianRO: "ro-RO",
	// Russian
	language.Russian:   "ru-RU",
	language.RussianRU: "ru-RU",
	// Serbian
	language.Serbian:   "sr-RS",
	language.SerbianRS: "sr-RS",
	// Slovak
	language.Slovak:   "sk-SK",
	language.SlovakSK: "sk-SK",
	// Slovenian
	language.Slovenian:   "sl-SI",
	language.SlovenianSI: "sl-SI",
	// Spanish
	language.Spanish:   "es-ES",
	language.SpanishES: "es-ES",
	language.SpanishUS: "es-US",
	// Swahili
	language.Swahili:   "sw-KE",
	language.SwahiliKE: "sw-KE",
	// Swedish
	language.Swedish:   "sv-SE",
	language.SwedishSE: "sv-SE",
	// Tamil
	language.Tamil:   "ta-IN",
	language.TamilIN: "ta-IN",
	// Telugu
	language.Telugu:   "te-IN",
	language.TeluguIN: "te-IN",
	// Thai
	language.Thai:   "th-TH",
	language.ThaiTH: "th-TH",
	// Turkish
	language.Turkish:   "tr-TR",
	language.TurkishTR: "tr-TR",
	// Ukrainian
	language.Ukrainian:   "uk-UA",
	language.UkrainianUA: "uk-UA",
	// Urdu
	language.Urdu:   "ur-IN",
	language.UrduIN: "ur-IN",
	// Vietnamese
	language.Vietnamese:   "vi-VN",
	language.VietnameseVN: "vi-VN",
}

// TTSLanguage names a language the way Cloud Text-to-Speech does. A language
// the table does not list is passed on as its full code, and logged.
func TTSLanguage(l language.Language) string {
	return language.Resolve(l, ttsLanguages, false)
}
