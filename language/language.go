// Package language defines a canonical Language type and helpers so an
// application can name a language once and let each STT/TTS service map it to
// that provider's own code. The value of each Language is a BCP-47 code (an
// ISO-639 base, optionally with a region or script subtag).
//
// Each service that takes a language provides its own mapper (for example
// deepgram and elevenlabs in their packages), because providers disagree on the
// exact codes: Deepgram wants BCP-47 ("fr-CA"), ElevenLabs wants the base code
// ("fr"). Resolve is the shared shape those mappers take, so a language a
// provider has not been verified against still reaches it rather than being
// dropped.
package language

import (
	"log/slog"
	"strings"
)

// Language is a BCP-47 language code (an ISO-639 base plus an optional region or
// script subtag).
type Language string

// The languages a service can be asked for. The constant's value is its BCP-47
// code, and the groups are the languages themselves: a group's bare constant is
// the language with no region, and the rest name one region or script of it.
const (
	// Afrikaans.
	Afrikaans   Language = "af"
	AfrikaansZA Language = "af-ZA"

	// Amharic.
	Amharic   Language = "am"
	AmharicET Language = "am-ET"

	// Arabic.
	Arabic    Language = "ar"
	ArabicAE  Language = "ar-AE"
	ArabicBH  Language = "ar-BH"
	ArabicDZ  Language = "ar-DZ"
	ArabicEG  Language = "ar-EG"
	ArabicIQ  Language = "ar-IQ"
	ArabicJO  Language = "ar-JO"
	ArabicKW  Language = "ar-KW"
	ArabicLB  Language = "ar-LB"
	ArabicLY  Language = "ar-LY"
	ArabicMA  Language = "ar-MA"
	ArabicOM  Language = "ar-OM"
	ArabicQA  Language = "ar-QA"
	ArabicSA  Language = "ar-SA"
	ArabicSY  Language = "ar-SY"
	ArabicTN  Language = "ar-TN"
	ArabicXA  Language = "ar-XA"
	ArabicYE  Language = "ar-YE"
	Arabic001 Language = "ar-001"

	// Assamese.
	Assamese   Language = "as"
	AssameseIN Language = "as-IN"

	// Asturian.
	Asturian Language = "ast"

	// Azerbaijani.
	Azerbaijani   Language = "az"
	AzerbaijaniAZ Language = "az-AZ"

	// Bashkir.
	Bashkir Language = "ba"

	// Belarusian.
	Belarusian   Language = "be"
	BelarusianBY Language = "be-BY"

	// Bulgarian.
	Bulgarian   Language = "bg"
	BulgarianBG Language = "bg-BG"

	// Bengali.
	Bengali   Language = "bn"
	BengaliBD Language = "bn-BD"
	BengaliIN Language = "bn-IN"

	// Tibetan.
	Tibetan Language = "bo"

	// Breton.
	Breton Language = "br"

	// Bosnian.
	Bosnian   Language = "bs"
	BosnianBA Language = "bs-BA"

	// Catalan.
	Catalan   Language = "ca"
	CatalanES Language = "ca-ES"

	// Cebuano.
	Cebuano   Language = "ceb"
	CebuanoPH Language = "ceb-PH"

	// Mandarin Chinese.
	MandarinChinese   Language = "cmn"
	MandarinChineseCN Language = "cmn-CN"

	// Czech.
	Czech   Language = "cs"
	CzechCZ Language = "cs-CZ"

	// Welsh.
	Welsh   Language = "cy"
	WelshGB Language = "cy-GB"

	// Danish.
	Danish   Language = "da"
	DanishDK Language = "da-DK"

	// German.
	German   Language = "de"
	GermanAT Language = "de-AT"
	GermanBE Language = "de-BE"
	GermanCH Language = "de-CH"
	GermanDE Language = "de-DE"

	// Greek.
	Greek   Language = "el"
	GreekGR Language = "el-GR"

	// English.
	English   Language = "en"
	EnglishAU Language = "en-AU"
	EnglishCA Language = "en-CA"
	EnglishGB Language = "en-GB"
	EnglishGH Language = "en-GH"
	EnglishHK Language = "en-HK"
	EnglishIE Language = "en-IE"
	EnglishIN Language = "en-IN"
	EnglishKE Language = "en-KE"
	EnglishNG Language = "en-NG"
	EnglishNZ Language = "en-NZ"
	EnglishPH Language = "en-PH"
	EnglishSG Language = "en-SG"
	EnglishTZ Language = "en-TZ"
	EnglishUS Language = "en-US"
	EnglishZA Language = "en-ZA"

	// Esperanto.
	Esperanto Language = "eo"

	// Spanish.
	Spanish    Language = "es"
	SpanishAR  Language = "es-AR"
	SpanishBO  Language = "es-BO"
	SpanishCL  Language = "es-CL"
	SpanishCO  Language = "es-CO"
	SpanishCR  Language = "es-CR"
	SpanishCU  Language = "es-CU"
	SpanishDO  Language = "es-DO"
	SpanishEC  Language = "es-EC"
	SpanishES  Language = "es-ES"
	SpanishGQ  Language = "es-GQ"
	SpanishGT  Language = "es-GT"
	SpanishHN  Language = "es-HN"
	SpanishMX  Language = "es-MX"
	SpanishNI  Language = "es-NI"
	SpanishPA  Language = "es-PA"
	SpanishPE  Language = "es-PE"
	SpanishPR  Language = "es-PR"
	SpanishPY  Language = "es-PY"
	SpanishSV  Language = "es-SV"
	SpanishUS  Language = "es-US"
	SpanishUY  Language = "es-UY"
	SpanishVE  Language = "es-VE"
	Spanish419 Language = "es-419"

	// Estonian.
	Estonian   Language = "et"
	EstonianEE Language = "et-EE"

	// Basque.
	Basque   Language = "eu"
	BasqueES Language = "eu-ES"

	// Persian.
	Persian   Language = "fa"
	PersianIR Language = "fa-IR"

	// Fulah.
	Fulah Language = "ff"

	// Finnish.
	Finnish   Language = "fi"
	FinnishFI Language = "fi-FI"

	// Filipino.
	Filipino   Language = "fil"
	FilipinoPH Language = "fil-PH"

	// Faroese.
	Faroese Language = "fo"

	// French.
	French   Language = "fr"
	FrenchBE Language = "fr-BE"
	FrenchCA Language = "fr-CA"
	FrenchCH Language = "fr-CH"
	FrenchFR Language = "fr-FR"

	// Irish.
	Irish   Language = "ga"
	IrishIE Language = "ga-IE"

	// Gaelic.
	Gaelic Language = "gd"

	// Galician.
	Galician   Language = "gl"
	GalicianES Language = "gl-ES"

	// Gujarati.
	Gujarati   Language = "gu"
	GujaratiIN Language = "gu-IN"

	// Hausa.
	Hausa Language = "ha"

	// Hawaiian.
	Hawaiian Language = "haw"

	// Hebrew.
	Hebrew   Language = "he"
	HebrewIL Language = "he-IL"

	// Hindi.
	Hindi   Language = "hi"
	HindiIN Language = "hi-IN"

	// Croatian.
	Croatian   Language = "hr"
	CroatianHR Language = "hr-HR"

	// Haitian Creole.
	HaitianCreole   Language = "ht"
	HaitianCreoleHT Language = "ht-HT"

	// Hungarian.
	Hungarian   Language = "hu"
	HungarianHU Language = "hu-HU"

	// Armenian.
	Armenian   Language = "hy"
	ArmenianAM Language = "hy-AM"

	// Indonesian.
	Indonesian   Language = "id"
	IndonesianID Language = "id-ID"

	// Igbo.
	Igbo Language = "ig"

	// Icelandic.
	Icelandic   Language = "is"
	IcelandicIS Language = "is-IS"

	// Italian.
	Italian   Language = "it"
	ItalianIT Language = "it-IT"
	ItalianCH Language = "it-CH"

	// Inuktitut.
	InuktitutCans   Language = "iu-Cans"
	InuktitutCansCA Language = "iu-Cans-CA"
	InuktitutLatn   Language = "iu-Latn"
	InuktitutLatnCA Language = "iu-Latn-CA"

	// Japanese.
	Japanese   Language = "ja"
	JapaneseJP Language = "ja-JP"

	// Javanese.
	Javanese   Language = "jv"
	JavaneseID Language = "jv-ID"
	JavaneseJV Language = "jv-JV"
	JavaneseJW Language = "jw"

	// Georgian.
	Georgian   Language = "ka"
	GeorgianGE Language = "ka-GE"

	// Kabuverdianu.
	Kabuverdianu Language = "kea"

	// Kazakh.
	Kazakh   Language = "kk"
	KazakhKZ Language = "kk-KZ"

	// Khmer.
	Khmer   Language = "km"
	KhmerKH Language = "km-KH"

	// Kannada.
	Kannada   Language = "kn"
	KannadaIN Language = "kn-IN"

	// Konkani.
	Konkani   Language = "kok"
	KonkaniIN Language = "kok-IN"

	// Korean.
	Korean   Language = "ko"
	KoreanKR Language = "ko-KR"

	// Kurdish.
	Kurdish Language = "ku"

	// Kyrgyz.
	Kyrgyz   Language = "ky"
	KyrgyzKG Language = "ky-KG"

	// Latin.
	Latin   Language = "la"
	LatinVA Language = "la-VA"

	// Luxembourgish.
	Luxembourgish   Language = "lb"
	LuxembourgishLU Language = "lb-LU"

	// Lingala.
	Lingala Language = "ln"

	// Lao.
	Lao   Language = "lo"
	LaoLA Language = "lo-LA"

	// Lithuanian.
	Lithuanian   Language = "lt"
	LithuanianLT Language = "lt-LT"

	// Ganda.
	Ganda Language = "lg"

	// Luo.
	Luo Language = "luo"

	// Latvian.
	Latvian   Language = "lv"
	LatvianLV Language = "lv-LV"

	// Malagasy.
	Malagasy   Language = "mg"
	MalagasyMG Language = "mg-MG"

	// Maori.
	Maori Language = "mi"

	// Macedonian.
	Macedonian   Language = "mk"
	MacedonianMK Language = "mk-MK"

	// Maithili.
	Maithili   Language = "mai"
	MaithiliIN Language = "mai-IN"

	// Malayalam.
	Malayalam   Language = "ml"
	MalayalamIN Language = "ml-IN"

	// Mongolian.
	Mongolian   Language = "mn"
	MongolianMN Language = "mn-MN"

	// Marathi.
	Marathi   Language = "mr"
	MarathiIN Language = "mr-IN"

	// Malay.
	Malay   Language = "ms"
	MalayMY Language = "ms-MY"

	// Maltese.
	Maltese   Language = "mt"
	MalteseMT Language = "mt-MT"

	// Burmese.
	Burmese     Language = "my"
	BurmeseMM   Language = "my-MM"
	BurmeseMymr Language = "mymr"

	// Norwegian.
	NorwegianBokmal    Language = "nb"
	NorwegianBokmalNO  Language = "nb-NO"
	Norwegian          Language = "no"
	NorwegianNynorsk   Language = "nn"
	NorwegianNynorskNO Language = "nn-NO"

	// Nepali.
	Nepali   Language = "ne"
	NepaliNP Language = "ne-NP"

	// Dutch.
	Dutch   Language = "nl"
	DutchBE Language = "nl-BE"
	DutchNL Language = "nl-NL"

	// Northern Sotho.
	NorthernSotho Language = "nso"

	// Chichewa.
	Chichewa Language = "ny"

	// Occitan.
	Occitan Language = "oc"

	// Odia.
	Odia   Language = "or"
	OdiaIN Language = "or-IN"

	// Punjabi.
	Punjabi   Language = "pa"
	PunjabiIN Language = "pa-IN"

	// Polish.
	Polish   Language = "pl"
	PolishPL Language = "pl-PL"

	// Pashto.
	Pashto   Language = "ps"
	PashtoAF Language = "ps-AF"

	// Portuguese.
	Portuguese   Language = "pt"
	PortugueseBR Language = "pt-BR"
	PortuguesePT Language = "pt-PT"

	// Romanian.
	Romanian   Language = "ro"
	RomanianRO Language = "ro-RO"

	// Russian.
	Russian   Language = "ru"
	RussianRU Language = "ru-RU"

	// Sanskrit.
	Sanskrit Language = "sa"

	// Sindhi.
	Sindhi   Language = "sd"
	SindhiIN Language = "sd-IN"

	// Sinhala.
	Sinhala   Language = "si"
	SinhalaLK Language = "si-LK"

	// Slovak.
	Slovak   Language = "sk"
	SlovakSK Language = "sk-SK"

	// Slovenian.
	Slovenian   Language = "sl"
	SlovenianSI Language = "sl-SI"

	// Shona.
	Shona Language = "sn"

	// Somali.
	Somali   Language = "so"
	SomaliSO Language = "so-SO"

	// Albanian.
	Albanian   Language = "sq"
	AlbanianAL Language = "sq-AL"

	// Serbian.
	Serbian       Language = "sr"
	SerbianRS     Language = "sr-RS"
	SerbianLatn   Language = "sr-Latn"
	SerbianLatnRS Language = "sr-Latn-RS"

	// Sundanese.
	Sundanese   Language = "su"
	SundaneseID Language = "su-ID"

	// Swedish.
	Swedish   Language = "sv"
	SwedishSE Language = "sv-SE"

	// Swahili.
	Swahili   Language = "sw"
	SwahiliKE Language = "sw-KE"
	SwahiliTZ Language = "sw-TZ"

	// Tamil.
	Tamil   Language = "ta"
	TamilIN Language = "ta-IN"
	TamilLK Language = "ta-LK"
	TamilMY Language = "ta-MY"
	TamilSG Language = "ta-SG"

	// Telugu.
	Telugu   Language = "te"
	TeluguIN Language = "te-IN"

	// Tajik.
	Tajik Language = "tg"

	// Thai.
	Thai   Language = "th"
	ThaiTH Language = "th-TH"

	// Turkmen.
	Turkmen Language = "tk"

	// Tagalog.
	Tagalog Language = "tl"

	// Turkish.
	Turkish   Language = "tr"
	TurkishTR Language = "tr-TR"

	// Tatar.
	Tatar Language = "tt"

	// Uyghur.
	Uyghur Language = "ug"

	// Ukrainian.
	Ukrainian   Language = "uk"
	UkrainianUA Language = "uk-UA"

	// Umbundu.
	Umbundu Language = "umb"

	// Urdu.
	Urdu   Language = "ur"
	UrduIN Language = "ur-IN"
	UrduPK Language = "ur-PK"

	// Uzbek.
	Uzbek   Language = "uz"
	UzbekUZ Language = "uz-UZ"

	// Vietnamese.
	Vietnamese   Language = "vi"
	VietnameseVN Language = "vi-VN"

	// Wolof.
	Wolof Language = "wo"

	// Wu Chinese.
	WuChinese   Language = "wuu"
	WuChineseCN Language = "wuu-CN"

	// Yiddish.
	Yiddish Language = "yi"

	// Yoruba.
	Yoruba Language = "yo"

	// Yue Chinese (Cantonese).
	YueChineseCantonese   Language = "yue"
	YueChineseCantoneseCN Language = "yue-CN"

	// Chinese.
	Chinese           Language = "zh"
	ChineseCN         Language = "zh-CN"
	ChineseCNGuangxi  Language = "zh-CN-guangxi"
	ChineseCNHenan    Language = "zh-CN-henan"
	ChineseCNLiaoning Language = "zh-CN-liaoning"
	ChineseCNShaanxi  Language = "zh-CN-shaanxi"
	ChineseCNShandong Language = "zh-CN-shandong"
	ChineseCNSichuan  Language = "zh-CN-sichuan"
	ChineseHK         Language = "zh-HK"
	ChineseTW         Language = "zh-TW"

	// Xhosa.
	Xhosa Language = "xh-ZA"

	// Zulu.
	Zulu   Language = "zu"
	ZuluZA Language = "zu-ZA"
)

// Code returns the BCP-47 code (the constant's value).
func (l Language) Code() string { return string(l) }

// BaseCode returns the language without any region or script subtag (for
// example "fr" for FrenchCA, "zh" for ChineseCN).
func (l Language) BaseCode() string {
	base, _, _ := strings.Cut(string(l), "-")
	return base
}

// Resolve maps a language to the code one service takes.
//
// A language the service has been verified against is looked up in codes. One
// that is not is still sent, derived from the language itself and reported, so
// an unverified language reaches the provider and is named in the log rather
// than being silently dropped: providers accept far more than any map lists, and
// the ones they refuse are worth seeing.
//
// useBaseCode says which form the service takes. A service wanting the base
// alone ("fr") gets BaseCode; one wanting the full code ("fr-CA") gets that.
func Resolve(l Language, codes map[Language]string, useBaseCode bool) string {
	if code, ok := codes[l]; ok {
		return code
	}
	if useBaseCode {
		base := strings.ToLower(l.BaseCode())
		slog.Warn("language not verified for this service, using its base code",
			"language", string(l), "base_code", base)
		return base
	}
	slog.Warn("language not verified for this service, using it as it stands",
		"language", string(l))
	return string(l)
}

// All returns every language the catalog names, in the order it names them.
//
// It is what a Python enum gives for free by being iterable: a caller building a
// service's map, or listing what an application can be asked for, needs the set
// rather than the names one at a time. The slice is fresh on each call, so a
// caller sorting or filtering it cannot edit the catalog.
func All() []Language {
	return []Language{
		Afrikaans,
		AfrikaansZA,
		Amharic,
		AmharicET,
		Arabic,
		ArabicAE,
		ArabicBH,
		ArabicDZ,
		ArabicEG,
		ArabicIQ,
		ArabicJO,
		ArabicKW,
		ArabicLB,
		ArabicLY,
		ArabicMA,
		ArabicOM,
		ArabicQA,
		ArabicSA,
		ArabicSY,
		ArabicTN,
		ArabicXA,
		ArabicYE,
		Arabic001,
		Assamese,
		AssameseIN,
		Asturian,
		Azerbaijani,
		AzerbaijaniAZ,
		Bashkir,
		Belarusian,
		BelarusianBY,
		Bulgarian,
		BulgarianBG,
		Bengali,
		BengaliBD,
		BengaliIN,
		Tibetan,
		Breton,
		Bosnian,
		BosnianBA,
		Catalan,
		CatalanES,
		Cebuano,
		CebuanoPH,
		MandarinChinese,
		MandarinChineseCN,
		Czech,
		CzechCZ,
		Welsh,
		WelshGB,
		Danish,
		DanishDK,
		German,
		GermanAT,
		GermanBE,
		GermanCH,
		GermanDE,
		Greek,
		GreekGR,
		English,
		EnglishAU,
		EnglishCA,
		EnglishGB,
		EnglishGH,
		EnglishHK,
		EnglishIE,
		EnglishIN,
		EnglishKE,
		EnglishNG,
		EnglishNZ,
		EnglishPH,
		EnglishSG,
		EnglishTZ,
		EnglishUS,
		EnglishZA,
		Esperanto,
		Spanish,
		SpanishAR,
		SpanishBO,
		SpanishCL,
		SpanishCO,
		SpanishCR,
		SpanishCU,
		SpanishDO,
		SpanishEC,
		SpanishES,
		SpanishGQ,
		SpanishGT,
		SpanishHN,
		SpanishMX,
		SpanishNI,
		SpanishPA,
		SpanishPE,
		SpanishPR,
		SpanishPY,
		SpanishSV,
		SpanishUS,
		SpanishUY,
		SpanishVE,
		Spanish419,
		Estonian,
		EstonianEE,
		Basque,
		BasqueES,
		Persian,
		PersianIR,
		Fulah,
		Finnish,
		FinnishFI,
		Filipino,
		FilipinoPH,
		Faroese,
		French,
		FrenchBE,
		FrenchCA,
		FrenchCH,
		FrenchFR,
		Irish,
		IrishIE,
		Gaelic,
		Galician,
		GalicianES,
		Gujarati,
		GujaratiIN,
		Hausa,
		Hawaiian,
		Hebrew,
		HebrewIL,
		Hindi,
		HindiIN,
		Croatian,
		CroatianHR,
		HaitianCreole,
		HaitianCreoleHT,
		Hungarian,
		HungarianHU,
		Armenian,
		ArmenianAM,
		Indonesian,
		IndonesianID,
		Igbo,
		Icelandic,
		IcelandicIS,
		Italian,
		ItalianIT,
		ItalianCH,
		InuktitutCans,
		InuktitutCansCA,
		InuktitutLatn,
		InuktitutLatnCA,
		Japanese,
		JapaneseJP,
		Javanese,
		JavaneseID,
		JavaneseJV,
		JavaneseJW,
		Georgian,
		GeorgianGE,
		Kabuverdianu,
		Kazakh,
		KazakhKZ,
		Khmer,
		KhmerKH,
		Kannada,
		KannadaIN,
		Konkani,
		KonkaniIN,
		Korean,
		KoreanKR,
		Kurdish,
		Kyrgyz,
		KyrgyzKG,
		Latin,
		LatinVA,
		Luxembourgish,
		LuxembourgishLU,
		Lingala,
		Lao,
		LaoLA,
		Lithuanian,
		LithuanianLT,
		Ganda,
		Luo,
		Latvian,
		LatvianLV,
		Malagasy,
		MalagasyMG,
		Maori,
		Macedonian,
		MacedonianMK,
		Maithili,
		MaithiliIN,
		Malayalam,
		MalayalamIN,
		Mongolian,
		MongolianMN,
		Marathi,
		MarathiIN,
		Malay,
		MalayMY,
		Maltese,
		MalteseMT,
		Burmese,
		BurmeseMM,
		BurmeseMymr,
		NorwegianBokmal,
		NorwegianBokmalNO,
		Norwegian,
		NorwegianNynorsk,
		NorwegianNynorskNO,
		Nepali,
		NepaliNP,
		Dutch,
		DutchBE,
		DutchNL,
		NorthernSotho,
		Chichewa,
		Occitan,
		Odia,
		OdiaIN,
		Punjabi,
		PunjabiIN,
		Polish,
		PolishPL,
		Pashto,
		PashtoAF,
		Portuguese,
		PortugueseBR,
		PortuguesePT,
		Romanian,
		RomanianRO,
		Russian,
		RussianRU,
		Sanskrit,
		Sindhi,
		SindhiIN,
		Sinhala,
		SinhalaLK,
		Slovak,
		SlovakSK,
		Slovenian,
		SlovenianSI,
		Shona,
		Somali,
		SomaliSO,
		Albanian,
		AlbanianAL,
		Serbian,
		SerbianRS,
		SerbianLatn,
		SerbianLatnRS,
		Sundanese,
		SundaneseID,
		Swedish,
		SwedishSE,
		Swahili,
		SwahiliKE,
		SwahiliTZ,
		Tamil,
		TamilIN,
		TamilLK,
		TamilMY,
		TamilSG,
		Telugu,
		TeluguIN,
		Tajik,
		Thai,
		ThaiTH,
		Turkmen,
		Tagalog,
		Turkish,
		TurkishTR,
		Tatar,
		Uyghur,
		Ukrainian,
		UkrainianUA,
		Umbundu,
		Urdu,
		UrduIN,
		UrduPK,
		Uzbek,
		UzbekUZ,
		Vietnamese,
		VietnameseVN,
		Wolof,
		WuChinese,
		WuChineseCN,
		Yiddish,
		Yoruba,
		YueChineseCantonese,
		YueChineseCantoneseCN,
		Chinese,
		ChineseCN,
		ChineseCNGuangxi,
		ChineseCNHenan,
		ChineseCNLiaoning,
		ChineseCNShaanxi,
		ChineseCNShandong,
		ChineseCNSichuan,
		ChineseHK,
		ChineseTW,
		Xhosa,
		Zulu,
		ZuluZA,
	}
}
