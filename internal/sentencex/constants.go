package sentencex

import (
	"cmp"
	"regexp"
	"slices"
	"strings"
	"sync"
)

// romanNumerals are the lowercase Roman numerals one to twenty.
//
//nolint:gochecknoglobals // fixed lookup table
var romanNumerals = [...]string{
	"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv",
	"xvi", "xvii", "xviii", "xix", "xx",
}

// quotePair is a quoted-region delimiter pair recognized by quotesFindAll.
type quotePair struct {
	open  string
	close string
	// ambiguous is set for pairs whose delimiters are ambiguous with non-quote
	// uses: ' clashes with contractions and possessives, ` with code ticks (so
	// don`t with a backtick instead of an apostrophe could trigger a false
	// match). The matcher surrounds these with word-boundary guards.
	// Unambiguous symbols like « or 「 need no special handling.
	ambiguous bool
}

// quotePairs lists the recognized quote pairs. Order matters: the matcher
// tries alternatives left to right, so longer openers and closers must come
// before shorter ones (” before ').
//
//nolint:gochecknoglobals // fixed lookup table
var quotePairs = [...]quotePair{
	{open: "``", close: "''"},
	{open: "``", close: "``"},
	{open: "''", close: "''"},
	{open: "`", close: "'", ambiguous: true},
	{open: "`", close: "`", ambiguous: true},
	{open: "'", close: "'", ambiguous: true},
	{open: "\"", close: "\""},
	{open: "«", close: "»"},
	{open: "‘", close: "’"},
	{open: "‚", close: "‚"},
	{open: "“", close: "”"},
	{open: "‛", close: "‛"},
	{open: "„", close: "“"},
	{open: "»", close: "«"},
	{open: "‟", close: "‟"},
	{open: "‹", close: "›"},
	{open: "《", close: "》"},
	{open: "「", close: "」"},
}

// whitespaceClass is the regular expression class of the Unicode White_Space
// property: the ASCII controls tab to carriage return, U+0085 and the
// separator categories. Go's \s covers ASCII only.
const whitespaceClass = `\t-\r\x{85}\p{Z}`

// Compiled regular expressions shared by every language.
//
//nolint:gochecknoglobals // compiled once
var (
	parensRegex = regexp.MustCompile(`[\(（<{\[](?:[^\)\]}>）]|\\[\)\]}>）])*[\)\]}>）]`)

	emailRegex = regexp.MustCompile(`[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}`)

	// numberedReferenceRegex matches one or more bracketed citation numbers,
	// as in "[1][2]". The digit class is Unicode Nd.
	numberedReferenceRegex = regexp.MustCompile(`^(?:[` + whitespaceClass + `]*\[\p{Nd}+\])+`)

	spaceAfterSeparator = regexp.MustCompile(`^[` + whitespaceClass + `]+`)
)

// quoteClosersByLen lists the quote closers, longest first. Only adjacent
// duplicates are removed, so a closer can appear twice.
//
//nolint:gochecknoglobals // built once
var quoteClosersByLen = sync.OnceValue(func() []string {
	v := make([]string, 0, len(quotePairs))
	for i := range quotePairs {
		v = append(v, quotePairs[i].close)
	}
	slices.SortStableFunc(v, func(a, b string) int { return cmp.Compare(len(b), len(a)) })
	return slices.Compact(v)
})

// exclamationWords are names that end, or contain, an exclamation mark that
// does not end a sentence.
//
//nolint:gochecknoglobals // fixed lookup table
var exclamationWords = [...]string{
	"!Xũ",
	"!Kung",
	"ǃʼOǃKung",
	"!Xuun",
	"!Kung-Ekoka",
	"ǃHu",
	"ǃKhung",
	"ǃKu",
	"ǃung",
	"ǃXo",
	"ǃXû",
	"ǃXung",
	"ǃXũ",
	"!Xun",
	"Yahoo!",
	"Y!J",
	"Yum!",
}

// globalSentenceTerminators are the characters that can end a sentence: the
// Sentence_Break STerm and ATerm characters, plus two manual entries. See
// https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/SentenceBreakProperty.txt.
//
//nolint:gochecknoglobals // fixed lookup table
var globalSentenceTerminators = [...]rune{
	'!', // U+00021 BC=ON BLK=Basic_Latin SC=Common EXCLAMATION MARK
	'.', // U+0002E BC=CS BLK=Basic_Latin SC=Common FULL STOP
	'?', // U+0003F BC=ON BLK=Basic_Latin SC=Common QUESTION MARK
	'։', // U+00589 BC=L BLK=Armenian SC=Armenian ARMENIAN FULL STOP
	'؝', // U+0061D BC=AL BLK=Arabic SC=Arabic ARABIC END OF TEXT MARK
	'؞', // U+0061E BC=AL BLK=Arabic SC=Arabic ARABIC TRIPLE DOT PUNCTUATION MARK
	'؟', // U+0061F BC=AL BLK=Arabic SC=Common ARABIC QUESTION MARK
	'۔', // U+006D4 BC=AL BLK=Arabic SC=Arabic ARABIC FULL STOP
	'܀', // U+00700 BC=AL BLK=Syriac SC=Syriac SYRIAC END OF PARAGRAPH
	'܁', // U+00701 BC=AL BLK=Syriac SC=Syriac SYRIAC SUPRALINEAR FULL STOP
	'܂', // U+00702 BC=AL BLK=Syriac SC=Syriac SYRIAC SUBLINEAR FULL STOP
	'߹', // U+007F9 BC=ON BLK=NKo SC=Nko NKO EXCLAMATION MARK
	'࠷', // U+00837 BC=R BLK=Samaritan SC=Samaritan SAMARITAN PUNCTUATION MELODIC QITSA
	'࠹', // U+00839 BC=R BLK=Samaritan SC=Samaritan SAMARITAN PUNCTUATION QITSA
	'࠽', // U+0083D BC=R BLK=Samaritan SC=Samaritan SAMARITAN PUNCTUATION SOF MASHFAAT
	'࠾', // U+0083E BC=R BLK=Samaritan SC=Samaritan SAMARITAN PUNCTUATION ANNAAU
	'।', // U+00964 BC=L BLK=Devanagari SC=Common DEVANAGARI DANDA
	'॥', // U+00965 BC=L BLK=Devanagari SC=Common DEVANAGARI DOUBLE DANDA
	'၊', // U+0104A BC=L BLK=Myanmar SC=Myanmar MYANMAR SIGN LITTLE SECTION
	'။', // U+0104B BC=L BLK=Myanmar SC=Myanmar MYANMAR SIGN SECTION
	'።', // U+01362 BC=L BLK=Ethiopic SC=Ethiopic ETHIOPIC FULL STOP
	'፧', // U+01367 BC=L BLK=Ethiopic SC=Ethiopic ETHIOPIC QUESTION MARK
	'፨', // U+01368 BC=L BLK=Ethiopic SC=Ethiopic ETHIOPIC PARAGRAPH SEPARATOR
	'᙮', // U+0166E BC=L BLK=Unified_Canadian_Aboriginal_Syllabics SC=Canadian_Aboriginal CANADIAN SYLLABICS FULL STOP
	'᜵', // U+01735 BC=L BLK=Hanunoo SC=Common PHILIPPINE SINGLE PUNCTUATION
	'᜶', // U+01736 BC=L BLK=Hanunoo SC=Common PHILIPPINE DOUBLE PUNCTUATION
	'᠃', // U+01803 BC=ON BLK=Mongolian SC=Common MONGOLIAN FULL STOP
	'᠉', // U+01809 BC=ON BLK=Mongolian SC=Mongolian MONGOLIAN MANCHU FULL STOP
	'᥄', // U+01944 BC=ON BLK=Limbu SC=Limbu LIMBU EXCLAMATION MARK
	'᥅', // U+01945 BC=ON BLK=Limbu SC=Limbu LIMBU QUESTION MARK
	'᪨', // U+01AA8 BC=L BLK=Tai_Tham SC=Tai_Tham TAI THAM SIGN KAAN
	'᪩', // U+01AA9 BC=L BLK=Tai_Tham SC=Tai_Tham TAI THAM SIGN KAANKUU
	'᪪', // U+01AAA BC=L BLK=Tai_Tham SC=Tai_Tham TAI THAM SIGN SATKAAN
	'᪫', // U+01AAB BC=L BLK=Tai_Tham SC=Tai_Tham TAI THAM SIGN SATKAANKUU
	'᭚', // U+01B5A BC=L BLK=Balinese SC=Balinese BALINESE PANTI
	'᭛', // U+01B5B BC=L BLK=Balinese SC=Balinese BALINESE PAMADA
	'᭞', // U+01B5E BC=L BLK=Balinese SC=Balinese BALINESE CARIK SIKI
	'᭟', // U+01B5F BC=L BLK=Balinese SC=Balinese BALINESE CARIK PAREREN
	'᭽', // U+01B7D BC=L BLK=Balinese SC=Balinese BALINESE PANTI LANTANG
	'᭾', // U+01B7E BC=L BLK=Balinese SC=Balinese BALINESE PAMADA LANTANG
	'᰻', // U+01C3B BC=L BLK=Lepcha SC=Lepcha LEPCHA PUNCTUATION TA-ROL
	'᰼', // U+01C3C BC=L BLK=Lepcha SC=Lepcha LEPCHA PUNCTUATION NYET THYOOM TA-ROL
	'᱾', // U+01C7E BC=L BLK=Ol_Chiki SC=Ol_Chiki OL CHIKI PUNCTUATION MUCAAD
	'᱿', // U+01C7F BC=L BLK=Ol_Chiki SC=Ol_Chiki OL CHIKI PUNCTUATION DOUBLE MUCAAD
	'․', // U+02024 BC=ON BLK=General_Punctuation SC=Common ONE DOT LEADER
	'‼', // U+0203C BC=ON BLK=General_Punctuation SC=Common DOUBLE EXCLAMATION MARK
	'‽', // U+0203D BC=ON BLK=General_Punctuation SC=Common INTERROBANG
	'⁇', // U+02047 BC=ON BLK=General_Punctuation SC=Common DOUBLE QUESTION MARK
	'⁈', // U+02048 BC=ON BLK=General_Punctuation SC=Common QUESTION EXCLAMATION MARK
	'⁉', // U+02049 BC=ON BLK=General_Punctuation SC=Common EXCLAMATION QUESTION MARK
	'⸮', // U+02E2E BC=ON BLK=Supplemental_Punctuation SC=Common REVERSED QUESTION MARK
	'⸼', // U+02E3C BC=ON BLK=Supplemental_Punctuation SC=Common STENOGRAPHIC FULL STOP
	'⹓', // U+02E53 BC=ON BLK=Supplemental_Punctuation SC=Common MEDIEVAL EXCLAMATION MARK
	'⹔', // U+02E54 BC=ON BLK=Supplemental_Punctuation SC=Common MEDIEVAL QUESTION MARK
	'꓿', // U+0A4FF BC=L BLK=Lisu SC=Lisu LISU PUNCTUATION FULL STOP
	'꘎', // U+0A60E BC=ON BLK=Vai SC=Vai VAI FULL STOP
	'꘏', // U+0A60F BC=ON BLK=Vai SC=Vai VAI QUESTION MARK
	'꛳', // U+0A6F3 BC=L BLK=Bamum SC=Bamum BAMUM FULL STOP
	'꛷', // U+0A6F7 BC=L BLK=Bamum SC=Bamum BAMUM QUESTION MARK
	'꡶', // U+0A876 BC=ON BLK=Phags-pa SC=Phags_Pa PHAGS-PA MARK SHAD
	'꡷', // U+0A877 BC=ON BLK=Phags-pa SC=Phags_Pa PHAGS-PA MARK DOUBLE SHAD
	'꣎', // U+0A8CE BC=L BLK=Saurashtra SC=Saurashtra SAURASHTRA DANDA
	'꣏', // U+0A8CF BC=L BLK=Saurashtra SC=Saurashtra SAURASHTRA DOUBLE DANDA
	'꤯', // U+0A92F BC=L BLK=Kayah_Li SC=Kayah_Li KAYAH LI SIGN SHYA
	'꧈', // U+0A9C8 BC=L BLK=Javanese SC=Javanese JAVANESE PADA LINGSA
	'꧉', // U+0A9C9 BC=L BLK=Javanese SC=Javanese JAVANESE PADA LUNGSI
	'꩝', // U+0AA5D BC=L BLK=Cham SC=Cham CHAM PUNCTUATION DANDA
	'꩞', // U+0AA5E BC=L BLK=Cham SC=Cham CHAM PUNCTUATION DOUBLE DANDA
	'꩟', // U+0AA5F BC=L BLK=Cham SC=Cham CHAM PUNCTUATION TRIPLE DANDA
	'꫰', // U+0AAF0 BC=L BLK=Meetei_Mayek_Extensions SC=Meetei_Mayek MEETEI MAYEK CHEIKHAN
	'꫱', // U+0AAF1 BC=L BLK=Meetei_Mayek_Extensions SC=Meetei_Mayek MEETEI MAYEK AHANG KHUDAM
	'꯫', // U+0ABEB BC=L BLK=Meetei_Mayek SC=Meetei_Mayek MEETEI MAYEK CHEIKHEI
	'﹒', // U+0FE52 BC=CS BLK=Small_Form_Variants SC=Common SMALL FULL STOP
	'﹖', // U+0FE56 BC=ON BLK=Small_Form_Variants SC=Common SMALL QUESTION MARK
	'﹗', // U+0FE57 BC=ON BLK=Small_Form_Variants SC=Common SMALL EXCLAMATION MARK
	'！', // U+0FF01 BC=ON BLK=Halfwidth_and_Fullwidth_Forms SC=Common FULLWIDTH EXCLAMATION MARK
	'．', // U+0FF0E BC=CS BLK=Halfwidth_and_Fullwidth_Forms SC=Common FULLWIDTH FULL STOP
	'？', // U+0FF1F BC=ON BLK=Halfwidth_and_Fullwidth_Forms SC=Common FULLWIDTH QUESTION MARK
	'𐩖', // U+10A56 BC=R BLK=Kharoshthi SC=Kharoshthi KHAROSHTHI PUNCTUATION DANDA
	'𐩗', // U+10A57 BC=R BLK=Kharoshthi SC=Kharoshthi KHAROSHTHI PUNCTUATION DOUBLE DANDA
	'𐽕', // U+10F55 BC=AL BLK=Sogdian SC=Sogdian SOGDIAN PUNCTUATION TWO VERTICAL BARS
	'𐽖', // U+10F56 BC=AL BLK=Sogdian SC=Sogdian SOGDIAN PUNCTUATION TWO VERTICAL BARS WITH DOTS
	'𐽗', // U+10F57 BC=AL BLK=Sogdian SC=Sogdian SOGDIAN PUNCTUATION CIRCLE WITH DOT
	'𐽘', // U+10F58 BC=AL BLK=Sogdian SC=Sogdian SOGDIAN PUNCTUATION TWO CIRCLES WITH DOTS
	'𐽙', // U+10F59 BC=AL BLK=Sogdian SC=Sogdian SOGDIAN PUNCTUATION HALF CIRCLE WITH DOT
	'𐾆', // U+10F86 BC=R BLK=Old_Uyghur SC=Old_Uyghur OLD UYGHUR PUNCTUATION BAR
	'𐾇', // U+10F87 BC=R BLK=Old_Uyghur SC=Old_Uyghur OLD UYGHUR PUNCTUATION TWO BARS
	'𐾈', // U+10F88 BC=R BLK=Old_Uyghur SC=Old_Uyghur OLD UYGHUR PUNCTUATION TWO DOTS
	'𐾉', // U+10F89 BC=R BLK=Old_Uyghur SC=Old_Uyghur OLD UYGHUR PUNCTUATION FOUR DOTS
	'𑁇', // U+11047 BC=L BLK=Brahmi SC=Brahmi BRAHMI DANDA
	'𑁈', // U+11048 BC=L BLK=Brahmi SC=Brahmi BRAHMI DOUBLE DANDA
	'𑂾', // U+110BE BC=L BLK=Kaithi SC=Kaithi KAITHI SECTION MARK
	'𑂿', // U+110BF BC=L BLK=Kaithi SC=Kaithi KAITHI DOUBLE SECTION MARK
	'𑃀', // U+110C0 BC=L BLK=Kaithi SC=Kaithi KAITHI DANDA
	'𑃁', // U+110C1 BC=L BLK=Kaithi SC=Kaithi KAITHI DOUBLE DANDA
	'𑅁', // U+11141 BC=L BLK=Chakma SC=Chakma CHAKMA DANDA
	'𑅂', // U+11142 BC=L BLK=Chakma SC=Chakma CHAKMA DOUBLE DANDA
	'𑅃', // U+11143 BC=L BLK=Chakma SC=Chakma CHAKMA QUESTION MARK
	'𑇅', // U+111C5 BC=L BLK=Sharada SC=Sharada SHARADA DANDA
	'𑇆', // U+111C6 BC=L BLK=Sharada SC=Sharada SHARADA DOUBLE DANDA
	'𑇍', // U+111CD BC=L BLK=Sharada SC=Sharada SHARADA SUTRA MARK
	'𑇞', // U+111DE BC=L BLK=Sharada SC=Sharada SHARADA SECTION MARK-1
	'𑇟', // U+111DF BC=L BLK=Sharada SC=Sharada SHARADA SECTION MARK-2
	'𑈸', // U+11238 BC=L BLK=Khojki SC=Khojki KHOJKI DANDA
	'𑈹', // U+11239 BC=L BLK=Khojki SC=Khojki KHOJKI DOUBLE DANDA
	'𑈻', // U+1123B BC=L BLK=Khojki SC=Khojki KHOJKI SECTION MARK
	'𑈼', // U+1123C BC=L BLK=Khojki SC=Khojki KHOJKI DOUBLE SECTION MARK
	'𑊩', // U+112A9 BC=L BLK=Multani SC=Multani MULTANI SECTION MARK
	'𑑋', // U+1144B BC=L BLK=Newa SC=Newa NEWA DANDA
	'𑑌', // U+1144C BC=L BLK=Newa SC=Newa NEWA DOUBLE DANDA
	'𑗂', // U+115C2 BC=L BLK=Siddham SC=Siddham SIDDHAM DANDA
	'𑗃', // U+115C3 BC=L BLK=Siddham SC=Siddham SIDDHAM DOUBLE DANDA
	'𑗉', // U+115C9 BC=L BLK=Siddham SC=Siddham SIDDHAM END OF TEXT MARK
	'𑗊', // U+115CA BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH TRIDENT AND U-SHAPED ORNAMENTS
	'𑗋', // U+115CB BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH TRIDENT AND DOTTED CRESCENTS
	'𑗌', // U+115CC BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH RAYS AND DOTTED CRESCENTS
	'𑗍', // U+115CD BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH RAYS AND DOTTED DOUBLE CRESCENTS
	'𑗎', // U+115CE BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH RAYS AND DOTTED TRIPLE CRESCENTS
	'𑗏', // U+115CF BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK DOUBLE RING
	'𑗐', // U+115D0 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK DOUBLE RING WITH RAYS
	'𑗑', // U+115D1 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH DOUBLE CRESCENTS
	'𑗒', // U+115D2 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH TRIPLE CRESCENTS
	'𑗓', // U+115D3 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH QUADRUPLE CRESCENTS
	'𑗔', // U+115D4 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH SEPTUPLE CRESCENTS
	'𑗕', // U+115D5 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH CIRCLES AND RAYS
	'𑗖', // U+115D6 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH CIRCLES AND TWO ENCLOSURES
	'𑗗', // U+115D7 BC=L BLK=Siddham SC=Siddham SIDDHAM SECTION MARK WITH CIRCLES AND FOUR ENCLOSURES
	'𑙁', // U+11641 BC=L BLK=Modi SC=Modi MODI DANDA
	'𑙂', // U+11642 BC=L BLK=Modi SC=Modi MODI DOUBLE DANDA
	'𑜼', // U+1173C BC=L BLK=Ahom SC=Ahom AHOM SIGN SMALL SECTION
	'𑜽', // U+1173D BC=L BLK=Ahom SC=Ahom AHOM SIGN SECTION
	'𑜾', // U+1173E BC=L BLK=Ahom SC=Ahom AHOM SIGN RULAI
	'𑥄', // U+11944 BC=L BLK=Dives_Akuru SC=Dives_Akuru DIVES AKURU DOUBLE DANDA
	'𑥆', // U+11946 BC=L BLK=Dives_Akuru SC=Dives_Akuru DIVES AKURU END OF TEXT MARK
	'𑩂', // U+11A42 BC=L BLK=Zanabazar_Square SC=Zanabazar_Square ZANABAZAR SQUARE MARK SHAD
	'𑩃', // U+11A43 BC=L BLK=Zanabazar_Square SC=Zanabazar_Square ZANABAZAR SQUARE MARK DOUBLE SHAD
	'𑪛', // U+11A9B BC=L BLK=Soyombo SC=Soyombo SOYOMBO MARK SHAD
	'𑪜', // U+11A9C BC=L BLK=Soyombo SC=Soyombo SOYOMBO MARK DOUBLE SHAD
	'𑱁', // U+11C41 BC=L BLK=Bhaiksuki SC=Bhaiksuki BHAIKSUKI DANDA
	'𑱂', // U+11C42 BC=L BLK=Bhaiksuki SC=Bhaiksuki BHAIKSUKI DOUBLE DANDA
	'𑻷', // U+11EF7 BC=L BLK=Makasar SC=Makasar MAKASAR PASSIMBANG
	'𑻸', // U+11EF8 BC=L BLK=Makasar SC=Makasar MAKASAR END OF SECTION
	'𑽃', // U+11F43 BC=L BLK=Kawi SC=Kawi KAWI DANDA
	'𑽄', // U+11F44 BC=L BLK=Kawi SC=Kawi KAWI DOUBLE DANDA
	'𖩮', // U+16A6E BC=L BLK=Mro SC=Mro MRO DANDA
	'𖩯', // U+16A6F BC=L BLK=Mro SC=Mro MRO DOUBLE DANDA
	'𖫵', // U+16AF5 BC=L BLK=Bassa_Vah SC=Bassa_Vah BASSA VAH FULL STOP
	'𖬷', // U+16B37 BC=L BLK=Pahawh_Hmong SC=Pahawh_Hmong PAHAWH HMONG SIGN VOS THOM
	'𖬸', // U+16B38 BC=L BLK=Pahawh_Hmong SC=Pahawh_Hmong PAHAWH HMONG SIGN VOS TSHAB CEEB
	'𖭄', // U+16B44 BC=L BLK=Pahawh_Hmong SC=Pahawh_Hmong PAHAWH HMONG SIGN XAUS
	'𖺘', // U+16E98 BC=L BLK=Medefaidrin SC=Medefaidrin MEDEFAIDRIN FULL STOP
	'𛲟', // U+1BC9F BC=L BLK=Duployan SC=Duployan DUPLOYAN PUNCTUATION CHINOOK FULL STOP
	'𝪈', // U+1DA88 BC=L BLK=Sutton_SignWriting SC=SignWriting SIGNWRITING FULL STOP
	// Additional manual entries.
	'。', // U+3002 IDEOGRAPHIC FULL STOP
	'｡', // U+FF61 HALFWIDTH IDEOGRAPHIC FULL STOP
}

// globalSentenceTerminatorsSet is globalSentenceTerminators as a set.
//
//nolint:gochecknoglobals // built once
var globalSentenceTerminatorsSet = sync.OnceValue(func() map[rune]struct{} {
	m := make(map[rune]struct{}, len(globalSentenceTerminators))
	for _, r := range globalSentenceTerminators {
		m[r] = struct{}{}
	}
	return m
})

// isSentenceTerminator reports whether c is one of globalSentenceTerminators.
func isSentenceTerminator(c rune) bool {
	// Fast check for ASCII to avoid the map lookup.
	if c < 0x80 {
		return c == '!' || c == '.' || c == '?'
	}
	_, ok := globalSentenceTerminatorsSet()[c]
	return ok
}

// quoteAlternativeKind is the shape of one alternative of the quote matcher.
type quoteAlternativeKind int

const (
	// quotePlain is open(?s:.*?)close: the opener, then the nearest closer.
	quotePlain quoteAlternativeKind = iota
	// quoteGuarded is \Bopen\b(?s:.*?[^\sclose])close\B: an ambiguous opener
	// that is not preceded by a word character and is followed by one, and a
	// closer that follows a character other than whitespace or the closer and
	// is not followed by a word character.
	quoteGuarded
	// quoteLineAnchored is (?m:^open.+?close$): an ambiguous opener at the
	// start of a line and a closer at the end of the same line. At line start
	// there is no preceding noun for a possessive to attach to, and the
	// symmetric pairing across the line is evidence of a quotation, so both
	// guards are dropped and whitespace is allowed just inside the pair
	// (' Hello, world. ').
	quoteLineAnchored
)

// quoteAlternative is one alternative of the quote matcher.
type quoteAlternative struct {
	pair *quotePair
	kind quoteAlternativeKind
}

// quoteAlternatives are the matcher's alternatives in priority order: one per
// quote pair, then a line-anchored one per ambiguous pair.
//
//nolint:gochecknoglobals // built once
var quoteAlternatives = sync.OnceValue(func() []quoteAlternative {
	alts := make([]quoteAlternative, 0, len(quotePairs)+3)
	for i := range quotePairs {
		kind := quotePlain
		if quotePairs[i].ambiguous {
			kind = quoteGuarded
		}
		alts = append(alts, quoteAlternative{pair: &quotePairs[i], kind: kind})
	}
	for i := range quotePairs {
		if quotePairs[i].ambiguous {
			alts = append(alts, quoteAlternative{pair: &quotePairs[i], kind: quoteLineAnchored})
		}
	}
	return alts
})

// quoteOpenerFirstBytes marks every byte a quote opener can begin with.
//
//nolint:gochecknoglobals // built once
var quoteOpenerFirstBytes = sync.OnceValue(func() *[256]bool {
	var t [256]bool
	for i := range quotePairs {
		t[quotePairs[i].open[0]] = true
	}
	return &t
})

// closerCursor remembers the first acceptable closer found at or after a
// position, so the searches of one pass over a text, whose start positions
// only grow, do not rescan the same bytes.
type closerCursor struct {
	from  int
	at    int
	valid bool
}

// next returns the first offset at or after from that find accepts, or -1.
// find must return the first accepted offset at or after its argument, with
// acceptance independent of the argument.
func (c *closerCursor) next(from int, find func(from int) int) int {
	if c.valid && from >= c.from && (c.at < 0 || c.at >= from) {
		return c.at
	}
	c.from, c.at, c.valid = from, find(from), true
	return c.at
}

// quotesFindAll returns the byte ranges of the quoted regions of text, the
// successive leftmost-first matches of the alternation of quoteAlternatives.
//
// A regular expression cannot express this in Go: the guarded alternatives
// need Unicode word boundaries (\b and \B), which RE2 only supports for ASCII.
// The matcher reproduces the leftmost-first semantics of the alternation
// exactly: at each start position the alternatives are tried in order, a
// lazy repetition takes the nearest closer that completes the match, and the
// next search resumes at the end of the previous match.
func quotesFindAll(text string) [][2]int {
	alts := quoteAlternatives()
	firstBytes := quoteOpenerFirstBytes()
	cursors := make([]closerCursor, len(alts))

	var out [][2]int
	for s := 0; s < len(text); {
		end := -1
		if firstBytes[text[s]] {
			for i := range alts {
				if end = alts[i].matchAt(text, s, &cursors[i]); end >= 0 {
					break
				}
			}
		}
		if end < 0 {
			s++
			continue
		}
		out = append(out, [2]int{s, end})
		s = end
	}
	return out
}

// matchAt returns the end of the alternative's match starting at s, or -1.
func (a quoteAlternative) matchAt(text string, s int, cursor *closerCursor) int {
	open, closer := a.pair.open, a.pair.close
	if !strings.HasPrefix(text[s:], open) {
		return -1
	}
	content := s + len(open)

	switch a.kind {
	case quotePlain:
		q := cursor.next(content, func(from int) int { return indexFrom(text, closer, from) })
		if q < 0 {
			return -1
		}
		return q + len(closer)

	case quoteGuarded:
		if !isNotWordBoundary(text, s) || isNotWordBoundary(text, content) {
			return -1
		}
		// The character class consumes at least one character, so the closer
		// starts one byte past the content start at the earliest.
		q := cursor.next(content+1, func(from int) int { return guardedCloser(text, closer, from) })
		if q < 0 {
			return -1
		}
		return q + len(closer)

	case quoteLineAnchored:
		if s > 0 && text[s-1] != '\n' {
			return -1
		}
		q := cursor.next(content+1, func(from int) int { return lineEndCloser(text, closer, from) })
		if q < 0 {
			return -1
		}
		// The lazy repetition cannot cross a newline, and needs one character.
		if nl := strings.IndexByte(text[content:], '\n'); nl >= 0 && content+nl < q {
			return -1
		}
		return q + len(closer)
	}
	return -1
}

// indexFrom returns the offset of the first occurrence of token in text at or
// after from, or -1.
func indexFrom(text, token string, from int) int {
	if from > len(text) {
		return -1
	}
	i := strings.Index(text[from:], token)
	if i < 0 {
		return -1
	}
	return from + i
}

// guardedCloser returns the first offset q at or after from where closer
// completes a guarded quote: the character before q is neither whitespace nor
// the closer, and no word boundary follows the closer.
func guardedCloser(text, closer string, from int) int {
	for q := indexFrom(text, closer, from); q >= 0; q = indexFrom(text, closer, q+1) {
		before, _ := lastRune(text[:q])
		if isWhitespace(before) || strings.ContainsRune(closer, before) {
			continue
		}
		if isNotWordBoundary(text, q+len(closer)) {
			return q
		}
	}
	return -1
}

// lineEndCloser returns the first offset q at or after from where closer ends
// a line: it is followed by the end of the text or a newline.
func lineEndCloser(text, closer string, from int) int {
	for q := indexFrom(text, closer, from); q >= 0; q = indexFrom(text, closer, q+1) {
		after := q + len(closer)
		if after == len(text) || text[after] == '\n' {
			return q
		}
	}
	return -1
}

// isNotWordBoundary reports \B at byte offset i of text: the characters on
// both sides are both word characters or both not. The outside of the text
// counts as not a word character.
func isNotWordBoundary(text string, i int) bool {
	before, ok := lastRune(text[:i])
	wordBefore := ok && isWordChar(before)
	after, ok := firstRune(text[i:])
	wordAfter := ok && isWordChar(after)
	return wordBefore == wordAfter
}
