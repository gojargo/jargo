package text

// FormatterOptions configures which transforms a VoiceFormatter applies. The
// zero value enables nothing; use DefaultFormatterOptions for the recommended
// set and toggle individual fields from there.
type FormatterOptions struct {
	// StripMarkdown removes Markdown formatting symbols.
	StripMarkdown bool
	// EmailToSpeech spells email addresses ("a@b.com" → "a at b dot com").
	EmailToSpeech bool
	// ExpandPhoneNumbers spaces out phone-number digits.
	ExpandPhoneNumbers bool
	// NormalizeDates expands ISO and US dates into spoken form.
	NormalizeDates bool
	// ExpandCurrency expands currency amounts ("$5" → "five dollars").
	ExpandCurrency bool
	// ExpandPercentages expands percentages ("50%" → "fifty percent").
	ExpandPercentages bool
	// ExpandUnits expands unit abbreviations ("5km" → "5 kilometers").
	ExpandUnits bool
	// NormalizeAcronyms letter-spaces all-caps acronyms ("API" → "A P I").
	NormalizeAcronyms bool
	// ExpandNumbers spells numeric digits as words. Off by default, since many
	// numbers (versions, codes, IDs) read better as digits.
	ExpandNumbers bool
	// NumberDigitCutoff is passed to ExpandNumbers when it is enabled: numbers
	// above it are read digit-by-digit. Zero or less spells every number.
	NumberDigitCutoff int
	// CustomReplacements are {pattern, replacement} regex rules applied last.
	CustomReplacements [][2]string
}

// DefaultFormatterOptions returns the recommended set: every transform except
// ExpandNumbers (which can mangle numbers better left as digits).
func DefaultFormatterOptions() FormatterOptions {
	return FormatterOptions{
		StripMarkdown:      true,
		EmailToSpeech:      true,
		ExpandPhoneNumbers: true,
		NormalizeDates:     true,
		ExpandCurrency:     true,
		ExpandPercentages:  true,
		ExpandUnits:        true,
		NormalizeAcronyms:  true,
		ExpandNumbers:      false,
	}
}

// VoiceFormatter applies an ordered pipeline of text transforms, implementing
// Filter. Build one with NewVoiceFormatter.
type VoiceFormatter struct {
	transforms []Transform
}

// NewVoiceFormatter builds a VoiceFormatter from opts. The transforms run in a
// deliberate order (structural cleanup, then language expansions, then user
// replacements) chosen so earlier steps do not hide patterns later ones match:
// email before phone and acronyms, units before acronyms. It returns an error
// only if a CustomReplacements pattern fails to compile.
func NewVoiceFormatter(opts FormatterOptions) (*VoiceFormatter, error) {
	var ts []Transform
	if opts.StripMarkdown {
		ts = append(ts, StripMarkdown)
	}
	// Email must run before phone (its digit-only domains match the phone
	// pattern) and before acronyms (all-caps local parts would be letter-spaced).
	if opts.EmailToSpeech {
		ts = append(ts, EmailToSpeech)
	}
	if opts.ExpandPhoneNumbers {
		ts = append(ts, ExpandPhoneNumbers)
	}
	if opts.NormalizeDates {
		ts = append(ts, NormalizeDates)
	}
	if opts.ExpandCurrency {
		ts = append(ts, ExpandCurrency)
	}
	if opts.ExpandPercentages {
		ts = append(ts, ExpandPercentages)
	}
	// Units before acronyms: uppercase abbreviations like "MB" or "MPH" would
	// otherwise be letter-spaced and no longer recognized.
	if opts.ExpandUnits {
		ts = append(ts, ExpandUnits)
	}
	if opts.NormalizeAcronyms {
		ts = append(ts, NormalizeAcronyms)
	}
	if opts.ExpandNumbers {
		ts = append(ts, ExpandNumbers(opts.NumberDigitCutoff))
	}
	if len(opts.CustomReplacements) > 0 {
		r, err := ReplaceText(opts.CustomReplacements)
		if err != nil {
			return nil, err
		}
		ts = append(ts, r)
	}
	return &VoiceFormatter{transforms: ts}, nil
}

// Filter applies every configured transform to text in order.
func (f *VoiceFormatter) Filter(text string) string {
	for _, t := range f.transforms {
		text = t(text)
	}
	return text
}
