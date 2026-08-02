// Package text normalizes written text into a form better suited to
// text-to-speech: it strips Markdown, expands numbers, currency, percentages,
// dates, units and acronyms into spoken words, spaces out phone-number digits,
// and spells email addresses. The transforms are plain string functions; a
// VoiceFormatter bundles a configurable ordered pipeline of them behind the
// Filter interface, which the TTS base applies to each sentence before synthesis.
//
// The expansions target English (the num2words "en" conventions): most TTS
// providers already normalize other languages server-side.
package text

// Filter transforms text before it is handed to a speech synthesizer. The TTS
// base calls Filter on each complete sentence just before synthesizing it.
type Filter interface {
	Filter(text string) string
}

// InterruptibleFilter is a Filter that carries state between calls and needs to
// be told when speech is cut off. Markdown structures such as code blocks and
// tables arrive split across several sentences, so a filter tracking one has to
// abandon it when the text that would have closed it never arrives. The TTS base
// calls HandleInterruption on an interruption, and ResetInterruption before
// filtering resumes.
type InterruptibleFilter interface {
	Filter
	HandleInterruption()
	ResetInterruption()
}

// Transform is a single text-normalization step. All the package's transforms
// have this shape, so they compose directly and a VoiceFormatter is just an
// ordered list of them.
type Transform func(text string) string

// Filter lets a bare Transform satisfy the Filter interface.
func (t Transform) Filter(text string) string { return t(text) }
