// Package text normalizes written text into a form better suited to
// text-to-speech: it strips Markdown, expands numbers, currency, percentages,
// dates, units and acronyms into spoken words, spaces out phone-number digits,
// and spells email addresses. The transforms are plain string functions; a
// VoiceFormatter bundles a configurable ordered pipeline of them, in the shape
// the TTS base's text transformers take, so it is registered with
// SetTextTransformers and reshapes each unit on its way to the provider.
//
// Reshaping for the provider is not the same as filtering the text: what a
// synthesizer needs to see ("A P I", "forty-two dollars") is not what the
// conversation should record.
//
// The expansions target English (the num2words "en" conventions): most TTS
// providers already normalize other languages server-side.
package text

// Transform is a single text-normalization step. All the package's transforms
// have this shape, so they compose directly and a VoiceFormatter is just an
// ordered list of them.
type Transform func(text string) string
