// Package tts holds how a TTS service performs expression tags, for an
// expressive conversation.
//
// Each TTS service has an expression adapter, an [ExpressionAdapter], the way
// each LLM service has an LLM adapter. The adapter declares what the service
// can perform and formats it into the text sent to the service: words in their
// delivery, and sounds and pauses. The base adapter performs nothing, so a
// service without one speaks only the words. Each provider's adapter lives in a
// package under this one.
//
// Alpha: the tag syntax and these types are likely to change between releases.
package tts

import "github.com/gojargo/jargo/expressive"

// ExpressionAdapter formats expression for one TTS service.
//
// Its capabilities are what the service can perform: the service broadcasts
// them at start, and reads and formats tags by them. The service calls the
// adapter only with what they say it supports: other delivery is taken off the
// words first, and other sounds are never formatted. Each name reaches the
// adapter as the service's own: a universal name is translated by the
// capabilities' ServiceVocabulary first, so "warm" reaches ElevenLabs' adapter
// as "warmly".
//
// Markup is set apart from the words by spaces, so the words a service with
// word timestamps reports line up with the text it was sent.
type ExpressionAdapter interface {
	// Capabilities is the expression the service can perform, or nil for none.
	// The service broadcasts it at start.
	Capabilities() *expressive.ExpressionCapabilities
	// FormatText formats words to speak in their delivery. text is the words as
	// prepared for the service; delivery is the supported delivery to speak them
	// in, by the service's names, empty for the voice's baseline;
	// previousDelivery is the delivery of the words before them in the
	// response, or nil for its first words, so markup can be emitted only where
	// the delivery changes. It returns the text to send to the service.
	FormatText(text string, delivery, previousDelivery []string) string
	// RequestSettings is the settings to send beside a request's text, for its
	// delivery, or nil for none. It is for a service that takes some of its
	// expression as request settings rather than as markup in the text, such
	// as a speed. The service merges them into its request; each adapter
	// documents where. delivery is the supported delivery the request's first
	// words are spoken in, by the service's names, empty for the baseline.
	RequestSettings(delivery []string) map[string]any
	// FormatEnd closes a sentence's request, given the delivery its last words
	// are spoken in, by the service's names. It is for a service whose markup
	// lasts past the request it is sent in, so the request can set the delivery
	// back for the text after it. It returns the text to end the request with,
	// or "".
	FormatEnd(delivery []string) string
	// FormatSound formats a supported sound, such as a laugh, by the service's
	// name for it, at its place in the text.
	FormatSound(name string) string
	// FormatPause formats a pause of durationMS milliseconds, within the
	// service's pause range, at its place in the text.
	FormatPause(durationMS int) string
}

// BaseExpressionAdapter performs nothing. A provider's adapter embeds it and
// overrides what its service performs.
type BaseExpressionAdapter struct{}

// Capabilities implements [ExpressionAdapter]: none.
func (BaseExpressionAdapter) Capabilities() *expressive.ExpressionCapabilities { return nil }

// FormatText implements [ExpressionAdapter]: the words as they are.
func (BaseExpressionAdapter) FormatText(text string, _, _ []string) string { return text }

// RequestSettings implements [ExpressionAdapter]: none.
func (BaseExpressionAdapter) RequestSettings([]string) map[string]any { return nil }

// FormatEnd implements [ExpressionAdapter]: nothing.
func (BaseExpressionAdapter) FormatEnd([]string) string { return "" }

// FormatSound implements [ExpressionAdapter]: nothing.
func (BaseExpressionAdapter) FormatSound(string) string { return "" }

// FormatPause implements [ExpressionAdapter]: nothing.
func (BaseExpressionAdapter) FormatPause(int) string { return "" }

var _ ExpressionAdapter = BaseExpressionAdapter{}

// DeliveryTags is a delivery's names as a provider's tags, in order, for an
// adapter that writes each as one. Emphasis is left out, since such a provider
// stresses words its own way. renames is the provider's tag for a name it calls
// something else, such as "slowly" for "slow".
func DeliveryTags(delivery []string, renames map[string]string) []string {
	var tags []string
	for _, name := range delivery {
		if name == expressive.EmphasisDelivery {
			continue
		}
		if renamed, ok := renames[name]; ok {
			name = renamed
		}
		tags = append(tags, name)
	}
	return tags
}
