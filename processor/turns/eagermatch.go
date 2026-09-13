package turns

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// EagerMatchPolicy decides whether a speculative reply is still the right
// answer.
//
// A speculative reply is generated from an eager transcript, before the service
// commits one. This decides whether the committed transcript is close enough to
// the eager one for that reply to stand.
type EagerMatchPolicy interface {
	// Matches reports whether the committed transcript still matches the eager
	// one: true to keep the speculative reply, false to discard it and answer
	// the committed transcript instead.
	Matches(eager, final string) bool
}

// ExactMatch keeps the reply only when the two transcripts are identical.
type ExactMatch struct{}

// Matches reports whether the two transcripts are identical.
func (ExactMatch) Matches(eager, final string) bool { return eager == final }

// NormalizedMatch keeps the reply when the transcripts differ only in
// formatting.
//
// It compares with case, punctuation and whitespace removed. Services commonly
// format the committed transcript, capitalizing it and punctuating it, while
// leaving the eager one raw, which ExactMatch counts as a difference.
type NormalizedMatch struct{}

// Matches reports whether the two transcripts match once formatting is removed.
func (NormalizedMatch) Matches(eager, final string) bool {
	return normalizeForMatch(eager) == normalizeForMatch(final)
}

// normalizeForMatch strips everything a service may add when it commits a
// transcript: a different unicode composition, capitalization, punctuation and
// the spacing around it.
//
// Apostrophes are dropped rather than spaced out, so a service writing "it's"
// matches one writing "its".
func normalizeForMatch(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	for _, r := range norm.NFKC.String(text) {
		switch {
		case r == '\'' || r == '’' || r == 'ʼ':
			// Dropped outright, joining the letters either side of it.
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_':
			b.WriteRune(unicode.ToLower(r))
		default:
			// Punctuation and whitespace alike become one separator, collapsed
			// below.
			b.WriteRune(' ')
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}
