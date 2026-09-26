package sentencex

import (
	"strings"
	"sync"
)

// Some abbreviations, like a.m., B.C., Ph.D., 5 ft. and NE., cannot always
// suppress a break. They usually trail a digit or a name, and breaking a
// sentence at them correctly needs some semantic context. These definitions
// give such marker words their own handling policies, as targeted carve-outs.
//
// Suppose p.m has a policy that breaks before an uppercase follower:
//   - "In the evening at 7 p.m. Tom wakes up." does not split: "In the
//     evening at" are all fronting (adverbial) words, so the marker reads as
//     mid-phrase;
//   - "The sun sets at 7 p.m. Tom wakes up then." splits into "The sun sets at
//     7 p.m. " and "Tom wakes up then.": sun is lowercase and not a fronting
//     word, and Tom is an uppercase follower, so the break stands.

// markerPolicy describes how a trailing marker is handled. See
// data/trailing_markers/en.txt for examples.
type markerPolicy struct {
	// digitBreaks is whether a digit follower forces a sentence break.
	digitBreaks bool
	// uppercaseBreaks is whether an uppercase non-starter follower forces a
	// sentence break. It is not useful for English, but might be for other
	// languages like German.
	uppercaseBreaks bool
}

// markerDef is one trailing marker and its policy.
type markerDef struct {
	matcher suffixMatcher
	policy  markerPolicy
}

// suffixMatcher describes how a marker's suffix is recognized in the tail.
type suffixMatcher struct {
	suffix     string
	ignoreCase bool
	digitOnly  bool
}

// strip returns the text before the suffix when head ends with it, and false
// otherwise. Case-sensitive markers compare exactly; case-insensitive ones
// compare the tail bytes ignoring ASCII case.
func (m *suffixMatcher) strip(head string) (string, bool) {
	if m.ignoreCase {
		idx := len(head) - len(m.suffix)
		if idx < 0 || !equalFoldASCII(head[idx:], m.suffix) {
			return "", false
		}
		return head[:idx], true
	}
	return strings.CutSuffix(head, m.suffix)
}

// equalFoldASCII reports whether a and b are equal ignoring ASCII case.
func equalFoldASCII(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range len(a) {
		if asciiLower(a[i]) != asciiLower(b[i]) {
			return false
		}
	}
	return true
}

func asciiLower(b byte) byte {
	if b >= 'A' && b <= 'Z' {
		return b + 'a' - 'A'
	}
	return b
}

// markerTable is a language's trailing-marker lookup table.
type markerTable struct {
	markers []markerDef
	// twoByteFilter rejects tails whose final two bytes match no marker, so
	// the linear scan over markers runs only for real candidates.
	twoByteFilter twoByteFilter
}

// emptyMarkerTable is the table of a language with no trailing markers.
//
//nolint:gochecknoglobals // built once
var emptyMarkerTable = sync.OnceValue(func() *markerTable { return &markerTable{} })

// buildMarkerTable builds the lookup table of markers.
func buildMarkerTable(markers []markerDef) *markerTable {
	return &markerTable{
		twoByteFilter: buildTwoByteFilter(markers),
		markers:       markers,
	}
}

// twoByteKey is the folded (lowercased) last-two-byte lookup key. Folding
// makes the filter case-insensitive; case-sensitive markers are checked
// precisely by strip.
func twoByteKey(secondLast, last byte) int {
	return int(asciiLower(secondLast))<<8 | int(asciiLower(last))
}

// twoByteFilter is a 65536-bit set of twoByteKey values.
type twoByteFilter [1024]uint64

func (f *twoByteFilter) insert(key int) {
	f[key>>6] |= 1 << (key & 63)
}

func (f *twoByteFilter) contains(key int) bool {
	return f[key>>6]&(1<<(key&63)) != 0
}

// buildTwoByteFilter marks the folded last two bytes of every marker, or every
// pair ending in the last byte of a one-byte marker.
func buildTwoByteFilter(markers []markerDef) twoByteFilter {
	var filter twoByteFilter

	for _, marker := range markers {
		suffix := marker.matcher.suffix
		last := suffix[len(suffix)-1]

		if len(suffix) >= 2 {
			filter.insert(twoByteKey(suffix[len(suffix)-2], last))
			continue
		}
		for hi := range 256 {
			filter.insert(twoByteKey(byte(hi), last))
		}
	}

	return filter
}

// markerMatch is a trailing marker found at the end of a head: the text before
// it and its definition.
type markerMatch struct {
	prefix string
	def    *markerDef
}

// stripMarkerSuffix finds the first marker headTrimmed ends with. The suffix
// must be preceded by whitespace, a digit or the start of the text, so
// "clause 5.a.m" does not match a.m.
func stripMarkerSuffix(headTrimmed string, markers []markerDef) (markerMatch, bool) {
	for i := range markers {
		marker := &markers[i]
		prefix, ok := marker.matcher.strip(headTrimmed)
		if !ok {
			continue
		}

		var predecessorOK bool
		if marker.matcher.digitOnly {
			last, ok := lastRune(trimEnd(prefix))
			predecessorOK = ok && isASCIIDigit(last)
		} else {
			last, ok := lastRune(prefix)
			predecessorOK = !ok || isWhitespace(last) || isASCIIDigit(last)
		}

		if predecessorOK {
			return markerMatch{prefix: prefix, def: marker}, true
		}
	}
	return markerMatch{}, false
}

// classifyTrailingMarker returns the trailing marker head ends with, and false
// when there is none.
func classifyTrailingMarker(head string, table *markerTable) (markerMatch, bool) {
	trimmed := trimEnd(head)

	// A marker is at least two bytes, so a shorter head cannot match one.
	// Reject on the folded last two bytes: words whose final pair matches no
	// marker never reach the scan.
	if len(trimmed) < 2 {
		return markerMatch{}, false
	}

	key := twoByteKey(trimmed[len(trimmed)-2], trimmed[len(trimmed)-1])
	if !table.twoByteFilter.contains(key) {
		return markerMatch{}, false
	}

	return stripMarkerSuffix(trimmed, table.markers)
}

// markerBypassesSuppression decides whether a matched marker breaks the
// sentence (true) or keeps suppressing the break (false), from the first
// character of the next word:
//   - a single-letter capital initial (P., J. R. R.) and a lowercase letter or
//     punctuation never break;
//   - a digit defers to the policy's digitBreaks;
//   - an uppercase follower breaks when it is a sentence starter, or when
//     uppercaseBreaks holds and fronting does not veto it.
func markerBypassesSuppression(marker markerMatch, nextWord string, nextIsStarter bool, l *language) bool {
	policy := marker.def.policy
	rest := trimStart(nextWord)
	first, ok := firstRune(rest)

	// A single-letter capital initial (P.D.T., J. R. R.) continues.
	if ok && isASCIIUpper(first) {
		if second, hasSecond := firstRune(rest[1:]); hasSecond && second == '.' {
			return false
		}
	}

	switch {
	case !ok:
		return false
	case isASCIIDigit(first):
		return policy.digitBreaks
	case isUppercase(first):
		return nextIsStarter ||
			(policy.uppercaseBreaks &&
				!wordBeforeMarkerIsCapitalised(marker.prefix) &&
				!prefixIsPurelyFronting(marker.prefix, l))
	default:
		return false
	}
}
