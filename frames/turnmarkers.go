package frames

import "strings"

// The markers a model is instructed to begin each response with when its
// replies are gated on whether the user had finished speaking.
//
// They travel on an LLMMarkerFrame and are written into the conversation with
// the reply they prefix, so the model can see its own earlier verdicts. They are
// protocol rather than prose: anything building a transcript for people strips
// them out.
// The fill level tracks how much of the user's turn has arrived: full is a
// finished turn, half is one cut off mid-thought, empty is a user who has not
// started answering. Each is a single token in every major tokenizer, which
// matters because the complete marker is generated ahead of any speakable text.
const (
	// UserTurnCompleteMarker means the user's turn was complete; answer
	// normally.
	UserTurnCompleteMarker = "●"
	// UserTurnIncompleteShortMarker means the user was cut off and will likely
	// continue within seconds.
	UserTurnIncompleteShortMarker = "◐"
	// UserTurnIncompleteLongMarker means the user needs longer to think.
	UserTurnIncompleteLongMarker = "○"
)

// userTurnMarkers is every marker of the protocol, for stripping them out.
//
//nolint:gochecknoglobals // a fixed set, read only
var userTurnMarkers = []string{
	UserTurnCompleteMarker,
	UserTurnIncompleteShortMarker,
	UserTurnIncompleteLongMarker,
}

// StripUserTurnMarkers removes the turn-completion markers from text, and trims
// the whitespace they leave behind.
//
// The markers prefix a reply so the model can see its own earlier verdicts, but
// they are not something the bot said. Anything reporting a turn as a transcript
// therefore strips them, while what is written to the conversation keeps them.
//
// The defaults are removed, along with any extra markers given: a service
// configured with a set of its own has those stripped too.
//
// Text carrying no marker is returned exactly as it came, whitespace included:
// trimming it would change a reply that nothing was stripped from.
func StripUserTurnMarkers(text string, extra ...string) string {
	found := false
	for _, marker := range append(userTurnMarkers, extra...) {
		if marker == "" {
			continue
		}
		if strings.Contains(text, marker) {
			text = strings.ReplaceAll(text, marker, "")
			found = true
		}
	}
	if !found {
		return text
	}
	return strings.TrimSpace(text)
}
