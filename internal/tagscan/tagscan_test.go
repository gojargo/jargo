package tagscan_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/internal/tagscan"
)

// tagEvent is one tag the scanner reported.
type tagEvent struct {
	tag   string
	value string
}

// feedAll pushes each chunk through the scanner and returns the emitted text
// (including whatever Flush releases) plus the tags seen, in order.
func feedAll(s *tagscan.Scanner, chunks ...string) (string, []tagEvent) {
	var out strings.Builder
	var tags []tagEvent
	for _, c := range chunks {
		out.WriteString(s.Feed(c, func(tag, value string) {
			tags = append(tags, tagEvent{tag, value})
		}))
	}
	out.WriteString(s.Flush())
	return out.String(), tags
}

func TestFeedExtractsTags(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []string
		wantText string
		wantTags []tagEvent
	}{
		{
			name:     "no tags passes through",
			chunks:   []string{"hello there"},
			wantText: "hello there",
		},
		{
			name:     "single tag is extracted and stripped",
			chunks:   []string{"press <dtmf>1</dtmf> now"},
			wantText: "press  now",
			wantTags: []tagEvent{{"dtmf", "1"}},
		},
		{
			name:     "several tags in one chunk, in order",
			chunks:   []string{"<dtmf>1</dtmf>then<dtmf>2</dtmf>"},
			wantText: "then",
			wantTags: []tagEvent{{"dtmf", "1"}, {"dtmf", "2"}},
		},
		{
			name:     "empty tag value",
			chunks:   []string{"a<dtmf></dtmf>b"},
			wantText: "ab",
			wantTags: []tagEvent{{"dtmf", ""}},
		},
		{
			name:     "unknown tag is left in the text",
			chunks:   []string{"say <bold>hi</bold>"},
			wantText: "say <bold>hi</bold>",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, tags := feedAll(tagscan.New("dtmf"), tt.chunks...)
			if text != tt.wantText {
				t.Errorf("text = %q, want %q", text, tt.wantText)
			}
			assertTags(t, tags, tt.wantTags)
		})
	}
}

// TestFeedAcrossChunks is the reason this package exists: a streamed LLM
// response splits tags at arbitrary token boundaries, and a tag must still be
// recognized once its pieces arrive.
func TestFeedAcrossChunks(t *testing.T) {
	tests := []struct {
		name     string
		chunks   []string
		wantText string
		wantTags []tagEvent
	}{
		{
			name:     "split mid-tag",
			chunks:   []string{"press <dt", "mf>1</dtm", "f> now"},
			wantText: "press  now",
			wantTags: []tagEvent{{"dtmf", "1"}},
		},
		{
			name:     "split one character at a time",
			chunks:   strings.Split("go <dtmf>5</dtmf> on", ""),
			wantText: "go  on",
			wantTags: []tagEvent{{"dtmf", "5"}},
		},
		{
			name:     "value split across chunks",
			chunks:   []string{"<voicemail>ye", "s</voicemail>"},
			wantText: "",
			wantTags: []tagEvent{{"voicemail", "yes"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			text, tags := feedAll(tagscan.New("dtmf", "voicemail"), tt.chunks...)
			if text != tt.wantText {
				t.Errorf("text = %q, want %q", text, tt.wantText)
			}
			assertTags(t, tags, tt.wantTags)
		})
	}
}

// TestFeedHoldsBackPartialTag checks text is withheld from the moment a '<'
// appears, so half a tag is never spoken aloud while the rest is still in
// flight.
func TestFeedHoldsBackPartialTag(t *testing.T) {
	s := tagscan.New("dtmf")

	if got := s.Feed("press <dt", noTags); got != "press " {
		t.Errorf("Feed = %q, want only the text before the partial tag", got)
	}
	// Still incomplete: nothing more may be emitted.
	if got := s.Feed("mf>9</dt", noTags); got != "" {
		t.Errorf("Feed = %q, want nothing while the tag is incomplete", got)
	}

	var seen []tagEvent
	got := s.Feed("mf> done", func(tag, value string) { seen = append(seen, tagEvent{tag, value}) })
	if got != " done" {
		t.Errorf("Feed = %q, want the text after the completed tag", got)
	}
	assertTags(t, seen, []tagEvent{{"dtmf", "9"}})
}

// TestFlushReleasesIncompleteTag checks that when a response ends mid-tag the
// held-back text is released rather than silently dropped.
func TestFlushReleasesIncompleteTag(t *testing.T) {
	s := tagscan.New("dtmf")

	if got := s.Feed("almost <dtm", noTags); got != "almost " {
		t.Errorf("Feed = %q", got)
	}
	if got := s.Flush(); got != "<dtm" {
		t.Errorf("Flush = %q, want the held-back partial tag", got)
	}
	// Flush empties the buffer, so a second call yields nothing.
	if got := s.Flush(); got != "" {
		t.Errorf("second Flush = %q, want empty", got)
	}
}

// TestScannerReusableAcrossResponses checks a flushed scanner starts clean, so
// one response cannot leak a partial tag into the next.
func TestScannerReusableAcrossResponses(t *testing.T) {
	s := tagscan.New("dtmf")
	s.Feed("leftover <dt", noTags)
	s.Flush()

	text, tags := feedAll(s, "<dtmf>3</dtmf>ok")
	if text != "ok" {
		t.Errorf("text = %q, want %q", text, "ok")
	}
	assertTags(t, tags, []tagEvent{{"dtmf", "3"}})
}

// TestMultipleTagNames checks a scanner built for several names matches each of
// them, which is how the voicemail detector distinguishes its verdicts.
func TestMultipleTagNames(t *testing.T) {
	text, tags := feedAll(
		tagscan.New("voicemail", "human", "conversation"),
		"<human></human>", "hi there", "<voicemail></voicemail>",
	)
	if text != "hi there" {
		t.Errorf("text = %q", text)
	}
	assertTags(t, tags, []tagEvent{{"human", ""}, {"voicemail", ""}})
}

// noTags is the callback for cases that expect no tag to fire.
func noTags(string, string) {}

func assertTags(t *testing.T, got, want []tagEvent) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("tags = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("tag %d = %v, want %v", i, got[i], want[i])
		}
	}
}
