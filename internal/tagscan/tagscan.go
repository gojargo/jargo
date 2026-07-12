// Package tagscan extracts control tags like <dtmf>1</dtmf> from a streamed text
// response, so a processor can act on the tags and speak the rest. It buffers
// across chunks, since a tag may span several, and holds back a trailing partial
// tag until it completes.
package tagscan

import (
	"regexp"
	"strings"
)

// Scanner extracts <tag>value</tag> control tags for a fixed set of tag names
// from a stream of text chunks, returning the text with the tags removed.
type Scanner struct {
	re  *regexp.Regexp
	buf string
}

// New builds a Scanner recognizing the given tag names.
func New(tags ...string) *Scanner {
	alt := strings.Join(tags, "|")
	return &Scanner{re: regexp.MustCompile(`<(` + alt + `)>([^<]*)</(?:` + alt + `)>`)}
}

// Feed appends chunk, invokes onTag for each complete tag found (in order), and
// returns the text safe to emit now: tag content removed, with a trailing
// partial tag held back until a later Feed or Flush completes it.
func (s *Scanner) Feed(chunk string, onTag func(tag, value string)) string {
	s.buf += chunk
	for {
		loc := s.re.FindStringSubmatchIndex(s.buf)
		if loc == nil {
			break
		}
		onTag(s.buf[loc[2]:loc[3]], s.buf[loc[4]:loc[5]])
		s.buf = s.buf[:loc[0]] + s.buf[loc[1]:]
	}
	// Hold back from the first '<', which may begin an incomplete tag.
	if i := strings.IndexByte(s.buf, '<'); i >= 0 {
		out := s.buf[:i]
		s.buf = s.buf[i:]
		return out
	}
	out := s.buf
	s.buf = ""
	return out
}

// Flush returns any held-back text and clears the buffer. Call it at the end of
// a response, when no more chunks will complete a partial tag.
func (s *Scanner) Flush() string {
	out := s.buf
	s.buf = ""
	return out
}
