// Package spoken tells the expression of the words a TTS service reports
// speaking.
//
// Part of expressive conversation (alpha). A TTS service that reports word
// timestamps tells which words it has spoken, and word tracking gives each word
// the stretch of the LLM's text it covers, tags included, as its raw text. A
// word covers the text from the end of the word before it, so tags written
// between two words arrive with the second.
//
// [SpokenExpression] keeps the text sent in one audio context, with the
// formatter.ExpressionSpan list formatter.ExpressionFormatter.Format returned
// for it. Looking up each spoken word's text in it gives the delivery the word
// is spoken in and the sounds performed before it, which the TTS service
// reports to clients as the word is spoken.
//
// Alpha: the tag syntax and these types are likely to change between releases.
package spoken

import (
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/gojargo/jargo/expressive/formatter"
)

// SpokenExpression tells the delivery and sounds of the words a TTS service
// reports speaking in one audio context.
//
// Each of the service's synthesis contexts holds one, so it lasts as long as
// its audio context. Units are added as they are sent, and words are looked up
// as they are spoken, in order: each lookup searches from where the last one
// ended.
//
//	s := spoken.New()
//	s.Add("<<warm>> It's [[laugh]] ^^back^^ up.", spans)
//
//	s.Spoken("<<warm>> It's")    // ["warm"], no sounds
//	s.Spoken("[[laugh]] ^^back") // ["warm", "emphasis"], the laugh
//	s.Spoken("up.")              // ["warm"], no sounds
//
// It is safe for concurrent use: units are added on the service's frame
// goroutine and words looked up as its audio plays.
type SpokenExpression struct {
	mu    sync.Mutex
	text  string
	spans []formatter.ExpressionSpan
	// reported is the sounds and pauses already reported, by their index in
	// the spans.
	reported map[int]bool
	// position is how far into the text the service has reported speaking.
	position int
	delivery []string
}

// New builds an empty SpokenExpression.
func New() *SpokenExpression {
	return &SpokenExpression{reported: map[int]bool{}}
}

// Delivery is the delivery of the text last reported spoken, and false when no
// text was added.
func (s *SpokenExpression) Delivery() ([]string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.text == "" {
		return nil, false
	}
	return deliveryOf(s.delivery), true
}

// Add keeps a unit sent to the service, as the LLM wrote it, for the words it
// reports later, with what each stretch of the unit performs. Units follow one
// another in the context's text, and the spans are moved to where the unit
// falls in it.
func (s *SpokenExpression) Add(unit string, spans []formatter.ExpressionSpan) {
	s.mu.Lock()
	defer s.mu.Unlock()
	offset := len(s.text)
	s.text += unit
	for _, span := range spans {
		span.Start += offset
		span.End += offset
		s.spans = append(s.spans, span)
	}
}

// Spoken reads the delivery of text the service reports speaking, and the
// sounds before it.
//
// text is the LLM's text it covers, tags included, such as a word's raw text,
// or a whole sentence's from a service without word timestamps. Its delivery
// is that of its first letter that falls in words, so a delivery tag's own
// name, such as "warm" in "<<warm>> It's", isn't taken for a word. A sound
// already reported by [SpokenExpression.SpokenMarkup] isn't reported again,
// and one written after the text's last word, such as the laugh in
// "back [[laugh]]!", is left for the service's markup or the next text to
// report, since it is performed after the word.
//
// It returns the delivery, and false when no text was added, and the sounds
// and pauses performed since the text before it. Text that can't be found
// keeps the delivery of the text before it.
func (s *SpokenExpression) Spoken(text string) ([]string, bool, []formatter.ReportedSound) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.text == "" || text == "" {
		return nil, false, nil
	}
	found := strings.Index(s.text[s.position:], text)
	if found < 0 {
		return deliveryOf(s.delivery), true, nil
	}
	start := s.position + found
	end := start + len(text)
	position, sounds := s.soundsBefore(end, s.lastWordIn(start, end))
	for i, r := range s.text[start:end] {
		if !isAlnum(r) {
			continue
		}
		if words := s.wordsAt(start + i); words != nil {
			s.delivery = deliveryOf(words.Delivery)
			break
		}
	}
	s.position = position
	return deliveryOf(s.delivery), true, sounds
}

// lastWordIn is where the last letter or digit of words between start and end
// is, or -1 when there is none.
func (s *SpokenExpression) lastWordIn(start, end int) int {
	for i := end; i > start; {
		r, size := utf8.DecodeLastRuneInString(s.text[start:i])
		i -= size
		if isAlnum(r) && s.inWords(i) {
			return i
		}
	}
	return -1
}

// soundsBefore reports the sounds and pauses performed since the last text
// spoken, up to end, and where the next lookup starts. A sound after the text's
// last word, at lastWord, is left for what follows.
func (s *SpokenExpression) soundsBefore(end, lastWord int) (int, []formatter.ReportedSound) {
	position := end
	var sounds []formatter.ReportedSound
	for i, span := range s.spans {
		if span.Sound == nil || s.reported[i] || span.Start < s.position || span.Start >= end {
			continue
		}
		if lastWord >= 0 && span.Start > lastWord {
			position = min(position, span.Start)
			continue
		}
		s.reported[i] = true
		sounds = append(sounds, *span.Sound)
	}
	return position, sounds
}

// Unreported reads the sounds and pauses not yet reported, once the context's
// audio has ended. A sound after the last words, such as the laugh in
// "Bye [[laugh]]", has no later text to report it, and a service may not
// report its markup.
func (s *SpokenExpression) Unreported() []formatter.ReportedSound {
	s.mu.Lock()
	defer s.mu.Unlock()
	var sounds []formatter.ReportedSound
	for i, span := range s.spans {
		if span.Sound != nil && !s.reported[i] && span.Start >= s.position {
			s.reported[i] = true
			sounds = append(sounds, *span.Sound)
		}
	}
	return sounds
}

// SpokenMarkup reads markup the service reports speaking as a word, such as
// "[laughs]", and false when the markup stands for no sound.
//
// Some services include their markup in their word timestamps. It covers none
// of the LLM's text, so it has no raw text, and is matched by position instead:
// it stands for the next sound or pause still to be spoken, which is reported
// with it, at its time, as written in its tag. Markup reached before any
// sound, such as a delivery's "[warmly]", stands for none, and isn't spoken:
//
//	s.Spoken("<<warm>> It's")
//	s.SpokenMarkup()             // "[laughs]" reported as a word: the laugh
//	s.Spoken("[[laugh]] ^^back") // ["warm", "emphasis"], already reported
func (s *SpokenExpression) SpokenMarkup() (formatter.ReportedSound, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, span := range s.spans {
		if span.End <= s.position || s.reported[i] {
			continue
		}
		if span.Sound != nil {
			s.reported[i] = true
			return *span.Sound, true
		}
		words := s.text[max(s.position, span.Start):span.End]
		if strings.IndexFunc(words, isAlnum) >= 0 {
			return formatter.ReportedSound{}, false
		}
	}
	return formatter.ReportedSound{}, false
}

// inWords reports whether the character at i is a letter or digit of words,
// not of a tag.
func (s *SpokenExpression) inWords(i int) bool {
	return s.wordsAt(i) != nil
}

// wordsAt is the span of words the byte at i falls in, or nil.
func (s *SpokenExpression) wordsAt(i int) *formatter.ExpressionSpan {
	for j := range s.spans {
		span := &s.spans[j]
		if span.Delivery != nil && span.Start <= i && i < span.End {
			return span
		}
	}
	return nil
}

// deliveryOf is a delivery as reported, never nil: an empty one is the
// baseline.
func deliveryOf(d []string) []string {
	return append([]string{}, d...)
}

func isAlnum(r rune) bool { return unicode.IsLetter(r) || unicode.IsNumber(r) }
