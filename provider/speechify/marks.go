package speechify

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"strconv"
	"strings"
	"unicode"

	"github.com/gojargo/jargo/text/align"
)

// markSpan returns a mark's character span within the synthesized text, if it
// has one.
//
// start and end are offsets into the request's input. They are the authority
// on a mark's text: a mark's value is the word as Speechify pronounced it,
// which sometimes normalizes characters (typographic apostrophes become ASCII,
// for instance), and downstream word tracking matches words against the text
// that was sent, where a normalized word derails the match for the rest of the
// sentence.
func markSpan(mark map[string]any, text []rune) (int, int, bool) {
	start, okStart := integer(mark["start"])
	end, okEnd := integer(mark["end"])
	if okStart && okEnd && start >= 0 && start < end && end <= len(text) {
		return start, end, true
	}
	return 0, 0, false
}

// integer reads a JSON integer. A number written with a fraction or an
// exponent, a string, or anything else is not one.
func integer(v any) (int, bool) {
	n, ok := v.(json.Number)
	if !ok {
		return 0, false
	}
	i, err := strconv.Atoi(n.String())
	return i, err == nil
}

// number reads a JSON number, or 0 when v is none.
func number(v any) float64 {
	n, ok := v.(json.Number)
	if !ok {
		return 0
	}
	f, _ := n.Float64()
	return f
}

// pendingWord is the word an accumulator is still assembling: its span in the
// text and the time its first mark starts at.
type pendingWord struct {
	start, end int
	startTime  float64
}

// speechMarkAccumulator assembles a request's speech marks into whole words.
//
// Speechify marks intra-word punctuation separately, and the resulting run of
// marks can straddle the boundary between two speech.chunk events. Marks that
// abut in the source text (the next starting exactly where the previous
// ended) are joined into one word. A word is released as soon as the text shows
// it cannot continue (the next character is whitespace, or the text ends),
// which is every word not split this way.
//
// Example marks for "text-to-speech is well-known.", split across two events:
//
//	event 1: "text"  "-"  "to"  "-"
//	event 2: "speech"  "is"  "well"  "-"  "known."
//
// Which are assembled into:
//
//	["text-to-speech", "is", "well-known."]
type speechMarkAccumulator struct {
	text       []rune
	timeOffset float64
	pending    *pendingWord
	// endTime is the last mark's end, in seconds and without the offset
	// applied.
	endTime float64
}

// newSpeechMarkAccumulator builds an accumulator for the text sent for
// synthesis, which the marks index into. timeOffset is the seconds added to
// each word, carrying over the duration of the utterances already synthesized
// in this turn.
func newSpeechMarkAccumulator(text string, timeOffset float64) *speechMarkAccumulator {
	return &speechMarkAccumulator{text: []rune(text), timeOffset: timeOffset}
}

// add takes one event's speech marks, the speech_marks payload of a
// speech.chunk event (absent on an audio-only event), and returns the words
// they complete.
func (a *speechMarkAccumulator) add(speechMarks any) []align.WordTiming {
	marks, _ := speechMarks.([]any)
	var words []align.WordTiming
	for _, raw := range marks {
		mark, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		// The stream carries word marks; skip any other type Speechify may add.
		if kind, present := mark["type"]; present && kind != "word" {
			continue
		}

		a.endTime = max(a.endTime, number(mark["end_time"])/1000)
		startTime := a.timeOffset + number(mark["start_time"])/1000

		start, end, ok := markSpan(mark, a.text)
		if !ok {
			words = append(words, a.flush()...)
			value, _ := mark["value"].(string)
			if value = strings.TrimSpace(value); value != "" {
				words = append(words, align.WordTiming{Word: value, Offset: startTime})
			}
			continue
		}

		if a.pending != nil && a.pending.end == start {
			a.pending.end = end
		} else {
			words = append(words, a.flush()...)
			a.pending = &pendingWord{start: start, end: end, startTime: startTime}
		}

		if end >= len(a.text) || unicode.IsSpace(a.text[end]) {
			words = append(words, a.flush()...)
		}
	}
	return words
}

// flush releases the word still being assembled, if any.
func (a *speechMarkAccumulator) flush() []align.WordTiming {
	if a.pending == nil {
		return nil
	}
	p := a.pending
	a.pending = nil
	word := strings.TrimSpace(string(a.text[p.start:p.end]))
	if word == "" {
		return nil
	}
	return []align.WordTiming{{Word: word, Offset: p.startTime}}
}

// sseEvent is one server-sent event: its name and its decoded data, with the
// numbers kept as written.
type sseEvent struct {
	name string
	data map[string]any
}

// parseSSEEvent parses one server-sent events block, its lines newline-joined
// and without the terminating blank line. It reports false for a block that
// carries no decodable data.
func parseSSEEvent(block string) (sseEvent, bool) {
	var name string
	var dataLines []string
	for line := range strings.Lines(block) {
		line = strings.TrimRight(line, "\r\n")
		field, value, _ := strings.Cut(line, ":")
		if field == "" { // A comment line.
			continue
		}
		value = strings.TrimPrefix(value, " ")
		switch field {
		case "event":
			name = value
		case "data":
			dataLines = append(dataLines, value)
		}
	}
	if len(dataLines) == 0 {
		return sseEvent{}, false
	}
	dec := json.NewDecoder(strings.NewReader(strings.Join(dataLines, "\n")))
	dec.UseNumber()
	var data map[string]any
	if err := dec.Decode(&data); err != nil {
		slog.Warn("speechify tts: failed to parse SSE event", "err", err)
		return sseEvent{}, false
	}
	return sseEvent{name: name, data: data}, true
}

// readSSEEvents reads the server-sent events of a response, handing each to
// handle in turn. A line is as long as the audio it carries, so it is read
// whole rather than through a scanner's bounded buffer.
func readSSEEvents(r io.Reader, handle func(sseEvent) error) error {
	reader := bufio.NewReader(r)
	var block []string
	emit := func() error {
		if len(block) == 0 {
			return nil
		}
		event, ok := parseSSEEvent(strings.Join(block, "\n"))
		block = block[:0]
		if !ok {
			return nil
		}
		return handle(event)
	}
	for {
		raw, readErr := reader.ReadBytes('\n')
		if len(raw) > 0 {
			line := string(bytes.TrimRight(raw, "\r\n"))
			if line != "" {
				block = append(block, line)
			} else if err := emit(); err != nil {
				return err
			}
		}
		if errors.Is(readErr, io.EOF) {
			return emit()
		}
		if readErr != nil {
			return readErr
		}
	}
}
