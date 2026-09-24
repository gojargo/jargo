package text_test

import (
	"fmt"
	"regexp"
	"strings"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

// aggregateChunks feeds chunks to a sentence aggregator and returns every
// sentence it completed, including what Flush returns at the end.
func aggregateChunks(t *testing.T, a *text.SimpleAggregator, chunks []string) []string {
	t.Helper()
	var got []string
	for _, c := range chunks {
		for _, agg := range a.Aggregate(c) {
			got = append(got, agg.Text)
		}
	}
	if tail, ok := a.Flush(); ok {
		got = append(got, tail.Text)
	}
	if _, ok := a.Flush(); ok {
		t.Fatal("a second Flush returned text")
	}
	return got
}

// chunkBy splits s into chunks of size runes.
func chunkBy(s string, size int) []string {
	rs := []rune(s)
	var out []string
	for i := 0; i < len(rs); i += size {
		out = append(out, string(rs[i:min(i+size, len(rs))]))
	}
	return out
}

// Where a stream is cut into chunks does not change the sentences it is
// aggregated into, including around an ellipsis and repeated punctuation.
func TestSimpleAggregatorChunkingDoesNotChangeSentences(t *testing.T) {
	words := regexp.MustCompile(`\S+\s*`)
	cases := [][]string{
		{"Dr. Smith is here.", "It costs $29.95."},
		{"Albert I. Jones is here.", "Hello."},
		{"Hello!?", "Next."},
		{"That's strange... but possible.", "Let's check."},
		{"No!!!", "Stop!"},
		{"Your IP address is 192.168.1.1.", "Enter it in the browser."},
	}
	for _, sentences := range cases {
		joined := strings.Join(sentences, " ")
		chunkings := map[string][]string{
			"whole": {joined},
			"word":  words.FindAllString(joined, -1),
			"1":     chunkBy(joined, 1),
			"3":     chunkBy(joined, 3),
		}
		for name, chunks := range chunkings {
			t.Run(fmt.Sprintf("%s/%s", sentences[0], name), func(t *testing.T) {
				got := aggregateChunks(t, newAggregator(t, frames.AggregationSentence), chunks)
				if strings.Join(got, "|") != strings.Join(sentences, "|") {
					t.Fatalf("sentences = %q, want %q", got, sentences)
				}
			})
		}
	}
}

// An ellipsis is one mark: its later dots are not lookahead, so the
// aggregator never cuts a sentence between them, and never emits a chunk of
// punctuation alone. Whether the ellipsis then ends the sentence is the
// tokenizer's call.
func TestSimpleAggregatorKeepsAnEllipsisWhole(t *testing.T) {
	words := regexp.MustCompile(`\S+\s*`)
	for _, input := range []string{
		"Wait... Then continue.",
		"Oh, they have no respect at all... Had you spent long choosing them?",
		"Oh, they have no respect at all… Had you spent long choosing them?",
	} {
		whole := aggregateChunks(t, newAggregator(t, frames.AggregationSentence), []string{input})
		for name, chunks := range map[string][]string{
			"word": words.FindAllString(input, -1),
			"1":    chunkBy(input, 1),
			"3":    chunkBy(input, 3),
		} {
			got := aggregateChunks(t, newAggregator(t, frames.AggregationSentence), chunks)
			if strings.Join(got, "|") != strings.Join(whole, "|") {
				t.Errorf("%q by %s = %q, want %q as when whole", input, name, got, whole)
			}
		}
		for _, sentence := range whole {
			if strings.Contains(sentence, "..") && !strings.Contains(sentence, "...") {
				t.Errorf("%q: sentence %q cuts the ellipsis", input, sentence)
			}
			if strings.Trim(sentence, ".…!?; ") == "" {
				t.Errorf("%q: sentence %q has nothing to speak", input, sentence)
			}
		}
		if strings.Join(whole, " ") != input {
			t.Errorf("%q aggregated to %q, want the text unchanged", input, whole)
		}
	}
}

// wordEndTokenizer finds a boundary after "I." only once the word after it has
// ended, the case a tokenizer cannot settle at the first character.
func wordEndTokenizer() *recordingTokenizer {
	return &recordingTokenizer{boundary: func(s string) int {
		i := strings.Index(s, "I. ")
		if i < 0 || !strings.ContainsAny(s[i+3:], " ?") {
			return 0
		}
		return i + len("I.")
	}}
}

// A boundary the tokenizer cannot resolve at the first character after it is
// retried when the following word ends, and never at a partial word.
func TestSimpleAggregatorRetriesAtTheEndOfTheNextWord(t *testing.T) {
	tok := wordEndTokenizer()
	a := text.NewSimpleAggregator(frames.AggregationSentence, tok)
	if got := a.Aggregate("We make a good team, you and I. D"); len(got) != 0 {
		t.Fatalf("aggregations = %+v, want none before the word ends", got)
	}
	if got := a.Aggregate("id"); len(got) != 0 {
		t.Fatalf("aggregations = %+v, want none before the word ends", got)
	}
	if len(tok.calls) != 1 {
		t.Fatalf("tokenizer calls = %q, want none inside the word", tok.calls)
	}
	got := a.Aggregate(" ")
	if len(got) != 1 || got[0].Text != "We make a good team, you and I." {
		t.Fatalf("aggregations = %+v, want the sentence at the word's end", got)
	}
	if rest := a.Text().Text; rest != "Did" {
		t.Fatalf("buffer = %q, want %q", rest, "Did")
	}
}

// Punctuation ending the lookahead word is checked without itself, so the
// earlier boundary is found, and then waits for lookahead of its own.
func TestSimpleAggregatorChecksTheEarlierBoundaryFirst(t *testing.T) {
	tok := wordEndTokenizer()
	a := text.NewSimpleAggregator(frames.AggregationSentence, tok)
	got := a.Aggregate("We make a good team, you and I. Did?")
	if len(got) != 0 {
		t.Fatalf("aggregations = %+v, want none: the word ends at the \"?\"", got)
	}
	if last := tok.calls[len(tok.calls)-1]; last != "We make a good team, you and I. Did" {
		t.Fatalf("last check = %q, want it without the \"?\"", last)
	}
	got = a.Aggregate(" ")
	if len(got) != 0 {
		t.Fatalf("aggregations = %+v, want none", got)
	}
}

// A clear boundary is still released at the first character after it, without
// waiting for the rest of the word.
func TestSimpleAggregatorEmitsAtTheFirstCharacter(t *testing.T) {
	a := newAggregator(t, frames.AggregationSentence)
	if got := a.Aggregate("Hello. "); len(got) != 0 {
		t.Fatalf("aggregations = %+v, want none", got)
	}
	got := a.Aggregate("N")
	if len(got) != 1 || got[0].Text != "Hello." {
		t.Fatalf("aggregations = %+v, want %q", got, "Hello.")
	}
	if rest := a.Text().Text; rest != "N" {
		t.Fatalf("buffer = %q, want %q", rest, "N")
	}
}

// A retry pending when a response ends does not carry into the next one.
func TestSimpleAggregatorClearsAPendingRetryBetweenResponses(t *testing.T) {
	const pending = "Albert I. Did"
	endings := map[string]func(a *text.SimpleAggregator){
		"reset":        (*text.SimpleAggregator).Reset,
		"interruption": (*text.SimpleAggregator).HandleInterruption,
		"flush": func(a *text.SimpleAggregator) {
			if tail, _ := a.Flush(); tail.Text != pending {
				t.Errorf("flush = %q, want %q", tail.Text, pending)
			}
		},
	}
	for name, end := range endings {
		t.Run(name, func(t *testing.T) {
			tok := wordEndTokenizer()
			a := text.NewSimpleAggregator(frames.AggregationSentence, tok)
			if got := a.Aggregate(pending); len(got) != 0 {
				t.Fatalf("aggregations = %+v, want none", got)
			}
			end(a)
			// A retry left pending would ask the tokenizer at this first space.
			before := len(tok.calls)
			a.Aggregate("Next ")
			if len(tok.calls) != before {
				t.Fatalf("tokenizer asked %q after the response ended", tok.calls[before:])
			}
			if tail, _ := a.Flush(); tail.Text != "Next" {
				t.Fatalf("flush = %q, want %q", tail.Text, "Next")
			}
		})
	}
}

// recordingTokenizer records every text it is asked about and answers with
// boundary.
type recordingTokenizer struct {
	calls    []string
	boundary func(text string) int
}

func (r *recordingTokenizer) MatchEndOfSentence(s string) int {
	r.calls = append(r.calls, s)
	return r.boundary(s)
}

// However long the text after a candidate runs without resolving it, the
// tokenizer is asked about it twice: at the first character and at the end of
// the word.
func TestSimpleAggregatorBoundsTokenizerWork(t *testing.T) {
	tok := &recordingTokenizer{boundary: func(string) int { return 0 }}
	a := text.NewSimpleAggregator(frames.AggregationSentence, tok)
	long := strings.Repeat("x", 10_000)
	input := "I. " + long + " more words without punctuation"
	if got := a.Aggregate(input); len(got) != 0 {
		t.Fatalf("aggregations = %+v, want none", got)
	}
	want := []string{"I. x", "I. " + long + " "}
	if strings.Join(tok.calls, "|") != strings.Join(want, "|") {
		t.Fatalf("tokenizer calls = %d, want the two checks", len(tok.calls))
	}
	if tail, _ := a.Flush(); tail.Text != input {
		t.Fatal("flush did not return the whole buffer")
	}
}

// A long word after a candidate is retried at the delimiter that ends it.
func TestSimpleAggregatorRetriesALongWordAtItsDelimiter(t *testing.T) {
	for _, n := range []int{64, 65, 10_000} {
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			tok := &recordingTokenizer{boundary: func(s string) int {
				if strings.HasSuffix(s, " ") {
					return len("I.")
				}
				return 0
			}}
			a := text.NewSimpleAggregator(frames.AggregationSentence, tok)
			word := strings.Repeat("x", n)
			if got := a.Aggregate("I. " + word); len(got) != 0 {
				t.Fatalf("aggregations = %+v, want none", got)
			}
			if len(tok.calls) != 1 || tok.calls[0] != "I. x" {
				t.Fatalf("tokenizer calls = %d, want one at the first character", len(tok.calls))
			}
			got := a.Aggregate(" ")
			if len(got) != 1 || got[0].Text != "I." {
				t.Fatalf("aggregations = %+v, want %q", got, "I.")
			}
			if len(tok.calls) != 2 || tok.calls[1] != "I. "+word+" " {
				t.Fatalf("tokenizer calls = %d, want a retry at the word's end", len(tok.calls))
			}
			if tail, _ := a.Flush(); tail.Text != word {
				t.Fatal("flush did not return the word")
			}
		})
	}
}

// A lookahead that opens on a character outside a word, such as a quote, is
// checked there and retried only once the word after it ends.
func TestSimpleAggregatorRetriesOnlyAfterTheFollowingWord(t *testing.T) {
	for _, prefix := range []string{`"`, "“", "—", "👋 "} {
		t.Run(prefix, func(t *testing.T) {
			tok := &recordingTokenizer{boundary: func(string) int { return 0 }}
			a := text.NewSimpleAggregator(frames.AggregationSentence, tok)
			first, _ := firstRune(prefix)
			if got := a.Aggregate("I. " + prefix); len(got) != 0 {
				t.Fatalf("aggregations = %+v, want none", got)
			}
			if len(tok.calls) != 1 || tok.calls[0] != "I. "+first {
				t.Fatalf("tokenizer calls = %q, want one at the first character", tok.calls)
			}
			a.Aggregate("Douglas")
			if len(tok.calls) != 1 {
				t.Fatalf("tokenizer calls = %q, want no check inside the word", tok.calls)
			}
			a.Aggregate(" ")
			if len(tok.calls) != 2 || tok.calls[1] != "I. "+prefix+"Douglas " {
				t.Fatalf("tokenizer calls = %q, want a retry at the word's end", tok.calls)
			}
			if tail, _ := a.Flush(); tail.Text != "I. "+prefix+"Douglas" {
				t.Fatalf("flush = %q", tail.Text)
			}
		})
	}
}

func firstRune(s string) (string, bool) {
	for _, r := range s {
		return string(r), true
	}
	return "", false
}
