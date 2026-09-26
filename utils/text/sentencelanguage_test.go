package text_test

import (
	"fmt"
	"regexp"
	"slices"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/utils/text"
)

// The boundary is a byte offset into the text as given, whatever the script.
func TestMatchEndOfSentenceSourceOffsets(t *testing.T) {
	cases := []struct{ text, language, prefix string }{
		{"Hello.   Next", "en", "Hello."},
		{"Hello.\n\nNext", "en", "Hello."},
		{"\n\nHello. Next", "en", "\n\nHello."},
		{"👋 Hello. 😀 Next", "en", "👋 Hello."},
		{"Café is open. Next", "en", "Café is open."},
		{"こんにちは。次", "ja", "こんにちは。"},
		{"你好。下一句", "zh", "你好。"},
		{"नमस्ते। यह", "hi", "नमस्ते।"},
		{"هل أنت بخير؟ نعم", "ar", "هل أنت بخير؟"},
		{"Hello. Next", "unrecognized", "Hello."},
	}
	for _, c := range cases {
		if got := text.MatchEndOfSentence(c.text, c.language); got != len(c.prefix) {
			t.Errorf("MatchEndOfSentence(%q, %q) = %d, want %d (%q)", c.text, c.language, got, len(c.prefix), c.prefix)
		}
	}
}

// aggregateInChunks runs text through an aggregator in chunks of size runes and
// returns the sentences it produced, the flushed tail included.
func aggregateInChunks(language, input string, size int) []string {
	a := text.NewSimpleAggregator(frames.AggregationSentence, language)
	runes := []rune(input)
	var out []string
	for i := 0; i < len(runes); i += size {
		for _, agg := range a.Aggregate(string(runes[i:min(i+size, len(runes))])) {
			out = append(out, agg.Text)
		}
	}
	if tail, ok := a.Flush(); ok {
		out = append(out, tail.Text)
	}
	return out
}

func wantSentences(t *testing.T, language string, sentences []string, sizes ...int) {
	t.Helper()
	input := joinSentences(sentences)
	for _, size := range sizes {
		if got := aggregateInChunks(language, input, size); !slices.Equal(got, sentences) {
			t.Errorf("%s, chunks of %d: got %q, want %q", language, size, got, sentences)
		}
	}
}

func joinSentences(sentences []string) string {
	out := ""
	for i, s := range sentences {
		if i > 0 {
			out += " "
		}
		out += s
	}
	return out
}

// However the text is cut into chunks, the same sentences come out, in each
// language's rules.
func TestChunkBoundariesDoNotChangeSentences(t *testing.T) {
	cases := []struct {
		language  string
		sentences []string
	}{
		{"en", []string{"Dr. Smith is here.", "It costs $29.95."}},
		{"de-DE", []string{"Das ist z.B. wichtig.", "Weiter geht es."}},
		{"pt-BR", []string{"O Sr. Silva chegou.", "Depois saiu."}},
		{"fr", []string{"Voir p. 12 pour les détails.", "Ensuite continuez."}},
		{"pl", []string{"Prof. Kowalski przyszedł.", "Potem wyszedł."}},
		{"it", []string{"Il dott. Rossi arriva.", "Poi parte."}},
		{"nl", []string{"Dr. Jansen komt.", "Daarna vertrekt hij."}},
		{"ja", []string{"こんにちは。", "次の文です。"}},
	}
	for _, c := range cases {
		wantSentences(t, c.language, c.sentences, 1, 3, 1000)
	}
}

func TestPronounBoundaryWithStreamedSentenceStarter(t *testing.T) {
	wantSentences(t, "en", []string{"We make a good team, you and I.", "Did you see him?"}, 1000)
}

// The retry at the end of the word after a candidate keeps sentences whole.
func TestLookaheadRetryPreservesSentences(t *testing.T) {
	for _, sentences := range [][]string{
		{"We make a good team, you and I.", "Did you see him?"},
		{"Albert I. Jones is here.", "Hello."},
		{"Albert I. Douglas is here.", "Hello."},
		{`He said "Hello."`, "Then he left."},
		{"This is the U.S. Department of State.", "Next."},
		{"Hello!?", "Next."},
		{"We make a good team, you and I.", "Did?"},
	} {
		wantSentences(t, "en", sentences, 1, 3, 1000)
	}
}

// A retry pending when a generation ends does not carry into the next one.
func TestPendingRetryIsClearedBetweenGenerations(t *testing.T) {
	const pending = "We make a good team, you and I. Did"
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
			a := text.NewSimpleAggregator(frames.AggregationSentence, "")
			if got := a.Aggregate(pending); len(got) != 0 {
				t.Fatalf("aggregations = %+v, want none", got)
			}
			end(a)
			got := a.Aggregate("Dr. Smith is here. Next")
			if len(got) != 1 || got[0].Text != "Dr. Smith is here." {
				t.Fatalf("aggregations = %+v, want the one sentence", got)
			}
			if tail, _ := a.Flush(); tail.Text != "Next" {
				t.Fatalf("flush = %q, want Next", tail.Text)
			}
		})
	}
}

// An opening quote does not hide an abbreviation after it, and the boundary is
// found without waiting for the quote to close.
func TestOpenQuotePreservesAbbreviations(t *testing.T) {
	for _, quote := range []string{`"`, "'", "“", "‘", "«", "「"} {
		for _, size := range []int{1, 3, 1000} {
			input := fmt.Sprintf("She said, %sDr. Smith is here. Next sentence", quote)
			a := text.NewSimpleAggregator(frames.AggregationSentence, "")
			runes := []rune(input)
			var got []string
			for i := 0; i < len(runes); i += size {
				for _, agg := range a.Aggregate(string(runes[i:min(i+size, len(runes))])) {
					got = append(got, agg.Text)
				}
			}
			want := []string{fmt.Sprintf("She said, %sDr. Smith is here.", quote)}
			if !slices.Equal(got, want) {
				t.Errorf("quote %s, chunks of %d: got %q, want %q", quote, size, got, want)
			}
			if tail, _ := a.Flush(); tail.Text != "Next sentence" {
				t.Errorf("quote %s, chunks of %d: flush = %q", quote, size, tail.Text)
			}
		}
	}
}

// The recheck after a quoted word leaves the other boundaries alone.
func TestQuotationProbePreservesOtherBoundaries(t *testing.T) {
	for _, sentences := range [][]string{
		{`She said, "Dr. Smith is here."`, "Then she left."},
		{"She said, “Dr. Smith is here.”", "Then she left."},
		{`She said, "Hello.`, `Next sentence."`},
		{"Don't call Dr. Smith.", "He's busy."},
		{"The doctor's here.", "Let's go."},
	} {
		wantSentences(t, "en", sentences, 1)
	}
}

func TestQuotedAbbreviationOffsets(t *testing.T) {
	for _, quote := range []string{`"`, "“", "'", "«"} {
		in := fmt.Sprintf("👋 She said, %sDr. Smith is here. Next", quote)
		want := len(fmt.Sprintf("👋 She said, %sDr. Smith is here.", quote))
		if got := text.MatchEndOfSentence(in, ""); got != want {
			t.Errorf("MatchEndOfSentence(%q) = %d, want %d", in, got, want)
		}
		if got := text.MatchEndOfSentence(fmt.Sprintf("She said, %sDr. S", quote), ""); got != 0 {
			t.Errorf("an unfinished quoted abbreviation ended a sentence at %d", got)
		}
	}
}

// English sentence boundaries under fragmented streaming input.
func TestEnglishBoundariesPreserveTextAcrossChunks(t *testing.T) {
	cases := map[string][]string{
		"sentence final time":         {"We met at 5 p.m.", "Then we left."},
		"sentence final acronym":      {"He lives in the U.S.", "His sister lives in Canada."},
		"mid sentence abbreviation":   {"We sell pens, paper, etc. in the shop.", "Come inside."},
		"sentence final abbreviation": {"We sell pens, paper, etc.", "The shop closes at six."},
		"initials":                    {"W. E. B. Du Bois wrote extensively.", "His work remains influential."},
		"attributed exclamation":      {`"Stop!" he shouted.`, "The car halted."},
		"attributed question":         {"“Are you ready?” she asked.", "I nodded."},
		"ellipsis boundary":           {"Wait...", "Then continue."},
		"mid sentence ellipsis":       {"That's strange... but possible.", "Let's check."},
		"repeated exclamation":        {"No!!!", "Stop!"},
		"ip address":                  {"Your IP address is 192.168.1.1.", "Enter it in the browser."},
		"url":                         {"Go to https://example.com/help.", "Then click Support."},
		"email":                       {"Email support@example.com.", "We'll reply soon."},
	}
	words := regexp.MustCompile(`\S+\s*`)
	for name, sentences := range cases {
		t.Run(name, func(t *testing.T) {
			input := joinSentences(sentences)
			chunkings := map[string][]string{
				"whole": {input},
				"word":  words.FindAllString(input, -1),
			}
			for _, size := range []int{1, 3} {
				var chunks []string
				runes := []rune(input)
				for i := 0; i < len(runes); i += size {
					chunks = append(chunks, string(runes[i:min(i+size, len(runes))]))
				}
				chunkings[fmt.Sprintf("%d characters", size)] = chunks
			}
			for chunking, chunks := range chunkings {
				a := text.NewSimpleAggregator(frames.AggregationSentence, "")
				var got []string
				for _, c := range chunks {
					for _, agg := range a.Aggregate(c) {
						got = append(got, agg.Text)
					}
				}
				if tail, ok := a.Flush(); ok {
					got = append(got, tail.Text)
				}
				if !slices.Equal(got, sentences) {
					t.Errorf("%s: got %q, want %q", chunking, got, sentences)
				}
				if _, ok := a.Flush(); ok {
					t.Errorf("%s: a second flush returned text", chunking)
				}
			}
		})
	}
}

func TestResolveSentenceTokenizerLanguage(t *testing.T) {
	cases := map[string]string{
		"de": "de", "de-DE": "de", "de_AT": "de", "pt-BR": "pt", "nb-NO": "nb", "nn": "nn",
		"sl": "sl", "ml": "ml", "unknown": "unknown", "ja": "ja", "auto": "en", "": "en",
	}
	for in, want := range cases {
		if got := text.ResolveSentenceTokenizerLanguage(in); got != want {
			t.Errorf("ResolveSentenceTokenizerLanguage(%q) = %q, want %q", in, got, want)
		}
	}
}

// Two aggregators in different languages keep their own rules.
func TestStreamingLanguagesAreIndependent(t *testing.T) {
	german := text.NewSimpleAggregator(frames.AggregationSentence, "de")
	english := text.NewSimpleAggregator(frames.AggregationSentence, "en")
	if got := german.Aggregate("Das ist bzw. w"); len(got) != 0 {
		t.Fatalf("german = %+v, want nothing: bzw. is an abbreviation", got)
	}
	if got := english.Aggregate("Das ist bzw. w"); len(got) != 1 || got[0].Text != "Das ist bzw." {
		t.Fatalf("english = %+v, want the sentence", got)
	}
	if got := german.Aggregate("ichtig. Weiter"); len(got) != 1 || got[0].Text != "Das ist bzw. wichtig." {
		t.Fatalf("german = %+v, want the whole sentence", got)
	}
	if got := text.MatchEndOfSentence("こんにちは。次", "ja"); got != len("こんにちは。") {
		t.Fatalf("japanese boundary = %d", got)
	}
}
