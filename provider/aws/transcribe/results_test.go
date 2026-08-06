package transcribe

import (
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/transcribestreaming/types"
)

// segment builds one Transcribe result carrying a single alternative.
func segment(text string, partial bool, lang types.LanguageCode) types.Result {
	return types.Result{
		IsPartial:    partial,
		LanguageCode: lang,
		Alternatives: []types.Alternative{{Transcript: &text}},
	}
}

// TestResultsMarksFinalSegments covers the distinction the rest of the pipeline
// turns on. A partial segment is the service revising itself and must not close
// anything; a settled one is the transcript for that stretch of speech, and this
// service settles per segment, so it ends the segment's transcription with it.
func TestResultsMarksFinalSegments(t *testing.T) {
	s := &stream{lang: "en-US"}

	partial := s.results([]types.Result{segment("hello th", true, "")})
	if len(partial) != 1 {
		t.Fatalf("mapped %d results, want 1", len(partial))
	}
	if partial[0].Final || partial[0].EndOfTurn {
		t.Error("a partial segment was reported as settled")
	}

	settled := s.results([]types.Result{segment("hello there", false, "")})
	if len(settled) != 1 {
		t.Fatalf("mapped %d results, want 1", len(settled))
	}
	if !settled[0].Final || !settled[0].EndOfTurn {
		t.Error("a settled segment was reported as still being revised")
	}
	if settled[0].Text != "hello there" {
		t.Errorf("text = %q, want %q", settled[0].Text, "hello there")
	}
}

// TestResultsReadsTheFirstSegment covers an event carrying more than one
// segment. One event describes one stretch of speech, so the first segment is
// the transcript for it and the rest are not more of the same utterance.
func TestResultsReadsTheFirstSegment(t *testing.T) {
	s := &stream{lang: "en-US"}

	got := s.results([]types.Result{
		segment("the first", false, ""),
		segment("something else", false, ""),
	})

	if len(got) != 1 {
		t.Fatalf("mapped %d results, want 1", len(got))
	}
	if got[0].Text != "the first" {
		t.Errorf("text = %q, want %q", got[0].Text, "the first")
	}
}

// TestResultsSkipsSegmentsWithNothingInThem covers what arrives between
// utterances. A segment with no alternative, or one whose transcript is empty,
// carries no speech, and forwarding it would put an empty transcript into the
// conversation.
func TestResultsSkipsSegmentsWithNothingInThem(t *testing.T) {
	s := &stream{lang: "en-US"}

	tests := []struct {
		name    string
		segment types.Result
	}{
		{"no alternatives at all", types.Result{}},
		{"an alternative with no transcript", types.Result{Alternatives: []types.Alternative{{}}}},
		{"an empty transcript", segment("", false, "")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := s.results([]types.Result{tt.segment}); len(got) != 0 {
				t.Errorf("mapped %d results, want none: the segment carried no speech", len(got))
			}
		})
	}

	if got := s.results(nil); len(got) != 0 {
		t.Errorf("mapped %d results from an event with no segments, want none", len(got))
	}
}

// TestResultsLabelWithTheSessionLanguage covers how a transcript is labeled. The
// session was opened to transcribe one language, and that is the language the
// transcript is reported in, whatever a segment names itself.
func TestResultsLabelWithTheSessionLanguage(t *testing.T) {
	s := &stream{lang: "en-US"}

	got := s.results([]types.Result{segment("hello", false, "fr-FR")})
	if len(got) != 1 {
		t.Fatalf("mapped %d results, want 1", len(got))
	}
	if got[0].Language != "en-US" {
		t.Errorf("language = %q, want the session's en-US", got[0].Language)
	}
}

// TestLoadOptionsFallsBackToTheDefaultChain covers how the service is
// authenticated. Static credentials are used when both halves are given;
// anything less falls back to the AWS chain (environment, shared config, an IAM
// role), so a deployment that authenticates by role is not overridden by a
// half-filled config.
func TestLoadOptionsFallsBackToTheDefaultChain(t *testing.T) {
	tests := []struct {
		name string
		cfg  Config
		want int
	}{
		{"nothing set", Config{}, 0},
		{"region only", Config{Region: "us-east-1"}, 1},
		{"a key with no secret", Config{AccessKeyID: "id"}, 0},
		{"a secret with no key", Config{SecretAccessKey: "secret"}, 0},
		{"both halves", Config{AccessKeyID: "id", SecretAccessKey: "secret"}, 1},
		{"region and credentials", Config{
			Region: "us-east-1", AccessKeyID: "id", SecretAccessKey: "secret",
		}, 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := len(tt.cfg.loadOptions()); got != tt.want {
				t.Errorf("built %d load options, want %d", got, tt.want)
			}
		})
	}
}
