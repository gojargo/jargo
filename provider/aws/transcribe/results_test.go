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

	got := s.results([]types.Result{
		segment("hello th", true, ""),
		segment("hello there", false, ""),
	})

	if len(got) != 2 {
		t.Fatalf("mapped %d results, want 2", len(got))
	}
	if got[0].Final || got[0].EndOfTurn {
		t.Error("a partial segment was reported as settled")
	}
	if !got[1].Final || !got[1].EndOfTurn {
		t.Error("a settled segment was reported as still being revised")
	}
	if got[1].Text != "hello there" {
		t.Errorf("text = %q, want %q", got[1].Text, "hello there")
	}
}

// TestResultsSkipsSegmentsWithNothingInThem covers what arrives between
// utterances. A segment with no alternative, or one whose transcript is empty,
// carries no speech, and forwarding it would put an empty transcript into the
// conversation.
func TestResultsSkipsSegmentsWithNothingInThem(t *testing.T) {
	s := &stream{lang: "en-US"}

	got := s.results([]types.Result{
		{IsPartial: false}, // no alternatives at all
		{IsPartial: false, Alternatives: []types.Alternative{{}}}, // an alternative with no transcript
		segment("", false, ""), // an empty transcript
	})

	if len(got) != 0 {
		t.Errorf("mapped %d results, want none: none of the segments carried speech", len(got))
	}
}

// TestResultsPrefersTheDetectedLanguage covers automatic language
// identification. The configured language is what the session was opened with,
// but a segment that names its own language was recognized as that one, and the
// transcript has to be labeled with what was actually spoken.
func TestResultsPrefersTheDetectedLanguage(t *testing.T) {
	s := &stream{lang: "en-US"}

	got := s.results([]types.Result{
		segment("bonjour", false, "fr-FR"),
		segment("hello", false, ""),
	})

	if len(got) != 2 {
		t.Fatalf("mapped %d results, want 2", len(got))
	}
	if got[0].Language != "fr-FR" {
		t.Errorf("language = %q, want the detected fr-FR", got[0].Language)
	}
	if got[1].Language != "en-US" {
		t.Errorf("language = %q, want the session's en-US where none was detected", got[1].Language)
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
