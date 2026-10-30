package elevenlabs

import (
	"net/http"
	"reflect"
	"testing"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	ttselevenlabs "github.com/gojargo/jargo/adapter/tts/elevenlabs"
)

// The expression adapter follows the model: audio tags on Eleven v3 and v4,
// and none on the others.
func TestHTTPExpressionAdapterFollowsTheModel(t *testing.T) {
	for _, tt := range []struct {
		model   string
		adapter ttsadapter.ExpressionAdapter
	}{
		{"eleven_flash_v2_5", ttsadapter.BaseExpressionAdapter{}},
		{"eleven_multilingual_v2", ttsadapter.BaseExpressionAdapter{}},
		{"eleven_v3", ttselevenlabs.ElevenLabsAudioTagExpressionAdapter{}},
		{"eleven_v4_turbo", ttselevenlabs.ElevenLabsAudioTagExpressionAdapter{}},
	} {
		s := &synthesizer{cfg: Config{Model: tt.model}}
		if got := s.ExpressionAdapter(); reflect.TypeOf(got) != reflect.TypeOf(tt.adapter) {
			t.Errorf("%s: adapter %T, want %T", tt.model, got, tt.adapter)
		}
	}
}

func expressiveSynthesizer(t *testing.T, chunks ...string) *synthesizer {
	t.Helper()
	lines := make([]string, 0, len(chunks))
	for _, c := range chunks {
		lines = append(lines, alignmentJSON(t, c, nil))
	}
	srv, _ := newTTSServer(t, lines)
	cfg := ttsCfg(srv)
	cfg.Model = "eleven_v3"
	return &synthesizer{cfg: cfg, http: &http.Client{}, expressive: func() bool { return true }}
}

// The audio tags expressive conversation sends are reported as words, which
// mark when their sounds are performed.
func TestHTTPWordTimestampsReportAudioTagsAsWords(t *testing.T) {
	s := expressiveSynthesizer(t, "It's ", "[excited]", " back [sh", "ort pause] up.")

	got := spokenWords(speak(t, s, "It's [excited] back [short pause] up."))

	if want := []string{"It's", "[excited]", "back", "[short pause]"}; len(got) < 4 || !equalWords(got[:4], want) {
		t.Fatalf("words = %q, want %q first", got, want)
	}
}

func TestHTTPAudioTagEndingTheReplyIsReportedWhole(t *testing.T) {
	s := expressiveSynthesizer(t, "Thanks. [inhales ", "deeply]")

	got := spokenWords(speak(t, s, "Thanks. [inhales deeply]"))

	if want := []string{"Thanks.", "[inhales deeply]"}; !equalWords(got, want) {
		t.Fatalf("words = %q, want %q", got, want)
	}
}

func TestHTTPAlignmentIsLeftAsIsWithoutExpressiveConversation(t *testing.T) {
	srv, _ := newTTSServer(t, []string{alignmentJSON(t, "Press [ to open the menu now.", nil)})
	cfg := ttsCfg(srv)
	cfg.Model = "eleven_v3"

	got := spokenWords(speak(t, &synthesizer{cfg: cfg, http: &http.Client{}}, "Press [ to open the menu now."))

	if want := []string{"Press", "[", "to", "open", "the", "menu", "now."}; !equalWords(got, want) {
		t.Fatalf("words = %q, want %q", got, want)
	}
}
