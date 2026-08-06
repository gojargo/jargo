package inworld

import (
	"encoding/base64"
	"errors"
	"strings"
	"testing"

	"github.com/gojargo/jargo/language"
)

// errDownstreamGone stands in for the pipeline going away mid-stream.
var errDownstreamGone = errors.New("downstream is gone")

// audioLine builds one line of the response, carrying base64 audio the way the
// service sends it.
func audioLine(pcm []byte) string {
	return `{"result":{"audioContent":"` + base64.StdEncoding.EncodeToString(pcm) + `"}}` + "\n"
}

// wavOf wraps pcm in a minimal WAV container, which is what the service returns
// when it is not asked for raw samples.
func wavOf(pcm []byte) []byte {
	out := make([]byte, wavHeaderSize+len(pcm))
	copy(out, "RIFF")
	copy(out[8:], "WAVE")
	copy(out[wavHeaderSize:], pcm)
	return out
}

// collect runs a response body through the stream and returns everything it
// emitted, concatenated.
func collect(t *testing.T, body string) []byte {
	t.Helper()
	var got []byte
	if err := stream(strings.NewReader(body), func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	}); err != nil {
		t.Fatalf("stream: %v", err)
	}
	return got
}

// TestStreamEmitsEachLinesAudio covers the ordinary response: it arrives as one
// JSON object per line, and the audio in each is played in the order it came.
func TestStreamEmitsEachLinesAudio(t *testing.T) {
	body := audioLine([]byte{1, 2}) + audioLine([]byte{3, 4})

	got := collect(t, body)
	if string(got) != string([]byte{1, 2, 3, 4}) {
		t.Errorf("emitted % x, want % x", got, []byte{1, 2, 3, 4})
	}
}

// TestStreamStripsTheWAVHeader covers audio the service wrapped in a container.
// The pipeline plays raw samples, so a header left in front of them would be
// heard as a burst of noise at the start of every chunk.
func TestStreamStripsTheWAVHeader(t *testing.T) {
	pcm := []byte{9, 8, 7, 6}
	got := collect(t, audioLine(wavOf(pcm)))

	if string(got) != string(pcm) {
		t.Errorf("emitted % x, want the samples % x with the header stripped", got, pcm)
	}
}

// TestStreamSkipsLinesWithNoAudio covers what a response carries besides audio:
// blank lines, lines that are not JSON at all, and results that hold no audio.
// None of them is playable, and none may stop the response being read.
func TestStreamSkipsLinesWithNoAudio(t *testing.T) {
	body := "\n" +
		"not json\n" +
		`{"result":{}}` + "\n" +
		`{"result":{"audioContent":""}}` + "\n" +
		audioLine([]byte{5, 5})

	got := collect(t, body)
	if string(got) != string([]byte{5, 5}) {
		t.Errorf("emitted % x, want only the one playable line % x", got, []byte{5, 5})
	}
}

// TestStreamEmitsALastLineWithoutANewline covers a response that ends the moment
// its last object does. The line is complete, so its audio has to be played
// rather than left in the reader.
func TestStreamEmitsALastLineWithoutANewline(t *testing.T) {
	body := strings.TrimSuffix(audioLine([]byte{7, 7}), "\n")

	if got := collect(t, body); string(got) != string([]byte{7, 7}) {
		t.Errorf("emitted % x, want the final line % x", got, []byte{7, 7})
	}
}

// TestStreamStopsOnAFailedEmit covers the pipeline going away underneath the
// response: the failure is reported rather than swallowed, so the turn ends
// instead of reading out the rest of a reply nobody is listening to.
func TestStreamStopsOnAFailedEmit(t *testing.T) {
	body := audioLine([]byte{1, 1}) + audioLine([]byte{2, 2})

	calls := 0
	err := stream(strings.NewReader(body), func([]byte) error {
		calls++
		return errDownstreamGone
	})
	if !errors.Is(err, errDownstreamGone) {
		t.Errorf("stream returned %v, want the emit failure", err)
	}
	if calls != 1 {
		t.Errorf("emitted %d times, want 1: the response was read past a failed emit", calls)
	}
}

// TestLanguageKeepsTheRegionItWasGiven covers naming the language. A verified
// language takes the tag the service names it by, which carries a region of its
// own; anything else is passed through as it was given, so a caller asking for a
// region is answered in that region rather than in the verified one.
func TestLanguageKeepsTheRegionItWasGiven(t *testing.T) {
	tests := []struct {
		name string
		lang language.Language
		want string
	}{
		{"unset", "", ""},
		{"a verified language takes its tag", language.English, "en-US"},
		{"another verified language", language.French, "fr-FR"},
		{"a region is kept", language.EnglishGB, "en-GB"},
		{"another region is kept", language.FrenchCA, "fr-CA"},
		{"an unverified language is passed through", "sv-SE", "sv-SE"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := inworldLanguage(tt.lang); got != tt.want {
				t.Errorf("%q named %q, want %q", tt.lang, got, tt.want)
			}
		})
	}
}
