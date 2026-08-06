package minimax

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

// errDownstreamGone stands in for the pipeline going away mid-stream.
var errDownstreamGone = errors.New("downstream is gone")

// collect runs a stream through the scanner and returns each payload it handed
// on, as text.
func collect(t *testing.T, body string) []string {
	t.Helper()
	var got []string
	if err := scanSSE(strings.NewReader(body), func(data []byte) error {
		got = append(got, string(data))
		return nil
	}); err != nil {
		t.Fatalf("scan: %v", err)
	}
	return got
}

// TestScanSSEReadsDataPayloads covers the ordinary stream: each event's payload
// is handed on, in order, with the field prefix and its spacing removed.
func TestScanSSEReadsDataPayloads(t *testing.T) {
	body := "data: {\"a\":1}\n\ndata:{\"b\":2}\n\n"

	got := collect(t, body)
	want := []string{`{"a":1}`, `{"b":2}`}
	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] {
		t.Errorf("payloads = %q, want %q", got, want)
	}
}

// TestScanSSESkipsWhatIsNotAPayload covers everything else a stream carries:
// blank lines between events, the end marker, an empty payload, and any other
// field. None of them is audio, and none may stop the stream being read.
func TestScanSSESkipsWhatIsNotAPayload(t *testing.T) {
	body := "\n" +
		": keep-alive\n" +
		"event: message\n" +
		"data:\n" +
		"data: [DONE]\n" +
		"data: {\"real\":true}\n"

	got := collect(t, body)
	if len(got) != 1 || got[0] != `{"real":true}` {
		t.Errorf("payloads = %q, want just the one real payload", got)
	}
}

// TestScanSSEStopsOnAFailedHandler covers the pipeline going away underneath the
// stream: the failure is reported rather than swallowed, so the turn ends
// instead of reading out a reply nobody is listening to.
func TestScanSSEStopsOnAFailedHandler(t *testing.T) {
	body := "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n"

	calls := 0
	err := scanSSE(strings.NewReader(body), func([]byte) error {
		calls++
		return errDownstreamGone
	})
	if !errors.Is(err, errDownstreamGone) {
		t.Errorf("scan returned %v, want the handler's failure", err)
	}
	if calls != 1 {
		t.Errorf("handled %d payloads, want 1: the stream was read past a failure", calls)
	}
}

// TestSummaryEventIsRecognized covers the event that closes the stream. Its
// audio repeats what has already been played, so playing it would say the whole
// utterance a second time. It names itself two ways and either has to be enough.
func TestSummaryEventIsRecognized(t *testing.T) {
	tests := []struct {
		name string
		raw  string
		want bool
	}{
		{"an ordinary audio event", `{"data":{"audio":"00ff","status":1}}`, false},
		{"the summary named by its status", `{"data":{"audio":"00ff","status":2}}`, true},
		{"the summary named by its extra info", `{"extra_info":{"audio_length":42},"data":{"audio":"00ff"}}`, true},
		{"both at once", `{"extra_info":{},"data":{"audio":"00ff","status":2}}`, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var c t2aChunk
			if err := json.Unmarshal([]byte(tt.raw), &c); err != nil {
				t.Fatalf("decode: %v", err)
			}
			if got := c.summary(); got != tt.want {
				t.Errorf("summary = %v, want %v", got, tt.want)
			}
		})
	}
}

// TestAudioIsCarriedAsHex covers the encoding the audio arrives in: the samples
// are hex text on the wire and have to be decoded before they are played.
func TestAudioIsCarriedAsHex(t *testing.T) {
	var c t2aChunk
	if err := json.Unmarshal([]byte(`{"data":{"audio":"00ff7f","status":1}}`), &c); err != nil {
		t.Fatalf("decode: %v", err)
	}

	pcm, err := hex.DecodeString(c.Data.Audio)
	if err != nil {
		t.Fatalf("decode audio: %v", err)
	}
	want := []byte{0x00, 0xff, 0x7f}
	if string(pcm) != string(want) {
		t.Errorf("audio = % x, want % x", pcm, want)
	}
}
