package llm_test

import (
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/gojargo/jargo/service/llm"
)

// Tests for the Server-Sent Events scanner every streaming provider reads its
// completions through. It hands on the JSON payloads and nothing else, so a
// provider parses chunks rather than the framing around them.

// The failures the tests below inject.
//
//nolint:gochecknoglobals // sentinel errors for the tests
var (
	errParse  = errors.New("cannot parse")
	errBroken = errors.New("connection reset")
)

// collect runs the scanner over s and returns the payloads it yielded.
func collect(t *testing.T, s string) []string {
	t.Helper()
	var got []string
	if err := llm.ScanSSE(strings.NewReader(s), func(data string) error {
		got = append(got, data)
		return nil
	}); err != nil {
		t.Fatalf("ScanSSE: %v", err)
	}
	return got
}

func TestScanSSE(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{
			name: "one payload per data line",
			in:   "data: {\"a\":1}\ndata: {\"b\":2}\n",
			want: []string{`{"a":1}`, `{"b":2}`},
		},
		{
			name: "the done sentinel ends nothing and is not a payload",
			in:   "data: {\"a\":1}\ndata: [DONE]\n",
			want: []string{`{"a":1}`},
		},
		{
			name: "blank lines between events are skipped",
			in:   "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n",
			want: []string{`{"a":1}`, `{"b":2}`},
		},
		{
			name: "a data line with nothing after it is skipped",
			in:   "data:\ndata: \ndata: {\"a\":1}\n",
			want: []string{`{"a":1}`},
		},
		{
			name: "the other SSE fields are not payloads",
			in:   "event: message\nid: 7\nretry: 100\n: a comment\ndata: {\"a\":1}\n",
			want: []string{`{"a":1}`},
		},
		{
			name: "the space after the colon is optional",
			in:   "data:{\"a\":1}\n",
			want: []string{`{"a":1}`},
		},
		{
			name: "carriage returns are not part of the payload",
			in:   "data: {\"a\":1}\r\ndata: {\"b\":2}\r\n",
			want: []string{`{"a":1}`, `{"b":2}`},
		},
		{
			name: "a final line without a newline still arrives",
			in:   "data: {\"a\":1}",
			want: []string{`{"a":1}`},
		},
		{
			name: "an empty stream yields nothing",
			in:   "",
		},
		{
			name: "a stream of only framing yields nothing",
			in:   "event: ping\n\n: keep-alive\n\ndata: [DONE]\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := collect(t, tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("got %d payloads %q, want %d %q", len(got), got, len(tt.want), tt.want)
			}
			for i := range tt.want {
				if got[i] != tt.want[i] {
					t.Errorf("payload %d = %q, want %q", i, got[i], tt.want[i])
				}
			}
		})
	}
}

// TestScanSSEStopsOnHandlerError checks a handler that fails ends the scan and
// its error is what the caller sees, so a provider that cannot parse a chunk is
// not fed the rest of the stream.
func TestScanSSEStopsOnHandlerError(t *testing.T) {
	var seen []string
	err := llm.ScanSSE(strings.NewReader("data: one\ndata: two\ndata: three\n"),
		func(data string) error {
			seen = append(seen, data)
			if data == "two" {
				return errParse
			}
			return nil
		})

	if !errors.Is(err, errParse) {
		t.Fatalf("ScanSSE = %v, want the handler's error", err)
	}
	if len(seen) != 2 {
		t.Errorf("the handler saw %q, want it to stop at the failing payload", seen)
	}
}

// failingReader fails partway through, standing in for a connection dropped
// mid-completion.
type failingReader struct {
	data string
	err  error
	done bool
}

func (r *failingReader) Read(p []byte) (int, error) {
	if r.done {
		return 0, r.err
	}
	r.done = true
	return copy(p, r.data), nil
}

// TestScanSSEReportsAReadError checks a stream that breaks mid-completion is
// reported rather than looking like a stream that simply ended.
func TestScanSSEReportsAReadError(t *testing.T) {
	r := &failingReader{data: "data: one\n", err: errBroken}

	var seen []string
	err := llm.ScanSSE(r, func(data string) error {
		seen = append(seen, data)
		return nil
	})

	if !errors.Is(err, errBroken) {
		t.Fatalf("ScanSSE = %v, want the read error", err)
	}
	if len(seen) != 1 || seen[0] != "one" {
		t.Errorf("the handler saw %q, want the payload read before the break", seen)
	}
}

// TestScanSSEHandlesALargePayload checks a chunk far past the scanner's initial
// buffer still arrives whole. A streamed completion carrying a long tool-call
// argument is one line, and bufio's default 64 KiB would cut it short.
func TestScanSSEHandlesALargePayload(t *testing.T) {
	big := strings.Repeat("x", 512*1024)

	got := collect(t, "data: "+big+"\n")
	if len(got) != 1 {
		t.Fatalf("got %d payloads, want 1", len(got))
	}
	if got[0] != big {
		t.Errorf("the payload came back %d bytes, want %d", len(got[0]), len(big))
	}
}

// TestScanSSERejectsAnUnboundedLine checks the buffer is bounded: a line that
// never ends is reported rather than read until the process runs out of memory.
func TestScanSSERejectsAnUnboundedLine(t *testing.T) {
	huge := strings.Repeat("y", 2<<20)

	err := llm.ScanSSE(strings.NewReader("data: "+huge+"\n"), func(string) error {
		t.Error("a payload past the bound was handed on")
		return nil
	})
	if err == nil {
		t.Fatal("ScanSSE accepted a line past its bound")
	}
	if !errors.Is(err, io.ErrShortBuffer) && !strings.Contains(err.Error(), "too long") {
		t.Fatalf("ScanSSE = %v, want it to report the line as too long", err)
	}
}
