package groq

import (
	"bytes"
	"encoding/binary"
	"errors"
	"strings"
	"testing"
)

// errDownstreamGone stands in for the pipeline going away mid-response.
var errDownstreamGone = errors.New("downstream is gone")

// wavChunk builds one RIFF chunk: its four-character id, its length, and its
// payload.
func wavChunk(id string, payload []byte) []byte {
	out := make([]byte, 8, 8+len(payload))
	copy(out, id)
	binary.LittleEndian.PutUint32(out[4:], uint32(len(payload)))
	return append(out, payload...)
}

// wavOf builds a WAV response carrying pcm, with any extra chunks placed ahead
// of the audio the way a real encoder writes them.
func wavOf(pcm []byte, before ...[]byte) []byte {
	body := []byte("RIFF")
	body = binary.LittleEndian.AppendUint32(body, 0) // length, which the reader skips
	body = append(body, "WAVE"...)
	for _, c := range before {
		body = append(body, c...)
	}
	return append(body, wavChunk("data", pcm)...)
}

// collect runs a response through the parser and returns everything it emitted.
func collect(t *testing.T, body []byte) []byte {
	t.Helper()
	var got []byte
	if err := streamWAV(bytes.NewReader(body), func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	}); err != nil {
		t.Fatalf("stream: %v", err)
	}
	return got
}

// TestStreamWAVEmitsTheAudioPayload covers the ordinary response: the container
// is unwrapped and what reaches the pipeline is the samples alone, since that is
// what it plays.
func TestStreamWAVEmitsTheAudioPayload(t *testing.T) {
	pcm := bytes.Repeat([]byte{0x01, 0x02}, 64)

	if got := collect(t, wavOf(pcm)); !bytes.Equal(got, pcm) {
		t.Errorf("emitted %d bytes, want the %d samples unchanged", len(got), len(pcm))
	}
}

// TestStreamWAVWalksPastOtherChunks covers a container that carries more than
// audio. The format chunk, and anything else an encoder writes ahead of the
// samples, has to be walked past rather than played.
func TestStreamWAVWalksPastOtherChunks(t *testing.T) {
	pcm := bytes.Repeat([]byte{0x07}, 32)
	fmtChunk := wavChunk("fmt ", make([]byte, 16))
	listChunk := wavChunk("LIST", []byte("INFOsome metadata!!")[:16]) // even length, so no pad byte

	if got := collect(t, wavOf(pcm, fmtChunk, listChunk)); !bytes.Equal(got, pcm) {
		t.Errorf("emitted % x, want the samples % x", got, pcm)
	}
}

// TestStreamWAVWalksPastAnOddLengthChunk covers a chunk whose payload has an odd
// length. The container pads it to an even boundary, so a reader that trusted
// the stated length alone would start reading the audio one byte late and play
// noise.
func TestStreamWAVWalksPastAnOddLengthChunk(t *testing.T) {
	pcm := bytes.Repeat([]byte{0x09}, 16)
	odd := wavChunk("LIST", []byte("odd"))
	odd = append(odd, 0) // the pad byte the container writes

	if got := collect(t, wavOf(pcm, odd)); !bytes.Equal(got, pcm) {
		t.Errorf("emitted % x, want the samples % x: the pad byte was misread", got, pcm)
	}
}

// TestStreamWAVEmitsLargeAudioInChunks covers a response longer than one read.
// It arrives in pieces, and all of it has to reach the pipeline in order.
func TestStreamWAVEmitsLargeAudioInChunks(t *testing.T) {
	pcm := make([]byte, ttsReadChunk*2+123)
	for i := range pcm {
		pcm[i] = byte(i)
	}

	var chunks int
	var got []byte
	err := streamWAV(bytes.NewReader(wavOf(pcm)), func(c []byte) error {
		chunks++
		got = append(got, c...)
		return nil
	})
	if err != nil {
		t.Fatalf("stream: %v", err)
	}
	if !bytes.Equal(got, pcm) {
		t.Errorf("emitted %d bytes, want %d, and in the order they were sent", len(got), len(pcm))
	}
	if chunks < 2 {
		t.Errorf("emitted in %d chunks, want it streamed rather than held whole", chunks)
	}
}

// TestStreamWAVRejectsSomethingElse covers a response that is not the format
// that was asked for. Playing its bytes as samples would be noise, so it is
// reported instead.
func TestStreamWAVRejectsSomethingElse(t *testing.T) {
	err := streamWAV(strings.NewReader("<html>not audio at all</html>"), func([]byte) error {
		t.Error("something that was not audio was played")
		return nil
	})
	if !errors.Is(err, errFormat) {
		t.Errorf("error = %v, want the format error", err)
	}
}

// TestStreamWAVReportsATruncatedResponse covers a connection cut mid-header.
// There is no audio to be had, and the failure is reported rather than passed
// off as a response that simply ended.
func TestStreamWAVReportsATruncatedResponse(t *testing.T) {
	err := streamWAV(bytes.NewReader([]byte("RIFF")), func([]byte) error { return nil })
	if err == nil {
		t.Error("a response cut short mid-header was accepted")
	}
}

// TestStreamWAVStopsOnAFailedEmit covers the pipeline going away underneath the
// response: the failure is reported rather than swallowed, so the turn ends
// instead of reading out a reply nobody is listening to.
func TestStreamWAVStopsOnAFailedEmit(t *testing.T) {
	pcm := make([]byte, ttsReadChunk*3)

	calls := 0
	err := streamWAV(bytes.NewReader(wavOf(pcm)), func([]byte) error {
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
