package mistral

import (
	"encoding/base64"
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
)

// errDownstreamGone stands in for the pipeline going away mid-stream.
var errDownstreamGone = errors.New("downstream is gone")

// f32PCM packs float32 samples the way the service sends them: little-endian,
// base64 over the wire.
func f32PCM(samples ...float32) string {
	raw := make([]byte, len(samples)*4)
	for i, s := range samples {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(s))
	}
	return base64.StdEncoding.EncodeToString(raw)
}

// int16At reads the i-th 16-bit little-endian sample.
func int16At(pcm []byte, i int) int16 {
	return int16(binary.LittleEndian.Uint16(pcm[i*2:]))
}

// collect runs a stream and returns everything it emitted, concatenated.
func collect(t *testing.T, stream string) []byte {
	t.Helper()
	var got []byte
	if err := streamEvents(strings.NewReader(stream), func(pcm []byte) error {
		got = append(got, pcm...)
		return nil
	}); err != nil {
		t.Fatalf("stream events: %v", err)
	}
	return got
}

// TestFloat32ToInt16Scales covers the conversion the service's audio needs: it
// sends float32, the pipeline works in 16-bit. Full scale maps to the top of the
// 16-bit range and anything beyond it is clamped rather than wrapping, which
// would turn a loud passage into noise.
func TestFloat32ToInt16Scales(t *testing.T) {
	tests := []struct {
		name   string
		sample float32
		want   int16
	}{
		{"silence", 0, 0},
		{"half scale", 0.5, 16383},
		{"full scale", 1, 32767},
		{"full scale negative", -1, -32767},
		{"above full scale is clamped", 2, 32767},
		{"below full scale is clamped", -2, -32768},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := make([]byte, 4)
			binary.LittleEndian.PutUint32(raw, math.Float32bits(tt.sample))

			pcm := float32ToInt16(raw)
			if len(pcm) != 2 {
				t.Fatalf("converted %d bytes, want 2", len(pcm))
			}
			if got := int16At(pcm, 0); got != tt.want {
				t.Errorf("%v converted to %d, want %d", tt.sample, got, tt.want)
			}
		})
	}
}

// TestStreamEventsEmitsAudioDeltas covers the ordinary stream: each audio delta
// is decoded and passed on in order, and the run of them makes up the utterance.
func TestStreamEventsEmitsAudioDeltas(t *testing.T) {
	stream := "event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(0, 0.5) + "\"}\n\n" +
		"event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(1, -1) + "\"}\n\n" +
		"event: speech.audio.done\n" +
		`data: {"usage":{"characters":9}}` + "\n\n"

	got := collect(t, stream)
	if len(got) != 8 {
		t.Fatalf("emitted %d bytes, want 8: four samples across two deltas", len(got))
	}
	want := []int16{0, 16383, 32767, -32767}
	for i, w := range want {
		if got := int16At(got, i); got != w {
			t.Errorf("sample %d = %d, want %d", i, got, w)
		}
	}
}

// TestStreamEventsIgnoresEverythingButDeltas covers the event selection. Only
// the audio delta carries speech, so an event that happens to hold an audio
// field without being one is not played.
func TestStreamEventsIgnoresEverythingButDeltas(t *testing.T) {
	audio := f32PCM(0.5, 0.5)
	stream := "event: speech.audio.done\n" +
		`data: {"audio_data":"` + audio + "\"}\n\n" +
		"event: speech.something.else\n" +
		`data: {"audio_data":"` + audio + "\"}\n\n"

	if got := collect(t, stream); len(got) != 0 {
		t.Errorf("emitted %d bytes from events that were not audio deltas, want none", len(got))
	}
}

// TestStreamEventsReadsTheNestedShape covers the other place the audio can sit.
// The event names itself in its payload and carries the audio one level down,
// and it has to be read the same either way.
func TestStreamEventsReadsTheNestedShape(t *testing.T) {
	stream := `data: {"event":"speech.audio.delta","data":{"audio_data":"` + f32PCM(1) + "\"}}\n\n"

	got := collect(t, stream)
	if len(got) != 2 {
		t.Fatalf("emitted %d bytes, want 2", len(got))
	}
	if s := int16At(got, 0); s != 32767 {
		t.Errorf("sample = %d, want 32767", s)
	}
}

// TestStreamEventsStopsAtDone covers the end marker: nothing after it is read,
// so audio from a stream that keeps talking past its own end is not played.
func TestStreamEventsStopsAtDone(t *testing.T) {
	stream := "event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(1) + "\"}\n\n" +
		"data: [DONE]\n\n" +
		"event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(1, 1, 1) + "\"}\n\n"

	if got := collect(t, stream); len(got) != 2 {
		t.Errorf("emitted %d bytes, want 2: the stream ended at the done marker", len(got))
	}
}

// TestStreamEventsSkipsNoise covers what a long-running stream carries between
// its events: keep-alive comments, and payloads that cannot be read. None of it
// is audio, and none of it may stop the stream.
func TestStreamEventsSkipsNoise(t *testing.T) {
	stream := ": keep-alive\n\n" +
		"event: speech.audio.delta\n" +
		"data: not json\n\n" +
		"event: speech.audio.delta\n" +
		`data: {"audio_data":"not base64!!"}` + "\n\n" +
		"event: speech.audio.delta\n" +
		`data: {"audio_data":""}` + "\n\n" +
		"event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(0.5) + "\"}\n\n"

	got := collect(t, stream)
	if len(got) != 2 {
		t.Fatalf("emitted %d bytes, want 2: only the one readable delta", len(got))
	}
	if s := int16At(got, 0); s != 16383 {
		t.Errorf("sample = %d, want 16383", s)
	}
}

// TestStreamEventsEmitsTheLastEventWithoutATrailingBlankLine covers a stream
// that ends as soon as its last event does. The event is complete, so its audio
// has to be played rather than left in the parser.
func TestStreamEventsEmitsTheLastEventWithoutATrailingBlankLine(t *testing.T) {
	stream := "event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(1) + "\"}"

	if got := collect(t, stream); len(got) != 2 {
		t.Errorf("emitted %d bytes, want 2: the final event was dropped", len(got))
	}
}

// TestStreamEventsStopsOnAFailedEmit covers the pipeline going away underneath
// the stream: the error is reported rather than swallowed, so the turn ends
// instead of reading the rest of a response nobody is listening to.
func TestStreamEventsStopsOnAFailedEmit(t *testing.T) {
	stream := "event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(1) + "\"}\n\n" +
		"event: speech.audio.delta\n" +
		`data: {"audio_data":"` + f32PCM(1) + "\"}\n\n"

	calls := 0
	err := streamEvents(strings.NewReader(stream), func([]byte) error {
		calls++
		return errDownstreamGone
	})
	if !errors.Is(err, errDownstreamGone) {
		t.Errorf("stream events returned %v, want the emit failure", err)
	}
	if calls != 1 {
		t.Errorf("emitted %d times, want 1: the stream carried on past a failed emit", calls)
	}
}
