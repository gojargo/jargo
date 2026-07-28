package pionrtc

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/transport"
)

// newTestOutput builds an output transport over an unconnected peer connection.
// Writes go to a track with nothing bound to it, which is enough to exercise the
// sender's own behavior.
func newTestOutput(t *testing.T) *outputTransport {
	t.Helper()
	conn, err := NewConnection(WithICEServers())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = conn.Close() })

	params := transport.DefaultParams()
	params.AudioOutSampleRate = opus.SampleRate
	out := newOutput(conn, params)
	if err := out.startSending(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(out.stopSending)
	return out
}

// The sender takes whatever is queued at the moment it comes round, so the writer
// has to be allowed to run ahead of it. A writer that could only ever have one
// frame outstanding would leave the queue empty every time the sender looked,
// and any hesitation upstream would put silence in the middle of a word.
func TestWriteAudioRunsAheadOfTheSender(t *testing.T) {
	out := newTestOutput(t)
	frameBytes := opus.FrameBytes(1)

	// Comfortably inside the cushion: this must not wait for playout.
	frames := queuedFrames / 2
	pcm := make([]byte, frameBytes*frames)

	started := time.Now()
	if err := out.WriteAudio(context.Background(), pcm); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(started)

	// Handing frames to a queue with room in it is bookkeeping, so this should
	// not take so much as one frame's playout. Waiting even roughly that long
	// means each frame is being waited on individually.
	if elapsed >= opus.FrameDuration {
		t.Fatalf("writing %d frames into a cushion of %d took %v, at least a frame's "+
			"playout: the writer cannot run ahead, so the sender will starve",
			frames, queuedFrames, elapsed)
	}
}

// The cushion is slack, not a buffer to dump a whole utterance into: once it is
// full the writer has to wait, which is what paces the pipeline to real time.
func TestWriteAudioBlocksOncePastTheCushion(t *testing.T) {
	out := newTestOutput(t)
	frameBytes := opus.FrameBytes(1)

	frames := queuedFrames * 3
	pcm := make([]byte, frameBytes*frames)

	started := time.Now()
	if err := out.WriteAudio(context.Background(), pcm); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(started)

	// Everything past the cushion has to be played before it can be accepted.
	least := time.Duration(frames-queuedFrames) * opus.FrameDuration
	// Allow for the sender having drained a frame or two while we measured.
	least -= 2 * opus.FrameDuration
	if elapsed < least {
		t.Fatalf("writing %d frames returned in %v, less than the %v of playout it "+
			"should have waited for: output is not paced", frames, elapsed, least)
	}
}
