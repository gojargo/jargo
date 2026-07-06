//go:build rnnoise

package rnnoise

import (
	"context"
	"testing"
)

func TestDenoiseFullFrame(t *testing.T) {
	f, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	if err := f.Start(ctx, 48000); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer func() { _ = f.Stop(ctx) }()

	// One exact 480-sample (960-byte) frame at 48 kHz yields one frame out.
	out, err := f.Filter(ctx, make([]byte, 960))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if len(out) != 960 {
		t.Fatalf("full frame out = %d bytes, want 960", len(out))
	}
}

func TestDenoiseBuffersPartialFrame(t *testing.T) {
	f, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	if err := f.Start(ctx, 48000); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer func() { _ = f.Stop(ctx) }()

	// A sub-frame chunk is buffered, so nothing is emitted yet.
	out, err := f.Filter(ctx, make([]byte, 100))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if out != nil {
		t.Fatalf("partial frame should buffer, got %d bytes", len(out))
	}

	// Completing the frame (100 + 860 = 960) emits exactly one frame.
	out, err = f.Filter(ctx, make([]byte, 860))
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if len(out) != 960 {
		t.Fatalf("completed frame out = %d bytes, want 960", len(out))
	}
}

func TestDenoiseResamplesOtherRate(t *testing.T) {
	f, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	if err := f.Start(ctx, 16000); err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer func() { _ = f.Stop(ctx) }()

	// 160 samples at 16 kHz upsamples to 480 at 48 kHz — one frame. Both the
	// up- and down-resamplers have several calls of soxr warm-up latency, so
	// feed enough chunks to clear it and require audio to eventually flow.
	var total int
	for range 30 {
		out, err := f.Filter(ctx, make([]byte, 320))
		if err != nil {
			t.Fatalf("Filter: %v", err)
		}
		total += len(out)
	}
	if total == 0 {
		t.Fatal("no audio emitted across thirty 16 kHz chunks")
	}
}
