//go:build !rnnoise

package rnnoise

import (
	"bytes"
	"context"
	"testing"
)

func TestPassthrough(t *testing.T) {
	f, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	ctx := context.Background()
	if err := f.Start(ctx, 48000); err != nil {
		t.Fatalf("Start: %v", err)
	}
	in := []byte{1, 2, 3, 4}
	out, err := f.Filter(ctx, in)
	if err != nil {
		t.Fatalf("Filter: %v", err)
	}
	if !bytes.Equal(out, in) {
		t.Fatalf("passthrough changed audio: got %v, want %v", out, in)
	}
	if err := f.Stop(ctx); err != nil {
		t.Fatalf("Stop: %v", err)
	}
}
