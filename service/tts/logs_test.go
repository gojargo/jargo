package tts_test

import (
	"bytes"
	"sync"
)

// syncBuffer collects log output written from the pipeline's own goroutines,
// which go on logging while a test reads what was logged so far.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
