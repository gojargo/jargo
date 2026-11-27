// Package sse reads Server-Sent Events streams.
package sse

import (
	"bufio"
	"io"
	"strings"
)

const (
	// initBuffer is the scanner's initial buffer size.
	initBuffer = 4096
	// maxBuffer bounds a single line; streamed completions can be large.
	maxBuffer = 1 << 20
)

// Scan reads a Server-Sent Events stream from r, invoking fn for each
// non-empty "data:" payload until the stream ends or fn returns an error. The
// OpenAI "[DONE]" sentinel and empty payloads are skipped, so fn only sees JSON
// chunks.
func Scan(r io.Reader, fn func(data string) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, initBuffer), maxBuffer)
	for scanner.Scan() {
		data, ok := strings.CutPrefix(scanner.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "" || data == "[DONE]" {
			continue
		}
		if err := fn(data); err != nil {
			return err
		}
	}
	return scanner.Err()
}
