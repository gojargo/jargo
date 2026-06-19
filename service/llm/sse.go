package llm

import (
	"bufio"
	"io"
	"strings"
)

const (
	// sseInitBuffer is the SSE scanner's initial buffer size.
	sseInitBuffer = 4096
	// sseMaxBuffer bounds a single SSE line; streamed completions can be large.
	sseMaxBuffer = 1 << 20
)

// ScanSSE reads a Server-Sent Events stream from r, invoking fn for each
// non-empty "data:" payload until the stream ends or fn returns an error. The
// OpenAI "[DONE]" sentinel and empty payloads are skipped, so fn only sees JSON
// chunks.
func ScanSSE(r io.Reader, fn func(data string) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, sseInitBuffer), sseMaxBuffer)
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
