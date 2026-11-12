package perception

import (
	"testing"
	"time"
)

// SetSpeechEndWait shortens how long a transcription waits for the user to
// stop speaking, for the test.
func SetSpeechEndWait(t *testing.T, d time.Duration) {
	t.Helper()
	old := speechEndWait
	speechEndWait = d
	t.Cleanup(func() { speechEndWait = old })
}
