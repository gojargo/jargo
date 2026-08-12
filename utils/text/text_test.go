package text_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/utils/text"
)

// A bare transform satisfies the Filter interface, so the package's own
// normalization steps compose with anything the TTS base accepts.
func TestTransformIsAFilter(t *testing.T) {
	var f text.Filter = text.Transform(strings.ToUpper)
	if got := f.Filter("speak up"); got != "SPEAK UP" {
		t.Errorf("Filter() = %q, want SPEAK UP", got)
	}
}
