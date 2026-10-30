package expressive

import (
	"strconv"
	"strings"
)

// Perception is how the user said what they said (alpha). A perception
// producer, such as an STT service or a processor after it, reads each user
// turn and sets what it perceives as the transcription's Perception: signals,
// such as a tone with the likeliest labels and how sure it is of each, or a
// measure such as speech rate. It declares the signals it reports with
// [PerceptionCapabilities], and broadcasts them at start in a
// frames.PerceptionMetadataFrame, so the LLM can be told how to read them.

// PerceptionLabel is one label of a signal read as labels, with how sure the
// reading is, from 0 to 1.
type PerceptionLabel struct {
	// Label is the label, such as "frustrated".
	Label string
	// Confidence is how sure the reading is, from 0 to 1.
	Confidence float64
}

// PerceptionSignal is one reading of how the user spoke. A signal is read
// either as labels or as a value.
type PerceptionSignal struct {
	// Name is what is read, such as "tone" or "speech_rate".
	Name string
	// Labels is the likeliest labels, likeliest first, each with how sure the
	// reading is, such as frustrated at 0.86 and neutral at 0.10. Empty for a
	// signal read as a value.
	Labels []PerceptionLabel
	// Value is the measure, for a signal read as a value, such as words per
	// second. Nil for a signal read as labels.
	Value *float64
}

// PerceptionTerm is one signal a perception producer reports.
type PerceptionTerm struct {
	// Name is the signal's name, as in PerceptionSignal.Name.
	Name string
	// Description is a short line on how to read it, such as its unit or
	// range, or empty.
	Description string
	// Labels is the labels it can report, for a signal read as labels, or
	// empty for a signal read as a value.
	Labels []string
}

// PerceptionCapabilities is the signals a perception producer reports.
type PerceptionCapabilities struct {
	// Signals is each signal the producer can report.
	Signals []PerceptionTerm
}

// FormatPerception formats a reading as written at the head of the user's
// message to the LLM.
//
// Each signal is a category in brackets: its labels with their confidences, so
// the LLM can tell a clear reading from a guess, or its measure as a value, as
// in ([tone frustrated=0.86, neutral=0.10],[speech_rate value=4.1]). A signal
// with neither labels nor a value is written with its name alone.
func FormatPerception(signals []PerceptionSignal) string {
	categories := make([]string, 0, len(signals))
	for _, signal := range signals {
		var pairs string
		switch {
		case len(signal.Labels) > 0:
			labels := make([]string, 0, len(signal.Labels))
			for _, l := range signal.Labels {
				labels = append(labels, l.Label+"="+strconv.FormatFloat(l.Confidence, 'f', 2, 64))
			}
			pairs = " " + strings.Join(labels, ", ")
		case signal.Value != nil:
			// The shortest of six significant digits, as %g writes a measure.
			pairs = " value=" + strconv.FormatFloat(*signal.Value, 'g', 6, 64)
		}
		categories = append(categories, "["+signal.Name+pairs+"]")
	}
	return "(" + strings.Join(categories, ",") + ")"
}
