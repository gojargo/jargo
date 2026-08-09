package frames

import (
	"fmt"
	"strings"
)

// KeypadEntry is a single DTMF keypad key: the digits 0-9, the symbols * and #,
// and the letters A-D.
type KeypadEntry string

// The DTMF keypad entries.
const (
	KeypadZero  KeypadEntry = "0"
	KeypadOne   KeypadEntry = "1"
	KeypadTwo   KeypadEntry = "2"
	KeypadThree KeypadEntry = "3"
	KeypadFour  KeypadEntry = "4"
	KeypadFive  KeypadEntry = "5"
	KeypadSix   KeypadEntry = "6"
	KeypadSeven KeypadEntry = "7"
	KeypadEight KeypadEntry = "8"
	KeypadNine  KeypadEntry = "9"
	KeypadStar  KeypadEntry = "*"
	KeypadPound KeypadEntry = "#"
	KeypadA     KeypadEntry = "A"
	KeypadB     KeypadEntry = "B"
	KeypadC     KeypadEntry = "C"
	KeypadD     KeypadEntry = "D"
)

// Valid reports whether e is one of the keypad entries. A transport reading a
// key off the wire uses it to tell a keypress from whatever else arrives.
func (e KeypadEntry) Valid() bool {
	switch e {
	case KeypadZero, KeypadOne, KeypadTwo, KeypadThree, KeypadFour, KeypadFive,
		KeypadSix, KeypadSeven, KeypadEight, KeypadNine,
		KeypadStar, KeypadPound, KeypadA, KeypadB, KeypadC, KeypadD:
		return true
	default:
		return false
	}
}

// InputDTMFFrame is a DTMF keypress received from the transport — for example a
// phone caller pressing a key. It is a system frame so it is delivered with
// priority and in order.
type InputDTMFFrame struct {
	BaseSystemFrame
	// Button is the key that was pressed.
	Button KeypadEntry
}

// NewInputDTMFFrame builds an InputDTMFFrame.
func NewInputDTMFFrame(button KeypadEntry) *InputDTMFFrame {
	return &InputDTMFFrame{
		BaseSystemFrame: NewBaseSystemFrame("InputDTMFFrame"),
		Button:          button,
	}
}

// String implements fmt.Stringer.
func (f *InputDTMFFrame) String() string {
	return fmt.Sprintf("%s(button: %s)", f.Name(), f.Button)
}

// OutputDTMFFrame requests the transport play the DTMF tones for Buttons, to
// navigate an IVR menu for example. It is a control frame, delivered in order
// behind the audio already queued, so the keys land where the caller meant them
// rather than over whatever is still being said.
type OutputDTMFFrame struct {
	BaseControlFrame
	// Buttons are the keys to emit, in order.
	Buttons []KeypadEntry
}

// NewOutputDTMFFrame builds an OutputDTMFFrame for one key.
func NewOutputDTMFFrame(button KeypadEntry) *OutputDTMFFrame {
	return &OutputDTMFFrame{
		BaseControlFrame: NewBaseControlFrame("OutputDTMFFrame"),
		Buttons:          []KeypadEntry{button},
	}
}

// NewOutputDTMFSequenceFrame builds an OutputDTMFFrame for a run of keys, as a
// caller entering an account number sends them.
func NewOutputDTMFSequenceFrame(buttons []KeypadEntry) *OutputDTMFFrame {
	return &OutputDTMFFrame{
		BaseControlFrame: NewBaseControlFrame("OutputDTMFFrame"),
		Buttons:          buttons,
	}
}

// String implements fmt.Stringer.
func (f *OutputDTMFFrame) String() string {
	return fmt.Sprintf("%s(buttons: %s)", f.Name(), KeypadString(f.Buttons))
}

// OutputDTMFUrgentFrame requests the same tones as OutputDTMFFrame but sends
// them at once, ahead of the audio already queued. It is a system frame, for a
// keypress that answers a prompt still playing.
type OutputDTMFUrgentFrame struct {
	BaseSystemFrame
	// Buttons are the keys to emit, in order.
	Buttons []KeypadEntry
}

// NewOutputDTMFUrgentFrame builds an OutputDTMFUrgentFrame for one key.
func NewOutputDTMFUrgentFrame(button KeypadEntry) *OutputDTMFUrgentFrame {
	return &OutputDTMFUrgentFrame{
		BaseSystemFrame: NewBaseSystemFrame("OutputDTMFUrgentFrame"),
		Buttons:         []KeypadEntry{button},
	}
}

// String implements fmt.Stringer.
func (f *OutputDTMFUrgentFrame) String() string {
	return fmt.Sprintf("%s(buttons: %s)", f.Name(), KeypadString(f.Buttons))
}

// KeypadString renders a run of keys the way they were pressed, so a log reads
// as the caller typed it.
func KeypadString(buttons []KeypadEntry) string {
	var b strings.Builder
	for _, e := range buttons {
		b.WriteString(string(e))
	}
	return b.String()
}

// DTMFOutput is implemented by both output DTMF frames, so a transport can take
// either without caring which queue it arrived on.
type DTMFOutput interface {
	Frame
	// Keys are the keys to emit, in order.
	Keys() []KeypadEntry
}

// Keys implements DTMFOutput.
func (f *OutputDTMFFrame) Keys() []KeypadEntry { return f.Buttons }

// Keys implements DTMFOutput.
func (f *OutputDTMFUrgentFrame) Keys() []KeypadEntry { return f.Buttons }

// Compile-time interface checks.
var (
	_ SystemFrame  = (*InputDTMFFrame)(nil)
	_ ControlFrame = (*OutputDTMFFrame)(nil)
	_ SystemFrame  = (*OutputDTMFUrgentFrame)(nil)
	_ DTMFOutput   = (*OutputDTMFFrame)(nil)
	_ DTMFOutput   = (*OutputDTMFUrgentFrame)(nil)
)
