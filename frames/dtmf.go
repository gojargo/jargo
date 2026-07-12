package frames

import "fmt"

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

// OutputDTMFFrame requests the transport play a DTMF tone for Button — for
// example to navigate an IVR menu. It is a control frame, delivered in order
// after preceding audio.
type OutputDTMFFrame struct {
	BaseControlFrame
	// Button is the key to emit.
	Button KeypadEntry
}

// NewOutputDTMFFrame builds an OutputDTMFFrame.
func NewOutputDTMFFrame(button KeypadEntry) *OutputDTMFFrame {
	return &OutputDTMFFrame{
		BaseControlFrame: NewBaseControlFrame("OutputDTMFFrame"),
		Button:           button,
	}
}

// String implements fmt.Stringer.
func (f *OutputDTMFFrame) String() string {
	return fmt.Sprintf("%s(button: %s)", f.Name(), f.Button)
}

// Compile-time interface checks.
var (
	_ SystemFrame  = (*InputDTMFFrame)(nil)
	_ ControlFrame = (*OutputDTMFFrame)(nil)
)
