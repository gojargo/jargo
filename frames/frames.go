// Package frames defines the Frame type and the core frame categories — system,
// data and control — that flow through a jargo pipeline.
//
// # Categories
//
// Every frame belongs to exactly one category, which decides how the pipeline
// schedules it and whether an interruption may drop it:
//
//   - [SystemFrame] is delivered ahead of queued work and is never dropped by an
//     interruption. Lifecycle and out-of-band signals: [StartFrame],
//     [CancelFrame], [InterruptionFrame], [MetricsFrame].
//   - [DataFrame] is processed in order and is canceled by an interruption. The
//     payload frames: [TextFrame], [OutputAudioRawFrame], [LLMContextFrame].
//   - [ControlFrame] is processed in order like a data frame and is likewise
//     canceled, but carries instructions rather than payload: [EndFrame],
//     [TTSStartedFrame], [FunctionCallResultFrame].
//
// A frame joins a category by embedding [BaseSystemFrame], [BaseDataFrame] or
// [BaseControlFrame]; assert the matching interface to test the category. The
// [Uninterruptible] marker is orthogonal: embed [UninterruptibleMixin] alongside
// a data or control base to keep a frame queued through an interruption.
//
// # Ownership
//
// A frame carries mutable state behind pointer receivers and is deliberately not
// synchronized. Exactly one goroutine owns a frame at a time: a processor may
// read and mutate a frame until it pushes the frame onward, and must not touch
// it afterwards. Do not mutate a frame that is being pushed in both directions
// at once — the two ends run on separate goroutines, so a shared frame must be
// treated as read-only by both.
//
// [LLMContext] is the exception. It is a long-lived aggregate shared by the
// aggregators and the LLM service rather than a frame, and it is safe for
// concurrent use.
package frames

import (
	"fmt"
	"strconv"
	"sync/atomic"
)

// idCounter mints a process-unique, monotonically increasing frame id. A single
// lock-free atomic; ids start at 1, so the zero value means "no id assigned".
//
//nolint:gochecknoglobals // process-wide id source
var idCounter atomic.Uint64

func nextID() uint64 { return idCounter.Add(1) }

// formatPTS renders a frame's presentation timestamp for String output,
// returning the nanosecond value or "none" when the timestamp is unset.
func formatPTS(f Frame) string {
	if pts, ok := f.Base().PTS(); ok {
		return strconv.FormatInt(pts, 10)
	}
	return "none"
}

// Frame is implemented by every frame that flows through a pipeline. Concrete
// frames embed BaseFrame (directly or via BaseSystemFrame, BaseDataFrame or
// BaseControlFrame), which supplies these methods.
//
// The unexported isFrame marker means a type must embed BaseFrame to satisfy
// Frame; this guarantees every Frame has a valid id and name. Frames carry
// mutable state and are passed as pointers.
//
// The interface is deliberately narrow: identity is all a pipeline needs to
// route, log and correlate a frame. The optional per-frame state — presentation
// timestamp, metadata, transport source and destination, broadcast sibling id —
// lives on [BaseFrame] and is reached through Base. Those accessors are also
// promoted onto every concrete frame, so a caller holding a concrete type can
// keep calling them directly.
type Frame interface {
	fmt.Stringer

	// ID is a process-unique identifier for this frame instance.
	ID() uint64
	// Name is a human-readable label, "<TypeName>#<n>".
	Name() string

	// Base exposes the embedded BaseFrame and the optional state it carries.
	Base() *BaseFrame

	isFrame()
}

// BaseFrame is embedded by every concrete frame and implements Frame. Construct
// it with NewBaseFrame so the id and name are initialized.
type BaseFrame struct {
	id                 uint64
	typeName           string
	pts                int64
	hasPTS             bool
	broadcastSiblingID *uint64
	metadata           map[string]any
	transportSource    string
	transportDest      string
}

// NewBaseFrame initializes a BaseFrame for a concrete frame whose type is named
// typeName (e.g. "TextFrame"). It assigns a unique id; the "<typeName>#<id>"
// name is formatted on demand.
func NewBaseFrame(typeName string) BaseFrame {
	return BaseFrame{id: nextID(), typeName: typeName}
}

// Base implements Frame, returning the BaseFrame itself so a caller holding the
// Frame interface can reach the optional per-frame state.
func (f *BaseFrame) Base() *BaseFrame { return f }

// ID implements Frame.
func (f *BaseFrame) ID() uint64 { return f.id }

// Name implements Frame. The label "<typeName>#<id>" is formatted on demand.
func (f *BaseFrame) Name() string { return f.typeName + "#" + strconv.FormatUint(f.id, 10) }

// String implements fmt.Stringer and returns Name.
func (f *BaseFrame) String() string { return f.Name() }

// PTS implements Frame.
func (f *BaseFrame) PTS() (int64, bool) { return f.pts, f.hasPTS }

// SetPTS implements Frame.
func (f *BaseFrame) SetPTS(pts int64) {
	f.pts = pts
	f.hasPTS = true
}

// BroadcastSiblingID implements Frame.
func (f *BaseFrame) BroadcastSiblingID() (uint64, bool) {
	if f.broadcastSiblingID == nil {
		return 0, false
	}
	return *f.broadcastSiblingID, true
}

// SetBroadcastSiblingID implements Frame.
func (f *BaseFrame) SetBroadcastSiblingID(id uint64) { f.broadcastSiblingID = &id }

// Metadata implements Frame. The map is allocated on first use, so a frame that
// carries no metadata costs nothing. Like the rest of a frame's state it is
// unsynchronized: only the goroutine that owns the frame may call this.
func (f *BaseFrame) Metadata() map[string]any {
	if f.metadata == nil {
		f.metadata = map[string]any{}
	}
	return f.metadata
}

// TransportSource implements Frame.
func (f *BaseFrame) TransportSource() string { return f.transportSource }

// SetTransportSource implements Frame.
func (f *BaseFrame) SetTransportSource(source string) { f.transportSource = source }

// TransportDestination implements Frame.
func (f *BaseFrame) TransportDestination() string { return f.transportDest }

// SetTransportDestination implements Frame.
func (f *BaseFrame) SetTransportDestination(dest string) { f.transportDest = dest }

func (f *BaseFrame) isFrame() {}

//
// Categories
//

// SystemFrame takes priority over other frames and is not affected by user
// interruptions; system frames are handled in order. Assert a Frame to
// SystemFrame to test its category. Embed BaseSystemFrame to define one.
type SystemFrame interface {
	Frame
	isSystemFrame()
}

// DataFrame is processed in order and is canceled by user interruptions. It
// usually carries data such as LLM context, text, audio or images. Embed
// BaseDataFrame to define one.
type DataFrame interface {
	Frame
	isDataFrame()
}

// ControlFrame is processed in order like a DataFrame and is canceled by user
// interruptions; it carries control information such as settings updates or a
// request to end the pipeline once everything is flushed. Embed BaseControlFrame
// to define one.
type ControlFrame interface {
	Frame
	isControlFrame()
}

// BaseSystemFrame is embedded by system frames. Construct with NewBaseSystemFrame.
type BaseSystemFrame struct{ BaseFrame }

func (*BaseSystemFrame) isSystemFrame() {}

// NewBaseSystemFrame initializes a BaseSystemFrame for the named concrete type.
func NewBaseSystemFrame(typeName string) BaseSystemFrame {
	return BaseSystemFrame{NewBaseFrame(typeName)}
}

// BaseDataFrame is embedded by data frames. Construct with NewBaseDataFrame.
type BaseDataFrame struct{ BaseFrame }

func (*BaseDataFrame) isDataFrame() {}

// NewBaseDataFrame initializes a BaseDataFrame for the named concrete type.
func NewBaseDataFrame(typeName string) BaseDataFrame {
	return BaseDataFrame{NewBaseFrame(typeName)}
}

// BaseControlFrame is embedded by control frames. Construct with NewBaseControlFrame.
type BaseControlFrame struct{ BaseFrame }

func (*BaseControlFrame) isControlFrame() {}

// NewBaseControlFrame initializes a BaseControlFrame for the named concrete type.
func NewBaseControlFrame(typeName string) BaseControlFrame {
	return BaseControlFrame{NewBaseFrame(typeName)}
}

//
// Mixins
//

// Uninterruptible marks a data or control frame that must survive interruptions:
// it stays queued and any task processing it is never canceled, guaranteeing
// delivery and completion. Embed UninterruptibleMixin (alongside a category
// base) and assert with this interface.
type Uninterruptible interface {
	isUninterruptible()
}

// UninterruptibleMixin is embedded to mark a frame Uninterruptible.
type UninterruptibleMixin struct{}

func (UninterruptibleMixin) isUninterruptible() {}

// Compile-time guarantees that the base structs satisfy their interfaces.
var (
	_ Frame        = (*BaseFrame)(nil)
	_ SystemFrame  = (*BaseSystemFrame)(nil)
	_ DataFrame    = (*BaseDataFrame)(nil)
	_ ControlFrame = (*BaseControlFrame)(nil)
)
