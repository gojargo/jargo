// Package frames defines the Frame type and the core frame categories — system,
// data and control — that flow through a jargo pipeline.
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
	if pts, ok := f.PTS(); ok {
		return strconv.FormatInt(pts, 10)
	}
	return "none"
}

// Frame is implemented by every frame that flows through a pipeline. Concrete
// frames embed BaseFrame (directly or via BaseSystemFrame, BaseDataFrame or
// BaseControlFrame), which supplies all of these methods.
//
// The unexported isFrame marker means a type must embed BaseFrame to satisfy
// Frame; this guarantees every Frame has a valid id, name and metadata map.
// Frames carry mutable state and are passed as pointers.
type Frame interface {
	fmt.Stringer

	// ID is a process-unique identifier for this frame instance.
	ID() uint64
	// Name is a human-readable label, "<TypeName>#<n>".
	Name() string

	// PTS is the presentation timestamp in nanoseconds; ok is false if unset.
	PTS() (pts int64, ok bool)
	// SetPTS sets the presentation timestamp in nanoseconds.
	SetPTS(pts int64)

	// BroadcastSiblingID is the id of the paired frame when this frame was
	// broadcast in both directions; ok is false if unset.
	BroadcastSiblingID() (id uint64, ok bool)
	// SetBroadcastSiblingID records the paired frame id.
	SetBroadcastSiblingID(id uint64)

	// Metadata is an arbitrary, mutable per-frame metadata map.
	Metadata() map[string]any

	// TransportSource is the name of the transport that created this frame.
	TransportSource() string
	// SetTransportSource sets the transport source.
	SetTransportSource(source string)
	// TransportDestination is the name of the transport this frame targets.
	TransportDestination() string
	// SetTransportDestination sets the transport destination.
	SetTransportDestination(dest string)

	isFrame()
}

// BaseFrame is embedded by every concrete frame and implements Frame. Construct
// it with NewBaseFrame so the id, name and metadata map are initialized.
type BaseFrame struct {
	id                 uint64
	typeName           string
	pts                *int64
	broadcastSiblingID *uint64
	metadata           map[string]any
	transportSource    string
	transportDest      string
}

// NewBaseFrame initializes a BaseFrame for a concrete frame whose type is named
// typeName (e.g. "TextFrame"). It assigns a unique id; the "<typeName>#<id>"
// name is formatted on demand.
func NewBaseFrame(typeName string) BaseFrame {
	return BaseFrame{
		id:       nextID(),
		typeName: typeName,
		metadata: map[string]any{},
	}
}

// ID implements Frame.
func (f *BaseFrame) ID() uint64 { return f.id }

// Name implements Frame. The label "<typeName>#<id>" is formatted on demand.
func (f *BaseFrame) Name() string { return f.typeName + "#" + strconv.FormatUint(f.id, 10) }

// String implements fmt.Stringer and returns Name.
func (f *BaseFrame) String() string { return f.Name() }

// PTS implements Frame.
func (f *BaseFrame) PTS() (int64, bool) {
	if f.pts == nil {
		return 0, false
	}
	return *f.pts, true
}

// SetPTS implements Frame.
func (f *BaseFrame) SetPTS(pts int64) { f.pts = &pts }

// BroadcastSiblingID implements Frame.
func (f *BaseFrame) BroadcastSiblingID() (uint64, bool) {
	if f.broadcastSiblingID == nil {
		return 0, false
	}
	return *f.broadcastSiblingID, true
}

// SetBroadcastSiblingID implements Frame.
func (f *BaseFrame) SetBroadcastSiblingID(id uint64) { f.broadcastSiblingID = &id }

// Metadata implements Frame. It lazily initializes the map so a zero BaseFrame
// is still usable.
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
