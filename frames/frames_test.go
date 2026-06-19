package frames_test

import (
	"fmt"
	"testing"

	"github.com/gojargo/jargo/frames"
)

// Sample concrete frames, defined the way downstream code will: embed a
// category base and call its constructor with the type name.

type sampleTextFrame struct {
	frames.BaseDataFrame
	Text string
}

func newSampleTextFrame(text string) *sampleTextFrame {
	return &sampleTextFrame{
		BaseDataFrame: frames.NewBaseDataFrame("sampleTextFrame"),
		Text:          text,
	}
}

type sampleStartFrame struct {
	frames.BaseSystemFrame
}

func newSampleStartFrame() *sampleStartFrame {
	return &sampleStartFrame{BaseSystemFrame: frames.NewBaseSystemFrame("sampleStartFrame")}
}

// sampleEndFrame is a control frame that is also uninterruptible.
type sampleEndFrame struct {
	frames.BaseControlFrame
	frames.UninterruptibleMixin
}

func newSampleEndFrame() *sampleEndFrame {
	return &sampleEndFrame{BaseControlFrame: frames.NewBaseControlFrame("sampleEndFrame")}
}

// Compile-time checks that the samples satisfy the expected interfaces.
var (
	_ frames.Frame           = (*sampleTextFrame)(nil)
	_ frames.DataFrame       = (*sampleTextFrame)(nil)
	_ frames.SystemFrame     = (*sampleStartFrame)(nil)
	_ frames.ControlFrame    = (*sampleEndFrame)(nil)
	_ frames.Uninterruptible = (*sampleEndFrame)(nil)
)

func TestIDsAreUniqueAndIncreasing(t *testing.T) {
	const n = 1000
	seen := make(map[uint64]bool, n)
	var prev uint64
	for i := range n {
		f := newSampleTextFrame("x")
		id := f.ID()
		if seen[id] {
			t.Fatalf("duplicate id %d", id)
		}
		if i > 0 && id <= prev {
			t.Fatalf("id not increasing: %d after %d", id, prev)
		}
		seen[id] = true
		prev = id
	}
}

func TestNameFormat(t *testing.T) {
	const typeName = "TestNameFormatFrame"
	bf := frames.NewBaseDataFrame(typeName)
	want := fmt.Sprintf("%s#%d", typeName, bf.ID())
	if got := bf.Name(); got != want {
		t.Errorf("Name() = %q, want %q", got, want)
	}
}

func TestStringEqualsName(t *testing.T) {
	f := newSampleTextFrame("hi")
	if f.String() != f.Name() {
		t.Errorf("String() = %q, want Name() %q", f.String(), f.Name())
	}
}

func TestCategoriesAreExclusive(t *testing.T) {
	var data frames.Frame = newSampleTextFrame("x")
	if _, ok := data.(frames.DataFrame); !ok {
		t.Error("text frame should be a DataFrame")
	}
	if _, ok := data.(frames.SystemFrame); ok {
		t.Error("text frame should not be a SystemFrame")
	}
	if _, ok := data.(frames.ControlFrame); ok {
		t.Error("text frame should not be a ControlFrame")
	}

	var sys frames.Frame = newSampleStartFrame()
	if _, ok := sys.(frames.SystemFrame); !ok {
		t.Error("start frame should be a SystemFrame")
	}
	if _, ok := sys.(frames.DataFrame); ok {
		t.Error("start frame should not be a DataFrame")
	}
}

func TestUninterruptibleMarker(t *testing.T) {
	var end frames.Frame = newSampleEndFrame()
	if _, ok := end.(frames.ControlFrame); !ok {
		t.Error("end frame should be a ControlFrame")
	}
	if _, ok := end.(frames.Uninterruptible); !ok {
		t.Error("end frame should be Uninterruptible")
	}
	// A plain data frame must not be Uninterruptible.
	var data frames.Frame = newSampleTextFrame("x")
	if _, ok := data.(frames.Uninterruptible); ok {
		t.Error("text frame should not be Uninterruptible")
	}
}

func TestPTSOptional(t *testing.T) {
	f := newSampleTextFrame("x")
	if _, ok := f.PTS(); ok {
		t.Error("PTS should be unset on a new frame")
	}
	f.SetPTS(1_500)
	if pts, ok := f.PTS(); !ok || pts != 1_500 {
		t.Errorf("PTS() = (%d, %v), want (1500, true)", pts, ok)
	}
}

func TestBroadcastSiblingOptional(t *testing.T) {
	f := newSampleTextFrame("x")
	if _, ok := f.BroadcastSiblingID(); ok {
		t.Error("broadcast sibling id should be unset on a new frame")
	}
	f.SetBroadcastSiblingID(42)
	if id, ok := f.BroadcastSiblingID(); !ok || id != 42 {
		t.Errorf("BroadcastSiblingID() = (%d, %v), want (42, true)", id, ok)
	}
}

func TestMetadataMutable(t *testing.T) {
	f := newSampleTextFrame("x")
	if f.Metadata() == nil {
		t.Fatal("metadata map should be initialized")
	}
	f.Metadata()["k"] = "v"
	if got := f.Metadata()["k"]; got != "v" {
		t.Errorf("metadata[k] = %v, want v", got)
	}
}

func TestTransportSourceDest(t *testing.T) {
	f := newSampleTextFrame("x")
	f.SetTransportSource("mic")
	f.SetTransportDestination("speaker")
	if f.TransportSource() != "mic" || f.TransportDestination() != "speaker" {
		t.Errorf("transport src/dest = %q/%q, want mic/speaker",
			f.TransportSource(), f.TransportDestination())
	}
}
