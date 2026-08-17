// Package observers provides pipeline observers: components that watch the
// frames flowing through a pipeline to derive turn, latency and startup
// metrics, or to log the stream, without modifying it. Register them via
// pipeline.WorkerConfig.Observers.
//
// Every handover between two processors is reported, not only what reaches the
// ends of the pipeline, so an observer sees where each frame came from. Each
// observer here is safe for concurrent use: a pipeline's processors each run on
// their own goroutine.
package observers

import (
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// skipBroadcastSibling reports whether a frame is the upstream half of a
// broadcast pair and should be ignored. A broadcast builds a distinct frame for
// each direction, paired by BroadcastSiblingID, so an observer that watched both
// would report one event twice. Counting only the downstream half reports it
// once, and the pairing is what makes the two halves recognizable: their ids
// deliberately differ, so the id deduper below cannot catch them.
func skipBroadcastSibling(f frames.Frame, dir processor.Direction) bool {
	_, paired := f.Base().BroadcastSiblingID()
	return paired && dir != processor.Downstream
}

// defaultMaxFrames is how many recent frame ids an observer remembers to
// recognize one it has already counted.
const defaultMaxFrames = 100

// deduper drops a frame already seen, keeping a bounded window of recent ids. It
// catches the same instance arriving twice, a frame pushed on and later echoed
// back, which is distinct from the broadcast pairing above.
type deduper struct {
	seen  map[uint64]struct{}
	order []uint64
	max   int
}

// newDeduper builds a deduper remembering the last n ids; 0 uses the default.
func newDeduper(n int) deduper {
	if n <= 0 {
		n = defaultMaxFrames
	}
	return deduper{seen: map[uint64]struct{}{}, max: n}
}

// seenBefore reports whether id was already observed, recording it otherwise.
func (d *deduper) seenBefore(id uint64) bool {
	if _, ok := d.seen[id]; ok {
		return true
	}
	d.seen[id] = struct{}{}
	d.order = append(d.order, id)
	if len(d.order) > d.max {
		old := d.order[0]
		d.order = d.order[1:]
		delete(d.seen, old)
	}
	return false
}

// defaultTurnEndTimeout is how long after the bot stops speaking a turn is
// considered ended, absent a new user turn.
const defaultTurnEndTimeout = 2500 * time.Millisecond
