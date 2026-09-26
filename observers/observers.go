// Package observers provides pipeline observers: components that watch the
// frames flowing through a pipeline without modifying it. They derive turn,
// latency and startup metrics, report the conversation's speaking lifecycle,
// the function calls it makes, the failures it runs into and what each service
// spent and consumed, or log the stream. Register them via
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
// once, and the pairing is what makes the two halves recognizable: they are two
// frames, each pushed for the first time once.
func skipBroadcastSibling(f frames.Frame, dir processor.Direction) bool {
	_, paired := f.Base().BroadcastSiblingID()
	return paired && dir != processor.Downstream
}

// defaultTurnEndTimeout is how long after the bot stops speaking a turn is
// considered ended, absent a new user turn.
const defaultTurnEndTimeout = 2500 * time.Millisecond
