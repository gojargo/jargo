package llm

import (
	"log/slog"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// speculationState is what the gate is doing with the frames passing through it.
type speculationState int

const (
	// speculationOpen emits everything.
	speculationOpen speculationState = iota
	// speculationHolding holds a speculative reply until it is confirmed.
	speculationHolding
	// speculationDropping discards the rest of a withdrawn speculative reply.
	speculationDropping
)

// gatedFrame is a frame paired with the direction it travels in.
type gatedFrame struct {
	frame frames.Frame
	dir   processor.Direction
}

// speculationGate decides which frames of a speculative reply may be emitted,
// and when.
//
// A speculative reply is generated from an eager end of turn, a provisional
// guess that the user has finished talking, so it may answer a transcript the
// user never actually completed. The gate holds everything such a reply produces
// until the turn is confirmed, then releases it, or discards it if the guess is
// withdrawn.
//
// The host calls beginSpeculation when it starts an inference for a turn that
// may not have ended, and the reply that follows is held from its
// LLMFullResponseStartFrame onward. A UserStoppedSpeakingFrame releases it, the
// turn it answers having ended, and an EagerEndOfTurnCancelFrame discards it.
// Neither names a reply: only one speculation is ever in flight, since producing
// one takes a whole user turn.
//
// Whether an inference is still answering an unconfirmed turn is therefore the
// gate's to answer, through isSpeculating, rather than something the host tracks
// alongside it. A host keeping its own copy would have to clear it on every path
// that settles a speculation, and the two would disagree on the paths it missed.
//
// The gate also tracks whether the turn has ended, because a confirmation can
// arrive before the inference it confirms (system frames are dispatched ahead of
// the queued frames they pass) and a reply for a turn already over is not held
// at all.
//
// Both resolving frames reach the gate ahead of anything queued behind them,
// which is what keeps a withdrawal in front of the turn end that follows it. A
// withdrawal arriving second would find the reply already released.
//
// While holding, everything is held in arrival order except system frames, which
// are out of band throughout the pipeline, and which carry the verdicts the gate
// is waiting for: holding them would deadlock it. That includes uninterruptible
// frames, which are ordered like any other; discarding a speculation keeps them
// and emits them on, since they can belong to work started before it.
//
// This decides rather than processes frames: process is synchronous and returns
// the frames its caller should push, in order. A host can therefore push from
// several goroutines at once, since every state transition completes without
// yielding for another to interleave with.
//
// Holding is unbounded here. Whoever started the speculation bounds it and
// withdraws it if it goes unresolved, so every way a speculation ends reaches
// this gate as a frame and the gate never has to abandon one on its own.
type speculationGate struct {
	name  string
	state speculationState
	// speculating reports whether the inference in flight answers a turn that
	// has not been confirmed. It is set when the inference starts and cleared
	// when the turn settles it either way, so it outlives any one reply frame.
	speculating bool
	// turnEnded reports whether the turn ended since it was last seen to start.
	// A speculation registered while this is set answers a turn that is already
	// over, so it is not held. It starts false, so a gate that never sees turn
	// frames holds rather than speaks, and the bound on the speculation recovers
	// it.
	turnEnded bool
	buffer    []gatedFrame
}

// newSpeculationGate builds a gate labeled with the owning service's name.
func newSpeculationGate(name string) *speculationGate {
	return &speculationGate{name: name}
}

// isSpeculating reports whether the inference in flight answers an unconfirmed
// turn.
//
// It is set from beginSpeculation until the turn settles it, so it covers the
// whole inference rather than only the stretch with frames in flight. False
// means whatever is generating now answers a turn that has ended, and its side
// effects can be let through.
func (g *speculationGate) isSpeculating() bool { return g.speculating }

// beginSpeculation takes note of an inference whose turn may not have ended.
//
// It is called when the inference starts rather than when its first reply frame
// arrives, so a turn confirmed in between is already accounted for by the time
// the reply shows up.
func (g *speculationGate) beginSpeculation(speculation bool) {
	// A turn that ended before the inference reached us leaves nothing to hold
	// for: the reply answers a turn that is already over.
	g.speculating = speculation && !g.turnEnded
}

// process decides what a frame passing through the gate releases. It returns the
// frames to push, in order: empty while a frame is held or dropped, and longer
// than one frame when a verdict releases what was held behind it.
func (g *speculationGate) process(f frames.Frame, dir processor.Direction) []gatedFrame {
	if dir != processor.Downstream {
		return []gatedFrame{{f, dir}}
	}

	if _, isSystem := f.(frames.SystemFrame); isSystem {
		// Emitted before the verdict is applied, so a released reply still
		// follows the frame that ended the turn it answers.
		emitted := []gatedFrame{{f, dir}}
		switch f.(type) {
		case *frames.EagerEndOfTurnCancelFrame, *frames.InterruptionFrame:
			emitted = append(emitted, g.discard()...)
		case *frames.UserStoppedSpeakingFrame:
			emitted = append(emitted, g.release()...)
		case *frames.UserStartedSpeakingFrame:
			// A turn is open again, so an inference started from here on answers
			// something that may not have ended.
			g.turnEnded = false
		}
		return emitted
	}

	if _, isEnd := f.(*frames.EndFrame); isEnd {
		// Uninterruptible, and the worker awaits it: holding it hangs shutdown.
		// Discarding first delivers whatever is held that has to outlive the
		// speculation, ahead of it.
		return append(g.discard(), gatedFrame{f, dir})
	}

	var emitted []gatedFrame
	if _, starts := f.(*frames.LLMFullResponseStartFrame); starts {
		emitted = append(emitted, g.begin()...)
	}

	switch g.state {
	case speculationHolding:
		// Held in arrival order, uninterruptible frames included: they are
		// ordered like any other, and the buffer preserves them when the
		// speculation around them is discarded.
		g.buffer = append(g.buffer, gatedFrame{f, dir})
	case speculationDropping:
		if _, uninterruptible := f.(frames.Uninterruptible); uninterruptible {
			// Not part of the reply being dropped, and nothing is being held
			// back, so emitting it keeps it in order.
			emitted = append(emitted, gatedFrame{f, dir})
		} else if _, ends := f.(*frames.LLMFullResponseEndFrame); ends {
			g.state = speculationOpen
		}
	case speculationOpen:
		emitted = append(emitted, gatedFrame{f, dir})
	}

	return emitted
}

// begin decides what to do with the reply opening here.
func (g *speculationGate) begin() []gatedFrame {
	var emitted []gatedFrame

	if g.state != speculationOpen {
		// A new reply supersedes the one being held or dropped. A withdrawn
		// reply may never send its end frame, its generation having been
		// canceled mid-flight, and an unconfirmed one is void once something
		// else starts answering. Nothing of it can still be queued behind this
		// frame, so there is no tail left to drop.
		emitted = append(emitted, g.dropHeld("superseded by a new reply", false)...)
	}

	// Whether this reply is speculative was settled when its inference started,
	// so a turn confirmed since then has already cleared it.
	if !g.speculating {
		return emitted
	}

	g.state = speculationHolding
	return emitted
}

// release releases whatever the turn ending confirms, in arrival order.
func (g *speculationGate) release() []gatedFrame {
	// Remembered so an inference arriving after this is not held: the turn it
	// answers is already over, and nothing follows to release it.
	g.turnEnded = true

	// Nothing this inference still produces is speculative either, including a
	// tool call it has yet to reach.
	g.speculating = false

	if g.state != speculationHolding {
		return nil
	}

	slog.Debug("releasing the speculative reply", "service", g.name, "frames", len(g.buffer))
	g.state = speculationOpen
	return g.flush()
}

// discard discards the reply a withdrawal or an interruption voids, returning
// whatever it was holding back that has to be delivered anyway.
//
// A withdrawal arriving before the reply it voids needs no memory: the reply is
// held on arrival, and whatever answers the turn instead supersedes it, or the
// bound on the hold runs out if nothing does.
func (g *speculationGate) discard() []gatedFrame {
	// The speculation is void, so nothing the inference still produces answers a
	// turn worth holding for.
	g.speculating = false
	return g.dropHeld("withdrawn", true)
}

// dropHeld drops the held reply, returning the frames that must be delivered
// anyway, in arrival order.
//
// keepDropping says whether the rest of the reply may still be queued behind us
// and has to be dropped as it arrives. It is false when something already past
// it proves there is no tail left.
func (g *speculationGate) dropHeld(reason string, keepDropping bool) []gatedFrame {
	if g.state != speculationHolding {
		g.state = speculationOpen
		return nil
	}

	slog.Debug("discarding the speculative reply",
		"service", g.name, "frames", len(g.buffer), "reason", reason)

	// A reply that already ended has no tail left to drop.
	complete := false
	for _, held := range g.buffer {
		if _, ends := held.frame.(*frames.LLMFullResponseEndFrame); ends {
			complete = true
			break
		}
	}

	// Drops the speculative reply and keeps anything that must always be
	// delivered, which is then emitted rather than discarded with it.
	kept := g.buffer[:0]
	for _, held := range g.buffer {
		if _, uninterruptible := held.frame.(frames.Uninterruptible); uninterruptible {
			kept = append(kept, held)
		}
	}
	g.buffer = kept

	g.state = speculationOpen
	if keepDropping && !complete {
		g.state = speculationDropping
	}
	return g.flush()
}

// flush takes everything the buffer still holds, in the order it arrived.
func (g *speculationGate) flush() []gatedFrame {
	emitted := g.buffer
	g.buffer = nil
	return emitted
}
