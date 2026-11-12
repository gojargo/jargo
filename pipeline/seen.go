package pipeline

import (
	"runtime"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// seenFrames is the set of frames already pushed, by id, so the copies of a
// frame that come out of more than one branch, or out of a branch later, are
// pushed once.
//
// It holds its frames weakly: a frame nothing holds any more cannot come out of
// a branch again, so its entry goes when the frame is collected. The set stays
// as large as the frames still in flight rather than growing with every frame
// of the session.
//
// The zero value is ready to use. It is safe for concurrent use.
type seenFrames struct {
	mu  sync.Mutex
	ids map[uint64]struct{}
}

// add records f as pushed and reports whether it was not already.
func (s *seenFrames) add(f frames.Frame) bool {
	id := f.ID()
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.ids[id]; ok {
		return false
	}
	if s.ids == nil {
		s.ids = make(map[uint64]struct{})
	}
	s.ids[id] = struct{}{}
	// The frame embeds its base, so the base lives exactly as long as it does.
	runtime.AddCleanup(f.Base(), s.forget, id)
	return true
}

// forget drops a frame that has been collected.
func (s *seenFrames) forget(id uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.ids, id)
}

// len is how many frames the set holds.
func (s *seenFrames) len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.ids)
}
