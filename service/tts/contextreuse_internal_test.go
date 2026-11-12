package tts

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
)

// idSynth hands each unit's audio to its context, as a provider whose audio
// arrives on its own does, and records the context each unit was spoken on.
// closeEach ends every context once its unit is spoken, for a provider that
// ends each request on its own.
type idSynth struct {
	host      AudioContextHost
	closeEach bool

	mu  sync.Mutex
	ids []string
}

func (s *idSynth) SampleRate() int { return wsSampleRate }

func (s *idSynth) SetAudioContextHost(h AudioContextHost) { s.host = h }

func (s *idSynth) RunTTS(_ context.Context, _, contextID string, _ func(frames.Frame) error) error {
	s.mu.Lock()
	s.ids = append(s.ids, contextID)
	s.mu.Unlock()
	s.host.AppendToAudioContext(contextID, frames.NewTTSAudioRawFrame(make([]byte, 320), wsSampleRate, 1))
	if s.closeEach {
		s.host.RemoveAudioContext(contextID)
	}
	return nil
}

func (s *idSynth) contextIDs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.ids...)
}

// TestContextReuseWithinTurn checks the sentences of a turn share its context
// by default, and each get one of their own when the service does not reuse it.
func TestContextReuseWithinTurn(t *testing.T) {
	for _, reuse := range []bool{true, false} {
		syn := &idSynth{closeEach: !reuse}
		b := New("ContextReuseTTS", syn)
		b.stopFrameTimeout = 200 * time.Millisecond
		b.SetReuseContextIDWithinTurn(reuse)
		down := runBase(t, b,
			frames.NewLLMFullResponseStartFrame(),
			frames.NewLLMTextFrame("One. Two. "),
			frames.NewLLMFullResponseEndFrame(),
		)
		ids := syn.contextIDs()
		if len(ids) != 2 {
			t.Fatalf("reuse %v: spoke %d units, want 2", reuse, len(ids))
		}
		if shared := ids[0] == ids[1]; shared != reuse {
			t.Errorf("reuse %v: contexts %q, shared = %v", reuse, ids, shared)
		}
		started := 0
		for _, f := range down {
			if _, ok := f.(*frames.TTSStartedFrame); ok {
				started++
			}
		}
		if want := map[bool]int{true: 1, false: 2}[reuse]; started != want {
			t.Errorf("reuse %v: %d started frames, want %d", reuse, started, want)
		}
	}
}
