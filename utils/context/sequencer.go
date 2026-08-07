package context

import (
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// aggregatedFrameSlot is one AggregatedTextFrame's place in the ordered queue.
//
// Every frame handed to the synthesizer takes a slot, spoken or skipped. A
// skipped frame (a code block excluded from synthesis, say) waits at its
// position and goes downstream only once every spoken slot ahead of it is done,
// which is what keeps the context in the order it was written.
type aggregatedFrameSlot struct {
	frame                *frames.AggregatedTextFrame
	contextID            string
	spoken               bool
	tracker              *WordCompletionTracker
	transportDestination string
	complete             bool
	includesInterFrame   bool
}

// bufferedWord is a word timing held until the sentence it belongs to has been
// promoted into a slot. Streaming a turn token by token means a word can arrive
// before the sentence it is part of has been recognized.
type bufferedWord struct {
	word               string
	pts                int64
	contextID          string
	includesInterFrame bool
}

// streamingContext is the per-context state of a sentence being assembled from
// tokens. Each live context keeps its own, so two in flight never share a
// pending sentence.
type streamingContext struct {
	aggregator   *parallelSentenceAggregator
	appendToCtx  bool
	buildTracker bool
}

// AggregatedFrameSequencer orders the frames of a synthesis so the conversation
// context is written in the order the text was spoken.
//
// It holds a queue of spoken and skipped slots. A spoken slot is tracked by a
// WordCompletionTracker and completes as its words come back; a skipped slot
// waits until every spoken slot ahead of it is complete, then goes downstream.
//
// Contexts can be live at once, so the state is kept in three tiers that never
// bleed into each other: slots is the single ordered timeline across all
// contexts, contextAppend marks a context live and says whether its words are
// written to the conversation, and streaming holds the transient pending
// sentence of a context still assembling one from tokens.
type AggregatedFrameSequencer struct {
	// mu guards everything below it. The sequencer is reached from the goroutine
	// processing frames and from the one draining a context's audio: a barge-in
	// clears it while the audio still playing is completing slots against it.
	//
	// The upstream project needs no such guard, running all of this on one event
	// loop; the guard is here because these are genuinely separate goroutines.
	mu sync.Mutex

	name          string
	streamingMode bool
	slots         []*aggregatedFrameSlot
	contextAppend map[string]bool
	streaming     map[string]*streamingContext
	buffered      []bufferedWord
	tokenizer     SentenceTokenizer
}

// SentenceTokenizer finds sentence boundaries. It is the same contract the text
// package defines, restated here so this package does not depend on it.
type SentenceTokenizer interface {
	MatchEndOfSentence(text string) int
}

// NewAggregatedFrameSequencer builds a sequencer labeled name.
//
// Set streaming when each register call carries one token rather than a whole
// unit: tokens are then assembled back into sentences and a slot appears only
// once a boundary is confirmed. Streaming requires the caller to reuse one
// context id for a whole turn, since a sentence built from several tokens is
// registered under a single id and every one of its word timings must arrive
// tagged with that same id.
func NewAggregatedFrameSequencer(name string, streaming bool, tokenizer SentenceTokenizer) *AggregatedFrameSequencer {
	return &AggregatedFrameSequencer{
		name:          name,
		streamingMode: streaming,
		contextAppend: map[string]bool{},
		streaming:     map[string]*streamingContext{},
		tokenizer:     tokenizer,
	}
}

// RegisterSpoken records a frame handed to the synthesizer and returns whatever
// the call unblocks.
//
// Not streaming, this registers a slot at once. Streaming, it feeds the token to
// the context's sentence assembler and registers a slot only once a boundary is
// confirmed. buildTracker is false for a service with no word timings, whose
// slot completes on CompleteSpokenSlot instead.
func (s *AggregatedFrameSequencer) registerSpokenLocked(
	frame *frames.AggregatedTextFrame,
	contextID, ttsText string,
	appendToContext, buildTracker, includesInterFrame bool,
) []frames.Frame {
	if !s.streamingMode {
		var tracker *WordCompletionTracker
		if buildTracker {
			tracker = NewWordCompletionTracker(ttsText, frame.Text, rawOr(frame))
		}
		s.appendSpokenSlot(frame, contextID, tracker, appendToContext, includesInterFrame)
		return nil
	}
	sc, ok := s.streaming[contextID]
	if !ok {
		sc = &streamingContext{
			aggregator:   newParallelSentenceAggregator(s.tokenizer),
			appendToCtx:  appendToContext,
			buildTracker: buildTracker,
		}
		s.streaming[contextID] = sc
	}
	var out []frames.Frame
	for _, agg := range sc.aggregator.aggregate(ttsText, rawOr(frame), frame.Text) {
		out = append(out, s.promote(agg, contextID, sc.appendToCtx, sc.buildTracker)...)
	}
	return out
}

// RegisterSkipped records a frame that is not spoken and returns it once it is
// unblocked. Any sentence still pending for the context is finalized first, so a
// real spoken slot sits immediately ahead of the skipped one and blocks it until
// that sentence has actually been spoken.
func (s *AggregatedFrameSequencer) registerSkippedLocked(
	frame *frames.AggregatedTextFrame,
	contextID, transportDestination string,
) []frames.Frame {
	out := s.finalizeLocked(contextID)
	frame.ContextID = contextID
	s.slots = append(s.slots, &aggregatedFrameSlot{
		frame:                frame,
		contextID:            contextID,
		spoken:               false,
		transportDestination: transportDestination,
	})
	return append(out, s.flushLocked(0)...)
}

// Finalize promotes a context's still-pending sentence into a real slot, for the
// end of its text where no more tokens are coming. The context's live entry is
// kept: word timings for the slot just promoted arrive later, during playback,
// and have to be recognized.
func (s *AggregatedFrameSequencer) finalizeLocked(contextID string) []frames.Frame {
	if !s.streamingMode || contextID == "" {
		return nil
	}
	sc, ok := s.streaming[contextID]
	if !ok {
		return nil
	}
	delete(s.streaming, contextID)
	agg, ok := sc.aggregator.flush()
	if !ok {
		return nil
	}
	return s.promote(agg, contextID, sc.appendToCtx, sc.buildTracker)
}

// ProcessWord folds one word timing into the slot it belongs to and returns the
// frames to push. pts is when the word is spoken.
func (s *AggregatedFrameSequencer) processWordLocked(
	word string, pts int64, contextID string, includesInterFrame bool,
) []frames.Frame {
	// A word for a context that was never registered, was cleared by an
	// interruption, or has already finished is stale: a provider can deliver
	// timings seconds after its context was abandoned, and emitting them would
	// interleave them into the current turn. A context still assembling a
	// sentence is not stale; its words are buffered below.
	_, pending := s.streaming[contextID]
	if contextID != "" && !s.contextLive(contextID) && !pending {
		slog.Debug("sequencer dropping stale word", "sequencer", s.name, "word", word, "context", contextID)
		return nil
	}

	active := s.activeSlot(contextID)
	if handled, out := s.routeWord(active, word, pts, contextID, includesInterFrame); handled {
		return out
	}

	complete, overflow := false, ""
	if active != nil && active.tracker != nil {
		complete = active.tracker.AddWord(word)
		if over, ok := active.tracker.OverflowWord(); ok {
			overflow = over
		}
	}

	out := s.emitWord(active, word, pts, contextID, includesInterFrame)
	if complete && active != nil {
		active.complete = true
		out = append(out, s.flushLocked(pts)...)
		if overflow != "" {
			slog.Debug("sequencer emitting overflow word", "sequencer", s.name, "word", overflow)
			out = append(out, s.processWordLocked(overflow, pts, contextID, false)...)
		}
	}
	return out
}

// routeWord decides whether the word can be handed to active at all. It reports
// handled when the word was dealt with here: buffered until the sentence it
// belongs to is promoted, or passed straight through when no slot claims it.
func (s *AggregatedFrameSequencer) routeWord(
	active *aggregatedFrameSlot, word string, pts int64, contextID string, includesInterFrame bool,
) (handled bool, out []frames.Frame) {
	if active == nil {
		if s.streamingMode {
			s.buffered = append(s.buffered, bufferedWord{word, pts, contextID, includesInterFrame})
			return true, nil
		}
		return false, nil
	}
	if active.tracker == nil || active.tracker.WordBelongsHere(word) {
		return false, nil
	}
	next := s.nextActiveSlot(active, contextID)
	if next != nil && next.tracker != nil && next.tracker.WordBelongsHere(word) {
		// It belongs to the next slot, so this one is force-completed below.
		return false, nil
	}
	if s.streamingMode {
		// The sentence it belongs to may simply not be promoted yet.
		s.buffered = append(s.buffered, bufferedWord{word, pts, contextID, includesInterFrame})
		return true, nil
	}
	slog.Warn("sequencer word matched no slot, passing through", "sequencer", s.name, "word", word)
	return true, []frames.Frame{s.buildWordFrame(word, pts, contextID, "", false, includesInterFrame)}
}

// emitWord builds the frames for a word the active slot has taken.
func (s *AggregatedFrameSequencer) emitWord(
	active *aggregatedFrameSlot, word string, pts int64, contextID string, includesInterFrame bool,
) []frames.Frame {
	// The per-call flag wins, and is carried onto the slot so a force-complete
	// inherits it.
	if active != nil && includesInterFrame {
		active.includesInterFrame = true
	}
	slotIFS := includesInterFrame
	emitContextID := contextID
	frameText, rawText, suppress := word, "", false
	if active != nil {
		slotIFS = includesInterFrame || active.includesInterFrame
		emitContextID = active.contextID
	}
	if active != nil && active.tracker != nil {
		frameText = ""
		if fw, ok := active.tracker.FrameWord(); ok {
			frameText = fw
		}
		if rt, ok := active.tracker.RawText(); ok {
			rawText = rt
		}
		suppress = active.tracker.Suppress()
	}

	var out []frames.Frame
	if frameText != "" {
		out = append(out, s.buildWordFrame(frameText, pts, emitContextID, rawText, suppress, slotIFS))
	}
	if active != nil && active.tracker != nil && !suppress {
		out = append(out, s.buildProgressFrame(active, pts))
	}
	return out
}

// CompleteSpokenSlot marks the first pending spoken slot complete and returns
// the skipped frames that unblocks. It is for a service with no word timings,
// whose slots complete one at a time in the call that registers them.
func (s *AggregatedFrameSequencer) completeSpokenSlotLocked() []frames.Frame {
	for _, slot := range s.slots {
		if slot.spoken && !slot.complete {
			slot.complete = true
			break
		}
	}
	return s.flushLocked(0)
}

// Flush walks the queue and returns every skipped frame now unblocked. Complete
// spoken slots come off the head, then skipped slots whose spoken predecessors
// are all done. It stops at the first incomplete spoken slot. A non-zero
// lastWordPTS places the skipped frames right after the last spoken word.
func (s *AggregatedFrameSequencer) flushLocked(lastWordPTS int64) []frames.Frame {
	var out []frames.Frame
	for len(s.slots) > 0 {
		slot := s.slots[0]
		switch {
		case slot.spoken && slot.complete:
			s.slots = s.slots[1:]
		case !slot.spoken && !slot.complete:
			slot.frame.AppendToContext = true
			slot.frame.SetTransportDestination(slot.transportDestination)
			if lastWordPTS != 0 {
				slot.frame.SetPTS(lastWordPTS)
			}
			out = append(out, slot.frame)
			slot.complete = true
			s.slots = s.slots[1:]
		default:
			// Spoken but not yet complete: everything behind it waits.
			return out
		}
	}
	return out
}

// ForceComplete closes out a context whose audio has ended, for a provider that
// silently drops word timings. Each of the context's incomplete spoken slots
// emits a frame for its remaining unspoken text and is marked complete; slots of
// other contexts still in flight are left to their own words. The context is
// then forgotten, so any word arriving later is stale.
func (s *AggregatedFrameSequencer) forceCompleteLocked(contextID string, lastWordPTS int64) []frames.Frame {
	var out []frames.Frame
	for _, slot := range s.slots {
		if !slot.spoken || slot.complete || slot.contextID != contextID {
			continue
		}
		if slot.tracker != nil {
			remaining := slot.tracker.RemainingTTSText(true)
			rawRemaining, hasRaw := slot.tracker.RemainingRawTextOnly()
			if hasRaw && rawRemaining != "" && remaining != "" && !contains(rawRemaining, remaining) {
				slog.Warn("sequencer force-complete: remaining texts disagree, discarding the original",
					"sequencer", s.name, "raw", rawRemaining, "text", remaining)
				rawRemaining = ""
			}
			if remaining != "" {
				out = append(out, s.buildWordFrame(
					remaining, lastWordPTS, slot.contextID, rawRemaining, false, slot.includesInterFrame))
			}
		}
		slot.complete = true
	}
	out = append(out, s.flushLocked(lastWordPTS)...)
	delete(s.contextAppend, contextID)
	delete(s.streaming, contextID)
	return out
}

// Clear drops every slot and all context state, for an interruption.
func (s *AggregatedFrameSequencer) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.slots = nil
	s.contextAppend = map[string]bool{}
	s.buffered = nil
	s.streaming = map[string]*streamingContext{}
}

// appendSpokenSlot adds a registered spoken slot.
func (s *AggregatedFrameSequencer) appendSpokenSlot(
	frame *frames.AggregatedTextFrame,
	contextID string,
	tracker *WordCompletionTracker,
	appendToContext, includesInterFrame bool,
) {
	s.contextAppend[contextID] = appendToContext
	s.slots = append(s.slots, &aggregatedFrameSlot{
		frame:              frame,
		contextID:          contextID,
		spoken:             true,
		tracker:            tracker,
		includesInterFrame: includesInterFrame,
	})
}

// promote turns a sentence assembled from tokens into a real spoken slot.
//
// The sentence frame goes downstream first, marked as one that will be spoken:
// it is the streaming counterpart of the sentence frame a whole-unit service
// pushes before synthesis, and it is what later progress frames refer back to.
// It does not itself go into the conversation, which is built from the per-word
// frames instead.
//
// With a tracker the slot completes as words arrive. Without one there is no
// word signal to complete it, and by the time a sentence promotes every token
// making it up has already been sent, so it completes at once and a
// whole-sentence frame carries its text into the conversation.
func (s *AggregatedFrameSequencer) promote(
	agg parallelAggregation, contextID string, appendToContext, buildTracker bool,
) []frames.Frame {
	if trimSpace(agg.userFacing) == "" {
		return nil
	}
	frame := frames.NewAggregatedTextFrame(agg.userFacing, frames.AggregationSentence)
	frame.RawText = agg.llm
	frame.ContextID = contextID
	frame.WillBeSpoken = true
	frame.AppendToContext = false

	var tracker *WordCompletionTracker
	if buildTracker {
		tracker = NewWordCompletionTracker(agg.tts, agg.userFacing, agg.llm)
	}
	s.appendSpokenSlot(frame, contextID, tracker, appendToContext, false)
	out := []frames.Frame{frame}

	if !buildTracker {
		wordFrame := frames.NewTTSTextFrame(agg.userFacing)
		wordFrame.RawText = agg.llm
		wordFrame.AppendToContext = appendToContext
		out = append(out, wordFrame)
		out = append(out, s.completeSpokenSlotLocked()...)
	}
	return append(out, s.drainBufferedWords()...)
}

// drainBufferedWords replays words held back now that a new slot may match them.
// The buffer is taken before replaying, so a word that still matches nothing is
// re-buffered by ProcessWord and waits for the next promotion rather than
// looping here.
func (s *AggregatedFrameSequencer) drainBufferedWords() []frames.Frame {
	buffered := s.buffered
	s.buffered = nil
	var out []frames.Frame
	for _, w := range buffered {
		out = append(out, s.processWordLocked(w.word, w.pts, w.contextID, w.includesInterFrame)...)
	}
	return out
}

// contextLive reports whether the context has been registered and not finished.
func (s *AggregatedFrameSequencer) contextLive(contextID string) bool {
	_, ok := s.contextAppend[contextID]
	return ok
}

// slotMatches reports whether slot can take a word for contextID. An empty
// contextID matches any slot, for a provider that does not tag its words.
func slotMatches(slot *aggregatedFrameSlot, contextID string) bool {
	if !slot.spoken || slot.complete || slot.tracker == nil {
		return false
	}
	return contextID == "" || slot.contextID == contextID
}

// activeSlot returns the first incomplete tracked spoken slot for contextID.
func (s *AggregatedFrameSequencer) activeSlot(contextID string) *aggregatedFrameSlot {
	for _, slot := range s.slots {
		if slotMatches(slot, contextID) {
			return slot
		}
	}
	return nil
}

// nextActiveSlot returns the first such slot after current.
func (s *AggregatedFrameSequencer) nextActiveSlot(
	current *aggregatedFrameSlot, contextID string,
) *aggregatedFrameSlot {
	found := false
	for _, slot := range s.slots {
		if slot == current {
			found = true
			continue
		}
		if found && slotMatches(slot, contextID) {
			return slot
		}
	}
	return nil
}

// buildProgressFrame reports how much of a slot has been spoken.
func (s *AggregatedFrameSequencer) buildProgressFrame(
	slot *aggregatedFrameSlot, pts int64,
) *frames.AggregatedTextProgressFrame {
	f := frames.NewAggregatedTextProgressFrame(
		slot.frame.ID(),
		slot.contextID,
		slot.frame.Text,
		slot.frame.AggregatedBy,
		slot.tracker.AccumulatedUserFacingText(),
		slot.tracker.RemainingUserFacingText(false),
	)
	f.SetPTS(pts)
	return f
}

// buildWordFrame builds the frame for one spoken word.
func (s *AggregatedFrameSequencer) buildWordFrame(
	text string, pts int64, contextID, rawText string, suppress, includesInterFrame bool,
) *frames.TTSTextFrame {
	f := frames.NewTTSTextFrame(text)
	f.SetPTS(pts)
	f.ContextID = contextID
	f.RawText = rawText
	f.IncludesInterFrameSpaces = includesInterFrame
	switch {
	case suppress:
		f.AppendToContext = false
	case contextID != "":
		appendTo, ok := s.contextAppend[contextID]
		f.AppendToContext = !ok || appendTo
	default:
		f.AppendToContext = true
	}
	return f
}

// rawOr returns the frame's original text, falling back to its spoken text.
func rawOr(f *frames.AggregatedTextFrame) string {
	if f.RawText != "" {
		return f.RawText
	}
	return f.Text
}

// RegisterSpoken records a frame handed to the synthesizer. See
// registerSpokenLocked.
func (s *AggregatedFrameSequencer) RegisterSpoken(
	frame *frames.AggregatedTextFrame,
	contextID, ttsText string,
	appendToContext, buildTracker, includesInterFrame bool,
) []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.registerSpokenLocked(frame, contextID, ttsText, appendToContext, buildTracker, includesInterFrame)
}

// RegisterSkipped records a frame that is not spoken. See registerSkippedLocked.
func (s *AggregatedFrameSequencer) RegisterSkipped(
	frame *frames.AggregatedTextFrame,
	contextID, transportDestination string,
) []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.registerSkippedLocked(frame, contextID, transportDestination)
}

// Finalize closes out a context. See finalizeLocked.
func (s *AggregatedFrameSequencer) Finalize(contextID string) []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.finalizeLocked(contextID)
}

// ProcessWord records one spoken word. See processWordLocked.
func (s *AggregatedFrameSequencer) ProcessWord(
	word string, pts int64, contextID string, includesInterFrame bool,
) []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.processWordLocked(word, pts, contextID, includesInterFrame)
}

// CompleteSpokenSlot marks the active spoken slot complete. See
// completeSpokenSlotLocked.
func (s *AggregatedFrameSequencer) CompleteSpokenSlot() []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.completeSpokenSlotLocked()
}

// Flush returns every skipped frame now unblocked. See flushLocked.
func (s *AggregatedFrameSequencer) Flush(lastWordPTS int64) []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.flushLocked(lastWordPTS)
}

// ForceComplete completes a context's outstanding slots. See
// forceCompleteLocked.
func (s *AggregatedFrameSequencer) ForceComplete(contextID string, lastWordPTS int64) []frames.Frame {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.forceCompleteLocked(contextID, lastWordPTS)
}
