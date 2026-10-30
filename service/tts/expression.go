package tts

import (
	"context"
	"maps"
	"strings"

	ttsadapter "github.com/gojargo/jargo/adapter/tts"
	"github.com/gojargo/jargo/expressive/formatter"
	"github.com/gojargo/jargo/expressive/spoken"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	ttstext "github.com/gojargo/jargo/utils/text"
)

// Expressive conversation (alpha) lets an agent hear how something was said and
// answer in kind. With pipeline.Params.EnableExpressiveConversation set, the
// service reads the expression tags an LLM writes into its reply instead of
// speaking them: it formats each in its provider's markup, through its
// expression adapter, and reports the delivery of each word it speaks and the
// sounds and pauses it performs. See the expressive package.

// ExpressionAdapterProvider is an optional interface a Synthesizer implements
// to perform expression tags in an expressive conversation (alpha). Its
// adapter's capabilities are what the service can perform: what it broadcasts
// at start, and what it reads and formats. A provider whose expression depends
// on its settings, such as its model, chooses an adapter for them on each call.
// A Synthesizer that does not implement it performs none, and speaks only the
// words.
type ExpressionAdapterProvider interface {
	ExpressionAdapter() ttsadapter.ExpressionAdapter
}

// ExpressionAdapter is the adapter that formats this service's expression tags:
// its provider's, or the base adapter, which performs nothing. Part of
// expressive conversation (alpha).
func (b *Base) ExpressionAdapter() ttsadapter.ExpressionAdapter {
	if p, ok := b.syn.(ExpressionAdapterProvider); ok {
		if a := p.ExpressionAdapter(); a != nil {
			return a
		}
	}
	return ttsadapter.BaseExpressionAdapter{}
}

// ExpressiveConversation reports whether the pipeline has expressive
// conversation on (alpha), so the service reads the expression tags an LLM
// writes. It is known once the pipeline is set up.
func (b *Base) ExpressiveConversation() bool { return b.exprFormatter != nil }

// ExpressionSettings is the settings to send with the request being
// synthesized, set before each request from the service's expression adapter
// (ttsadapter.ExpressionAdapter.RequestSettings), for a provider that takes
// some of its expression as request settings rather than as markup in its
// text. Empty for none. Part of expressive conversation (alpha).
func (b *Base) ExpressionSettings() map[string]any {
	b.exprMu.Lock()
	defer b.exprMu.Unlock()
	return maps.Clone(b.exprSettings)
}

func (b *Base) setExpressionSettings(s map[string]any) {
	b.exprMu.Lock()
	defer b.exprMu.Unlock()
	b.exprSettings = s
}

// setupExpression has the service read expression tags, when the pipeline has
// expressive conversation on.
func (b *Base) setupExpression(s processor.Setup) {
	if !s.EnableExpressiveConversation {
		return
	}
	b.exprFormatter = formatter.NewExpressionFormatter(b.ExpressionAdapter, b.Name())
	b.sequencer.SetLLMMarkup(formatter.ExpressionTagPattern)
	// Each tag reaches the parser whole, however the text streams in, and so do
	// the tags the service keeps whole itself, such as <spell>.
	tags := formatter.ExpressionTags()
	if own, ok := b.aggregator.(*ttstext.SkipTagsAggregator); ok {
		tags = append(tags, own.Tags()...)
	}
	b.aggregator = ttstext.NewSkipTagsAggregator(b.aggregator.Type(), b.aggregator.Language(), tags)
}

// expressiveTextType reports whether text of this type is what an LLM writes
// its replies in, where expressive conversation reads expression tags. Other
// text, such as a pattern's match, is spoken as is.
func expressiveTextType(t frames.TextType) bool {
	return t == frames.TextTypeSentence || t == frames.TextTypeToken
}

// pushSpeech speaks a TTSSpeakFrame's text. With expressive conversation on it
// is read as text of its own, apart from any response it is spoken in the
// middle of.
func (b *Base) pushSpeech(ctx context.Context, agg ttstext.Aggregation, interruptible bool) error {
	if b.exprFormatter == nil || !expressiveTextType(agg.Type) {
		return b.pushTTSFrames(ctx, agg, interruptible, false)
	}
	b.exprFormatter.RestateDelivery()
	err := b.pushTTSFrames(ctx, agg, interruptible, true)
	b.exprFormatter.RestateDelivery()
	return err
}

// speakExpression speaks a unit of text with expression tags in it.
//
// The unit stays one frame: its words without the tags, for clients, and its
// text as the LLM wrote it, for the context. The request is built from its
// segments: the words transformed, and the tags formatted in the provider's
// markup. whole says the text stands on its own, apart from the response, such
// as a TTSSpeakFrame's.
func (b *Base) speakExpression(
	ctx context.Context, agg ttstext.Aggregation, contextID string, interruptible, whole bool,
) error {
	streaming := b.isStreamingTokens()
	text := agg.Text
	if agg.Type != frames.TextTypeToken {
		text = strings.TrimSpace(text)
	}
	segments := b.exprFormatter.Parse(text, whole)
	unit := ttstext.Aggregation{Text: formatter.ShownText(segments, agg.Type), Type: agg.Type, RawText: text}
	shown, ok := b.trimIncomingWhitespace(unit.Text, streaming)
	if !ok {
		shown = ""
	}

	transform := func(words string, by frames.TextType) (string, bool) { return b.transformUnit(ctx, words, by) }
	trim := func(words string) (string, bool) { return b.trimIncomingWhitespace(words, streaming) }
	request := b.exprFormatter.Format(segments, agg.Type, transform, trim)
	b.setExpressionSettings(request.Settings)
	prepared := request.Text
	if prepared != "" {
		// The formatter trims a sentence's request, so a provider that needs
		// its inputs to end in a space gets it back here.
		prepared = b.prepareText(prepared)
		// A TTSSpeakFrame sets up its context before its text gets here.
		b.ttsContextOrDefault(contextID, ttsContext{appendToContext: true, interruptible: interruptible})
		b.spokenExpression(contextID).Add(text, request.Spans)
		return b.speakText(ctx, unit, contextID, spokenText{text: shown, raw: text}, prepared, interruptible)
	}
	if strings.TrimSpace(text) == "" {
		return nil
	}
	// A unit with nothing to say, such as a tag alone, still reaches the context
	// as the LLM wrote it, in order with the text around it.
	tc := b.ttsContextOrDefault(contextID, ttsContext{appendToContext: true, interruptible: interruptible})
	aggregated := b.aggregatedFrame(unit, contextID)
	b.pushSequencerFrames(ctx, b.sequencer.RegisterSpoken(
		aggregated, contextID, "", tc.appendToContext, b.wordPath(), false, b.TextAggregationLanguage()),
		contextID)
	b.pushSpokenText(ctx, aggregated, spokenText{raw: text}, contextID, tc.appendToContext)
	return nil
}

// spokenExpression is the delivery and sounds of the words spoken in a context,
// created with it.
func (b *Base) spokenExpression(contextID string) *spoken.SpokenExpression {
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	c, ok := b.ttsContexts[contextID]
	if !ok {
		return nil
	}
	if c.spoken == nil {
		c.spoken = spoken.New()
		b.ttsContexts[contextID] = c
	}
	return c.spoken
}

// knownSpokenExpression is what spokenExpression returns for a context that
// has one already, without creating it.
func (b *Base) knownSpokenExpression(contextID string) *spoken.SpokenExpression {
	if contextID == "" {
		return nil
	}
	b.audioCtxMu.Lock()
	defer b.audioCtxMu.Unlock()
	return b.ttsContexts[contextID].spoken
}

// endTagsOnlyReply completes the text of a reply that is only tags. It speaks
// nothing, so no audio context ends it to complete its text: the end of the
// reply does.
func (b *Base) endTagsOnlyReply(ctx context.Context, contextID string) {
	if contextID == "" || b.AudioContextAvailable(contextID) {
		return
	}
	b.audioCtxMu.Lock()
	_, known := b.ttsContexts[contextID]
	b.audioCtxMu.Unlock()
	if !known {
		return
	}
	b.applyForceComplete(ctx, contextID)
	b.deleteTTSContext(contextID)
}

// reportUnreportedExpression reports the sounds and pauses no text reported,
// once a context's audio has ended.
func (b *Base) reportUnreportedExpression(
	ctx context.Context, stopped *frames.TTSStoppedFrame, dir processor.Direction,
) {
	if b.exprFormatter == nil {
		return
	}
	s := b.knownSpokenExpression(stopped.ContextID)
	if s == nil {
		return
	}
	for _, sound := range b.expressionSoundFrames(stopped, stopped.ContextID, s.Unreported()) {
		// Ordered after the context's last words, as the stop frame is.
		if last := b.lastWordPTS(); last > 0 {
			sound.SetPTS(int64(last) + 1)
		}
		_ = b.PushFrame(ctx, sound, dir)
	}
}

// reportExpression gives spoken text the delivery it is spoken in, and reports
// the sounds before it.
//
// The text the provider reports speaking covers part of the text the LLM wrote,
// tags included, which tells its delivery and the sounds performed since the
// text before it. Markup the provider reports as a word stands for the sound it
// performs, or isn't spoken. It returns a frame for each sound or pause
// performed before the text, and whether the frame itself is pushed.
func (b *Base) reportExpression(f frames.Frame) ([]*frames.TTSTextFrame, bool) {
	switch fr := f.(type) {
	case *frames.AggregatedTextProgressFrame:
		if isExpressionSound(fr.TextType) {
			return nil, true
		}
		fr.Delivery = nil
		if s := b.knownSpokenExpression(fr.ContextID); s != nil {
			if d, ok := s.Delivery(); ok {
				fr.Delivery = d
			}
		}
		return nil, true
	case *frames.TTSTextFrame:
		if isExpressionSound(fr.TextType) {
			return nil, true
		}
		s := b.knownSpokenExpression(fr.ContextID)
		if isMarkup(fr.Text) {
			return b.reportMarkup(fr, s)
		}
		return b.reportWords(fr, s), true
	}
	return nil, true
}

// reportMarkup reports markup the provider reported as a word: the sound it
// stands for, or nothing. It returns a frame for each sound or pause performed
// before it, and whether the frame itself is pushed.
func (b *Base) reportMarkup(f *frames.TTSTextFrame, s *spoken.SpokenExpression) ([]*frames.TTSTextFrame, bool) {
	var sound formatter.ReportedSound
	ok := false
	if s != nil {
		sound, ok = s.SpokenMarkup()
	}
	if !ok {
		// Markup that stands for no sound isn't spoken, unless it covers some of
		// the LLM's text.
		if f.RawText == "" {
			return nil, false
		}
		return b.reportWords(f, s), true
	}
	f.TextType, f.Text = sound.TextType, sound.Text
	// A sound the bot performed, in a segment of its own, as the sounds found
	// between words are reported; as a word it was neither.
	f.WillBeSpoken = true
	f.SegmentID = 0
	if f.RawText == "" || s == nil {
		f.AppendToContext = false
		return nil, true
	}
	// Markup can cover the LLM's text, as "[laughs]" covers "[[laughs]]!" when
	// the punctuation was written after the tag. The context records that text
	// with the sound.
	_, _, sounds := s.Spoken(f.RawText)
	return b.expressionSoundFrames(f, f.ContextID, sounds), true
}

// reportWords gives words the delivery they are spoken in, and returns a frame
// for each sound or pause performed before them.
func (b *Base) reportWords(f *frames.TTSTextFrame, s *spoken.SpokenExpression) []*frames.TTSTextFrame {
	if s == nil {
		return nil
	}
	delivery, ok, sounds := s.Spoken(f.RawText)
	if ok {
		f.Delivery = delivery
	}
	return b.expressionSoundFrames(f, f.ContextID, sounds)
}

// expressionSoundFrames builds frames for sounds and pauses performed at a
// frame's time.
func (b *Base) expressionSoundFrames(
	at frames.Frame, contextID string, sounds []formatter.ReportedSound,
) []*frames.TTSTextFrame {
	out := make([]*frames.TTSTextFrame, 0, len(sounds))
	for _, s := range sounds {
		sound := frames.NewTTSTextFrame(s.Text, s.TextType)
		if pts, has := at.Base().PTS(); has {
			sound.SetPTS(pts)
		}
		sound.ContextID = contextID
		sound.WillBeSpoken = true
		// The context records the sound's tag with the text around it.
		sound.AppendToContext = false
		out = append(out, sound)
	}
	return out
}

func isExpressionSound(t frames.TextType) bool {
	return t == frames.TextTypeExpressionSound || t == frames.TextTypeExpressionPause
}

// isMarkup reports whether a word a provider reports is its markup, rather
// than speech.
//
// The shape alone decides, and that is safe: a bracketed word that covers some
// of the LLM's text, such as a literal "[note]" it wrote, is still reported as
// that text. Only a word that covers none of it, which can only be the
// adapter's markup, is read as a sound or left out.
func isMarkup(text string) bool {
	text = strings.TrimSpace(text)
	return (strings.HasPrefix(text, "[") && strings.HasSuffix(text, "]")) ||
		(strings.HasPrefix(text, "<") && strings.HasSuffix(text, ">"))
}
