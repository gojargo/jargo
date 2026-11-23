package stt

import (
	"context"
	"sync"

	"github.com/gojargo/jargo/frames"
)

// The text types a service's text transforms see a transcript as. They are
// separate from the types of the bot's own text, which the RTVI protocol
// carries.
const (
	// TranscriptionFinal is the type of a final transcript.
	TranscriptionFinal frames.TextType = "transcription"
	// TranscriptionInterim is the type of an interim transcript.
	TranscriptionInterim frames.TextType = "interim_transcription"
)

// TextTransform rewrites a transcript before the service pushes it, and is told
// whether the transcript is final (TranscriptionFinal) or interim
// (TranscriptionInterim). It has the shape of the transforms a TTS service and
// the LLM text processor take, so one function serves all three.
//
// Where a transform is registered decides who sees the rewrite. On an STT
// service it rewrites what the user said before anything else sees it,
// observers included.
//
// Returning an error drops the transcript: pushing the text as heard instead is
// not safe, because a transform may exist precisely to take something out.
type TextTransform func(ctx context.Context, text string, textType frames.TextType) (string, error)

// TextTransformer pairs a transform with the transcripts it applies to.
type TextTransformer struct {
	// TextType names the transcripts this transform applies to:
	// TranscriptionFinal, TranscriptionInterim, or frames.AnyTextType for
	// both. Empty means both.
	TextType frames.TextType
	// Transform is the rewrite itself.
	Transform TextTransform
}

// textTransforms holds the transforms a service rewrites its transcripts with.
// Both kinds of service embed it.
//
// Everything after the service sees only the rewritten text, observers
// included, so a transform can keep text such as a card number out of the
// context, the client and the logs. A transcription whose text a transform
// changed loses its Result, the provider's response with the text as heard.
type textTransforms struct {
	transformsMu sync.Mutex
	transforms   []TextTransformer
}

// SetTextTransformers sets the transforms each transcript is rewritten with
// before it is pushed, replacing any set before. They run in order, each one
// seeing what the one before it produced.
//
// Replacing the list is also how a transform is withdrawn: Go function values
// cannot be compared, so there is no removing one by naming it.
func (t *textTransforms) SetTextTransformers(ts ...TextTransformer) {
	t.transformsMu.Lock()
	defer t.transformsMu.Unlock()
	t.transforms = append([]TextTransformer(nil), ts...)
}

// AddTextTransformer appends one transform to those already set.
func (t *textTransforms) AddTextTransformer(tr TextTransformer) {
	t.transformsMu.Lock()
	defer t.transformsMu.Unlock()
	t.transforms = append(t.transforms, tr)
}

// transformTranscription rewrites a transcription's text with the transforms
// that apply to it. Any other frame is left alone.
func (t *textTransforms) transformTranscription(ctx context.Context, f frames.Frame) error {
	t.transformsMu.Lock()
	transforms := t.transforms
	t.transformsMu.Unlock()
	if len(transforms) == 0 {
		return nil
	}

	var text *string
	var result *any
	var textType frames.TextType
	switch fr := f.(type) {
	case *frames.TranscriptionFrame:
		text, result, textType = &fr.Text, &fr.Result, TranscriptionFinal
	case *frames.InterimTranscriptionFrame:
		text, result, textType = &fr.Text, &fr.Result, TranscriptionInterim
	default:
		return nil
	}

	rewritten := *text
	for _, tr := range transforms {
		if tr.Transform == nil {
			continue
		}
		if tr.TextType != "" && tr.TextType != textType && tr.TextType != frames.AnyTextType {
			continue
		}
		var err error
		if rewritten, err = tr.Transform(ctx, rewritten, textType); err != nil {
			return err
		}
	}
	if rewritten != *text {
		*text = rewritten
		*result = nil
	}
	return nil
}
