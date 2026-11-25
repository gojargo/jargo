package stt

import (
	"context"
	"errors"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
)

const cardNumber = "4111 1111 1111 1111"

// errTransform is the failure of a transform that could not rewrite its text.
var errTransform = errors.New("no")

func maskCard(_ context.Context, text string, _ frames.TextType) (string, error) {
	return strings.ReplaceAll(text, cardNumber, "[card number]"), nil
}

// heard is a transcription as a provider yields it, final or interim, carrying
// the provider's own result.
func heard(text string, final bool) frames.Frame {
	result := map[string]any{"transcript": text}
	if final {
		f := frames.NewTranscriptionFrame(text, "user", "now")
		f.Result = result
		return f
	}
	f := frames.NewInterimTranscriptionFrame(text, "user", "now")
	f.Result = result
	return f
}

// transcriptOf is the text and the result of a transcription frame.
func transcriptOf(t *testing.T, f frames.Frame) (string, any) {
	t.Helper()
	switch fr := f.(type) {
	case *frames.TranscriptionFrame:
		return fr.Text, fr.Result
	case *frames.InterimTranscriptionFrame:
		return fr.Text, fr.Result
	}
	t.Fatalf("pushed a %T, want a transcription", f)
	return "", nil
}

func textsOf(t *testing.T, fs []frames.Frame) []string {
	t.Helper()
	out := make([]string, 0, len(fs))
	for _, f := range fs {
		text, _ := transcriptOf(t, f)
		out = append(out, text)
	}
	return out
}

func pushDown(t *testing.T, s *StreamService, f frames.Frame) {
	t.Helper()
	if err := s.PushFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}
}

func TestTextTransformsRewriteATranscriptBeforeItIsPushed(t *testing.T) {
	t.Parallel()
	for name, tc := range map[string]struct {
		final    bool
		textType frames.TextType
	}{
		"final":   {true, TranscriptionFinal},
		"interim": {false, TranscriptionInterim},
	} {
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			var types []frames.TextType
			s := newConfirmingService(t)
			s.SetTextTransformers(TextTransformer{
				TextType: frames.AnyTextType,
				Transform: func(ctx context.Context, text string, textType frames.TextType) (string, error) {
					types = append(types, textType)
					return maskCard(ctx, text, textType)
				},
			})
			pushed := pushedFrames(s)

			pushDown(t, s, heard("It's "+cardNumber, tc.final))

			got := pushed()
			if texts := textsOf(t, got); !slices.Equal(texts, []string{"It's [card number]"}) {
				t.Fatalf("pushed %q, want the card number masked", texts)
			}
			if !slices.Equal(types, []frames.TextType{tc.textType}) {
				t.Errorf("the transform saw %v, want %v", types, tc.textType)
			}
			// The provider's result has the transcript as heard, so it goes.
			if _, result := transcriptOf(t, got[0]); result != nil {
				t.Errorf("result = %v, want none", result)
			}
		})
	}
}

func TestATranscriptATransformLeavesAloneKeepsItsResult(t *testing.T) {
	t.Parallel()
	s := newConfirmingService(t)
	s.SetTextTransformers(TextTransformer{TextType: frames.AnyTextType, Transform: maskCard})
	pushed := pushedFrames(s)

	pushDown(t, s, heard("Hello", true))

	got := pushed()
	if len(got) != 1 {
		t.Fatalf("pushed %v, want one transcription", got)
	}
	text, result := transcriptOf(t, got[0])
	if text != "Hello" || !reflect.DeepEqual(result, map[string]any{"transcript": "Hello"}) {
		t.Errorf("pushed (%q, %v), want the transcript and its result as given", text, result)
	}
}

func TestTextTransformsApplyToTheirTextTypeInOrder(t *testing.T) {
	t.Parallel()
	upper := func(_ context.Context, text string, _ frames.TextType) (string, error) {
		return strings.ToUpper(text), nil
	}
	exclaim := func(_ context.Context, text string, _ frames.TextType) (string, error) {
		return text + "!", nil
	}
	s := newConfirmingService(t)
	s.SetTextTransformers(
		TextTransformer{TextType: TranscriptionFinal, Transform: upper},
		TextTransformer{TextType: frames.AnyTextType, Transform: exclaim},
	)
	pushed := pushedFrames(s)

	pushDown(t, s, heard("hi", true))
	pushDown(t, s, heard("hi", false))

	if texts := textsOf(t, pushed()); !slices.Equal(texts, []string{"HI!", "hi!"}) {
		t.Errorf("pushed %q, want [HI! hi!]", texts)
	}
}

// A transform is withdrawn by replacing the list: Go function values cannot
// be compared, so there is no removing one by naming it.
func TestTextTransformersCanBeAddedAndWithdrawn(t *testing.T) {
	t.Parallel()
	s := newConfirmingService(t)
	pushed := pushedFrames(s)

	s.AddTextTransformer(TextTransformer{Transform: maskCard})
	pushDown(t, s, heard(cardNumber, true))
	s.SetTextTransformers()
	pushDown(t, s, heard(cardNumber, true))

	if texts := textsOf(t, pushed()); !slices.Equal(texts, []string{"[card number]", cardNumber}) {
		t.Errorf("pushed %q, want the first masked and the second as heard", texts)
	}
}

func TestATranscriptionGoingUpstreamIsNotTransformed(t *testing.T) {
	t.Parallel()
	s := newConfirmingService(t)
	s.SetTextTransformers(TextTransformer{TextType: frames.AnyTextType, Transform: maskCard})
	pushed := pushedFrames(s)

	if err := s.PushFrame(context.Background(), heard(cardNumber, true), processor.Upstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}

	if texts := textsOf(t, pushed()); !slices.Equal(texts, []string{cardNumber}) {
		t.Errorf("pushed %q, want the transcript as heard", texts)
	}
}

// (Not a port of an upstream test: a Go transform can fail.) A transcript
// whose transform failed is not pushed as heard.
func TestATranscriptWhoseTransformFailedIsNotPushed(t *testing.T) {
	t.Parallel()
	s := newConfirmingService(t)
	s.SetTextTransformers(TextTransformer{
		Transform: func(context.Context, string, frames.TextType) (string, error) { return "", errTransform },
	})
	pushed := pushedFrames(s)

	err := s.PushFrame(context.Background(), heard(cardNumber, true), processor.Downstream)
	if !errors.Is(err, errTransform) {
		t.Fatalf("PushFrame = %v, want the transform's error", err)
	}
	if got := pushed(); len(got) != 0 {
		t.Errorf("pushed %v, want nothing", got)
	}
}

// The segmented service rewrites its transcripts the same way.
func TestSegmentServiceTransformsItsTranscripts(t *testing.T) {
	t.Parallel()
	s := NewSegment("FakeSTT", &settingsTranscriber{}, 16000)
	s.SetTextTransformers(TextTransformer{TextType: TranscriptionFinal, Transform: maskCard})
	var mu sync.Mutex
	var got []frames.Frame
	s.Events().Add(processor.EventAfterPushFrame, func(_ context.Context, _ any, args ...any) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, pushedFrame(args))
	})

	if err := s.PushFrame(context.Background(), heard("It's "+cardNumber, true), processor.Downstream); err != nil {
		t.Fatalf("PushFrame: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if texts := textsOf(t, got); !slices.Equal(texts, []string{"It's [card number]"}) {
		t.Errorf("pushed %q, want the card number masked", texts)
	}
}

// transcriptObserver collects the text of every transcription it sees pushed.
type transcriptObserver struct {
	mu    sync.Mutex
	texts []string
}

func (o *transcriptObserver) OnPushFrame(data processor.FramePushed) {
	if fr, ok := data.Frame.(*frames.TranscriptionFrame); ok {
		o.mu.Lock()
		o.texts = append(o.texts, fr.Text)
		o.mu.Unlock()
	}
}

// An observer, such as the RTVI client's transcript, sees a frame as it is
// pushed.
func TestObserversSeeOnlyTheTransformedTranscript(t *testing.T) {
	t.Parallel()
	s := newConfirmingService(t)
	s.SetTextTransformers(TextTransformer{TextType: frames.AnyTextType, Transform: maskCard})
	observer := &transcriptObserver{}
	worker := pipeline.NewWorker(pipeline.New(s), pipeline.WorkerConfig{
		Observers:               []pipeline.Observer{observer},
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	started := make(chan struct{})
	var once sync.Once
	event.On(&worker.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		if _, ok := f.(*frames.StartFrame); ok {
			once.Do(func() { close(started) })
		}
	})
	done := make(chan error, 1)
	go func() { done <- worker.Run(context.Background()) }()
	select {
	case <-started:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not start")
	}

	pushDown(t, s, heard("It's "+cardNumber, true))
	worker.StopWhenDone()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("the pipeline did not finish")
	}

	observer.mu.Lock()
	defer observer.mu.Unlock()
	if len(observer.texts) == 0 {
		t.Fatal("the observer saw no transcription")
	}
	for _, text := range observer.texts {
		if text != "It's [card number]" {
			t.Errorf("the observer saw %q, want only the masked transcript", text)
		}
	}
}
