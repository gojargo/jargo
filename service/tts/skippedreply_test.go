package tts_test

import (
	"context"
	"reflect"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/service/tts"
	ttstext "github.com/gojargo/jargo/text"
)

// skipTTSResponse is an LLM response the LLM marked to skip TTS, as it streams
// it.
func skipTTSResponse(chunks ...string) []frames.Frame {
	skip := true
	start := frames.NewLLMFullResponseStartFrame()
	start.SkipTTS = &skip
	out := []frames.Frame{start}
	for _, c := range chunks {
		f := frames.NewLLMTextFrame(c)
		f.SkipTTS = &skip
		out = append(out, f)
	}
	end := frames.NewLLMFullResponseEndFrame()
	end.SkipTTS = &skip
	return append(out, end)
}

// runFrames runs feed through a pipeline of procs and returns what reached the
// end of it, in order.
func runFrames(t *testing.T, feed []frames.Frame, procs ...processor.Processor) []frames.Frame {
	t.Helper()
	var mu sync.Mutex
	var got []frames.Frame
	task := pipeline.NewWorker(pipeline.New(procs...), pipeline.WorkerConfig{ReachedDownstreamFilter: pipeline.AnyFrame})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		// The frames a reply is made of; what the worker adds around them (its
		// own messages to the client, the service's metadata) is left out.
		switch f.(type) {
		case *frames.LLMFullResponseStartFrame, *frames.LLMFullResponseEndFrame, *frames.LLMTextFrame,
			*frames.TextFrame, *frames.AggregatedTextFrame, *frames.TTSTextFrame,
			*frames.TTSStartedFrame, *frames.TTSStoppedFrame, *frames.TTSAudioRawFrame:
			mu.Lock()
			got = append(got, f)
			mu.Unlock()
		}
	})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrames(feed)
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("the pipeline did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	return append([]frames.Frame(nil), got...)
}

// typesOf names the type of each frame.
func typesOf(fs []frames.Frame) []string {
	out := make([]string, len(fs))
	for i, f := range fs {
		out[i] = reflect.TypeOf(f).Elem().Name()
	}
	return out
}

// TestASkippedReplyIsPushedAsUnspokenSentences checks a reply that is not
// spoken is still cut into sentences, in token mode too, so it reaches a client
// as the bot's output, and nothing of it is synthesized.
func TestASkippedReplyIsPushedAsUnspokenSentences(t *testing.T) {
	for _, mode := range []frames.TextType{frames.TextTypeSentence, frames.TextTypeToken} {
		t.Run(string(mode), func(t *testing.T) {
			synth := &recordingSynth{}
			base := tts.New("SkipTTS", synth)
			base.SetTextAggregator(ttstext.NewSimpleAggregator(mode, ""))

			down := runFrames(t, skipTTSResponse("Hello there. How", " are you?"), base)

			want := []string{
				"LLMFullResponseStartFrame", "AggregatedTextFrame", "AggregatedTextFrame",
				"LLMFullResponseEndFrame",
			}
			if got := typesOf(down); !slices.Equal(got, want) {
				t.Fatalf("frames = %v, want %v", got, want)
			}
			var texts []string
			for _, f := range down {
				if af, ok := f.(*frames.AggregatedTextFrame); ok {
					texts = append(texts, af.Text)
					if af.SkipTTS == nil || !*af.SkipTTS || af.WillBeSpoken {
						t.Errorf("%q is not marked as left out of speech", af.Text)
					}
				}
			}
			if !slices.Equal(texts, []string{"Hello there.", "How are you?"}) {
				t.Errorf("sentences = %q", texts)
			}
			if len(synth.texts()) != 0 {
				t.Errorf("synthesized %q, want nothing", synth.texts())
			}
		})
	}
}

// TestASkippedReplyOfOnlyWhitespaceHasNoSentences checks a reply that is not
// spoken and holds only whitespace pushes no text.
func TestASkippedReplyOfOnlyWhitespaceHasNoSentences(t *testing.T) {
	down := runFrames(t, skipTTSResponse("\n"), tts.New("SkipTTS", &recordingSynth{}))

	if got := typesOf(down); !slices.Equal(got, []string{"LLMFullResponseStartFrame", "LLMFullResponseEndFrame"}) {
		t.Errorf("frames = %v, want the brackets alone", got)
	}
}

// TestSkippedTextThatIsNotLLMTextPassesThrough checks text another processor
// marked to skip TTS, such as an IVR tag, is pushed on as it is.
func TestSkippedTextThatIsNotLLMTextPassesThrough(t *testing.T) {
	skip := true
	tag := frames.NewTextFrame("<dtmf>1</dtmf>")
	tag.SkipTTS = &skip

	down := runFrames(t, []frames.Frame{tag}, tts.New("SkipTTS", &recordingSynth{}))

	if len(down) != 1 || down[0] != frames.Frame(tag) {
		t.Errorf("frames = %v, want the tag as it was", down)
	}
}

// TestASkippedReplyReachesTheContextOnce checks the conversation gets the reply
// once, as the sentences it was cut into.
func TestASkippedReplyReachesTheContextOnce(t *testing.T) {
	convo := frames.NewLLMContext("")
	pair := aggregators.New(convo)

	runFrames(t, skipTTSResponse("Hello there. How", " are you?"), tts.New("SkipTTS", &recordingSynth{}), pair.Assistant())

	msgs := convo.Messages()
	if len(msgs) != 1 || msgs[0].Role != frames.RoleAssistant || msgs[0].Text != "Hello there. How are you?" {
		t.Errorf("messages = %+v, want the reply once", msgs)
	}
}

// TestASkippedReplyComesOutAfterTheAudioStillPlaying checks a reply that is not
// spoken waits for earlier speech, then its frames come out in order.
func TestASkippedReplyComesOutAfterTheAudioStillPlaying(t *testing.T) {
	feed := append([]frames.Frame{frames.NewTTSSpeakFrame("One moment.")}, skipTTSResponse("Hello there.")...)

	down := runFrames(t, feed, tts.New("SkipTTS", &recordingSynth{}))

	index := func(match func(frames.Frame) bool) int {
		return slices.IndexFunc(down, match)
	}
	audio := index(func(f frames.Frame) bool { _, ok := f.(*frames.TTSAudioRawFrame); return ok })
	start := index(func(f frames.Frame) bool { _, ok := f.(*frames.LLMFullResponseStartFrame); return ok })
	sentence := index(func(f frames.Frame) bool {
		af, ok := f.(*frames.AggregatedTextFrame)
		return ok && af.SkipTTS != nil && *af.SkipTTS
	})
	end := index(func(f frames.Frame) bool { _, ok := f.(*frames.LLMFullResponseEndFrame); return ok })
	if audio < 0 || audio >= start || start >= sentence || sentence >= end {
		t.Errorf("audio %d, start %d, sentence %d, end %d: want them in that order (%v)",
			audio, start, sentence, end, typesOf(down))
	}
}
