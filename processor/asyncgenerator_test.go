package processor_test

import (
	"context"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// Upstream has no tests for this processor; these are jargo's own.

// textEncoder serializes a text frame as its text and sends nothing else, which
// is enough to tell a frame that was serialized from one that was skipped.
type textEncoder struct{}

func (textEncoder) Serialize(f frames.Frame) ([]byte, error) {
	if tf, ok := f.(*frames.TextFrame); ok {
		return []byte(tf.Text), nil
	}
	return nil, nil
}

// The point of the processor: the pipeline is unaffected and a reader outside it
// still sees everything that went by.
func TestAsyncGeneratorPassesFramesOnAndSerializesThem(t *testing.T) {
	p := processor.NewAsyncGeneratorProcessor("Gen", textEncoder{})
	_, down := linkAndStart(t, p)

	ctx := context.Background()
	for _, text := range []string{"one", "two"} {
		if err := p.QueueFrame(ctx, frames.NewTextFrame(text), processor.Downstream); err != nil {
			t.Fatal(err)
		}
	}

	got := collect(t, p, 2)
	if len(got) != 2 || string(got[0]) != "one" || string(got[1]) != "two" {
		t.Errorf("serialized = %q, want one then two", got)
	}

	// The frames themselves carried on down the pipeline unchanged.
	mustReceive[*frames.TextFrame](t, down.got, "TextFrame")
	mustReceive[*frames.TextFrame](t, down.got, "TextFrame")
}

// A frame the encoder does not send contributes nothing, rather than an empty
// message a reader would have to filter out itself.
func TestAsyncGeneratorSkipsWhatTheEncoderDoesNotSend(t *testing.T) {
	p := processor.NewAsyncGeneratorProcessor("Gen", textEncoder{})
	linkAndStart(t, p)

	ctx := context.Background()
	if err := p.QueueFrame(ctx, frames.NewBotSpeakingFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if err := p.QueueFrame(ctx, frames.NewTextFrame("only this"), processor.Downstream); err != nil {
		t.Fatal(err)
	}

	got := collect(t, p, 1)
	if len(got) != 1 || string(got[0]) != "only this" {
		t.Errorf("serialized = %q, want only the text frame", got)
	}
}

// The stream ends with the pipeline, so a reader ranging over it stops rather
// than waiting for frames that are never coming.
func TestAsyncGeneratorEndsWithThePipeline(t *testing.T) {
	p := processor.NewAsyncGeneratorProcessor("Gen", textEncoder{})
	linkAndStart(t, p)

	ctx := context.Background()
	if err := p.QueueFrame(ctx, frames.NewTextFrame("last"), processor.Downstream); err != nil {
		t.Fatal(err)
	}
	if err := p.QueueFrame(ctx, frames.NewEndFrame(), processor.Downstream); err != nil {
		t.Fatal(err)
	}

	done := make(chan [][]byte, 1)
	go func() {
		var got [][]byte
		for data := range p.Frames(ctx) {
			got = append(got, data)
		}
		done <- got
	}()

	select {
	case got := <-done:
		// Everything queued ahead of the end is still delivered.
		if len(got) != 1 || string(got[0]) != "last" {
			t.Errorf("serialized = %q, want the frame queued before the end", got)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the stream did not end with the pipeline")
	}
}

// collect reads n serialized frames, failing if they do not arrive.
func collect(t *testing.T, p *processor.AsyncGeneratorProcessor, n int) [][]byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	var got [][]byte
	for data := range p.Frames(ctx) {
		got = append(got, data)
		if len(got) == n {
			return got
		}
	}
	t.Fatalf("read %d serialized frames, want %d", len(got), n)
	return nil
}
