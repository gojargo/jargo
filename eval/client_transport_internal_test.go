package eval

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsclient"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

const transportRate = 16000

// testClientTransport is a harness transport to nowhere, for its edges.
func testClientTransport(params transport.Params, recorder *ClientRecorder) *ClientTransport {
	return NewClientTransport("ws://localhost:0", wsclient.Params{Params: params}, evalws.ConnectionFlags{}, recorder)
}

// The output streams queued audio at real time, silence otherwise.
func TestEvalHarnessOutput(t *testing.T) {
	chunkBytes := frameBytes(transportRate) // one 40 ms frame, 16-bit mono

	t.Run("paces queued audio then silence", func(t *testing.T) {
		out := testClientTransport(transport.Params{AudioOutEnabled: true}, nil).output
		var (
			mu   sync.Mutex
			sent [][]byte
		)
		// Capture the frames instead of serializing and sending them; the
		// unlinked processor has no downstream.
		out.sendFrame = func(_ context.Context, f *frames.OutputAudioRawFrame) {
			mu.Lock()
			sent = append(sent, f.Audio)
			mu.Unlock()
		}
		out.pushFrame = func(context.Context, frames.Frame) {}

		utterance := bytes.Repeat([]byte{1, 2}, transportRate*12/100) // 120 ms: three 40 ms chunks
		out.pending = append(out.pending, utterance...)

		ctx, cancel := context.WithTimeout(t.Context(), 250*time.Millisecond)
		defer cancel()
		out.sendTaskHandler(ctx, transportRate)

		mu.Lock()
		defer mu.Unlock()
		silence := make([]byte, chunkBytes)
		var speech []byte
		sawSilence := false
		for _, pcm := range sent {
			if len(pcm) != chunkBytes {
				t.Fatalf("a frame of %d bytes, want %d", len(pcm), chunkBytes)
			}
			if bytes.Equal(pcm, silence) {
				sawSilence = true
				continue
			}
			speech = append(speech, pcm...)
		}
		if !bytes.Equal(speech, utterance) {
			t.Fatalf("speech of %d bytes, want the whole utterance in order", len(speech))
		}
		if !sawSilence {
			t.Fatal("no silence followed the utterance")
		}
		// Real-time pacing: about 0.25 s emits about 6 frames at 40 ms, not hundreds.
		if limit := int(250*time.Millisecond/FrameDuration) + 5; len(sent) >= limit {
			t.Fatalf("sent %d frames, want fewer than %d", len(sent), limit)
		}
	})

	t.Run("an interruption drops the unsent audio", func(t *testing.T) {
		recorder := NewClientRecorder(transportRate)
		out := testClientTransport(transport.Params{AudioOutEnabled: true}, recorder).output
		audio := bytes.Repeat([]byte{1, 0}, transportRate/10) // 100 ms
		recorder.AddUser(audio, transportRate)
		out.pending = append(out.pending, audio...)
		// Only the processor is set up: the transport's own setup would dial.
		if err := out.Base.Setup(t.Context(), processor.Setup{}); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = out.Base.Cleanup(context.Background()) })
		if err := out.ProcessFrame(t.Context(), frames.NewInterruptionFrame(), processor.Downstream); err != nil {
			t.Fatal(err)
		}
		if len(out.pending) != 0 {
			t.Fatalf("%d bytes still pending", len(out.pending))
		}
		if len(recorder.user.chunks) != 0 {
			t.Fatalf("the recorder kept %d chunks", len(recorder.user.chunks))
		}
	})
}

// A track lays its chunks out on a playout timeline; silence only for real
// pauses.
func TestRecorderTrack(t *testing.T) {
	const chunkDur = 40 * time.Millisecond
	chunk := bytes.Repeat([]byte{1, 0}, transportRate*int(chunkDur/time.Millisecond)/1000)
	base := time.Now()
	at := func(s float64) time.Time { return base.Add(time.Duration(s * float64(time.Second))) }
	track := func(arrivals []float64) *recorderTrack {
		tr := &recorderTrack{}
		for _, a := range arrivals {
			tr.addAt(at(a), chunk, transportRate)
		}
		return tr
	}
	paced := func(start float64, count int, step float64) []float64 {
		out := make([]float64, count)
		for i := range out {
			out[i] = start + float64(i)*step
		}
		return out
	}
	rendered := func(tr *recorderTrack, start float64) []byte {
		out, err := tr.rendered(transportRate, at(start))
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	// silenceS is the seconds of inserted silence: the output beyond the
	// chunks themselves.
	silenceS := func(tr *recorderTrack) float64 {
		first, _ := tr.first()
		out, err := tr.rendered(transportRate, first)
		if err != nil {
			t.Fatal(err)
		}
		return float64(len(out)-len(tr.chunks)*len(chunk)) / (transportRate * 2)
	}

	t.Run("a real-time stream is contiguous", func(t *testing.T) {
		jitter := []float64{0, 0.01, -0.005, 0.008, 0, -0.01, 0.012, 0}
		arrivals := paced(10, 8, 0.04)
		for i := range arrivals {
			arrivals[i] += jitter[i]
		}
		if got := rendered(track(arrivals), 10); !bytes.Equal(got, bytes.Repeat(chunk, 8)) {
			t.Fatalf("rendered %d bytes, want the 8 chunks back to back", len(got))
		}
	})
	t.Run("a pause between turns is silence", func(t *testing.T) {
		arrivals := append(paced(10, 5, 0.04), paced(10+5*0.04+1, 5, 0.04)...)
		if got := silenceS(track(arrivals)); math.Abs(got-1) > 0.001 {
			t.Fatalf("silence %.4fs, want 1s", got)
		}
	})
	t.Run("a receiver hiccup is absorbed by the source lead", func(t *testing.T) {
		// The bot sends at twice real time: 20 chunks (0.8 s of audio) in 0.4 s,
		// then a 0.3 s hole on the receiving side, then the rest. The hole is
		// shorter than the lead the source has built, so nothing in the
		// recording moves.
		arrivals := append(paced(10, 20, 0.02), paced(10.7, 10, 0.02)...)
		if got := rendered(track(arrivals), 10); !bytes.Equal(got, bytes.Repeat(chunk, 30)) {
			t.Fatalf("rendered %d bytes, want the 30 chunks back to back", len(got))
		}
	})
	t.Run("drop tail spans chunks", func(t *testing.T) {
		tr := track(paced(10, 3, 0.04))
		tr.dropTail(len(chunk) + 100)
		want := bytes.Repeat(chunk, 2)[:len(chunk)*2-100]
		if got := rendered(tr, 10); !bytes.Equal(got, want) {
			t.Fatalf("rendered %d bytes, want %d", len(got), len(want))
		}
	})
	t.Run("an unpaced pause counts from the end of playout", func(t *testing.T) {
		// A fast source (the user TTS) produces a turn in a burst; its playout
		// still takes the audio's duration, and the pause runs from that end.
		arrivals := append(paced(10, 10, 0.001), 20)
		if got := silenceS(track(arrivals)); math.Abs(got-(10-10*0.04)) > 0.001 {
			t.Fatalf("silence %.4fs, want 9.6s", got)
		}
	})
	t.Run("lead silence aligns to the recording start", func(t *testing.T) {
		got := rendered(track(paced(12, 2, 0.04)), 10)
		want := append(make([]byte, 2*transportRate*2), bytes.Repeat(chunk, 2)...)
		if !bytes.Equal(got, want) {
			t.Fatalf("rendered %d bytes, want %d", len(got), len(want))
		}
	})
}

// The input buffers the bot's audio and drops what is unplayed at an
// interruption.
func TestEvalHarnessInput(t *testing.T) {
	recorder := NewClientRecorder(transportRate)
	in := testClientTransport(transport.Params{AudioInEnabled: true}, recorder).input
	// The input has no pipeline here; a collector stands in for its neighbor.
	next := newCollector()
	in.Link(next)

	audio := bytes.Repeat([]byte{1, 0}, transportRate/10) // 100 ms
	in.PushAudioFrame(t.Context(), frames.NewInputAudioRawFrame(audio, transportRate, 1))
	in.PushAudioFrame(t.Context(), frames.NewInputAudioRawFrame(audio, transportRate, 1))
	if len(in.botPCM) != 2*len(audio) || len(recorder.bot.chunks) != 2 {
		t.Fatalf("buffered %d bytes, recorded %d chunks", len(in.botPCM), len(recorder.bot.chunks))
	}

	message, _ := json.Marshal(map[string]any{"label": rtvi.MessageLabel, "type": rtvi.TypeBotInterrupted})
	interrupted := frames.NewInputTransportMessageFrame(message)
	if err := in.PushFrame(t.Context(), interrupted, processor.Downstream); err != nil {
		t.Fatal(err)
	}

	if len(in.botPCM) != 0 {
		t.Fatalf("%d bytes still buffered", len(in.botPCM))
	}
	if len(recorder.bot.chunks) != 0 {
		t.Fatal("the recorder kept audio that was never played")
	}
	if got := next.seen(); len(got) == 0 || got[len(got)-1] != frames.Frame(interrupted) {
		t.Fatalf("pushed %v, want the interruption still reported", got)
	}
}

// eval-connect goes out first on the connection, ahead of client-ready.
func TestEvalClientTransportConnect(t *testing.T) {
	bot := newFakeRTVIServer(t)
	flags := evalws.ConnectionFlags{SkipTTS: true}
	tr := NewClientTransport(bot.url(), wsclient.Params{}, flags, nil)

	noRTVI := false
	worker := pipeline.NewWorker(pipeline.New(tr.Input(), tr.Output()), pipeline.WorkerConfig{EnableRTVI: &noRTVI})
	done := make(chan error, 1)
	go func() { done <- worker.Run(t.Context()) }()
	deadline := time.Now().Add(5 * time.Second)
	for len(bot.messages(rtvi.TypeClientReady)) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	worker.StopWhenDone()
	if err := <-done; err != nil {
		t.Fatal(err)
	}

	bot.mu.Lock()
	received := append([]rtvi.Incoming(nil), bot.received...)
	bot.mu.Unlock()
	if len(received) < 2 || received[0].ID != evalws.ConnectMessageType || received[1].Type != rtvi.TypeClientReady {
		t.Fatalf("received %+v, want eval-connect, then client-ready", received)
	}
	if got := bot.evalMessages(evalws.ConnectMessageType); len(got) != 1 || got[0]["skip_tts"] != true {
		t.Fatalf("eval-connect = %v, want the flags", got)
	}
}
