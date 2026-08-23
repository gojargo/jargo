package transport

import (
	"context"
	"log/slog"
	"sync"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// BaseInput is the head of a pipeline: it turns received audio into
// InputAudioRawFrames and pushes them downstream. A concrete transport embeds
// it and implements InputDriver to supply the media reading; the driver calls
// PushAudioFrame for each chunk of audio it receives.
type BaseInput struct {
	*processor.Base
	params Params
	self   InputDriver
	filter audio.Filter

	sampleRate   int
	filterActive bool

	// lifeMu serializes the streaming lifecycle (start, pause, stop) so that the
	// audio goroutine is only ever torn down and recreated by one caller at a
	// time. Without it a pause racing a teardown would add to the wait group
	// while the other was waiting on it. It is never held by PushAudioFrame, so
	// a driver blocked delivering audio cannot deadlock a stop.
	lifeMu sync.Mutex

	mu          sync.Mutex
	paused      bool
	audioIn     chan *frames.InputAudioRawFrame
	audioCtx    context.Context
	audioCancel context.CancelFunc
	audioWG     sync.WaitGroup
}

// NewBaseInput builds a BaseInput. self is the embedding transport, used to
// dispatch StartReading/StopReading and to process frames.
func NewBaseInput(name string, params Params, self InputDriver) *BaseInput {
	bi := &BaseInput{params: params, self: self, filter: params.AudioInFilter}
	bi.Base = processor.New(name, self)
	return bi
}

// SampleRate is the input sample rate in Hz, set when the transport starts.
func (bi *BaseInput) SampleRate() int { return bi.sampleRate }

// Params returns the transport parameters.
func (bi *BaseInput) Params() Params { return bi.params }

// StartReading is the default no-op; a concrete transport overrides it.
func (bi *BaseInput) StartReading(context.Context) error { return nil }

// StopReading is the default no-op; a concrete transport overrides it.
func (bi *BaseInput) StopReading(context.Context) error { return nil }

// StartAudioStreaming is the default no-op; a concrete transport overrides it to
// begin streaming audio from its source.
func (bi *BaseInput) StartAudioStreaming(context.Context) error { return nil }

// EnableAudioInStreamOnStart sets whether the transport streams audio as soon as
// it starts. A transport that reads Params.AudioInStreamsOnStart when it starts
// can be told otherwise through this, up until then.
func (bi *BaseInput) EnableAudioInStreamOnStart(enabled bool) {
	slog.Debug("audio streaming on start", "processor", bi.Name(), "enabled", enabled)
	bi.params.AudioInStreamOnStart = &enabled
}

// PushAudioFrame queues a received audio frame to be pushed downstream. The
// driver calls it for each chunk of audio it reads from the transport.
func (bi *BaseInput) PushAudioFrame(ctx context.Context, f *frames.InputAudioRawFrame) {
	if !bi.params.AudioInEnabled {
		return
	}
	bi.mu.Lock()
	paused, ch := bi.paused, bi.audioIn
	bi.mu.Unlock()
	if paused || ch == nil {
		return
	}
	sendAudio(ctx, ch, f)
}

// PushTransportMessage emits a message received from the client as an
// InputTransportMessageFrame. A concrete transport calls it when an application
// message arrives (for example off a data channel).
//
// It is broadcast rather than pushed one way, so whatever handles client
// messages hears them wherever it sits: a processor placed ahead of the input
// transport reads them traveling upstream, one placed behind it reads them
// traveling downstream. Only the copy going its way reaches it, so it handles
// the message once.
func (bi *BaseInput) PushTransportMessage(ctx context.Context, raw []byte) {
	_ = bi.Broadcast(ctx, func() frames.Frame {
		return frames.NewInputTransportMessageFrame(raw)
	})
}

// PushClientConnected reports that a remote participant has connected. A
// concrete transport calls it where its own connection tells it one arrived; on
// a transport a client dials into, that has already happened by the time the
// pipeline runs, so it is reported as the transport starts reading.
//
// It goes downstream from the head of the pipeline, in order with the StartFrame
// that opened the run, which is what lets an observer time how long the call
// took to become answerable.
func (bi *BaseInput) PushClientConnected(ctx context.Context) {
	_ = bi.PushFrame(ctx, frames.NewClientConnectedFrame(), processor.Downstream)
}

// PushBotConnected reports that the bot itself has joined the session. Only a
// transport where the bot joins something has this to report: a room on a media
// server, where the bot is a participant like any other. On a transport a client
// dials into directly there is nothing to join, and nothing is reported.
func (bi *BaseInput) PushBotConnected(ctx context.Context) {
	_ = bi.PushFrame(ctx, frames.NewBotConnectedFrame(), processor.Downstream)
}

// Setup resolves the rate the transport reads at. A rate configured on the
// transport wins; otherwise it takes the pipeline's input rate, which it knows
// from the moment it is set up rather than when the StartFrame arrives.
func (bi *BaseInput) Setup(ctx context.Context, s processor.Setup) error {
	if err := bi.Base.Setup(ctx, s); err != nil {
		return err
	}
	bi.sampleRate = pick(bi.params.AudioInSampleRate, s.AudioInSampleRate)
	return nil
}

// ProcessFrame handles the transport lifecycle and forwards frames.
func (bi *BaseInput) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := bi.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	switch fr := f.(type) {
	case *frames.StartFrame:
		// Push StartFrame before starting so every processor downstream is
		// initialized before audio begins to flow.
		if err := bi.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		return bi.startStreaming(ctx)
	case *frames.CancelFrame:
		bi.stopStreaming(ctx)
		return bi.PushFrame(ctx, f, dir)
	case *frames.InputAudioRawFrame:
		// Audio pushed in from upstream (by the RTVI processor, say) goes down
		// the same filtering path as audio read from the transport itself,
		// rather than being forwarded on as a plain frame.
		bi.PushAudioFrame(ctx, fr)
		return nil
	case *frames.EndFrame:
		// Push EndFrame before stopping, because stopping waits for the audio
		// goroutine to finish.
		if err := bi.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		bi.stopStreaming(ctx)
		return nil
	case *frames.StopFrame:
		if err := bi.PushFrame(ctx, f, dir); err != nil {
			return err
		}
		bi.pause(ctx)
		return nil
	case *frames.InputTransportStartAudioStreamingFrame:
		return bi.self.StartAudioStreaming(ctx)
	case *frames.FilterUpdateSettingsFrame:
		if bi.filter == nil {
			return nil
		}
		return bi.filter.ProcessFrame(ctx, fr)
	default:
		return bi.PushFrame(ctx, f, dir)
	}
}

// Cleanup stops the audio goroutine and the processor.
func (bi *BaseInput) Cleanup(ctx context.Context) error {
	bi.stopStreaming(ctx)
	return bi.Base.Cleanup(ctx)
}

func (bi *BaseInput) startStreaming(ctx context.Context) error {
	bi.lifeMu.Lock()
	defer bi.lifeMu.Unlock()

	// Start the input filter before the audio goroutine so audioLoop observes a
	// stable filterActive. A filter that fails to start is skipped, not fatal.
	if bi.filter != nil {
		if err := bi.filter.Start(ctx, bi.sampleRate); err != nil {
			bi.PushError(ctx, "audio input filter failed to start", err, false)
			bi.filterActive = false
		} else {
			bi.filterActive = true
		}
	}

	bi.mu.Lock()
	bi.paused = false
	bi.audioCtx, bi.audioCancel = context.WithCancel(ctx)
	bi.audioIn = make(chan *frames.InputAudioRawFrame, audioFrameChanCap)
	audioCtx := bi.audioCtx
	bi.mu.Unlock()

	bi.audioWG.Add(1)
	go bi.audioLoop(audioCtx)

	return bi.self.StartReading(audioCtx)
}

// pause stops received audio from reaching the pipeline while leaving the
// transport reading, so the pipeline can be stopped and started again without
// tearing the connection down. Canceling the audio goroutine and starting a
// fresh one also drops whatever was already queued, so a restart does not replay
// audio from before the stop.
func (bi *BaseInput) pause(ctx context.Context) {
	bi.lifeMu.Lock()
	defer bi.lifeMu.Unlock()

	bi.mu.Lock()
	bi.paused = true
	cancel := bi.audioCancel
	bi.audioCancel = nil
	bi.mu.Unlock()

	if cancel != nil {
		cancel()
		bi.audioWG.Wait()
	}

	bi.mu.Lock()
	bi.audioCtx, bi.audioCancel = context.WithCancel(ctx)
	bi.audioIn = make(chan *frames.InputAudioRawFrame, audioFrameChanCap)
	audioCtx := bi.audioCtx
	bi.mu.Unlock()

	bi.audioWG.Add(1)
	go bi.audioLoop(audioCtx)
}

func (bi *BaseInput) stopStreaming(ctx context.Context) {
	bi.lifeMu.Lock()
	defer bi.lifeMu.Unlock()

	_ = bi.self.StopReading(ctx)

	bi.mu.Lock()
	cancel := bi.audioCancel
	bi.audioCancel = nil
	bi.mu.Unlock()

	if cancel != nil {
		cancel()
		bi.audioWG.Wait()

		// Stop the filter here, inside the branch only the winning stop caller
		// enters, and after the audio goroutine has finished: this stops it
		// exactly once with no Filter call able to race with Stop.
		if bi.filterActive {
			_ = bi.filter.Stop(ctx)
			bi.filterActive = false
		}
	}
}

// audioLoop drains received audio frames, runs them through the input filter,
// and pushes them downstream when passthrough is set.
//
// The filter runs whatever passthrough is set to. It is stateful, so feeding it
// only the audio that happens to be forwarded would leave it working from a
// signal with holes in it.
func (bi *BaseInput) audioLoop(ctx context.Context) {
	defer bi.audioWG.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case f := <-bi.audioIn:
			if !bi.applyFilter(ctx, f) {
				continue
			}
			if !bi.params.AudioInPassthrough {
				continue
			}
			_ = bi.PushFrame(ctx, f, processor.Downstream)
		}
	}
}

// applyFilter runs the input filter over the frame's audio, in place, and
// reports whether there is anything left to push. It returns false when the
// filter buffered the audio with nothing to emit yet.
//
// The audio is replaced on the frame rather than copied onto a new one, so what
// travels on is the frame the transport produced: the source it names and the
// moment it carries belong to that frame and are not the filter's to drop.
func (bi *BaseInput) applyFilter(ctx context.Context, f *frames.InputAudioRawFrame) bool {
	if !bi.filterActive {
		return len(f.Audio) > 0
	}
	filtered, err := bi.filter.Filter(ctx, f.Audio)
	if err != nil {
		bi.PushError(ctx, "audio input filter failed", err, false)
		return len(f.Audio) > 0
	}
	f.Audio = filtered
	return len(filtered) > 0
}

func pick(a, b int) int {
	if a != 0 {
		return a
	}
	return b
}
