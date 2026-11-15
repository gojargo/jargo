package transport

import (
	"context"
	"encoding/json"
	"log/slog"
	"slices"
	"sync"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline/capabilities"
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

	sampleRate int
	// filterActive reports whether the input filter came up, and so whether the
	// audio goroutine runs audio through it. It is written in Setup, before that
	// goroutine exists, and in Cleanup, once it has finished.
	filterActive bool

	// lifeMu serializes the streaming lifecycle (start, pause, stop) so that the
	// audio goroutine is only ever torn down and recreated by one caller at a
	// time. Without it a pause racing a teardown would add to the wait group
	// while the other was waiting on it. It is never held by PushAudioFrame, so
	// a driver blocked delivering audio cannot deadlock a stop.
	lifeMu sync.Mutex

	mu          sync.Mutex
	paused      bool
	audioIn     chan frames.InputAudioFrame
	audioCtx    context.Context
	audioCancel context.CancelFunc
	audioWG     sync.WaitGroup
}

// NewBaseInput builds a BaseInput. self is the embedding transport, used to
// dispatch StartReading/StopReading and to process frames.
func NewBaseInput(name string, params Params, self InputDriver) *BaseInput {
	bi := &BaseInput{params: params, self: self, filter: params.AudioInFilter}
	bi.Base = processor.New(name, self)

	var unsupported []string
	for source := range params.VideoInSources {
		if !self.SupportsVideoInSource(source) {
			unsupported = append(unsupported, source)
		}
	}
	if len(unsupported) > 0 {
		slices.Sort(unsupported)
		slog.Warn("video_in_sources aren't supported and will be ignored",
			"processor", name, "sources", unsupported)
	}
	return bi
}

// SupportsVideoInSource reports whether this transport captures a video source
// listed in Params.VideoInSources as its user connects. A transport that does
// overrides it; a listed source it reports false for is ignored, with a
// warning. The default captures none.
//
// It is called while the transport is being built, so an override must answer
// from the source alone.
func (bi *BaseInput) SupportsVideoInSource(string) bool { return false }

// PushVideoFrame pushes an image read from the user's video downstream, when
// video input is enabled and the transport is not paused.
func (bi *BaseInput) PushVideoFrame(ctx context.Context, f frames.InputImageFrame) {
	if !bi.params.VideoInEnabled {
		return
	}
	bi.mu.Lock()
	paused := bi.paused
	bi.mu.Unlock()
	if paused {
		return
	}
	_ = bi.PushFrame(ctx, f, processor.Downstream)
}

// AnswerImageRequests completes image requests that no frame will answer, with
// an error result naming why. The tool call that made each request would
// otherwise never finish.
func (bi *BaseInput) AnswerImageRequests(
	ctx context.Context, requests []*frames.UserImageRequestFrame, reason string,
) {
	if len(requests) == 0 {
		return
	}
	slog.Warn(reason, "processor", bi.Name())
	result, err := json.Marshal(map[string]string{"error": reason})
	if err != nil {
		slog.Error("encode an image request error", "processor", bi.Name(), "err", err)
		return
	}
	for _, request := range requests {
		if request.ResultCallback == nil {
			continue
		}
		if err := request.ResultCallback(ctx, string(result), nil); err != nil {
			slog.Warn("image request result callback failed", "processor", bi.Name(), "err", err)
		}
	}
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
func (bi *BaseInput) PushAudioFrame(ctx context.Context, f frames.InputAudioFrame) {
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

// Capabilities is the media this transport receives from the user: audio and
// video as the parameters enable them, and screen shares.
//
// Whether screen shares are received is unknown when video input is enabled
// with no VideoInSources, since the application then decides which sources to
// capture, and when the transport does not capture screen shares through
// VideoInSources.
func (bi *BaseInput) Capabilities() capabilities.BotCapabilities {
	var screenIn *bool
	switch {
	case !bi.params.VideoInEnabled:
		screenIn = new(false)
	case bi.self.SupportsVideoInSource(ScreenVideoSource) && len(bi.params.VideoInSources) > 0:
		_, listed := bi.params.VideoInSources[ScreenVideoSource]
		screenIn = new(listed)
	}
	return capabilities.BotCapabilities{
		AudioIn:  new(bi.params.AudioInEnabled),
		VideoIn:  new(bi.params.VideoInEnabled),
		ScreenIn: screenIn,
	}
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
	bi.PushParticipantMessage(ctx, raw, "")
}

// PushParticipantMessage is PushTransportMessage for a transport that knows who
// sent the message. The frame carries participantID as its ParticipantID, so
// whatever handles the message can tell whose it was.
func (bi *BaseInput) PushParticipantMessage(ctx context.Context, raw []byte, participantID string) {
	_ = bi.Broadcast(ctx, func() frames.Frame {
		f := frames.NewInputTransportMessageFrame(raw)
		f.ParticipantID = participantID
		return f
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
	if err := bi.params.Validate(); err != nil {
		return err
	}
	bi.sampleRate = pick(bi.params.AudioInSampleRate, s.AudioInSampleRate)

	// The filter starts here and stops in Cleanup, the only two calls guaranteed
	// to happen exactly once. Started from the StartFrame instead, a session that
	// is stopped and started again would start it twice without an intervening
	// stop, which a filter holding a shared resource pays for by taking that
	// resource twice. A filter that fails to start is skipped, not fatal.
	if bi.filter != nil {
		if err := bi.filter.Start(ctx, bi.sampleRate); err != nil {
			bi.PushError(ctx, "audio input filter failed to start", err, false)
		} else {
			bi.lifeMu.Lock()
			bi.filterActive = true
			bi.lifeMu.Unlock()
		}
	}
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
	case frames.InputAudioFrame:
		// Audio pushed in from upstream (by the RTVI processor, say) goes down
		// the same filtering path as audio read from the transport itself,
		// rather than being forwarded on as a plain frame.
		//
		// It travels as a copy. The filter writes its output back into the
		// frame it is given, and this frame arrived by being pushed, so an
		// observer was handed it and reads it on its own goroutine: the write
		// and that read are the same field. Audio the transport read itself
		// has never been pushed when it reaches the filter, so it needs none
		// of this.
		bi.PushAudioFrame(ctx, unshared(fr))
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
	case frames.FilterControlFrame:
		// Every filter control (switching the filter on or off, or changing its
		// settings) is the input filter's to apply. With no filter here it is
		// meant for one further down, so it travels on.
		if bi.filter == nil {
			return bi.PushFrame(ctx, f, dir)
		}
		return bi.filter.ProcessFrame(ctx, fr)
	default:
		return bi.PushFrame(ctx, f, dir)
	}
}

// Cleanup stops the audio goroutine, the input filter and the processor.
func (bi *BaseInput) Cleanup(ctx context.Context) error {
	bi.stopStreaming(ctx)

	// Paired with the start in Setup. The audio goroutine has finished by now,
	// so no Filter call can race with this Stop.
	bi.lifeMu.Lock()
	active := bi.filterActive
	bi.filterActive = false
	bi.lifeMu.Unlock()
	if active {
		_ = bi.filter.Stop(ctx)
	}
	return bi.Base.Cleanup(ctx)
}

func (bi *BaseInput) startStreaming(ctx context.Context) error {
	bi.lifeMu.Lock()
	defer bi.lifeMu.Unlock()

	bi.mu.Lock()
	bi.paused = false
	bi.audioCtx, bi.audioCancel = context.WithCancel(ctx)
	bi.audioIn = make(chan frames.InputAudioFrame, audioFrameChanCap)
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
	bi.audioIn = make(chan frames.InputAudioFrame, audioFrameChanCap)
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
func (bi *BaseInput) applyFilter(ctx context.Context, f frames.InputAudioFrame) bool {
	audio := f.AudioData()
	if !bi.filterActive {
		return len(audio.Audio) > 0
	}
	filtered, err := bi.filter.Filter(ctx, audio.Audio)
	if err != nil {
		bi.PushError(ctx, "audio input filter failed", err, false)
		return len(audio.Audio) > 0
	}
	audio.Audio = filtered
	return len(filtered) > 0
}

// unshared returns a frame the filtering path may write into, for one that
// something else is already holding.
//
// It keeps the frame's identity, so what travels on is still the frame that was
// pushed in rather than a new one appearing mid-pipeline, and it shares the
// audio rather than duplicating it, which a filter only ever reads.
func unshared(f frames.InputAudioFrame) frames.InputAudioFrame {
	switch fr := f.(type) {
	case *frames.UserAudioRawFrame:
		cp := *fr
		return &cp
	case *frames.InputAudioRawFrame:
		cp := *fr
		return &cp
	default:
		return f
	}
}

func pick(a, b int) int {
	if a != 0 {
		return a
	}
	return b
}
