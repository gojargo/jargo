// Package transport defines the boundary between a pipeline and the outside
// world. A Transport exposes an input processor that turns received media into
// frames and an output processor that turns frames into sent media.
//
// This package holds the transport-agnostic base processors; a concrete
// transport (for example the WebRTC transport in transport/rtc) embeds
// them and supplies the media I/O.
package transport

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/utils/events"
)

// The events a transport raises about its clients.
const (
	// EventClientConnected fires when a client connects, with the client as
	// its argument. What a client is depends on the transport.
	EventClientConnected = "on_client_connected"
	// EventClientDisconnected fires when a client disconnects, with the client
	// as its argument.
	EventClientDisconnected = "on_client_disconnected"
	// EventAppMessage fires for an application message from a client, with the
	// message and its sender as arguments.
	EventAppMessage = "on_app_message"
)

// Params configures a transport's audio input and output.
type Params struct {
	// AudioInEnabled enables receiving audio.
	AudioInEnabled bool
	// AudioInSampleRate is the input sample rate in Hz; 0 uses the StartFrame's.
	AudioInSampleRate int
	// AudioInChannels is the number of input channels.
	AudioInChannels int
	// AudioInStreamOnStart starts streaming audio as soon as the transport
	// starts; nil enables it. Set it false to hold the audio back until
	// something asks for it with an InputTransportStartAudioStreamingFrame,
	// which is how a client that has not said it is ready yet is kept off the
	// pipeline.
	AudioInStreamOnStart *bool
	// AudioInPassthrough pushes received audio frames downstream.
	AudioInPassthrough bool
	// AudioInFilter, when set, transforms received audio (for example noise
	// reduction) before it is pushed downstream to VAD, STT and turn detection.
	AudioInFilter audio.Filter

	// AudioOutEnabled enables sending audio.
	AudioOutEnabled bool
	// AudioOutSampleRate is the output sample rate in Hz; 0 uses the StartFrame's.
	AudioOutSampleRate int
	// AudioOutChannels is the number of output channels.
	AudioOutChannels int
	// AudioOutBitrate is the output bitrate in bits per second; 0 uses the
	// codec default.
	AudioOutBitrate int
	// AudioOutAutoSilence fills the outgoing stream with silence while nothing
	// is queued for it, rather than letting it go quiet; nil enables it. A
	// transport whose stream has to keep flowing leaves it on; one that would
	// rather wait for audio turns it off.
	AudioOutAutoSilence *bool
	// AudioOut10msChunks is how many 10 ms chunks of audio are written at a
	// time; 0 writes four, so audio is handed to the transport 40 ms at a time.
	// A transport that frames the audio itself (Opus at 20 ms, say) re-splits
	// what it is given, so this sizes the buffering rather than the framing.
	AudioOut10msChunks int
	// AudioOutFilter, when set, transforms the audio the pipeline sends to the
	// default destination, before the mixer and the write. Audio the transport
	// generates itself (mixer audio, end silence, DTMF tones) is not filtered.
	// A FilterControlFrame reaches it in step with the audio queued around it.
	AudioOutFilter audio.Filter
	// AudioOutVolume is the volume of the output audio, as a multiplier on its
	// samples, applied after the filter and before the mixer; nil plays it at
	// 1.0. A VolumeFrame changes it while the transport runs, and a
	// VolumeGainFrame adjusts it for some audio.
	AudioOutVolume *float64
	// AudioOutMixer, when set, mixes auxiliary audio (for example background
	// music) into the outgoing audio before it is sent. It serves the default
	// destination only; use AudioOutMixers to mix into a named one.
	AudioOutMixer audio.Mixer
	// AudioOutMixers maps a destination name to the mixer serving it, for a
	// transport that sends several outgoing streams and wants different
	// auxiliary audio (or none) on each. It takes precedence over AudioOutMixer.
	AudioOutMixers map[string]audio.Mixer
	// AudioOutWriteTimeout bounds a single write to the transport before the peer
	// is taken to be gone. A peer that stops reading blocks the write on buffers
	// that never drain, which no transport can report as an error, so how long
	// the write takes is the only signal there is. Zero uses
	// DefaultAudioOutWriteTimeout.
	AudioOutWriteTimeout time.Duration
	// AudioOutEndSilenceSecs is how many seconds of silence are sent after the
	// last of the audio when the pipeline ends, so the closing words are not
	// clipped by whatever closes on top of them. 0 sends none.
	AudioOutEndSilenceSecs int
	// AudioOutDestinations names the additional outgoing audio streams the
	// transport serves, beyond the default unnamed one. A frame is routed to the
	// destination it carries, so each stream keeps its own buffer, mixer and
	// speaking state. Leave it empty for a transport with a single stream.
	AudioOutDestinations []string

	// VideoInEnabled enables receiving video.
	VideoInEnabled bool
	// VideoInSources are the video sources to capture from each user as they
	// connect, keyed by source ("camera", "screenVideo", or a source of the
	// transport's own). A source not listed is captured only when the
	// application asks for it, with the transport's CaptureParticipantVideo.
	// It requires VideoInEnabled. A transport that does not capture a listed
	// source logs a warning and ignores it, so check the logs if video does not
	// arrive as configured.
	VideoInSources map[string]VideoInSourceParams

	// VideoOutEnabled enables sending video.
	VideoOutEnabled bool
	// VideoOutIsLive sends each image as a frame of live video, paced at
	// VideoOutFramerate, rather than repeating the last image (or cycling the
	// last sprite) until another arrives.
	VideoOutIsLive bool
	// VideoOutWidth and VideoOutHeight are the size of the video sent, in
	// pixels; 0 uses 1024 by 768. An image of another size is resized to it.
	VideoOutWidth  int
	VideoOutHeight int
	// VideoOutFramerate is the rate video is sent at, in frames per second; 0
	// uses 30.
	VideoOutFramerate int
	// VideoOutColorFormat is the pixel format of the video sent; "" uses "RGB".
	VideoOutColorFormat string
	// VideoOutCodec is the codec the transport prefers for the video it sends,
	// such as "VP8" or "H264", or "" for the transport's own choice.
	VideoOutCodec string
	// VideoOutDestinations names the additional outgoing video streams the
	// transport serves, beyond the default unnamed one. A destination named in
	// both this and AudioOutDestinations is served by one sender carrying both.
	VideoOutDestinations []string
}

// VideoInSourceParams says how a transport captures one video source from the
// user.
type VideoInSourceParams struct {
	// Framerate is how many frames per second to pass on from the source; 0
	// passes on 30. It does not apply with OnRequestOnly.
	Framerate int
	// OnRequestOnly passes on frames only to answer a UserImageRequestFrame,
	// rather than continuously at Framerate.
	OnRequestOnly bool
}

// DefaultVideoInFramerate is the rate a video source is captured at when its
// parameters name none.
const DefaultVideoInFramerate = 30

// The video output shape when the parameters name none.
const (
	DefaultVideoOutWidth       = 1024
	DefaultVideoOutHeight      = 768
	DefaultVideoOutFramerate   = 30
	DefaultVideoOutColorFormat = "RGB"
)

// errVideoInSourcesWithoutVideoIn reports video sources configured on a
// transport that receives no video.
//
//nolint:gochecknoglobals // sentinel error
var errVideoInSourcesWithoutVideoIn = errors.New("transport: VideoInSources requires VideoInEnabled")

// errFramerateOnRequestOnly reports a framerate given for a source that passes
// on frames only to answer image requests.
//
//nolint:gochecknoglobals // sentinel error
var errFramerateOnRequestOnly = errors.New("transport: a framerate does not apply with OnRequestOnly")

// errNegativeVideoParam reports a video size or rate below zero.
//
//nolint:gochecknoglobals // sentinel error
var errNegativeVideoParam = errors.New("transport: a video size or framerate cannot be negative")

// Rate is how many frames per second to pass on from the source: the
// configured rate, or DefaultVideoInFramerate.
func (p VideoInSourceParams) Rate() int {
	if p.Framerate > 0 {
		return p.Framerate
	}
	return DefaultVideoInFramerate
}

// Validate checks the parameters: no negative sizes or rates, video sources
// only with video input enabled, and no framerate on a source captured only on
// request.
//
// The checks are written out rather than read from struct tags: the parameters
// hold the application's own filters and mixers, which a tag-driven validator
// would walk into while they are in use.
func (p Params) Validate() error {
	if p.VideoOutWidth < 0 || p.VideoOutHeight < 0 || p.VideoOutFramerate < 0 {
		return errNegativeVideoParam
	}
	if len(p.VideoInSources) > 0 && !p.VideoInEnabled {
		return errVideoInSourcesWithoutVideoIn
	}
	for source, sp := range p.VideoInSources {
		if sp.Framerate < 0 {
			return fmt.Errorf("%w: source %q", errNegativeVideoParam, source)
		}
		if sp.OnRequestOnly && sp.Framerate != 0 {
			return fmt.Errorf("%w: source %q", errFramerateOnRequestOnly, source)
		}
	}
	return nil
}

// VideoOutSize is the size of the video sent: the configured one, or 1024 by
// 768.
func (p Params) VideoOutSize() (width, height int) {
	width, height = p.VideoOutWidth, p.VideoOutHeight
	if width <= 0 {
		width = DefaultVideoOutWidth
	}
	if height <= 0 {
		height = DefaultVideoOutHeight
	}
	return width, height
}

// VideoOutRate is the rate video is sent at: the configured one, or 30 frames
// per second.
func (p Params) VideoOutRate() int {
	if p.VideoOutFramerate > 0 {
		return p.VideoOutFramerate
	}
	return DefaultVideoOutFramerate
}

// VideoOutFormat is the pixel format of the video sent: the configured one, or
// "RGB".
func (p Params) VideoOutFormat() string {
	if p.VideoOutColorFormat != "" {
		return p.VideoOutColorFormat
	}
	return DefaultVideoOutColorFormat
}

// DefaultParams returns Params with audio input and output enabled and mono
// audio each way.
//
// It asks for 20 ms output chunks rather than leaving the parameter at its own
// default of 40 ms: it is the shape a WebRTC transport writes in, so the audio
// is handed over already framed and the transport does not re-split it.
func DefaultParams() Params {
	return Params{
		AudioInEnabled:         true,
		AudioInChannels:        1,
		AudioInPassthrough:     true,
		AudioOutEnabled:        true,
		AudioOutChannels:       1,
		AudioOut10msChunks:     2,
		AudioOutEndSilenceSecs: 2,
	}
}

// DefaultAudioOutWriteTimeout bounds a single write to the transport before the
// peer is taken to be gone.
const DefaultAudioOutWriteTimeout = 10 * time.Second

// AudioOutWrite is how long one write to the transport may take before the peer
// is written off, which is the configured value or the default.
func (p Params) AudioOutWrite() time.Duration {
	if p.AudioOutWriteTimeout > 0 {
		return p.AudioOutWriteTimeout
	}
	return DefaultAudioOutWriteTimeout
}

// AudioOutStartVolume is the volume the output audio starts at: the configured
// one, or 1.0.
func (p Params) AudioOutStartVolume() float64 {
	if p.AudioOutVolume == nil {
		return 1.0
	}
	return *p.AudioOutVolume
}

// AudioInStreamsOnStart reports whether received audio reaches the pipeline as
// soon as the transport starts, which it does unless the parameter says
// otherwise.
func (p Params) AudioInStreamsOnStart() bool {
	return p.AudioInStreamOnStart == nil || *p.AudioInStreamOnStart
}

// AudioOutFillsSilence reports whether the outgoing stream is kept flowing with
// silence while nothing is queued for it, which it is unless the parameter says
// otherwise.
func (p Params) AudioOutFillsSilence() bool {
	return p.AudioOutAutoSilence == nil || *p.AudioOutAutoSilence
}

// Transport is a source and sink of media for a pipeline. Input and Output
// return the processors that sit at the head and tail of the pipeline.
type Transport interface {
	// Input returns the processor that emits frames from received media.
	Input() processor.Processor
	// Output returns the processor that sends frames as media.
	Output() processor.Processor
	// Events returns the registry of the events the transport raises, such as
	// EventClientConnected and EventClientDisconnected.
	Events() *events.Registry
	// GetClientID returns the id of a client, as passed to
	// EventClientConnected. It is the participant id CaptureParticipantVideo
	// and image requests take. A transport with client events implements it;
	// BaseTransport logs a warning and returns "".
	GetClientID(client any) string
	// CaptureParticipantVideo captures a participant's video source, passing on
	// framerate frames per second, or every frame for EveryFrame. With
	// onRequestOnly it passes on only the frames that answer image requests,
	// and framerate does not apply.
	//
	// It captures a source on demand, where Params.VideoInSources captures the
	// listed sources as each user connects, and it takes precedence over that
	// for the source. A transport that receives user video implements it;
	// BaseTransport logs a warning.
	CaptureParticipantVideo(
		ctx context.Context, participantID string, framerate int, videoSource string, onRequestOnly bool,
	) error
}

// The video sources a user can send.
const (
	// CameraVideoSource is the user's camera.
	CameraVideoSource = "camera"
	// ScreenVideoSource is the user's screen share.
	ScreenVideoSource = "screenVideo"
)

// BaseTransport is the part of a Transport every transport shares. A concrete
// transport embeds it and overrides what it supports: one with client events
// overrides GetClientID, and one that receives user video overrides
// CaptureParticipantVideo.
type BaseTransport struct {
	// Name labels the transport in the warnings the defaults log.
	Name string
}

// GetClientID is the default for a transport with no client ids: it logs a
// warning and returns "".
func (b BaseTransport) GetClientID(any) string {
	slog.Warn("getting a client id isn't supported", "transport", b.Name)
	return ""
}

// CaptureParticipantVideo is the default for a transport that receives no user
// video: it logs a warning and captures nothing.
func (b BaseTransport) CaptureParticipantVideo(context.Context, string, int, string, bool) error {
	slog.Warn("capturing participant video isn't supported", "transport", b.Name)
	return nil
}

// InputDriver is implemented by a concrete input transport so the base can
// start and stop the transport-specific media reading. A driver produces audio
// by calling BaseInput.PushAudioFrame.
type InputDriver interface {
	processor.Processor
	// StartReading begins reading media from the transport. It runs under ctx,
	// which is canceled when the transport stops.
	StartReading(ctx context.Context) error
	// StopReading stops reading media from the transport.
	StopReading(ctx context.Context) error
	// StartAudioStreaming begins streaming audio from the transport's source,
	// for a transport that does not start streaming as soon as it connects. It
	// is driven by an InputTransportStartAudioStreamingFrame; the default does
	// nothing.
	//
	// A transport with a source it can hold back reads Params.AudioInStreamOnStart
	// when it starts, leaves the source closed when it is off, and opens it here.
	StartAudioStreaming(ctx context.Context) error
	// SupportsVideoInSource reports whether the transport captures a video
	// source listed in Params.VideoInSources as its user connects. A listed
	// source it reports false for is ignored, with a warning. The default
	// captures none.
	SupportsVideoInSource(videoSource string) bool
}

// OutputDriver is implemented by a concrete output transport so the base can
// hand it audio chunks and messages to send.
type OutputDriver interface {
	processor.Processor
	// WriteAudio sends one chunk of audio over the transport. The frame carries
	// the interleaved S16LE PCM in AudioData, and the outgoing stream it belongs
	// to in TransportDestination, so a transport serving several streams can
	// send it on the right one.
	//
	// It must return once ctx is done, however far it had got. Stopping the
	// output cancels ctx and then waits for the send loop to finish, and the
	// loop is inside this call whenever it is sending, so a write that blocks
	// past cancellation holds the pipeline open for good. A transport that
	// paces its sends, or waits for room downstream, waits on ctx.Done()
	// alongside whatever else it waits for.
	//
	// It reports whether the transport took the chunk. A transport with nowhere
	// to put it returns false and no error: its track is not live yet, or its
	// stream has closed. Nothing failed, but the audio will not be heard, so the
	// base does not forward the frame downstream as though it had been. An error
	// is a genuine failure, which the base logs; audio that errored counts as
	// not taken either way.
	//
	// Taken is not the same as written. A transport whose wire format buffers
	// audio across chunks has taken one that produced no payload of its own: the
	// audio reaches the client inside a later message, so the chunk is still
	// paced and still travels downstream.
	WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (sent bool, err error)
	// SendMessage sends the frame's application message to the client (for
	// example over a data channel). The transport encodes the payload, because
	// only it knows what its wire carries: one whose format is supplied as a
	// serializer hands the frame over and lets the serializer decide, and one
	// that speaks JSON to its client encodes it here.
	SendMessage(ctx context.Context, f frames.OutputTransportMessage) error
	// SupportsNativeDTMF reports whether the transport signals a keypress over
	// its own protocol. A transport that does overrides this and
	// WriteDTMFNative; one that does not has its keys sounded as audio, which
	// is the only way a keypress travels a call that carries nothing but audio.
	SupportsNativeDTMF() bool
	// WriteDTMFNative signals the frame's keys over the transport's protocol.
	// It is called only when SupportsNativeDTMF reports true.
	WriteDTMFNative(ctx context.Context, f frames.DTMFOutput) error
	// WriteTransportFrame handles a queued frame that carries no audio, once the
	// audio ahead of it has been sent. A transport overrides it to act on frame
	// types it carries itself; the default does nothing.
	WriteTransportFrame(ctx context.Context, f frames.Frame) error
	// StartWriting opens the transport's outgoing media path, so that nothing is
	// sent before it can carry audio. The base calls it while starting, and only
	// once it returns does it start the senders and report the transport ready.
	// A transport with nothing to open leaves the default, which does nothing.
	StartWriting(ctx context.Context) error
	// RegisterAudioDestination opens the outgoing stream a destination names,
	// for a transport that serves more than one. It is called for each entry in
	// Params.AudioOutDestinations when the transport starts, and a destination
	// that fails to open is skipped rather than being sent to. The default does
	// nothing.
	RegisterAudioDestination(ctx context.Context, destination string) error
	// RegisterVideoDestination opens the outgoing video stream a destination
	// names, for a transport that serves more than one. It is called for each
	// entry in Params.VideoOutDestinations when the transport starts. The
	// default does nothing.
	RegisterVideoDestination(ctx context.Context, destination string) error
	// WriteVideoFrame sends one image as a frame of the outgoing video, sized to
	// the video output size. The frame names the stream it belongs to in
	// TransportDestination. It reports whether the transport took the image; a
	// transport that sends no video leaves the default, which takes none.
	//
	// Like WriteAudio, it must return once ctx is done.
	WriteVideoFrame(ctx context.Context, f *frames.OutputImageRawFrame) (sent bool, err error)
}

// audioFrameChanCap bounds the buffered audio channels between the media
// goroutines and the pipeline.
const audioFrameChanCap = 256

// sendAudio enqueues f on ch, dropping it if ctx is done.
func sendAudio[T frames.Frame](ctx context.Context, ch chan T, f T) {
	select {
	case ch <- f:
	case <-ctx.Done():
	}
}
