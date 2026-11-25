package rtc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
)

// readDeadline bounds a blocking RTP read so the read loop can notice
// cancellation between packets.
const readDeadline = 500 * time.Millisecond

// Transport is a WebRTC transport backed by a Pion connection. It provides the
// input and output processors for a pipeline.
//
// It raises transport.EventClientConnected when the peer connection is
// established (again after a restart), transport.EventClientDisconnected when
// the client goes away, and transport.EventAppMessage for each message the
// client sends. The client passed to the first two is the *Connection.
//
// With video input enabled, the client's camera is read from the start and its
// screen share once it is captured or an image is asked of it. Their pictures
// are pushed as UserImageRawFrames, sampled at the framerate each source is
// captured at (see CaptureParticipantVideo and Params.VideoInSources), and a
// UserImageRequestFrame is answered by the next picture of its source. With
// video output enabled, the images the pipeline shows are sent to the client
// as VP8 on the camera's transceiver.
type Transport struct {
	transport.BaseTransport
	in     *inputTransport
	out    *outputTransport
	conn   *Connection
	client *client
	// closing is set once the bot disconnects, so the connection closing then
	// is not reported as the client leaving.
	closing atomic.Bool
}

// NewTransport builds a WebRTC transport over conn.
func NewTransport(conn *Connection, params transport.Params) *Transport {
	t := &Transport{Name: "RTCTransport", conn: conn}
	t.client = &client{conn: conn, params: params, closing: &t.closing}
	t.in = newInput(conn, params, t.client)
	t.out = newOutput(conn, params, &t.closing, t.client)

	ev := t.in.Events()
	ev.Register(transport.EventAppMessage, false)
	ev.Register(transport.EventClientConnected, false)
	ev.Register(transport.EventClientDisconnected, false)

	conn.OnMessage(t.onAppMessage)
	conn.OnConnected(t.onClientConnected)
	conn.OnDisconnected(t.onPeerDisconnected)
	conn.OnClosed(t.onClientClosed)
	conn.OnTrackStatus(t.onTrackStatus)
	return t
}

// Input returns the input processor.
func (t *Transport) Input() processor.Processor { return t.in }

// Output returns the output processor.
func (t *Transport) Output() processor.Processor { return t.out }

// Events returns the registry of the events the transport raises.
func (t *Transport) Events() *event.Registry { return t.in.Events() }

// GetClientID returns the id of the peer connection a client event carries,
// which is the participant id image requests take. A client that is not a
// *Connection has none.
func (t *Transport) GetClientID(client any) string {
	if conn, ok := client.(*Connection); ok && conn != nil {
		return conn.PCID()
	}
	return ""
}

// onAppMessage surfaces a message from the client as a frame in the pipeline,
// then raises EventAppMessage with the message and the peer connection's id.
func (t *Transport) onAppMessage(raw []byte) {
	ctx := t.in.readContext()
	if ctx == nil {
		return
	}
	t.in.PushTransportMessage(ctx, raw)
	t.in.Events().Call(ctx, transport.EventAppMessage, t, json.RawMessage(raw), t.conn.PCID())
}

// onClientConnected picks up the client's tracks, captures the video sources
// Params.VideoInSources lists, raises EventClientConnected and reports the
// client to the pipeline. The connection raises it only once the pipeline is
// ready for it.
//
// The sources are captured before the event handlers run, so a handler that
// captures a source itself takes precedence.
func (t *Transport) onClientConnected() {
	ctx := t.in.readContext()
	if ctx == nil {
		return
	}
	t.client.handleClientConnected()
	params := t.in.Params()
	for _, videoSource := range configuredVideoSources(params) {
		if !t.in.SupportsVideoInSource(videoSource) {
			continue
		}
		sourceParams := params.VideoInSources[videoSource]
		framerate := sourceParams.Rate()
		if sourceParams.OnRequestOnly {
			framerate = 0
		}
		t.in.captureParticipantMedia(videoSource, framerate)
	}
	t.in.Events().Call(ctx, transport.EventClientConnected, t, t.conn)
	t.in.PushClientConnected(ctx)
}

// onPeerDisconnected forgets the client's tracks while the peer connection is
// down. They are picked up again when it reconnects.
func (t *Transport) onPeerDisconnected() {
	t.client.handlePeerDisconnected()
}

// onClientClosed forgets the client's tracks and, when the client went away
// rather than the bot disconnecting, answers the image requests waiting on its
// video and raises EventClientDisconnected.
func (t *Transport) onClientClosed() {
	t.client.handlePeerDisconnected()
	if t.closing.Load() {
		return
	}
	ctx := t.in.readContext()
	if ctx == nil {
		ctx = context.Background()
	}
	ctx = context.WithoutCancel(ctx)
	t.in.removeAllVideo(ctx)
	t.in.Events().Call(ctx, transport.EventClientDisconnected, t, t.conn)
}

// onTrackStatus handles the client turning one of its sources on or off: the
// image requests waiting on a video source turned off are answered.
func (t *Transport) onTrackStatus(receiverIndex int, enabled bool) {
	source, ok := sourcesByReceiver[receiverIndex]
	if !ok || source == micAudioSource || enabled {
		return
	}
	ctx := t.in.readContext()
	if ctx == nil {
		ctx = context.Background()
	}
	t.in.stopVideo(context.WithoutCancel(ctx), source)
}

// CaptureParticipantVideo captures the peer's camera or screen share,
// passing on framerate frames per second, every frame for
// transport.EveryFrame, or with onRequestOnly only the frames that answer image
// requests. The transport has one peer, so participantID does not select it.
// An empty videoSource is the camera.
func (t *Transport) CaptureParticipantVideo(
	_ context.Context, _ string, framerate int, videoSource string, onRequestOnly bool,
) error {
	if videoSource == "" {
		videoSource = camVideoSource
	}
	t.in.captureParticipantMedia(videoSource, transport.CaptureFramerate(framerate, onRequestOnly))
	return nil
}

// inputTransport reads Opus RTP from the connection, decodes it to PCM, and
// pushes InputAudioRawFrames into the pipeline at the input sample rate. It
// also reads the client's video sources, as UserImageRawFrames.
type inputTransport struct {
	*transport.BaseInput
	conn   *Connection
	client *client
	// videoSamplers decide which pictures of each video source are pushed.
	videoSamplers *transport.VideoInSamplers
	// receiveVideo holds the cancel function of each video source being read.
	receiveVideo map[string]context.CancelFunc
	videoWG      sync.WaitGroup
	// pushVideoFrame pushes a picture downstream: PushVideoFrame.
	pushVideoFrame func(context.Context, frames.InputImageFrame)
	dec            *opus.Decoder
	// resampler converts the 48 kHz the codec decodes to into the rate the
	// transport reads at. It passes audio through when the two match.
	resampler *resample.Resampler

	readWG     sync.WaitGroup
	mu         sync.Mutex // guards readCancel, readCtx and receiveVideo
	readCancel context.CancelFunc
	// readCtx is the context the input reads under, nil while it is not.
	readCtx context.Context //nolint:containedctx // the reading's lifetime, for the connection's callbacks
}

// readContext returns the context the input reads under, or nil when it is
// not reading.
func (in *inputTransport) readContext() context.Context {
	in.mu.Lock()
	defer in.mu.Unlock()
	return in.readCtx
}

func newInput(conn *Connection, params transport.Params, c *client) *inputTransport {
	in := &inputTransport{
		conn:          conn,
		client:        c,
		videoSamplers: transport.NewVideoInSamplers(),
		receiveVideo:  map[string]context.CancelFunc{},
	}
	in.BaseInput = transport.NewBaseInput("PionInput", params, in)
	in.pushVideoFrame = in.PushVideoFrame
	return in
}

// SupportsVideoInSource reports that the transport captures the camera and the
// screen share listed in Params.VideoInSources.
func (in *inputTransport) SupportsVideoInSource(videoSource string) bool {
	return videoSource == camVideoSource || videoSource == screenVideoSource
}

// ProcessFrame answers image requests, on top of what the base input does.
func (in *inputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := in.BaseInput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if request, ok := f.(*frames.UserImageRequestFrame); ok {
		in.requestParticipantImage(ctx, request)
	}
	return nil
}

// startReceiveVideo starts reading a video source, unless it is being read
// already or the input is not reading.
func (in *inputTransport) startReceiveVideo(videoSource string) {
	in.mu.Lock()
	defer in.mu.Unlock()
	if in.readCtx == nil || in.receiveVideo[videoSource] != nil {
		return
	}
	ctx, cancel := context.WithCancel(in.readCtx)
	in.receiveVideo[videoSource] = cancel
	in.videoWG.Go(func() { in.receiveVideoFrames(ctx, videoSource) })
}

// stopReceiveVideo stops reading every video source and waits for the reading
// to finish.
func (in *inputTransport) stopReceiveVideo() {
	in.mu.Lock()
	for source, cancel := range in.receiveVideo {
		cancel()
		delete(in.receiveVideo, source)
	}
	in.mu.Unlock()
	in.videoWG.Wait()
}

// receiveVideoFrames pushes the pictures of a video source its sampler passes
// on, each carrying the image request it answers, if any.
func (in *inputTransport) receiveVideoFrames(ctx context.Context, videoSource string) {
	for frame := range in.client.readVideoFrames(ctx, videoSource) {
		due, request := in.videoSamplers.Sample(peerID, videoSource)
		if !due {
			continue
		}
		if request != nil {
			frame.Text = request.Text
			frame.AppendToContext = request.AppendToContext
			frame.Request = request
		}
		in.pushVideoFrame(ctx, frame)
	}
}

// requestParticipantImage queues an image request, answered by the next
// picture of its source (the camera when it names none). A source the peer is
// not sending answers it at once, with an error.
//
// A request for a source not being captured starts capturing it, passing on
// every frame.
func (in *inputTransport) requestParticipantImage(ctx context.Context, request *frames.UserImageRequestFrame) {
	slog.Debug("rtc: requesting image from participant", "user", request.UserID)
	videoSource := request.VideoSource
	if videoSource == "" {
		videoSource = camVideoSource
	}
	if !in.client.videoSourceEnabled(videoSource) {
		in.AnswerImageRequests(ctx, []*frames.UserImageRequestFrame{request},
			fmt.Sprintf("The peer isn't sending %s video.", videoSource))
		return
	}
	if !in.videoSamplers.IsCapturing(peerID, videoSource) {
		in.videoSamplers.Capture(peerID, videoSource, transport.EveryFrame)
	}
	in.videoSamplers.AddRequest(peerID, videoSource, request)
	if in.SupportsVideoInSource(videoSource) && in.Params().VideoInEnabled {
		in.startReceiveVideo(videoSource)
	}
}

// stopVideo answers the image requests waiting on a video source the peer
// turned off. The source stays captured, so its frames are sampled again if it
// is turned back on.
func (in *inputTransport) stopVideo(ctx context.Context, videoSource string) {
	requests := in.videoSamplers.TakeRequests(peerID, videoSource)
	in.AnswerImageRequests(ctx, requests, fmt.Sprintf("The peer turned off its %s video.", videoSource))
}

// removeAllVideo stops sampling video and answers the waiting image requests,
// as when the peer disconnects.
func (in *inputTransport) removeAllVideo(ctx context.Context) {
	requests := in.videoSamplers.Clear()
	in.AnswerImageRequests(ctx, requests, "The peer disconnected.")
}

// captureParticipantMedia captures a source: a video source at framerate
// (0 for only the frames that answer image requests, transport.EveryFrame for
// every frame), read from now on. The microphone is read from the start.
func (in *inputTransport) captureParticipantMedia(source string, framerate int) {
	if !in.SupportsVideoInSource(source) {
		return
	}
	in.videoSamplers.Capture(peerID, source, framerate)
	if in.Params().VideoInEnabled {
		in.startReceiveVideo(source)
	}
}

func channels(n int) int {
	if n == 0 {
		return 1
	}
	return n
}

// StartReading decodes incoming audio on its own goroutine.
func (in *inputTransport) StartReading(ctx context.Context) error {
	ch := channels(in.Params().AudioInChannels)
	dec, err := opus.NewDecoder(ch)
	if err != nil {
		return err
	}
	// The decoded audio is one continuous stream, so the resampler keeps its
	// filter history for as long as the session lasts.
	r, err := resample.NewWithConfig(opus.SampleRate, in.sampleRate(), ch,
		resample.Config{ClearAfter: resample.NeverClear})
	if err != nil {
		return err
	}
	in.dec = dec
	in.resampler = r

	readCtx, cancel := context.WithCancel(ctx)
	in.mu.Lock()
	in.readCancel = cancel
	in.readCtx = readCtx
	in.mu.Unlock()
	in.readWG.Add(1)
	go in.readLoop(readCtx)

	// The camera is read from the start; the screen share once something
	// captures it.
	if in.Params().VideoInEnabled {
		in.startReceiveVideo(camVideoSource)
		if in.videoSamplers.IsCapturing(peerID, screenVideoSource) {
			in.startReceiveVideo(screenVideoSource)
		}
	}

	// The pipeline is ready for the connection now. The connected event is
	// raised now if the peer is already connected, or when it connects, and
	// the client's messages held until this point are released.
	in.conn.Connect()
	return nil
}

// StopReading stops the read goroutine.
func (in *inputTransport) StopReading(context.Context) error {
	in.mu.Lock()
	cancel := in.readCancel
	in.readCtx = nil
	in.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	in.readWG.Wait()
	in.stopReceiveVideo()
	if in.resampler != nil {
		in.resampler.Close()
		in.resampler = nil
	}
	return nil
}

// sampleRate is the rate the transport reads at, falling back to the codec's
// own rate when none was settled.
func (in *inputTransport) sampleRate() int {
	if r := in.SampleRate(); r > 0 {
		return r
	}
	return opus.SampleRate
}

func (in *inputTransport) readLoop(ctx context.Context) {
	defer in.readWG.Done()

	track, err := in.conn.RemoteAudioTrack(ctx)
	if err != nil {
		return
	}
	ch := channels(in.Params().AudioInChannels)
	rate := in.sampleRate()

	for {
		if ctx.Err() != nil {
			return
		}
		_ = track.SetReadDeadline(time.Now().Add(readDeadline))
		pkt, _, err := track.ReadRTP()
		if err != nil {
			// A deadline lets us re-check ctx.
			if errors.Is(err, os.ErrDeadlineExceeded) {
				continue
			}
			// Any other error means the track is gone. Reading it again would
			// fail the same way, so park until a restart brings a fresh one;
			// that is how the session survives the client restarting its peer
			// connection.
			slog.Warn("rtc: audio track stopped, waiting for a new one", "err", err)
			track, err = in.conn.nextRemoteAudioTrack(ctx, track)
			if err != nil {
				return
			}
			continue
		}
		if len(pkt.Payload) == 0 {
			continue
		}
		pcm, err := in.dec.Decode(pkt.Payload)
		if err != nil {
			continue
		}
		// The codec always decodes at 48 kHz; the pipeline gets the rate it
		// asked for.
		pcm = in.resampler.Process(pcm)
		if len(pcm) == 0 {
			continue
		}
		in.PushAudioFrame(ctx, frames.NewInputAudioRawFrame(pcm, rate, ch))
	}
}

// queuedFrames is how far the writer may run ahead of the sender. It has to be
// more than one: the sender takes whatever is queued at the instant it comes
// round, so a queue that only ever holds the frame being waited on leaves no
// slack, and any hesitation upstream — a goroutine not scheduled promptly, work
// done downstream between chunks — arrives too late and gets a frame of silence
// spliced into the middle of speech instead. The cushion is small enough that
// bot-speaking state stays close to what is actually playing.
const queuedFrames = 8

// starvationFrames is the longest run of silence that still counts as a gap in
// speech once real audio resumes after it. A longer run is taken as the end of
// an utterance.
const starvationFrames = 10

// gapTracker accounts for the silence the sender emits. Only silence that real
// audio resumes after is a fault: the writer fell behind and a word was chopped.
// Silence that nothing resumes is the talker having stopped, which is why a run
// stays pending until it is known which one it was — counting it on sight charges
// every utterance for its own ending.
type gapTracker struct {
	silence int64
	starved int64
	gaps    int64
	pending int64
	spoken  bool
	// sent is the frame the sender is on, so a gap can be placed in the session
	// rather than only counted at the end of it.
	sent int64
}

// real records a frame of real audio, settling any silence run before it.
func (g *gapTracker) real() {
	if g.pending > 0 && g.pending <= starvationFrames {
		g.starved += g.pending
		g.gaps++
		slog.Debug("audio gap spliced into speech",
			"gap", time.Duration(g.pending)*opus.FrameDuration,
			"at", time.Duration(g.sent)*opus.FrameDuration,
			"gaps", g.gaps)
	}
	g.pending = 0
	g.spoken = true
}

// quiet records a frame of silence sent because the queue was empty.
func (g *gapTracker) quiet() {
	g.silence++
	if g.spoken {
		g.pending++
	}
}

// outputTransport encodes outgoing PCM into Opus and writes it to the
// connection's audio track at real time.
//
// The sender goroutine owns the clock and writes on every frame boundary for as
// long as the session lasts, falling back to silence whenever nothing is queued
// — the same thing the local audio transport does in its playback callback, and
// what a device's pull callback forces on you. That is what keeps RTP timestamps
// honest: they advance one frame per packet whatever the wall clock did, so a
// sender that goes quiet during a gap and resumes leaves the audio after it
// timestamped as though the gap never happened. A receiver schedules playout from
// those timestamps, reads that as delay, conceals by repeating the last frame,
// then compresses once packets bunch up again — stuttered and clipped words. With
// the sender writing every frame, elapsed frames and elapsed time cannot diverge.
type outputTransport struct {
	*transport.BaseOutput
	conn   *Connection
	client *client
	enc    *opus.Encoder
	tail   []byte
	// frameBytes is the size of one 20 ms frame at the output sample rate. The
	// sender runs at that rate and the encoder brings it to the codec's own.
	frameBytes int

	mu      sync.Mutex
	queue   chan []byte
	cancel  context.CancelFunc
	sendWG  sync.WaitGroup
	running bool
	// queued counts frames handed over but not yet sent, so a graceful end can
	// wait for them rather than cutting the last words off.
	queued atomic.Int64

	disconnectOnce sync.Once
	// closing is the transport's flag that the bot is disconnecting.
	closing *atomic.Bool
}

func newOutput(conn *Connection, params transport.Params, closing *atomic.Bool, c *client) *outputTransport {
	out := &outputTransport{conn: conn, closing: closing, client: c}
	out.BaseOutput = transport.NewBaseOutput("PionOutput", params, out)
	return out
}

// ProcessFrame drives the sender's lifecycle around the base output: start it on
// the StartFrame so silence is already flowing before the first word, drop
// anything queued on a barge-in, and stop it when the session ends.
// StartWriting brings the sender up, so the outgoing track is running before the
// base starts queueing audio for it.
func (out *outputTransport) StartWriting(context.Context) error {
	return out.startSending()
}

func (out *outputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := out.BaseOutput.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if dir != processor.Downstream {
		return nil
	}
	switch f.(type) {
	case *frames.InterruptionFrame:
		out.discardQueued()
	case *frames.EndFrame:
		// The base output has handed over everything it had; let it play before
		// the sender goes away, or the farewell loses its last words.
		out.waitDrained(ctx)
		out.stopSending()
		out.disconnect()
	case *frames.CancelFrame:
		out.stopSending()
		out.disconnect()
	}
	return nil
}

// disconnect tells the client the bot has left and closes the connection, once.
// It happens here rather than on the input, which sees the EndFrame first:
// closing there would cut off the audio the output is still draining.
func (out *outputTransport) disconnect() {
	out.disconnectOnce.Do(func() {
		select {
		case <-out.conn.Done():
			// The client is already gone.
			return
		default:
		}
		slog.Info("rtc: disconnecting from the peer")
		out.closing.Store(true)
		_ = out.conn.Disconnect()
		out.client.handlePeerDisconnected()
	})
}

// WriteVideoFrame queues an image on the outgoing video track, already sized to
// the video output size by the base. It reports false when there is nowhere to
// send it: the client is not connected, the bot is leaving, or the transport
// sends no video.
func (out *outputTransport) WriteVideoFrame(_ context.Context, f *frames.OutputImageRawFrame) (bool, error) {
	return out.client.writeVideoFrame(f), nil
}

// SendMessage sends an application message over the data channel. The payload is
// encoded here rather than by the base, because what a wire carries is the
// transport's own business: this one speaks JSON to a browser client, which
// parses every message it receives, so every message is marshaled, a string
// included.
func (out *outputTransport) SendMessage(_ context.Context, f frames.OutputTransportMessage) error {
	data, err := encodeMessage(f.TransportMessage())
	if err != nil {
		return err
	}
	return out.conn.SendMessage(data)
}

// encodeMessage encodes an application message for the data channel as JSON.
func encodeMessage(message any) ([]byte, error) {
	return json.Marshal(message)
}

// startSending brings up the sender goroutine. It runs for the whole session, so
// the receiver is already being fed before the first word rather than having to
// build its buffer from a cold start mid-sentence.
func (out *outputTransport) startSending() error {
	out.mu.Lock()
	defer out.mu.Unlock()
	if out.running {
		return nil
	}
	ch := channels(out.Params().AudioOutChannels)
	p := out.Params()
	rate := out.SampleRate()
	if rate <= 0 {
		rate = opus.SampleRate
	}
	enc, err := opus.NewEncoder(opus.EncoderConfig{
		Channels:   ch,
		SampleRate: rate,
		Bitrate:    p.AudioOutBitrate,
	})
	if err != nil {
		return err
	}
	// The encoder is touched only by the sender goroutine from here on, so
	// packets cannot be sent in a different order than they were encoded.
	out.enc = enc
	out.frameBytes = opus.FrameBytesAt(rate, ch)
	out.queue = make(chan []byte, queuedFrames)
	ctx, cancel := context.WithCancel(context.Background())
	out.cancel = cancel
	out.running = true
	out.sendWG.Add(1)
	go out.sendLoop(ctx, out.frameBytes)
	return nil
}

// stopSending shuts the sender down and waits for it to finish.
func (out *outputTransport) stopSending() {
	out.mu.Lock()
	cancel := out.cancel
	out.cancel = nil
	out.running = false
	out.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	out.sendWG.Wait()
}

// discardQueued drops audio that has not been sent yet. A barge-in has to take
// effect now: anything already queued belongs to the turn the user just cut off.
func (out *outputTransport) discardQueued() {
	out.mu.Lock()
	q := out.queue
	out.tail = nil
	out.mu.Unlock()
	if q == nil {
		return
	}
	for {
		select {
		case <-q:
			out.queued.Add(-1)
		default:
			return
		}
	}
}

// waitDrained blocks until everything handed to the sender has gone out, so a
// graceful end plays the last of the audio instead of clipping it. Bounded, so a
// sender that has already stopped cannot hang the shutdown.
func (out *outputTransport) waitDrained(ctx context.Context) {
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(opus.FrameDuration)
	defer tick.Stop()
	for out.queued.Load() > 0 {
		select {
		case <-tick.C:
		case <-ctx.Done():
			return
		case <-out.conn.Done():
			return
		case <-deadline.C:
			return
		}
	}
}

// sendLoop writes one frame on every frame boundary until the session ends,
// sending queued audio when there is any and silence when there is not.
//
// The schedule comes from a fixed origin — frames sent times the frame duration —
// rather than from a ticker, which coalesces the ticks it misses and would let
// the stream fall permanently behind the wall clock while its timestamps claimed
// otherwise. Running late here instead sends back to back until the count catches
// up, so elapsed frames and elapsed time stay equal.
func (out *outputTransport) sendLoop(ctx context.Context, frameBytes int) {
	defer out.sendWG.Done()

	quiet := make([]byte, frameBytes)
	fillSilence := out.Params().AudioOutFillsSilence()
	start := time.Now()
	var sent int64
	var gap gapTracker

	defer func() {
		slog.Info("pion sender stopped", "processor", out.Name(), "frames", sent,
			"silence", gap.silence, "starved", gap.starved, "gaps", gap.gaps)
	}()

	for {
		if d := time.Until(start.Add(time.Duration(sent) * opus.FrameDuration)); d > 0 {
			timer := time.NewTimer(d)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-out.conn.Done():
				timer.Stop()
				return
			case <-timer.C:
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-out.conn.Done():
			return
		default:
		}

		gap.sent = sent
		pcm, ok := out.nextChunk(ctx, quiet, fillSilence, &gap)
		if !ok {
			return
		}

		packet, err := out.enc.Encode(pcm)
		if err == nil {
			err = out.conn.WriteAudio(packet, opus.FrameDuration)
		}
		if err != nil {
			slog.Error("write audio", "processor", out.Name(), "err", err)
		}
		sent++
	}
}

// nextChunk takes the next frame of audio to send: whatever is queued, or
// silence when nothing is and the stream is to be kept flowing.
//
// With the filling turned off it waits for audio instead. The timestamps go on
// advancing one frame per packet either way, so a receiver still schedules
// playout from a clock that never diverged from the wall clock. It reports false
// when the session ended while it waited.
func (out *outputTransport) nextChunk(
	ctx context.Context, quiet []byte, fillSilence bool, gap *gapTracker,
) ([]byte, bool) {
	select {
	case frame := <-out.queue:
		out.queued.Add(-1)
		gap.real()
		return frame, true
	default:
	}
	if fillSilence {
		gap.quiet()
		return quiet, true
	}
	select {
	case <-ctx.Done():
		return nil, false
	case <-out.conn.Done():
		return nil, false
	case frame := <-out.queue:
		out.queued.Add(-1)
		gap.real()
		return frame, true
	}
}

// WriteAudio hands PCM to the sender a frame at a time. It blocks only when the
// sender is already as far ahead as the queue allows, which is what paces the
// pipeline to real time; waiting on each frame individually would instead keep
// the queue empty and starve the sender. Audio that does not fill a whole frame
// is held until the next call.
func (out *outputTransport) WriteAudio(ctx context.Context, f frames.OutputAudioFrame) (bool, error) {
	pcm := f.AudioData().Audio

	out.mu.Lock()
	q, running := out.queue, out.running
	if !running {
		out.mu.Unlock()
		// The outgoing track is not live, so there is nowhere to send it.
		return false, nil
	}
	frameBytes := out.frameBytes
	out.tail = append(out.tail, pcm...)
	var batch [][]byte
	for len(out.tail) >= frameBytes {
		frame := make([]byte, frameBytes)
		copy(frame, out.tail[:frameBytes])
		batch = append(batch, frame)
		out.tail = out.tail[frameBytes:]
	}
	out.mu.Unlock()

	for _, frame := range batch {
		out.queued.Add(1)
		select {
		case q <- frame:
		case <-ctx.Done():
			out.queued.Add(-1)
			return false, ctx.Err()
		case <-out.conn.Done():
			out.queued.Add(-1)
			// The connection went away with the chunk part queued, so the rest
			// of it is never heard.
			return false, nil
		}
	}
	return true, nil
}
