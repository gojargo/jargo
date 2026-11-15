package eval

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/transport/wsclient"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// The harness's WebSocket transport and recorder.
//
// A live transport carries continuous audio both ways, speech or silence, and
// VADs, turn detection and streaming STT rely on that cadence. The eval bot
// sends and receives audio only while a TTS produces it, so both edges here
// reshape it: the output paces the user's audio to the bot one 40 ms frame per
// tick, silence in between, and the input re-emits the bot's audio at the same
// cadence, filling its gaps, so the harness's VAD finds the bot's real turn
// ends.
//
// The recording does not use the paced streams, whose jitter would make it
// stutter: ClientRecorder is fed the raw audio on both edges and lays each side
// out on its own timeline.

// FrameDuration is one audio frame per tick, on both edges. 40 ms matches the
// bot's output chunking (AudioOut10msChunks defaults to 4), so the receive side
// re-emits the bot's chunks one for one rather than splitting them, and it is
// still fine-grained enough for the VAD and turn models (well under their start
// and stop windows).
const FrameDuration = 40 * time.Millisecond

// recorderGap is how much later than its side's playout position a chunk has to
// arrive to be a real pause between turns; anything shorter is jitter within a
// contiguous turn and stays gapless.
const recorderGap = 200 * time.Millisecond

// frameBytes is the size of one FrameDuration frame of 16-bit mono audio at
// sampleRate.
func frameBytes(sampleRate int) int {
	return sampleRate * int(FrameDuration/time.Millisecond) / 1000 * 2
}

// sleepToNextTick sleeps to the next tick and returns the one after.
//
// When the loop has fallen behind, the next tick is measured from now rather
// than firing a burst of catch-up frames, which would compress the audio.
func sleepToNextTick(ctx context.Context, next time.Time) (time.Time, error) {
	wait := max(0, time.Until(next))
	if err := sleep(ctx, wait); err != nil {
		return next, err
	}
	if ctx.Err() != nil {
		return next, ctx.Err()
	}
	if wait == 0 {
		return time.Now().Add(FrameDuration), nil
	}
	return next.Add(FrameDuration), nil
}

// ClientRecorder builds the conversation recording from the raw audio, not the
// paced streams.
//
// The pacing tick cannot be held precisely, and a recording of the paced streams
// stutters. So each side is recorded as it was produced or received, laid out
// on its own timeline with silence only where a real pause was, and mixed to
// mono by Write. Audio the bot sent past an interruption is dropped, as a real
// client would drop it.
type ClientRecorder struct {
	rate int
	user *recorderTrack
	bot  *recorderTrack
}

// NewClientRecorder builds a recorder that writes at sampleRate; both sides are
// resampled to it on write.
func NewClientRecorder(sampleRate int) *ClientRecorder {
	return &ClientRecorder{rate: sampleRate, user: &recorderTrack{}, bot: &recorderTrack{}}
}

// AddUser records a chunk of the user's TTS audio (raw, before pacing to the
// bot), 16-bit mono PCM at inRate.
func (r *ClientRecorder) AddUser(pcm []byte, inRate int) { r.user.add(pcm, inRate) }

// AddBot records a chunk of the bot's audio (raw, before the gap-fill loop),
// 16-bit mono PCM at inRate.
func (r *ClientRecorder) AddBot(pcm []byte, inRate int) { r.bot.add(pcm, inRate) }

// DropBotTail drops the last numBytes of the bot's audio: sent, but never
// played.
func (r *ClientRecorder) DropBotTail(numBytes int) { r.bot.dropTail(numBytes) }

// DropUserTail drops the last numBytes of the user's audio: synthesized, but
// never sent.
func (r *ClientRecorder) DropUserTail(numBytes int) { r.user.dropTail(numBytes) }

// HasAudio reports whether any audio has been recorded on either side.
func (r *ClientRecorder) HasAudio() bool {
	_, user := r.user.first()
	_, bot := r.bot.first()
	return user || bot
}

// Write resamples both sides to a common rate, aligns them, mixes them to mono
// and writes a WAV file at path, creating its directory. It reports whether a
// file was written, which it is not when nothing was recorded.
func (r *ClientRecorder) Write(path string) (bool, error) {
	var (
		start time.Time
		found bool
	)
	for _, t := range []*recorderTrack{r.user, r.bot} {
		if first, ok := t.first(); ok && (!found || first.Before(start)) {
			start, found = first, true
		}
	}
	if !found {
		return false, nil
	}
	user, err := r.user.rendered(r.rate, start)
	if err != nil {
		return false, err
	}
	bot, err := r.bot.rendered(r.rate, start)
	if err != nil {
		return false, err
	}
	mixed := audio.MixAudio(user, bot)
	if err := os.MkdirAll(filepath.Dir(path), 0o750); err != nil {
		return false, err
	}
	if err := os.WriteFile(path, audio.PCMToWAV(mixed, r.rate, 1), 0o600); err != nil {
		return false, err
	}
	return true, nil
}

// recorderChunk is a raw chunk of audio and when it arrived.
type recorderChunk struct {
	arrived time.Time
	pcm     []byte
}

// recorderTrack is one side of the recording: raw audio chunks and when each
// arrived.
//
// Chunks are laid out back to back. Silence is inserted only when a chunk
// arrives later than the playout position by more than the gap threshold, since
// within a turn both sources run ahead of real time.
type recorderTrack struct {
	mu     sync.Mutex
	chunks []recorderChunk
	rate   int
}

// first is when the first chunk arrived, and false before any audio.
func (t *recorderTrack) first() (time.Time, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.chunks) == 0 {
		return time.Time{}, false
	}
	return t.chunks[0].arrived, true
}

// add records a raw audio chunk as of now.
func (t *recorderTrack) add(pcm []byte, inRate int) { t.addAt(time.Now(), pcm, inRate) }

// addAt records a raw audio chunk as arrived at at.
func (t *recorderTrack) addAt(at time.Time, pcm []byte, inRate int) {
	if len(pcm) == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.rate = inRate
	t.chunks = append(t.chunks, recorderChunk{arrived: at, pcm: append([]byte(nil), pcm...)})
}

// dropTail removes the last numBytes of audio, across as many chunks as it
// spans.
func (t *recorderTrack) dropTail(numBytes int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for numBytes > 0 && len(t.chunks) > 0 {
		last := &t.chunks[len(t.chunks)-1]
		if len(last.pcm) <= numBytes {
			numBytes -= len(last.pcm)
			t.chunks = t.chunks[:len(t.chunks)-1]
			continue
		}
		last.pcm = last.pcm[:len(last.pcm)-numBytes]
		numBytes = 0
	}
}

// rendered lays the track out, resamples it to outRate, and prepends its
// silence since start.
func (t *recorderTrack) rendered(outRate int, start time.Time) ([]byte, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.chunks) == 0 || t.rate == 0 {
		return nil, nil
	}
	pcm := t.layout()
	if t.rate != outRate {
		resampled, err := resample.Resample(pcm, t.rate, outRate, 1)
		if err != nil {
			return nil, err
		}
		pcm = resampled
	}
	lead := int(t.chunks[0].arrived.Sub(start).Seconds()*float64(outRate)*2) &^ 1
	return append(make([]byte, max(0, lead)), pcm...), nil
}

// layout concatenates the chunks, with silence for each real pause.
func (t *recorderTrack) layout() []byte {
	var out []byte
	// The playout position, as a wall-clock time.
	position := t.chunks[0].arrived
	bytesPerSecond := float64(t.rate * 2)
	for _, c := range t.chunks {
		late := c.arrived.Sub(position)
		switch {
		case late > recorderGap:
			out = append(out, make([]byte, int(late.Seconds()*bytesPerSecond)&^1)...)
			position = c.arrived
		case late > 0:
			position = c.arrived
		}
		out = append(out, c.pcm...)
		position = position.Add(time.Duration(float64(len(c.pcm)) / bytesPerSecond * float64(time.Second)))
	}
	return out
}

// ClientOutputTransport streams the user's audio to the bot as a continuous
// real-time stream.
//
// A real-time task sends one 40 ms frame per tick: queued TTS audio when there
// is some, silence otherwise, so the bot's VAD and turn detection see the
// silence they need to end a turn. It runs only when audio output is on.
type ClientOutputTransport struct {
	*wsclient.OutputTransport

	recorder *ClientRecorder

	mu      sync.Mutex
	pending []byte

	sendCancel context.CancelFunc
	sendDone   chan struct{}

	// sendFrame serializes and sends one paced frame; pushFrame pushes it on.
	// A test stands in for them.
	sendFrame func(ctx context.Context, f *frames.OutputAudioRawFrame)
	pushFrame func(ctx context.Context, f frames.Frame)
}

func newClientOutputTransport(t *wsclient.Transport, recorder *ClientRecorder) *ClientOutputTransport {
	out := &ClientOutputTransport{recorder: recorder}
	out.OutputTransport = wsclient.NewOutputTransport(t, out)
	out.sendFrame = out.sendToBot
	out.pushFrame = func(ctx context.Context, f frames.Frame) {
		_ = out.PushFrame(ctx, f, processor.Downstream)
	}
	return out
}

// ProcessFrame passes a frame on. In audio mode the start begins the real-time
// send stream, and an end or a cancel stops it before the transport. An
// interruption drops the user audio not yet sent, and the recorder drops the
// same bytes.
func (o *ClientOutputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	switch f.(type) {
	case *frames.InterruptionFrame:
		o.mu.Lock()
		if len(o.pending) > 0 {
			if o.recorder != nil {
				o.recorder.DropUserTail(len(o.pending))
			}
			o.pending = o.pending[:0]
		}
		o.mu.Unlock()
	case *frames.EndFrame, *frames.CancelFrame:
		o.cancelSendTask()
	}
	if err := o.OutputTransport.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok && o.Params().AudioOutEnabled {
		o.startSendTask(ctx)
	}
	return nil
}

// Cleanup stops the send stream, then cleans the transport up.
func (o *ClientOutputTransport) Cleanup(ctx context.Context) error {
	o.cancelSendTask()
	return o.OutputTransport.Cleanup(ctx)
}

// WriteAudio queues the user audio for the send task.
//
// It reports the audio as not sent, so the output does not push this unpaced
// frame downstream; the send task pushes the paced frames instead.
func (o *ClientOutputTransport) WriteAudio(_ context.Context, f frames.OutputAudioFrame) (bool, error) {
	session := o.Session()
	if session.IsClosing() || !session.IsConnected() {
		return false, nil
	}
	data := f.AudioData()
	// Record the raw TTS audio here (gapless), not the paced frames the send
	// task emits: pacing underruns and jitter would stutter the recording.
	if o.recorder != nil {
		o.recorder.AddUser(data.Audio, data.SampleRate)
	}
	o.mu.Lock()
	o.pending = append(o.pending, data.Audio...)
	o.mu.Unlock()
	return false, nil
}

// startSendTask starts the send stream, unless it is running.
func (o *ClientOutputTransport) startSendTask(ctx context.Context) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.sendCancel != nil {
		return
	}
	// The stream outlives the frame that started it; it is ended by the
	// pipeline's end or cancel.
	taskCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	o.sendCancel, o.sendDone = cancel, done
	rate := o.SampleRate()
	go func() {
		defer close(done)
		o.sendTaskHandler(taskCtx, rate)
	}()
}

// cancelSendTask stops the send stream and waits for it.
func (o *ClientOutputTransport) cancelSendTask() {
	o.mu.Lock()
	cancel, done := o.sendCancel, o.sendDone
	o.sendCancel, o.sendDone = nil, nil
	o.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
}

// sendTaskHandler sends one 40 ms frame every tick: queued audio, or silence
// when idle.
func (o *ClientOutputTransport) sendTaskHandler(ctx context.Context, sampleRate int) {
	chunk := frameBytes(sampleRate)
	channels := max(1, o.Params().AudioOutChannels)
	next := time.Now()
	for {
		o.mu.Lock()
		pcm := make([]byte, chunk)
		// A final partial chunk of the utterance is padded to a full frame.
		n := copy(pcm, o.pending)
		o.pending = o.pending[n:]
		o.mu.Unlock()

		frame := frames.NewOutputAudioRawFrame(pcm, sampleRate, channels)
		o.sendFrame(ctx, frame)
		// Every frame, audio and silence, goes downstream at this paced cadence:
		// whatever follows the output aligns by wall clock, so a continuous
		// stream keeps the user's turn at the right time.
		o.pushFrame(ctx, frame)

		var err error
		if next, err = sleepToNextTick(ctx, next); err != nil {
			return
		}
	}
}

// sendToBot serializes and sends one frame (raw PCM, through the RTVI
// serializer).
func (o *ClientOutputTransport) sendToBot(ctx context.Context, f *frames.OutputAudioRawFrame) {
	session := o.Session()
	if session.IsClosing() || !session.IsConnected() {
		return
	}
	_, _ = o.WriteFrame(ctx, f)
}

// ClientInputTransport feeds the bot's audio to the harness's VAD and STT as a
// continuous stream.
//
// The bot sends audio only while its TTS produces it, with gaps at its pauses.
// A gap reads as silence to a VAD, which would end the bot's turn mid-sentence,
// so the audio is buffered and re-emitted one 40 ms frame per tick, silence
// filling the gaps.
type ClientInputTransport struct {
	*wsclient.InputTransport

	recorder *ClientRecorder

	mu     sync.Mutex
	botPCM []byte

	fillCancel context.CancelFunc
	fillDone   chan struct{}
}

func newClientInputTransport(t *wsclient.Transport, recorder *ClientRecorder) *ClientInputTransport {
	in := &ClientInputTransport{recorder: recorder}
	in.InputTransport = wsclient.NewInputTransport(t, in)
	return in
}

// ProcessFrame passes a frame on. When audio is enabled the start begins the
// gap-fill stream, and an end or a cancel stops it before the transport.
func (i *ClientInputTransport) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	switch f.(type) {
	case *frames.EndFrame, *frames.CancelFrame:
		i.cancelFillTask()
	}
	if err := i.InputTransport.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.StartFrame); ok && i.Params().AudioInEnabled {
		i.startFillTask(ctx)
	}
	return nil
}

// Cleanup stops the gap-fill stream, then cleans the transport up.
func (i *ClientInputTransport) Cleanup(ctx context.Context) error {
	i.cancelFillTask()
	return i.InputTransport.Cleanup(ctx)
}

// PushFrame pushes a frame on. When the bot reports an interruption, the audio
// it sent but a client would never play is dropped, and the recorder drops the
// same bytes.
func (i *ClientInputTransport) PushFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if mf, ok := f.(*frames.InputTransportMessageFrame); ok && dir == processor.Downstream && botInterrupted(mf) {
		i.mu.Lock()
		if i.recorder != nil {
			i.recorder.DropBotTail(len(i.botPCM))
		}
		i.botPCM = i.botPCM[:0]
		i.mu.Unlock()
	}
	return i.InputTransport.PushFrame(ctx, f, dir)
}

// botInterrupted reports whether a message is the bot's RTVI bot-interrupted.
func botInterrupted(f *frames.InputTransportMessageFrame) bool {
	var message struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(f.Message, &message); err != nil {
		return false
	}
	return message.Type == rtvi.TypeBotInterrupted
}

// PushAudioFrame buffers the bot's audio for the fill task.
func (i *ClientInputTransport) PushAudioFrame(_ context.Context, f frames.InputAudioFrame) {
	data := f.AudioData()
	i.mu.Lock()
	defer i.mu.Unlock()
	// Record the raw bot audio here (gapless within a turn), not the
	// gap-filled frames the fill task emits: those underruns would stutter the
	// recording.
	if i.recorder != nil {
		i.recorder.AddBot(data.Audio, data.SampleRate)
	}
	i.botPCM = append(i.botPCM, data.Audio...)
}

// startFillTask starts the gap-fill stream, unless it is running.
func (i *ClientInputTransport) startFillTask(ctx context.Context) {
	i.mu.Lock()
	defer i.mu.Unlock()
	if i.fillCancel != nil {
		return
	}
	// The stream outlives the frame that started it; it is ended by the
	// pipeline's end or cancel.
	taskCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	i.fillCancel, i.fillDone = cancel, done
	rate := i.SampleRate()
	go func() {
		defer close(done)
		i.fillTaskHandler(taskCtx, rate)
	}()
}

// cancelFillTask stops the gap-fill stream and waits for it.
func (i *ClientInputTransport) cancelFillTask() {
	i.mu.Lock()
	cancel, done := i.fillCancel, i.fillDone
	i.fillCancel, i.fillDone = nil, nil
	i.mu.Unlock()
	if cancel == nil {
		return
	}
	cancel()
	<-done
}

// fillTaskHandler pushes one 40 ms frame every tick: buffered bot audio, or
// silence.
func (i *ClientInputTransport) fillTaskHandler(ctx context.Context, sampleRate int) {
	chunk := frameBytes(sampleRate)
	next := time.Now()
	for {
		i.mu.Lock()
		pcm := make([]byte, chunk)
		n := copy(pcm, i.botPCM)
		i.botPCM = i.botPCM[n:]
		i.mu.Unlock()

		// The base pushes it through the normal input audio path (VAD, STT).
		i.InputTransport.PushAudioFrame(ctx, frames.NewInputAudioRawFrame(pcm, sampleRate, 1))

		var err error
		if next, err = sleepToNextTick(ctx, next); err != nil {
			return
		}
	}
}

// ClientTransport is the harness's RTVI client transport, with audio edges that
// behave like a live transport.
//
// The input fills the gaps in the bot's audio and the output paces the user's;
// both feed the recorder the raw audio when one is given. On connect it sends
// the connection's flags in an eval-connect message, ahead of the RTVI
// handshake, which is how a bot hosted by the development runner recognizes the
// harness and learns what the connection asks of it.
type ClientTransport struct {
	*wsclient.RTVITransport

	flags    evalws.ConnectionFlags
	recorder *ClientRecorder
	input    *ClientInputTransport
	output   *ClientOutputTransport
}

// NewClientTransport builds the harness's transport to the bot at uri, sending
// flags first on every connection. recorder is fed the raw audio on both edges,
// or is nil not to record. The serializer defaults to a ClientSerializer when
// params does not set one.
func NewClientTransport(
	uri string, params wsclient.Params, flags evalws.ConnectionFlags, recorder *ClientRecorder,
) *ClientTransport {
	if params.Serializer == nil {
		params.Serializer = NewClientSerializer()
	}
	t := &ClientTransport{RTVITransport: wsclient.NewRTVITransport(uri, params), flags: flags, recorder: recorder}
	t.SetOpeningMessages(wsserver.Message{Data: []byte(evalws.ConnectMessage(flags))})
	t.input = newClientInputTransport(t.RTVITransport.Transport, recorder)
	t.output = newClientOutputTransport(t.RTVITransport.Transport, recorder)
	t.SetInput(t.input.InputTransport)
	t.SetOutput(t.output.OutputTransport)
	return t
}

// Flags are the connection's flags, as sent in eval-connect.
func (t *ClientTransport) Flags() evalws.ConnectionFlags { return t.flags }
