// Package audiobuffer records a conversation's audio. A Processor taps the audio
// flowing through the pipeline — the user's InputAudioRawFrames and the bot's
// Output/TTS audio — keeps the two tracks time-aligned by padding silence for
// wall-clock gaps, and delivers the recording through callbacks: a merged mix
// (mono) or interleaved stereo (user left, bot right), and the separate tracks.
//
// Recording is off until started, either automatically at pipeline start
// (AutoStart) or by StartRecording / an AudioBufferStartRecordingFrame. It is
// stopped by StopRecording, an AudioBufferStopRecordingFrame, or pipeline
// shutdown, which flushes the final audio and fires OnRecordingStopped.
package audiobuffer

import (
	"context"
	"encoding/binary"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// silenceGapThreshold is the wall-clock gap, beyond a frame's own duration, that
// triggers silence padding. It sits safely above normal jitter.
const silenceGapThreshold = 200 * time.Millisecond

// Config configures a recording Processor.
type Config struct {
	// SampleRate overrides the recording sample rate in Hz; 0 uses the pipeline
	// output rate from the StartFrame.
	SampleRate int
	// NumChannels is 1 for a mono mix of user and bot, or 2 for stereo with the
	// user on the left and the bot on the right; 0 uses 1.
	NumChannels int
	// BufferSize is the byte threshold at which OnAudioData and OnTrackAudioData
	// fire with the audio accumulated so far; 0 emits only when recording stops.
	BufferSize int
	// AutoStart begins recording as soon as the pipeline starts.
	AutoStart bool

	// OnRecordingStarted is called when recording transitions to active.
	OnRecordingStarted func()
	// OnRecordingStopped is called after recording stops and the final audio has
	// been delivered.
	OnRecordingStopped func()
	// OnAudioData receives the merged recording (mono mix or interleaved stereo)
	// at each BufferSize boundary and once more on stop.
	OnAudioData func(audio []byte, sampleRate, numChannels int)
	// OnTrackAudioData receives the separate, time-aligned user and bot tracks at
	// each BufferSize boundary and once more on stop.
	OnTrackAudioData func(user, bot []byte, sampleRate, numChannels int)
}

// Processor records the pipeline's audio. It is placed downstream, typically
// after the output transport, so it sees both the user's input audio and the
// bot's output audio.
type Processor struct {
	*processor.Base
	cfg Config

	mu         sync.Mutex
	recording  bool
	sampleRate int
	bufSize    int
	channels   int

	userBuf      []byte
	botBuf       []byte
	userSpeaking bool
	botSpeaking  bool
	lastUser     *time.Time
	lastBot      *time.Time

	inRes   *resample.Resampler
	inRate  int
	outRes  *resample.Resampler
	outRate int
}

// New builds a recording Processor.
func New(cfg Config) *Processor {
	if cfg.NumChannels == 0 {
		cfg.NumChannels = 1
	}
	p := &Processor{cfg: cfg, channels: cfg.NumChannels}
	p.Base = processor.New("AudioBuffer", p)
	return p
}

// ProcessFrame taps audio for recording and forwards every frame untouched.
func (p *Processor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch f := f.(type) {
	case *frames.StartFrame:
		p.onStart(f)
	case *frames.AudioBufferStartRecordingFrame:
		p.startRecording()
	case *frames.AudioBufferStopRecordingFrame:
		p.stopRecording()
	}

	p.record(f)

	switch f.(type) {
	case *frames.CancelFrame, *frames.EndFrame:
		p.stopRecording()
	}

	return p.PushFrame(ctx, f, dir)
}

// Cleanup releases the resamplers after the processor stops.
func (p *Processor) Cleanup(ctx context.Context) error {
	err := p.Base.Cleanup(ctx)
	p.mu.Lock()
	if p.inRes != nil {
		p.inRes.Close()
		p.inRes = nil
	}
	if p.outRes != nil {
		p.outRes.Close()
		p.outRes = nil
	}
	p.mu.Unlock()
	return err
}

// StartRecording begins recording. It does nothing if already recording.
func (p *Processor) StartRecording() { p.startRecording() }

// StopRecording flushes the buffered audio and stops recording. It does nothing
// if not recording.
func (p *Processor) StopRecording() { p.stopRecording() }

// Recording reports whether recording is active.
func (p *Processor) Recording() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.recording
}

func (p *Processor) onStart(f *frames.StartFrame) {
	p.mu.Lock()
	p.sampleRate = p.cfg.SampleRate
	if p.sampleRate == 0 {
		p.sampleRate = f.AudioOutSampleRate
	}
	p.bufSize = p.cfg.BufferSize
	p.mu.Unlock()
	if p.cfg.AutoStart {
		p.startRecording()
	}
}

func (p *Processor) startRecording() {
	p.mu.Lock()
	if p.recording {
		p.mu.Unlock()
		return
	}
	p.recording = true
	p.reset(nil)
	p.mu.Unlock()
	if p.cfg.OnRecordingStarted != nil {
		p.cfg.OnRecordingStarted()
	}
}

func (p *Processor) stopRecording() {
	p.mu.Lock()
	if !p.recording {
		p.mu.Unlock()
		return
	}
	snap := p.snapshot()
	p.reset(nil)
	p.recording = false
	p.mu.Unlock()

	p.deliver(snap)
	if p.cfg.OnRecordingStopped != nil {
		p.cfg.OnRecordingStopped()
	}
}

// record buffers one frame's audio when recording, keeping the user and bot
// tracks time-aligned, and flushes when a buffer reaches BufferSize. It runs
// under the lock; callbacks fire after it is released.
func (p *Processor) record(f frames.Frame) {
	p.mu.Lock()
	if !p.recording {
		p.mu.Unlock()
		return
	}
	switch f.(type) {
	case *frames.UserStartedSpeakingFrame:
		p.userSpeaking = true
	case *frames.UserStoppedSpeakingFrame:
		p.userSpeaking = false
	case *frames.BotStartedSpeakingFrame:
		p.botSpeaking = true
	case *frames.BotStoppedSpeakingFrame:
		p.botSpeaking = false
	}

	switch audio, rate, ch, kind := classify(f); kind {
	case kindUser:
		if res := p.resampleInput(audio, rate, ch); len(res) > 0 {
			p.addUserAudio(res)
		}
	case kindBot:
		if res := p.resampleOutput(audio, rate, ch); len(res) > 0 {
			p.addBotAudio(res)
		}
	case kindNone:
		// Not an audio frame; nothing to buffer.
	}

	var snap *recording
	if p.bufSize > 0 && (len(p.userBuf) >= p.bufSize || len(p.botBuf) >= p.bufSize) {
		snap = p.snapshot()
		p.resetPrimary()
	}
	p.mu.Unlock()

	p.deliver(snap)
}

// addUserAudio appends resampled user audio, padding the silence gap since the
// last write and syncing the bot track to the same timestamp (unless the bot is
// speaking). It runs with the lock held.
func (p *Processor) addUserAudio(res []byte) {
	now := time.Now()
	p.userBuf = p.fillSilenceGap(p.userBuf, p.lastUser, now, len(res))
	p.lastUser = new(now)
	if !p.botSpeaking {
		p.botBuf = padTo(p.botBuf, len(p.userBuf))
		p.lastBot = new(time.Now())
	}
	p.userBuf = append(p.userBuf, res...)
}

// addBotAudio appends resampled bot audio, padding the silence gap and syncing
// the user track (unless the user is speaking). It runs with the lock held.
func (p *Processor) addBotAudio(res []byte) {
	now := time.Now()
	p.botBuf = p.fillSilenceGap(p.botBuf, p.lastBot, now, len(res))
	p.lastBot = new(now)
	if !p.userSpeaking {
		p.userBuf = padTo(p.userBuf, len(p.botBuf))
		p.lastUser = new(time.Now())
	}
	p.botBuf = append(p.botBuf, res...)
}

// deliver fires the audio-data callbacks for a snapshot outside the lock.
func (p *Processor) deliver(r *recording) {
	if r == nil {
		return
	}
	if p.cfg.OnAudioData != nil {
		p.cfg.OnAudioData(r.merged, r.sampleRate, r.channels)
	}
	if p.cfg.OnTrackAudioData != nil {
		p.cfg.OnTrackAudioData(r.user, r.bot, r.sampleRate, r.channels)
	}
}

// recording is a snapshot of the buffered audio to hand to the callbacks.
type recording struct {
	merged     []byte
	user       []byte
	bot        []byte
	sampleRate int
	channels   int
}

// snapshot aligns the two tracks and copies them out. It returns nil when there
// is no audio. It must be called with the lock held.
func (p *Processor) snapshot() *recording {
	if len(p.userBuf) == 0 && len(p.botBuf) == 0 {
		return nil
	}
	// Pad the shorter track so both end at the same timestamp.
	target := max(len(p.userBuf), len(p.botBuf))
	user := padTo(append([]byte(nil), p.userBuf...), target)
	bot := padTo(append([]byte(nil), p.botBuf...), target)
	var merged []byte
	if p.channels == 2 {
		merged = interleaveStereo(user, bot)
	} else {
		merged = mixAudio(user, bot)
	}
	return &recording{merged: merged, user: user, bot: bot, sampleRate: p.sampleRate, channels: p.channels}
}

// reset clears all recording state. The optional now sets the last-write
// timestamps; nil leaves them unset so the first audio injects no silence.
func (p *Processor) reset(now *time.Time) {
	p.userBuf = nil
	p.botBuf = nil
	p.lastUser = now
	p.lastBot = now
}

// resetPrimary clears the buffers after a flush and re-bases the timestamps to
// now, so the next audio measures its gap from the flush point.
func (p *Processor) resetPrimary() {
	now := time.Now()
	p.userBuf = nil
	p.botBuf = nil
	p.lastUser = new(now)
	p.lastBot = new(now)
}

// fillSilenceGap pads buf with silence for any wall-clock gap since it was last
// written, beyond the incoming frame's own duration. It must be called with the
// lock held.
func (p *Processor) fillSilenceGap(buf []byte, last *time.Time, now time.Time, frameBytes int) []byte {
	if last == nil || p.sampleRate == 0 {
		return buf
	}
	elapsed := now.Sub(*last)
	frameDur := time.Duration(float64(frameBytes) / float64(p.sampleRate*2) * float64(time.Second))
	gap := elapsed - frameDur
	if gap > silenceGapThreshold {
		sb := int(gap.Seconds() * float64(p.sampleRate) * 2)
		sb -= sb % 2 // keep 16-bit alignment
		if sb > 0 {
			buf = append(buf, make([]byte, sb)...)
		}
	}
	return buf
}

func (p *Processor) resampleInput(audio []byte, rate, channels int) []byte {
	if rate == p.sampleRate || rate == 0 {
		return audio
	}
	if p.inRes == nil || p.inRate != rate {
		if p.inRes != nil {
			p.inRes.Close()
		}
		r, err := resample.New(rate, p.sampleRate, channels)
		if err != nil {
			return audio
		}
		p.inRes, p.inRate = r, rate
	}
	return p.inRes.Process(audio)
}

func (p *Processor) resampleOutput(audio []byte, rate, channels int) []byte {
	if rate == p.sampleRate || rate == 0 {
		return audio
	}
	if p.outRes == nil || p.outRate != rate {
		if p.outRes != nil {
			p.outRes.Close()
		}
		r, err := resample.New(rate, p.sampleRate, channels)
		if err != nil {
			return audio
		}
		p.outRes, p.outRate = r, rate
	}
	return p.outRes.Process(audio)
}

type audioKind int

const (
	kindNone audioKind = iota
	kindUser
	kindBot
)

// classify reports whether a frame carries user or bot audio, returning the raw
// PCM, its sample rate and channel count.
func classify(f frames.Frame) (audio []byte, rate, channels int, kind audioKind) {
	switch fr := f.(type) {
	case *frames.InputAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels, kindUser
	case *frames.TTSAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels, kindBot
	case *frames.OutputAudioRawFrame:
		return fr.Audio, fr.SampleRate, fr.NumChannels, kindBot
	default:
		return nil, 0, 0, kindNone
	}
}

// padTo appends silence to buf until it reaches at least target bytes.
func padTo(buf []byte, target int) []byte {
	if len(buf) < target {
		buf = append(buf, make([]byte, target-len(buf))...)
	}
	return buf
}

// mixAudio sums two 16-bit PCM streams sample-by-sample, clipping to the 16-bit
// range. The shorter stream is treated as zero-padded.
func mixAudio(a, b []byte) []byte {
	n := max(len(a), len(b))
	n -= n % 2
	out := make([]byte, n)
	for i := 0; i+1 < n; i += 2 {
		var s int32
		if i+1 < len(a) {
			s += int32(int16(binary.LittleEndian.Uint16(a[i:])))
		}
		if i+1 < len(b) {
			s += int32(int16(binary.LittleEndian.Uint16(b[i:])))
		}
		if s > 32767 {
			s = 32767
		} else if s < -32768 {
			s = -32768
		}
		binary.LittleEndian.PutUint16(out[i:], uint16(int16(s)))
	}
	return out
}

// interleaveStereo weaves two mono 16-bit streams into stereo (L, R, L, R …),
// truncating to the shorter stream.
func interleaveStereo(left, right []byte) []byte {
	n := min(len(left), len(right))
	n -= n % 2
	out := make([]byte, n*2)
	for i := 0; i+1 < n; i += 2 {
		copy(out[i*2:], left[i:i+2])
		copy(out[i*2+2:], right[i:i+2])
	}
	return out
}
