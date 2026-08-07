package audiobuffer

// Ported from upstream's audio-buffer suite. The tests live inside the package
// because several of them set the last-write timestamps directly, which is how
// upstream simulates a gap without sleeping through it.

import (
	"bytes"
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/clock"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

const testRate = 16000

// capture collects what the callbacks were handed.
type capture struct {
	mu       sync.Mutex
	merged   []byte
	user     []byte
	bot      []byte
	rate     int
	channels int
	started  int
	stopped  int
	flushes  int
}

func (c *capture) tracks() (user, bot []byte) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.user, c.bot
}

// newProc builds a processor ready to record: the start frame has been through
// it, so the sample rate is configured, and recording is on unless the caller
// asked otherwise.
func newProc(t *testing.T, cfg Config, got *capture, start bool) *Processor {
	t.Helper()
	cfg.SampleRate = testRate
	if cfg.NumChannels == 0 {
		cfg.NumChannels = 2
	}
	cfg.OnRecordingStarted = func() {
		got.mu.Lock()
		got.started++
		got.mu.Unlock()
	}
	cfg.OnRecordingStopped = func() {
		got.mu.Lock()
		got.stopped++
		got.mu.Unlock()
	}
	cfg.OnAudioData = func(audio []byte, rate, channels int) {
		got.mu.Lock()
		got.merged = audio
		got.rate, got.channels = rate, channels
		got.flushes++
		got.mu.Unlock()
	}
	cfg.OnTrackAudioData = func(user, bot []byte, rate, channels int) {
		got.mu.Lock()
		got.user, got.bot = user, bot
		got.rate, got.channels = rate, channels
		got.mu.Unlock()
	}

	p := New(cfg)
	// Setup before the start frame, as a pipeline would: the base opens its
	// frame task there and needs the session context.
	if err := p.Setup(context.Background(), processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(func() { _ = p.Cleanup(context.Background()) })
	send(t, p, frames.NewStartFrame())
	if start {
		p.StartRecording()
	}
	return p
}

// send puts one frame through the processor.
func send(t *testing.T, p *Processor, f frames.Frame) {
	t.Helper()
	if err := p.ProcessFrame(context.Background(), f, processor.Downstream); err != nil {
		t.Fatalf("ProcessFrame(%T): %v", f, err)
	}
}

func userAudio(t *testing.T, p *Processor, pcm []byte) {
	t.Helper()
	send(t, p, frames.NewInputAudioRawFrame(pcm, testRate, 1))
}

func botAudio(t *testing.T, p *Processor, pcm []byte) {
	t.Helper()
	send(t, p, frames.NewOutputAudioRawFrame(pcm, testRate, 1))
}

// A flush delivers the two tracks aligned: audio the other side never produced
// is silence of the same length, and the merged stereo carries one on each
// channel.
func TestFlushUserAudioPadsBotTrack(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{BufferSize: 4}, got, true)

	user := []byte{0xe8, 0x03, 0x18, 0xfc}
	userAudio(t, p, user)

	gotUser, gotBot := got.tracks()
	if !bytes.Equal(gotUser, user) {
		t.Fatalf("user track = %x, want %x", gotUser, user)
	}
	if want := make([]byte, len(user)); !bytes.Equal(gotBot, want) {
		t.Fatalf("bot track = %x, want silence of the same length", gotBot)
	}

	got.mu.Lock()
	defer got.mu.Unlock()
	if got.rate != testRate || got.channels != 2 {
		t.Fatalf("delivered at %d Hz on %d channels, want %d on 2", got.rate, got.channels, testRate)
	}
	if len(got.merged) != len(user)*2 {
		t.Fatalf("merged length = %d, want %d for stereo", len(got.merged), len(user)*2)
	}
	// User left, bot right.
	if !bytes.Equal(got.merged[0:2], user[0:2]) || !bytes.Equal(got.merged[2:4], []byte{0, 0}) ||
		!bytes.Equal(got.merged[4:6], user[2:4]) || !bytes.Equal(got.merged[6:8], []byte{0, 0}) {
		t.Fatalf("merged = %x, want the user on the left and silence on the right", got.merged)
	}
	if len(p.userBuf) != 0 || len(p.botBuf) != 0 {
		t.Fatalf("buffers held %d/%d bytes after the flush, want them emptied",
			len(p.userBuf), len(p.botBuf))
	}
}

// The same the other way round.
func TestFlushBotAudioPadsUserTrack(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{BufferSize: 4}, got, true)

	bot := []byte{0xe0, 0xfc, 0x90, 0x01}
	botAudio(t, p, bot)

	gotUser, gotBot := got.tracks()
	if want := make([]byte, len(bot)); !bytes.Equal(gotUser, want) {
		t.Fatalf("user track = %x, want silence of the same length", gotUser)
	}
	if !bytes.Equal(gotBot, bot) {
		t.Fatalf("bot track = %x, want %x", gotBot, bot)
	}

	got.mu.Lock()
	defer got.mu.Unlock()
	if !bytes.Equal(got.merged[0:2], []byte{0, 0}) || !bytes.Equal(got.merged[2:4], bot[0:2]) ||
		!bytes.Equal(got.merged[4:6], []byte{0, 0}) || !bytes.Equal(got.merged[6:8], bot[2:4]) {
		t.Fatalf("merged = %x, want silence on the left and the bot on the right", got.merged)
	}
}

// Silence is never inserted into a track while that side is speaking: it would
// land in the middle of an utterance, which is heard as a crackle. The bot's
// audio has to start at the beginning of its track, with padding only at the
// end where the flush aligns the two.
func TestNoSilenceIntoTheBotTrackWhileTheBotSpeaks(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	bot := []byte{0xaa, 0xbb, 0xcc, 0xdd}
	userAudio(t, p, []byte{1, 2, 3, 4})
	send(t, p, frames.NewBotStartedSpeakingFrame())
	userAudio(t, p, []byte{5, 6, 7, 8})
	botAudio(t, p, bot)
	p.StopRecording()

	_, gotBot := got.tracks()
	if !bytes.Equal(gotBot[:4], bot) {
		t.Fatalf("bot track starts %x, want the bot's audio %x with no silence before it",
			gotBot[:4], bot)
	}
	if want := make([]byte, len(gotBot)-4); !bytes.Equal(gotBot[4:], want) {
		t.Fatalf("bot track tail = %x, want silence only", gotBot[4:])
	}
}

// The same guard on the user's side.
func TestNoSilenceIntoTheUserTrackWhileTheUserSpeaks(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	user := []byte{0xaa, 0xbb, 0xcc, 0xdd}
	botAudio(t, p, []byte{1, 2, 3, 4})
	send(t, p, frames.NewUserStartedSpeakingFrame())
	botAudio(t, p, []byte{5, 6, 7, 8})
	userAudio(t, p, user)
	p.StopRecording()

	gotUser, _ := got.tracks()
	if !bytes.Equal(gotUser[:4], user) {
		t.Fatalf("user track starts %x, want the user's audio %x with no silence before it",
			gotUser[:4], user)
	}
	if want := make([]byte, len(gotUser)-4); !bytes.Equal(gotUser[4:], want) {
		t.Fatalf("user track tail = %x, want silence only", gotUser[4:])
	}
}

// Once the bot stops speaking the guard lifts and its track is synced again, so
// the next user audio finds it padded up to the same position.
func TestSilenceResumesIntoTheBotTrackOnceItStopsSpeaking(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	userAudio(t, p, []byte{1, 2, 3, 4})
	send(t, p, frames.NewBotStartedSpeakingFrame())
	userAudio(t, p, []byte{5, 6, 7, 8})
	send(t, p, frames.NewBotStoppedSpeakingFrame())
	userAudio(t, p, []byte{9, 10, 11, 12})

	p.mu.Lock()
	botLen := len(p.botBuf)
	p.mu.Unlock()
	if botLen != 8 {
		t.Fatalf("bot track holds %d bytes, want 8: the sync resumes once the bot is quiet, "+
			"catching up to where the user was before the third frame", botLen)
	}
	p.StopRecording()
}

// And the same on the user's side.
func TestSilenceResumesIntoTheUserTrackOnceTheyStopSpeaking(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	botAudio(t, p, []byte{1, 2, 3, 4})
	send(t, p, frames.NewUserStartedSpeakingFrame())
	botAudio(t, p, []byte{5, 6, 7, 8})
	send(t, p, frames.NewUserStoppedSpeakingFrame())
	botAudio(t, p, []byte{9, 10, 11, 12})

	p.mu.Lock()
	userLen := len(p.userBuf)
	p.mu.Unlock()
	if userLen != 8 {
		t.Fatalf("user track holds %d bytes, want 8: the sync resumes once the user is quiet",
			userLen)
	}
	p.StopRecording()
}

// ago returns a timestamp d in the past, for a test that needs a gap without
// waiting one out.
func ago(d time.Duration) *time.Time {
	t := time.Now().Add(-d)
	return &t
}

// bytesPerSecond is what one second of the recording weighs: 16-bit mono.
const bytesPerSecond = testRate * 2

// silenceFor is the padding a gap of elapsed calls for, given the incoming
// frame's own duration. It is the rule the processor applies, restated so a
// test asserts against the rule rather than against a number.
func silenceFor(elapsed time.Duration, frameBytes int) int {
	frameDur := time.Duration(float64(frameBytes) / float64(bytesPerSecond) * float64(time.Second))
	gap := elapsed - frameDur
	if gap <= silenceGapThreshold {
		return 0
	}
	n := int(gap.Seconds() * bytesPerSecond)
	return n - n%2
}

// tolerance is how far a measured length may drift from the expected one: the
// tests set a timestamp in the past and let real time run, so a few
// milliseconds of it land in the answer. It stays far below the differences the
// tests discriminate, which are whole seconds.
const tolerance = 100 * testRate * 2 / 1000 // 100 ms

func closeTo(got, want int) bool { return got-want < tolerance && want-got < tolerance }

// The first frame of a recording has no earlier write to measure a gap from, so
// it is buffered as it came.
func TestNoSilenceWithoutAPriorWrite(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	userAudio(t, p, audio)

	p.mu.Lock()
	defer p.mu.Unlock()
	if !bytes.Equal(p.userBuf, audio) {
		t.Fatalf("user track = %x, want %x with no silence before it", p.userBuf, audio)
	}
}

// A gap shorter than the threshold is jitter, not silence.
func TestNoSilenceForAGapBelowTheThreshold(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastUser = ago(100 * time.Millisecond)
	p.mu.Unlock()
	userAudio(t, p, audio)

	p.mu.Lock()
	defer p.mu.Unlock()
	if !bytes.Equal(p.userBuf, audio) {
		t.Fatalf("user track = %x, want %x: a 100ms gap is below the threshold", p.userBuf, audio)
	}
}

// A gap the microphone was muted through is filled with silence of its own
// length, so two utterances seconds apart are not recorded back to back.
func TestSilenceProportionalToTheMuteGap(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastUser = ago(time.Second)
	p.mu.Unlock()
	userAudio(t, p, audio)

	want := silenceFor(time.Second, len(audio))
	p.mu.Lock()
	defer p.mu.Unlock()
	if !closeTo(len(p.userBuf), want+len(audio)) {
		t.Fatalf("user track holds %d bytes, want about %d", len(p.userBuf), want+len(audio))
	}
	silence := len(p.userBuf) - len(audio)
	if !bytes.Equal(p.userBuf[:silence], make([]byte, silence)) {
		t.Fatal("the gap was not filled with silence")
	}
	if !bytes.Equal(p.userBuf[silence:], audio) {
		t.Fatalf("the new audio landed as %x, want %x after the silence", p.userBuf[silence:], audio)
	}
}

// The original report: two utterances with a muted stretch between them must
// not sound like one.
func TestTwoUtterancesSeparatedByAMuteKeepTheirGap(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	utterance := []byte{0x11, 0x22, 0x33, 0x44}
	p.mu.Lock()
	p.userBuf = append(p.userBuf, utterance...)
	p.lastUser = ago(time.Second)
	p.mu.Unlock()
	userAudio(t, p, utterance)

	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.userBuf) <= len(utterance)*2 {
		t.Fatalf("user track holds %d bytes, want more than the two utterances back to back",
			len(p.userBuf))
	}
	if !bytes.Equal(p.userBuf[:len(utterance)], utterance) {
		t.Fatal("the first utterance is no longer at the start of the track")
	}
	tail := p.userBuf[len(p.userBuf)-len(utterance):]
	if !bytes.Equal(tail, utterance) {
		t.Fatalf("the track ends with %x, want the second utterance %x", tail, utterance)
	}
	between := p.userBuf[len(utterance) : len(p.userBuf)-len(utterance)]
	if !bytes.Equal(between, make([]byte, len(between))) {
		t.Fatal("what lies between the two utterances is not silence")
	}
}

// Bot audio that syncs the user track also advances the user's timestamp, so
// the silence that sync already wrote is not counted a second time when the
// user speaks again.
func TestBotAudioDuringAMuteAdvancesTheUserTimestamp(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastUser = ago(2 * time.Second)
	p.mu.Unlock()

	// The bot speaks a second into the mute, which syncs and re-bases the user.
	botAudio(t, p, audio)
	p.mu.Lock()
	p.lastUser = ago(time.Second)
	p.mu.Unlock()

	p.mu.Lock()
	before := len(p.userBuf)
	p.mu.Unlock()

	userAudio(t, p, audio)

	p.mu.Lock()
	defer p.mu.Unlock()
	added := len(p.userBuf) - before - len(audio)
	oneSecond := silenceFor(time.Second, len(audio))
	twoSeconds := silenceFor(2*time.Second, len(audio))
	if !closeTo(added, oneSecond) {
		t.Fatalf("the gap was filled with %d bytes, want about %d: it is measured from the "+
			"sync the bot's audio made, not from the last real user audio", added, oneSecond)
	}
	if closeTo(added, twoSeconds) {
		t.Fatalf("the gap was filled with %d bytes, which double-counts the silence the sync "+
			"had already written", added)
	}
}

// The bot track is synced to where the user track stands once the gap has been
// filled and before the new audio is added, so both share the same point in
// time.
func TestBotTrackSyncedToTheUserPositionAfterAGapFill(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastUser = ago(time.Second)
	p.mu.Unlock()
	userAudio(t, p, audio)

	want := silenceFor(time.Second, len(audio))
	p.mu.Lock()
	defer p.mu.Unlock()
	if !closeTo(len(p.botBuf), want) {
		t.Fatalf("bot track holds %d bytes, want about %d: the silence the user's gap wrote, "+
			"without the audio that followed it", len(p.botBuf), want)
	}
	if !bytes.Equal(p.botBuf, make([]byte, len(p.botBuf))) {
		t.Fatal("the bot track holds something other than silence")
	}
}

// Stopping clears the timestamps, so the first audio of the next recording has
// nothing to measure a gap from.
func TestStoppingClearsTheTimestamps(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	p.mu.Lock()
	p.lastUser = ago(time.Second)
	p.lastBot = ago(time.Second)
	p.mu.Unlock()

	p.StopRecording()

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.lastUser != nil || p.lastBot != nil {
		t.Fatal("the timestamps survived the stop, so the next recording would open with silence")
	}
}

// A flush re-bases the timestamps to the moment it happened, so the next audio
// measures its gap from there rather than from a point before the flush.
func TestAFlushRebasesTheTimestamps(t *testing.T) {
	got := &capture{}
	audio := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	p := newProc(t, Config{BufferSize: len(audio)}, got, true)

	p.mu.Lock()
	p.lastUser = ago(5 * time.Second)
	p.mu.Unlock()
	userAudio(t, p, audio)

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.lastUser == nil {
		t.Fatal("the flush left no timestamp, so the next audio would measure no gap at all")
	}
	if since := time.Since(*p.lastUser); since > time.Second {
		t.Fatalf("the timestamp is %v old, want it re-based to the flush", since)
	}
}

// hasAudio reports whether either track holds anything, upstream's has_audio.
func (p *Processor) hasAudio() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.userBuf) > 0 || len(p.botBuf) > 0
}

// A start frame turns recording on, so what follows it is recorded. It is how
// any processor upstream starts a recording from inside the frame flow.
func TestStartFrameEnablesRecording(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, false)
	if p.Recording() {
		t.Fatal("recording before anything asked for it")
	}

	send(t, p, frames.NewAudioBufferStartRecordingFrame())
	if !p.Recording() {
		t.Fatal("the start frame did not begin the recording")
	}

	audio := []byte{0xe8, 0x03, 0x18, 0xfc}
	userAudio(t, p, audio)
	p.StopRecording()

	gotUser, _ := got.tracks()
	if !bytes.Equal(gotUser, audio) {
		t.Fatalf("user track = %x, want %x", gotUser, audio)
	}
}

// Audio arriving before the recording starts is not kept.
func TestAudioBeforeTheStartFrameIsIgnored(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, false)

	userAudio(t, p, []byte{1, 2, 3, 4})

	if p.hasAudio() {
		t.Fatal("audio was buffered before the recording began")
	}
}

// A stop frame delivers what was recorded and turns recording off.
func TestStopFrameDeliversAndDisables(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{0xe8, 0x03, 0x18, 0xfc}
	userAudio(t, p, audio)
	send(t, p, frames.NewAudioBufferStopRecordingFrame())

	gotUser, _ := got.tracks()
	if !bytes.Equal(gotUser, audio) {
		t.Fatalf("user track = %x, want %x delivered by the stop frame", gotUser, audio)
	}
	if p.Recording() {
		t.Fatal("still recording after the stop frame")
	}
	if p.hasAudio() {
		t.Fatal("the buffers still hold audio after the stop frame")
	}
}

// The control frames travel on, so anything else watching for them reacts too.
func TestRecordingControlFramesTravelOn(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, false)

	var mu sync.Mutex
	var starts, stops int
	next := &collector{onFrame: func(f frames.Frame) {
		mu.Lock()
		defer mu.Unlock()
		switch f.(type) {
		case *frames.AudioBufferStartRecordingFrame:
			starts++
		case *frames.AudioBufferStopRecordingFrame:
			stops++
		}
	}}
	next.Base = processor.New("Collector", next)
	if err := next.Setup(context.Background(), processor.Setup{Clock: clock.NewSystem()}); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	t.Cleanup(func() { _ = next.Cleanup(context.Background()) })
	// Started on its own, since it is linked after the recorder saw the pipeline
	// start, and a processor only takes frames once it has.
	if err := next.ProcessFrame(context.Background(), frames.NewStartFrame(), processor.Downstream); err != nil {
		t.Fatalf("start the collector: %v", err)
	}
	p.Link(next)

	send(t, p, frames.NewAudioBufferStartRecordingFrame())
	send(t, p, frames.NewAudioBufferStopRecordingFrame())

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		done := starts == 1 && stops == 1
		mu.Unlock()
		if done {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	mu.Lock()
	defer mu.Unlock()
	t.Fatalf("%d start and %d stop frames reached the next processor, want one of each",
		starts, stops)
}

// collector records what reaches it.
type collector struct {
	*processor.Base
	onFrame func(frames.Frame)
}

func (c *collector) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := c.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	c.onFrame(f)
	return nil
}

// With auto-start the recording is already running by the time the pipeline is.
func TestAutoStartBeginsOnTheStartFrame(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{AutoStart: true}, got, false)
	if !p.Recording() {
		t.Fatal("auto-start did not begin the recording")
	}

	audio := []byte{0xe8, 0x03, 0x18, 0xfc}
	userAudio(t, p, audio)
	p.StopRecording()

	gotUser, _ := got.tracks()
	if !bytes.Equal(gotUser, audio) {
		t.Fatalf("user track = %x, want %x", gotUser, audio)
	}
}

// Without it, the pipeline starting records nothing.
func TestAutoStartIsOffByDefault(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, false)
	if p.Recording() {
		t.Fatal("the recording began without being asked to")
	}
	got.mu.Lock()
	defer got.mu.Unlock()
	if got.started != 0 {
		t.Fatalf("the started callback fired %d times, want none", got.started)
	}
}

// Auto-start announces itself like any other start.
func TestAutoStartAnnouncesItself(t *testing.T) {
	got := &capture{}
	newProc(t, Config{AutoStart: true}, got, false)
	got.mu.Lock()
	defer got.mu.Unlock()
	if got.started != 1 {
		t.Fatalf("the started callback fired %d times, want once", got.started)
	}
}

// Starting a recording that is already running changes nothing: it does not
// announce a second start, and it does not throw away what has been recorded.
func TestARedundantStartChangesNothing(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	userAudio(t, p, []byte{0xe8, 0x03, 0x18, 0xfc})
	if !p.hasAudio() {
		t.Fatal("the audio was not recorded")
	}

	p.StartRecording()

	if !p.hasAudio() {
		t.Fatal("starting again threw away what had been recorded")
	}
	got.mu.Lock()
	defer got.mu.Unlock()
	if got.started != 1 {
		t.Fatalf("the started callback fired %d times, want once", got.started)
	}
}

// Stopping delivers the recording and only then announces the stop, so a
// consumer acting on the announcement finds the audio already handed over.
func TestStoppingDeliversBeforeAnnouncing(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	userAudio(t, p, []byte{0xe8, 0x03, 0x18, 0xfc})
	p.StopRecording()

	got.mu.Lock()
	defer got.mu.Unlock()
	if got.flushes != 1 {
		t.Fatalf("the audio was delivered %d times, want once", got.flushes)
	}
	if got.stopped != 1 {
		t.Fatalf("the stopped callback fired %d times, want once", got.stopped)
	}
	if p.Recording() {
		t.Fatal("still recording after the stop")
	}
}

// A pipeline ending without a recording having started announces nothing.
func TestEndingWithoutARecordingAnnouncesNothing(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, false)

	send(t, p, frames.NewEndFrame())

	got.mu.Lock()
	defer got.mu.Unlock()
	if got.stopped != 0 {
		t.Fatalf("the stopped callback fired %d times with no recording to stop", got.stopped)
	}
}

// The bot's track answers to the same rules as the user's. Its gaps are idle
// stretches between utterances, a slow tool call for instance, rather than a
// muted microphone, and they are filled the same way.

func TestNoSilenceIntoTheBotTrackWithoutAPriorWrite(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	botAudio(t, p, audio)

	p.mu.Lock()
	defer p.mu.Unlock()
	if !bytes.Equal(p.botBuf, audio) {
		t.Fatalf("bot track = %x, want %x with no silence before it", p.botBuf, audio)
	}
}

func TestNoSilenceIntoTheBotTrackForAGapBelowTheThreshold(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastBot = ago(100 * time.Millisecond)
	p.mu.Unlock()
	botAudio(t, p, audio)

	p.mu.Lock()
	defer p.mu.Unlock()
	if !bytes.Equal(p.botBuf, audio) {
		t.Fatalf("bot track = %x, want %x: a 100ms gap is below the threshold", p.botBuf, audio)
	}
}

func TestSilenceProportionalToTheIdleGap(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastBot = ago(time.Second)
	p.mu.Unlock()
	botAudio(t, p, audio)

	want := silenceFor(time.Second, len(audio))
	p.mu.Lock()
	defer p.mu.Unlock()
	if !closeTo(len(p.botBuf), want+len(audio)) {
		t.Fatalf("bot track holds %d bytes, want about %d", len(p.botBuf), want+len(audio))
	}
	silence := len(p.botBuf) - len(audio)
	if !bytes.Equal(p.botBuf[:silence], make([]byte, silence)) {
		t.Fatal("the idle stretch was not filled with silence")
	}
}

func TestTwoBotUtterancesSeparatedByAPauseKeepTheirGap(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	utterance := []byte{0x11, 0x22, 0x33, 0x44}
	p.mu.Lock()
	p.botBuf = append(p.botBuf, utterance...)
	p.lastBot = ago(time.Second)
	p.mu.Unlock()
	botAudio(t, p, utterance)

	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.botBuf) <= len(utterance)*2 {
		t.Fatalf("bot track holds %d bytes, want more than the two utterances back to back",
			len(p.botBuf))
	}
	between := p.botBuf[len(utterance) : len(p.botBuf)-len(utterance)]
	if !bytes.Equal(between, make([]byte, len(between))) {
		t.Fatal("what lies between the two utterances is not silence")
	}
}

func TestUserAudioDuringAPauseAdvancesTheBotTimestamp(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastBot = ago(2 * time.Second)
	p.mu.Unlock()

	// The user speaks a second into the pause, which syncs and re-bases the bot.
	userAudio(t, p, audio)
	p.mu.Lock()
	p.lastBot = ago(time.Second)
	before := len(p.botBuf)
	p.mu.Unlock()

	botAudio(t, p, audio)

	p.mu.Lock()
	defer p.mu.Unlock()
	added := len(p.botBuf) - before - len(audio)
	oneSecond := silenceFor(time.Second, len(audio))
	twoSeconds := silenceFor(2*time.Second, len(audio))
	if !closeTo(added, oneSecond) {
		t.Fatalf("the pause was filled with %d bytes, want about %d: it is measured from the "+
			"sync the user's audio made", added, oneSecond)
	}
	if closeTo(added, twoSeconds) {
		t.Fatalf("the pause was filled with %d bytes, which double-counts the silence the sync "+
			"had already written", added)
	}
}

func TestUserTrackSyncedToTheBotPositionAfterAGapFill(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{1, 2, 3, 4}
	p.mu.Lock()
	p.lastBot = ago(time.Second)
	p.mu.Unlock()
	botAudio(t, p, audio)

	want := silenceFor(time.Second, len(audio))
	p.mu.Lock()
	defer p.mu.Unlock()
	if !closeTo(len(p.userBuf), want) {
		t.Fatalf("user track holds %d bytes, want about %d: the silence the bot's gap wrote, "+
			"without the audio that followed it", len(p.userBuf), want)
	}
}

// Audio a synthesizer produced is the bot's audio: it is output audio, of a
// kind that names the synthesis it came from, and it belongs on the bot's track
// like any other.
func TestSynthesizedAudioLandsOnTheBotTrack(t *testing.T) {
	got := &capture{}
	p := newProc(t, Config{}, got, true)

	audio := []byte{0xaa, 0xbb, 0xcc, 0xdd}
	send(t, p, frames.NewTTSAudioRawFrame(audio, testRate, 1))
	p.StopRecording()

	gotUser, gotBot := got.tracks()
	if !bytes.Equal(gotBot, audio) {
		t.Fatalf("bot track = %x, want the synthesized audio %x", gotBot, audio)
	}
	if !bytes.Equal(gotUser, make([]byte, len(audio))) {
		t.Fatalf("user track = %x, want silence of the same length", gotUser)
	}
}
