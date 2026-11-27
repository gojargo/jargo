package eval

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/stt"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
)

// Failures the harness's services and transport report in the tests below.
var (
	errBadVoice  = errors.New("bad voice")
	errConnReset = errors.New("connection reset")
	errBadFrame  = errors.New("bad frame")
)

func init() {
	RegisterFactory("evaltest.streaming_stt", func(map[string]any) (any, error) {
		return stt.NewStream("StreamingSTT", nil, 16000), nil
	})
}

// audioClient is the client of a session of scenario, or of an audio-mode
// scenario with no turns.
func audioClient(t *testing.T, scenario *ScriptScenario) *Client {
	t.Helper()
	if scenario == nil {
		scenario = &ScriptScenario{Name: "t"}
		scenario.BotAudio = true
	}
	return clientFor(t, scenario, "ws://localhost:0", SessionParams{})
}

// injected collects the user-turn frames a client hands its sink.
type injected struct {
	mu     sync.Mutex
	frames []frames.Frame
}

func (i *injected) inject(_ context.Context, f frames.Frame) error {
	i.mu.Lock()
	defer i.mu.Unlock()
	i.frames = append(i.frames, f)
	return nil
}

// captureInjected stands in for the client's sink, collecting the user-turn
// frames it is handed.
func captureInjected(c *Client) *injected {
	i := &injected{}
	c.injector = i
	return i
}

// sineTone is seconds of a 440 Hz tone at sampleRate, as 16-bit samples.
func sineTone(sampleRate int, seconds float64) []int16 {
	n := int(float64(sampleRate) * seconds)
	out := make([]int16, n)
	for i := range out {
		out[i] = int16(0.3 * math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)) * 32767)
	}
	return out
}

// pcmOf is samples as 16-bit little-endian PCM.
func pcmOf(samples []int16) []byte {
	out := make([]byte, 2*len(samples))
	for i, v := range samples {
		binary.LittleEndian.PutUint16(out[2*i:], uint16(v))
	}
	return out
}

// writeTone writes a tone WAV file of channels at sampleRate and returns its
// mono samples.
func writeTone(t *testing.T, path string, sampleRate int, seconds float64, channels int) []int16 {
	t.Helper()
	samples := sineTone(sampleRate, seconds)
	interleaved := make([]int16, 0, len(samples)*channels)
	for _, v := range samples {
		for range channels {
			interleaved = append(interleaved, v)
		}
	}
	if err := os.WriteFile(path, audio.PCMToWAV(pcmOf(interleaved), sampleRate, channels), 0o600); err != nil {
		t.Fatal(err)
	}
	return samples
}

// User audio is spoken by pushing a TTSSpeakFrame into the pipeline at the sink.
func TestAudioSender(t *testing.T) {
	c := audioClient(t, nil)
	got := captureInjected(c)
	if err := c.Say(t.Context(), "hello world"); err != nil {
		t.Fatal(err)
	}
	if len(got.frames) != 1 {
		t.Fatalf("injected %v, want one frame", got.frames)
	}
	if f, ok := got.frames[0].(*frames.TTSSpeakFrame); !ok || f.Text != "hello world" {
		t.Fatalf("injected %v, want the TTSSpeakFrame", got.frames[0])
	}
}

// wantUtterance asserts the frames are one TTS-bracketed utterance and returns
// its audio frame.
func wantUtterance(t *testing.T, got []frames.Frame) *frames.TTSAudioRawFrame {
	t.Helper()
	if len(got) != 3 {
		t.Fatalf("injected %v, want started, audio, stopped", got)
	}
	_, started := got[0].(*frames.TTSStartedFrame)
	af, isAudio := got[1].(*frames.TTSAudioRawFrame)
	_, stopped := got[2].(*frames.TTSStoppedFrame)
	if !started || !isAudio || !stopped {
		t.Fatalf("injected %v, want started, audio, stopped", got)
	}
	return af
}

// textVoice is a user voice whose every character is one non-silent sample.
type textVoice struct{}

func (textVoice) SampleRate() int { return 16000 }

func (textVoice) RunTTS(_ context.Context, text, _ string, yield func(frames.Frame) error) error {
	return yield(frames.NewTTSAudioRawFrame(bytes.Repeat([]byte{1, 0}, len(text)), 16000, 1))
}

// A turn's audio: file is played to the bot instead of being synthesized.
func TestAudioFileSender(t *testing.T) {
	t.Run("the file is spoken as one TTS utterance at its own rate", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "hi.wav")
		samples := writeTone(t, path, 16000, 2, 1)
		c := audioClient(t, nil)
		got := captureInjected(c)
		if err := c.Play(t.Context(), path); err != nil {
			t.Fatal(err)
		}
		// Bracketed like the user TTS's output, so the output flushes the
		// utterance's final partial chunk on the stop frame.
		af := wantUtterance(t, got.frames)
		if af.SampleRate != 16000 || af.NumChannels != 1 || !bytes.Equal(af.Audio, pcmOf(samples)) {
			t.Fatalf("audio at %d Hz, %d channels, %d bytes; want the file", af.SampleRate, af.NumChannels, len(af.Audio))
		}
	})
	t.Run("pauses are silence between the spoken parts", func(t *testing.T) {
		c := audioClient(t, nil)
		noCache := false
		voice, err := NewCachingTTSService(textVoice{}, CachingTTSConfig{UseCache: &noCache, SampleRate: 16000})
		if err != nil {
			t.Fatal(err)
		}
		c.userTTS = voice
		got := captureInjected(c)
		if err := c.SayWithPauses(t.Context(), []TurnPart{{Text: "Hi"}, {PauseMS: 250}, {Text: "there"}}); err != nil {
			t.Fatal(err)
		}
		// Spoken as one utterance, the 250 ms pause as 4000 silent samples.
		af := wantUtterance(t, got.frames)
		want := slices.Concat(bytes.Repeat([]byte{1, 0}, 2), make([]byte, 2*4000), bytes.Repeat([]byte{1, 0}, 5))
		if !bytes.Equal(af.Audio, want) || af.SampleRate != 16000 {
			t.Fatalf("audio of %d bytes at %d Hz, want %d at 16000", len(af.Audio), af.SampleRate, len(want))
		}
	})
	t.Run("a non-native rate is preserved", func(t *testing.T) {
		// The frame carries the file's rate (the output resamples it), so a
		// recording need not match the bot.
		path := filepath.Join(t.TempDir(), "hi.wav")
		writeTone(t, path, 44100, 0.5, 1)
		c := audioClient(t, nil)
		got := captureInjected(c)
		if err := c.Play(t.Context(), path); err != nil {
			t.Fatal(err)
		}
		if af := wantUtterance(t, got.frames); af.SampleRate != 44100 {
			t.Fatalf("rate = %d, want the file's", af.SampleRate)
		}
	})
	t.Run("stereo is downmixed to mono", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "s.wav")
		samples := writeTone(t, path, 16000, 0.5, 2)
		pcm, rate, err := LoadUserAudio(path)
		if err != nil {
			t.Fatal(err)
		}
		if rate != 16000 || !bytes.Equal(pcm, pcmOf(samples)) {
			t.Fatalf("rate %d, %d bytes; want the mono tone", rate, len(pcm))
		}
	})
	t.Run("an unreadable file reports the path", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "x.txt")
		if err := os.WriteFile(path, []byte("not audio"), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, _, err := LoadUserAudio(path); err == nil || !strings.Contains(err.Error(), "x.txt") {
			t.Fatalf("err = %v, want one naming the file", err)
		}
	})
}

// A turn that plays a file enables the user audio output.
func TestAudioFileEnablesUserAudio(t *testing.T) {
	t.Run("an audio turn enables the user audio output", func(t *testing.T) {
		// The harness streams user audio only when audio output is enabled, so a
		// turn that plays a file must enable it even though the scenario names
		// no TTS.
		scenario := &ScriptScenario{Name: "t", Turns: []ScriptTurn{{User: "hi", Audio: "hi.wav"}}}
		scenario.BotAudio, scenario.UserAudio = true, true
		if !audioClient(t, scenario).SendsUserAudio() {
			t.Fatal("the user audio output is off")
		}
	})
	t.Run("text turns do not", func(t *testing.T) {
		scenario := &ScriptScenario{Name: "t", Turns: []ScriptTurn{{User: "hi"}}}
		scenario.BotAudio = true
		if audioClient(t, scenario).SendsUserAudio() {
			t.Fatal("the user audio output is on")
		}
	})
}

// countingTurns counts the resets of the bot's turn.
type countingTurns struct{ resets int }

func (c *countingTurns) Reset() { c.resets++ }

// A send over a speaking bot drops what its open turn holds of the bot's
// earlier speech.
func TestInterruptingSendDropsOpenBotTurn(t *testing.T) {
	t.Run("a send over a speaking bot resets its turn", func(t *testing.T) {
		c := audioClient(t, nil)
		captureInjected(c)
		turns := &countingTurns{}
		c.botTurns = turns
		c.stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		if err := c.Say(t.Context(), "Actually, tell me a joke instead."); err != nil {
			t.Fatal(err)
		}
		if turns.resets != 1 {
			t.Fatalf("resets = %d, want 1", turns.resets)
		}
	})
	t.Run("a send to a quiet bot leaves its turn alone", func(t *testing.T) {
		c := audioClient(t, nil)
		captureInjected(c)
		turns := &countingTurns{}
		c.botTurns = turns
		if err := c.Say(t.Context(), "What is the capital of Germany?"); err != nil {
			t.Fatal(err)
		}
		if turns.resets != 0 {
			t.Fatalf("resets = %d, want 0", turns.resets)
		}
	})
}

// segmentTranscript is a transcription of a segment that began at startedAt.
func segmentTranscript(text string, startedAt time.Time) *frames.TranscriptionFrame {
	tf := frames.NewTranscriptionFrame(text, "", "")
	tf.Metadata()[segmentStartedAtKey] = startedAt
	return tf
}

// runGate passes fs through gate in dir and returns what it pushed on.
func runGate(t *testing.T, gate *botSpeechGate, dir processor.Direction, fs ...frames.Frame) []frames.Frame {
	t.Helper()
	ctx := t.Context()
	if err := gate.Setup(ctx, processor.Setup{}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = gate.Cleanup(context.Background()) })
	next, prev := newCollector(), newCollector()
	gate.Link(next)
	prev.Link(gate)
	for _, f := range fs {
		if err := gate.ProcessFrame(ctx, f, dir); err != nil {
			t.Fatal(err)
		}
	}
	if dir == processor.Upstream {
		return prev.seen()
	}
	return next.seen()
}

// transcripts is the text of each transcription among fs.
func transcripts(fs []frames.Frame) []string {
	var out []string
	for _, f := range fs {
		if tf, ok := f.(*frames.TranscriptionFrame); ok {
			out = append(out, tf.Text)
		}
	}
	return out
}

// After a send that talked over the bot, its speech from before its reply is
// dropped by the time its audio began.
func TestBotSpeechGate(t *testing.T) {
	t.Run("a segment from before an interrupting send is dropped however late it lands", func(t *testing.T) {
		stream := newStream(true)
		gate := newBotSpeechGate(stream, NewTrace(), &segmentStarts{})
		before := time.Now()
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil)) // the send talks over the bot
		stream.InputSent()
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStoppedSpeaking, nil))
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil)) // the reply
		after := time.Now()
		down := runGate(t, gate, processor.Downstream,
			segmentTranscript("He had seen sunken,", before),
			segmentTranscript("Why don't scientists trust atoms?", after))
		if got := transcripts(down); !slices.Equal(got, []string{"Why don't scientists trust atoms?"}) {
			t.Fatalf("passed %q", got)
		}
	})
	t.Run("a segment the bot began after the send but before its reply is dropped", func(t *testing.T) {
		// The bot goes on speaking until it notices the user talking over it.
		stream := newStream(true)
		gate := newBotSpeechGate(stream, NewTrace(), &segmentStarts{})
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		stream.InputSent()
		stillSpeaking := time.Now()
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStoppedSpeaking, nil))
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil)) // the reply
		reply := time.Now()
		down := runGate(t, gate, processor.Downstream,
			segmentTranscript("She swam past", stillSpeaking),
			segmentTranscript("Why don't oysters donate to charity?", reply))
		if got := transcripts(down); !slices.Equal(got, []string{"Why don't oysters donate to charity?"}) {
			t.Fatalf("passed %q", got)
		}
	})
	t.Run("a segment from before an observing turn is dropped", func(t *testing.T) {
		// The previous reply's last segment is transcribed after the observing
		// turn began; what the bot says once it speaks again gets through.
		stream := newStream(true)
		gate := newBotSpeechGate(stream, NewTrace(), &segmentStarts{})
		before := time.Now()
		stream.ObservingBoundary()
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil)) // a presence check
		after := time.Now()
		down := runGate(t, gate, processor.Downstream,
			segmentTranscript("Do you have cities in mind?", before),
			segmentTranscript("Are you still there?", after))
		if got := transcripts(down); !slices.Equal(got, []string{"Are you still there?"}) {
			t.Fatalf("passed %q", got)
		}
	})
	t.Run("an ordinary send drops nothing", func(t *testing.T) {
		// The bot was quiet at the send, which waited for its speech to be
		// transcribed; a transcript of earlier audio arriving anyway is kept.
		stream := newStream(true)
		gate := newBotSpeechGate(stream, NewTrace(), &segmentStarts{})
		before := time.Now()
		stream.InputSent()
		down := runGate(t, gate, processor.Downstream, segmentTranscript("earlier audio", before))
		if got := transcripts(down); !slices.Equal(got, []string{"earlier audio"}) {
			t.Fatalf("passed %q", got)
		}
	})
	t.Run("a plain transcript passes", func(t *testing.T) {
		stream := newStream(true)
		stream.RTVIMessageToEvent(rawMessage(t, rtvi.TypeBotStartedSpeaking, nil))
		stream.InputSent()
		gate := newBotSpeechGate(stream, NewTrace(), &segmentStarts{})
		down := runGate(t, gate, processor.Downstream, frames.NewTranscriptionFrame("untagged", "", ""))
		if got := transcripts(down); !slices.Equal(got, []string{"untagged"}) {
			t.Fatalf("passed %q", got)
		}
	})
	t.Run("the gate records when each segment began as it ends", func(t *testing.T) {
		starts := &segmentStarts{}
		stream := newStream(true)
		gate := newBotSpeechGate(stream, NewTrace(), starts)
		t0 := time.Now()
		runGate(t, gate, processor.Upstream,
			frames.NewVADUserStartedSpeakingFrame(0, time.Now()), frames.NewVADUserStoppedSpeakingFrame(0, time.Now()))
		if starts.len() != 1 {
			t.Fatalf("recorded %d starts, want 1", starts.len())
		}
		if at, _ := starts.pop(); at.Before(t0) {
			t.Fatalf("start %v is before the segment", at)
		}
		// The ended segment awaits transcription, so the bot's speech is not
		// yet fully accounted for.
		if stream.WaitBotTranscribed(t.Context(), 10*time.Millisecond) {
			t.Fatal("the bot's speech counts as transcribed")
		}
	})
}

// fixedRun is a segment's run that yields fs.
func fixedRun(fs ...frames.Frame) stt.RunSTTFunc {
	return func(context.Context, []byte, int) iter.Seq[frames.Frame] { return slices.Values(fs) }
}

// collectRun is everything one run of service yields for audio.
func collectRun(t *testing.T, service *stt.SegmentService, audio string) []frames.Frame {
	t.Helper()
	return slices.Collect(service.RunSTT()(t.Context(), []byte(audio), 16000))
}

// Each transcription run takes the start of the segment it transcribes, yield
// or not.
func TestTagBotSegments(t *testing.T) {
	t.Run("runs pair with starts in order even when one yields nothing", func(t *testing.T) {
		service := stt.NewSegment("t", fakeTranscriber{}, 16000)
		service.SetRunSTT(func(_ context.Context, audio []byte, _ int) iter.Seq[frames.Frame] {
			return func(yield func(frames.Frame) bool) {
				if string(audio) == "silence" {
					return
				}
				yield(frames.NewTranscriptionFrame(string(audio), "", ""))
			}
		})
		base := time.Now()
		starts := &segmentStarts{}
		for i := range 3 {
			starts.push(base.Add(time.Duration(i+1) * time.Second))
		}
		tagBotSegments(service, starts)

		first, none, third := collectRun(t, service, "first"), collectRun(t, service, "silence"),
			collectRun(t, service, "third")
		check := func(run []frames.Frame, text string, startedAt time.Time) {
			t.Helper()
			tf, ok := run[0].(*frames.TranscriptionFrame)
			if !ok || tf.Text != text {
				t.Fatalf("run = %v, want %q", run, text)
			}
			if at, _ := segmentStartedAt(tf); !at.Equal(startedAt) {
				t.Fatalf("%q started at %v, want %v", text, at, startedAt)
			}
		}
		check(first, "first", base.Add(time.Second))
		check(third, "third", base.Add(3*time.Second))
		if starts.len() != 0 {
			t.Fatalf("%d starts left", starts.len())
		}
		// Every run, the empty one included, ends by closing its segment.
		for _, run := range [][]frames.Frame{first, none, third} {
			if _, ok := run[len(run)-1].(*botSegmentTranscribedFrame); !ok {
				t.Fatalf("run = %v, want it closed", run)
			}
		}
	})
	t.Run("the sink reports a closed segment as transcribed", func(t *testing.T) {
		stream := newStream(true)
		stream.BotSegmentEnded()
		sink := newBotFrameSink(stream, nil, false)
		if err := sink.Setup(t.Context(), processor.Setup{}); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = sink.Cleanup(context.Background()) })
		next := newCollector()
		sink.Link(next)
		if err := sink.ProcessFrame(t.Context(), newBotSegmentTranscribedFrame(), processor.Downstream); err != nil {
			t.Fatal(err)
		}
		if !stream.WaitBotTranscribed(t.Context(), 10*time.Millisecond) {
			t.Fatal("the bot's speech does not count as transcribed")
		}
		if got := next.seen(); len(got) != 0 {
			t.Fatalf("pushed %v", got)
		}
	})
}

// The bot's speech is transcribed by a segmented STT or not at all.
func TestBotSTTMustBeSegmented(t *testing.T) {
	t.Run("a segmented STT or none is accepted", func(t *testing.T) {
		scenario := &ScriptScenario{Name: "t"}
		for _, botSTT := range []*stt.SegmentService{stt.NewSegment("t", fakeTranscriber{}, 16000), nil} {
			if _, err := NewScriptSession(scenario, "ws://localhost:0", SessionParams{}, nil, nil, botSTT); err != nil {
				t.Fatal(err)
			}
		}
	})
	t.Run("a transcription factory returning a streaming STT is refused", func(t *testing.T) {
		scenario := oneTurn("hi", Expectation{Event: EventResponse})
		scenario.BotAudio = true
		scenario.Transcriber = &TranscriptionConfig{Factory: "evaltest.streaming_stt"}
		_, err := ScriptSessionFromScenario(scenario, "ws://localhost:0", SessionParams{}, nil, nil, nil)
		if !errors.Is(err, ErrFactory) || !strings.Contains(err.Error(), "StreamService") {
			t.Fatalf("err = %v, want the streaming STT refused by name", err)
		}
	})
}

// runSegment drives one segment of speech through service, as the pipeline
// does, and returns what it pushes until a run closes.
func runSegment(t *testing.T, service *stt.SegmentService) []frames.Frame {
	t.Helper()
	ctx := t.Context()
	if err := service.Setup(ctx, processor.Setup{AudioInSampleRate: 16000}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = service.Cleanup(context.Background()) })
	next := newCollector()
	service.Link(next)
	for _, f := range []frames.Frame{
		frames.NewStartFrame(),
		frames.NewVADUserStartedSpeakingFrame(0, time.Now()),
		frames.NewInputAudioRawFrame(make([]byte, 3200), 16000, 1),
		frames.NewVADUserStoppedSpeakingFrame(0, time.Now()),
	} {
		if err := service.ProcessFrame(ctx, f, processor.Downstream); err != nil {
			t.Fatal(err)
		}
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		seen := next.seen()
		if len(seen) > 0 {
			if _, ok := seen[len(seen)-1].(*botSegmentTranscribedFrame); ok {
				return seen
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("the segment's run never closed")
	return nil
}

// A tagged segment's speaker runs keep their speakers, so only the last is
// finalized.
func TestTagBotSegmentsKeepsSpeakers(t *testing.T) {
	startedAt := time.Now().Add(-time.Second)
	t.Run("two speaker runs keep their speakers and only the last is finalized", func(t *testing.T) {
		service := stt.NewSegment("t", fakeTranscriber{}, 16000)
		hi := frames.NewTranscriptionFrame("Hi there.", "", "")
		hi.Speaker = &frames.TranscriptionSpeaker{Label: "0", Number: 1}
		hello := frames.NewTranscriptionFrame("Hello.", "", "")
		hello.Speaker = &frames.TranscriptionSpeaker{Label: "1", Number: 2}
		service.SetRunSTT(fixedRun(hi, hello))
		starts := &segmentStarts{}
		starts.push(startedAt)
		tagBotSegments(service, starts)

		var runs []string
		for _, f := range runSegment(t, service) {
			if tf, ok := f.(*frames.TranscriptionFrame); ok {
				runs = append(runs, fmt.Sprintf("%s|%s|%t", tf.Text, tf.Speaker.Label, tf.Finalized))
			}
		}
		if want := []string{"Hi there.|0|false", "Hello.|1|true"}; !slices.Equal(runs, want) {
			t.Fatalf("runs = %v, want %v", runs, want)
		}
	})
	t.Run("a tagged transcript keeps every field it was built with", func(t *testing.T) {
		original := frames.NewTranscriptionFrame("Hi there.", "bot", "2026-11-14T00:00:00Z")
		original.Language = "en"
		original.Result = map[string]any{"raw": true}
		original.Finalized = true
		original.Speaker = &frames.TranscriptionSpeaker{Label: "0", Number: 1}
		want := *original
		service := stt.NewSegment("t", fakeTranscriber{}, 16000)
		service.SetRunSTT(fixedRun(original))
		starts := &segmentStarts{}
		starts.push(startedAt)
		tagBotSegments(service, starts)

		tagged, ok := collectRun(t, service, "audio")[0].(*frames.TranscriptionFrame)
		if !ok {
			t.Fatal("the run yielded no transcription")
		}
		if at, _ := segmentStartedAt(tagged); !at.Equal(startedAt) {
			t.Fatalf("started at %v, want %v", at, startedAt)
		}
		if tagged.Text != want.Text || tagged.UserID != want.UserID || tagged.Timestamp != want.Timestamp ||
			tagged.Language != want.Language || tagged.Finalized != want.Finalized ||
			tagged.Speaker != want.Speaker || tagged.Result == nil {
			t.Fatalf("tagged = %+v, want the fields of %+v", tagged, want)
		}
	})
}

// silentBot is a bot that accepts the connection and never says it is ready.
func silentBot(ctx context.Context, conn *websocket.Conn) {
	for {
		if _, _, err := conn.Read(ctx); err != nil {
			return
		}
	}
}

// refusedRun plays a one-turn scenario against the bot bot serves, with voice
// as the user's voice when given, and returns the result and how long it took.
func refusedRun(
	t *testing.T, bot func(context.Context, *websocket.Conn), readyTimeout time.Duration,
	voice *CachingTTSService, hook func(*ClientTransport),
) (ScriptResult, time.Duration) {
	t.Helper()
	server := newFakeRTVIServer(t)
	server.onConnect = bot
	scenario := oneTurn("x", Expectation{Event: EventLLMStarted})
	scenario.UserAudio = voice != nil
	s, err := NewScriptSession(scenario, server.url(), SessionParams{BotReadyTimeout: readyTimeout}, nil, voice, nil)
	if err != nil {
		t.Fatal(err)
	}
	s.client.transportBuilt = hook
	started := time.Now()
	res, err := s.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	return res, time.Since(started)
}

// voiceThatConnects is a user voice that connects to its provider as it
// starts, as a streaming TTS does, and reports failing to.
func voiceThatConnects(t *testing.T, providerURL string, failure error) *CachingTTSService {
	t.Helper()
	voice := failingVoice(t)
	var once sync.Once
	voice.Events().Add(processor.EventAfterProcessFrame, func(ctx context.Context, _ any, args ...any) {
		if _, ok := args[0].(*frames.StartFrame); !ok {
			return
		}
		once.Do(func() {
			err := failure
			if err == nil {
				conn, resp, dialErr := websocket.Dial(ctx, providerURL, nil)
				if resp != nil && resp.Body != nil {
					_ = resp.Body.Close()
				}
				if dialErr == nil {
					_ = conn.CloseNow()
					return
				}
				err = dialErr
			}
			voice.PushError(ctx, fmt.Sprintf("%s connection failed: %v", voice.Name(), err), err, false)
		})
	})
	return voice
}

// wantOneFailure asserts the run failed once, as kind on event, naming reason.
func wantOneFailure(t *testing.T, res ScriptResult, kind, event, reason string) {
	t.Helper()
	if len(res.Failures) != 1 {
		t.Fatalf("failures = %+v, want one", res.Failures)
	}
	f := res.Failures[0]
	if f.Kind != kind || f.EventName != event || !strings.Contains(f.Reason, reason) {
		t.Fatalf("failure = %+v, want %s on %s naming %q", f, kind, event, reason)
	}
}

// A bot that refuses the run, or never answers, before bot-ready: the cases a
// harness service or its transport decide.
func TestBotUnavailableBeforeReady(t *testing.T) {
	t.Run("a harness service failing before bot-ready is the harness error at once", func(t *testing.T) {
		// The bot never says ready, so only the voice's failure can end the run
		// early.
		provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "insufficient credits", http.StatusPaymentRequired)
		}))
		defer provider.Close()
		// A provider's refusal is a connection error too, but not the bot's.
		for _, c := range []struct {
			failure error
			named   string
		}{{nil, "402"}, {errBadVoice, "bad voice"}} {
			t.Run(c.named, func(t *testing.T) {
				voice := voiceThatConnects(t, wsURL(provider.URL), c.failure)
				res, elapsed := refusedRun(t, silentBot, 30*time.Second, voice, nil)
				if elapsed > 2*time.Second {
					t.Fatalf("took %s, want it to fail at once", elapsed)
				}
				wantOneFailure(t, res, FailureError, RunEventHarness, c.named)
			})
		}
	})
	t.Run("only a connection error from the bot connection is connect failed", func(t *testing.T) {
		// A bug in the harness's own transport code is the harness's error, not
		// the bot's.
		for _, c := range []struct {
			err         error
			kind, event string
		}{
			{&net.OpError{Op: "read", Net: "tcp", Err: errConnReset}, FailureConnectFailed, RunEventConnect},
			{errBadFrame, FailureError, RunEventHarness},
		} {
			t.Run(c.err.Error(), func(t *testing.T) {
				hook := func(tr *ClientTransport) {
					var once sync.Once
					in := tr.input
					in.Events().Add(processor.EventAfterProcessFrame, func(ctx context.Context, _ any, args ...any) {
						if _, ok := args[0].(*frames.StartFrame); ok {
							once.Do(func() { in.PushError(ctx, fmt.Sprintf("%s: %v", in.Name(), c.err), c.err, false) })
						}
					})
				}
				res, elapsed := refusedRun(t, silentBot, 30*time.Second, nil, hook)
				if elapsed > 2*time.Second {
					t.Fatalf("took %s, want it to fail at once", elapsed)
				}
				wantOneFailure(t, res, c.kind, c.event, c.err.Error())
			})
		}
	})
	t.Run("the first startup failure is the one reported", func(t *testing.T) {
		// The voice fails as the run starts; the bot closes before bot-ready
		// after.
		closeAfterAMoment := func(ctx context.Context, conn *websocket.Conn) {
			for {
				_, data, err := conn.Read(ctx)
				if err != nil {
					return
				}
				if in, err := rtvi.ParseIncoming(data); err == nil && in.Type == rtvi.TypeClientReady {
					break
				}
			}
			time.Sleep(300 * time.Millisecond)
			_ = conn.Close(websocket.StatusInternalError, "Concurrency limit reached")
		}
		voice := voiceThatConnects(t, "ws://unused", errBadVoice)
		res, _ := refusedRun(t, closeAfterAMoment, 5*time.Second, voice, nil)
		wantOneFailure(t, res, FailureError, RunEventHarness, "bad voice")
	})
}

// A turn's audio file is sent, not its text synthesized.
func TestAudioTurnSendsTheFileNotSynthesizedText(t *testing.T) {
	const sampleRate = 16000
	path := filepath.Join(t.TempDir(), "hi.wav")
	samples := writeTone(t, path, sampleRate, 0.5, 1)

	bot := newFakeRTVIServer(t)
	bot.onText("hello there", reply("Hi!")...)
	scenario := &ScriptScenario{Name: "audio-turn", Turns: []ScriptTurn{{
		User: "hello there", Audio: path, Expect: []Expectation{{Event: EventLLMStarted, WithinMS: 2000}},
	}}}
	scenario.UserAudio = true

	// No user voice: the file is played, so nothing has to synthesize it.
	wantPassed(t, run(t, scenario, bot.url(), SessionParams{}, nil))

	audioMessages := bot.messages(rtvi.TypeRawAudio)
	if len(audioMessages) == 0 {
		t.Fatal("the turn's audio file never reached the bot")
	}
	if got := bot.sentTexts(); len(got) != 0 {
		t.Fatalf("sent text %q: an audio turn must not also be sent as text", got)
	}
	// The user's side is streamed continuously, with silence before and after
	// the turn, so the recording is one contiguous run inside that stream.
	var streamed []byte
	for _, m := range audioMessages {
		var d rtvi.RawAudioData
		if err := json.Unmarshal(m.Data, &d); err != nil {
			t.Fatal(err)
		}
		if d.SampleRate != sampleRate {
			t.Fatalf("a frame at %d Hz, want %d", d.SampleRate, sampleRate)
		}
		pcm, _ := base64.StdEncoding.DecodeString(d.Base64Audio)
		streamed = append(streamed, pcm...)
	}
	if !bytes.Contains(streamed, pcmOf(samples)) {
		t.Fatal("the file's audio is not in the stream")
	}
}

// An audio run with a record path saves the conversation, both sides mixed.
func TestAnAudioRunRecordsTheConversation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "rec", "audio-turn.wav")
	clip := filepath.Join(t.TempDir(), "hi.wav")
	writeTone(t, clip, 16000, 0.3, 1)

	bot := newFakeRTVIServer(t)
	bot.onText("hello there", reply("Hi!")...)
	scenario := &ScriptScenario{Name: "audio-turn", Turns: []ScriptTurn{{
		User: "hello there", Audio: clip, Expect: []Expectation{{Event: EventLLMStarted, WithinMS: 2000}},
	}}}
	scenario.UserAudio, scenario.BotAudio = true, true
	s, err := NewScriptSession(scenario, bot.url(), SessionParams{RecordPath: path}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	saved := false
	res, err := s.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	wantPassed(t, res)
	for _, line := range res.DebugLog {
		saved = saved || strings.Contains(line, "recording saved: "+path)
	}
	pcm, rate, err := LoadUserAudio(path)
	if err != nil || !saved {
		t.Fatalf("recording: %v (saved %t)", err, saved)
	}
	if rate != 16000 || len(pcm) < 2*16000*3/10 {
		t.Fatalf("recording of %d bytes at %d Hz, want the user's turn in it", len(pcm), rate)
	}
}

// energyVAD hears speech in any chunk that is not silent, and none otherwise.
type energyVAD struct {
	vad.Base

	params vad.Params
}

func (*energyVAD) SetSampleRate(int) error { return nil }

func (*energyVAD) AnalyzeAudio(buffer []byte) vad.State {
	if slices.ContainsFunc(buffer, func(b byte) bool { return b != 0 }) {
		return vad.StateSpeaking
	}
	return vad.StateQuiet
}

func (v *energyVAD) Params() vad.Params     { return v.params }
func (v *energyVAD) SetParams(p vad.Params) { v.params = p }
func (*energyVAD) Reset()                   {}
func (*energyVAD) Close() error             { return nil }

// cannedTranscriber hears the same words in every segment.
type cannedTranscriber struct{ text string }

func (c cannedTranscriber) Transcribe(context.Context, []byte, int) (string, error) {
	return c.text, nil
}

// botAudioMessage is the bot's eval-bot-audio carrying pcm at sampleRate.
func botAudioMessage(pcm []byte, sampleRate int) rtvi.Message {
	return botMessage(evalws.BotAudioType, map[string]any{
		"audio": base64.StdEncoding.EncodeToString(pcm), "sampleRate": sampleRate,
	})
}

// In audio mode the bot's speech is heard, transcribed and closed into a turn,
// which is the response.
func TestTheBotsSpeechIsTheResponse(t *testing.T) {
	bot := newFakeRTVIServer(t)
	replies := reply("Hello there.")
	speech := bytes.Repeat([]byte{0x20, 0x10}, 16000/2) // 500 ms
	for off := 0; off < len(speech); off += 1280 {
		replies = append(replies, botAudioMessage(speech[off:min(off+1280, len(speech))], 16000))
	}
	bot.onText("hi", replies...)

	scenario := oneTurn("hi", Expectation{Event: EventResponse, TextContains: "Hello", WithinMS: 10000})
	scenario.BotAudio = true
	s, err := NewScriptSession(scenario, bot.url(), SessionParams{}, nil, nil,
		stt.NewSegment("BotSTT", cannedTranscriber{text: "Hello there."}, 0))
	if err != nil {
		t.Fatal(err)
	}
	s.client.newBotVAD = func() (vad.Analyzer, error) {
		params := vad.DefaultParams()
		params.StopSecs = botVADStopSecs
		return &energyVAD{params: params}, nil
	}
	// The turn ends on a speech timeout rather than the default Smart Turn
	// model, which needs the ONNX runtime.
	s.client.botTurnStop = []turns.StopStrategy{
		turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 200 * time.Millisecond}),
	}
	res, err := s.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if !res.Passed() {
		t.Logf("trace:\n%s\nevents: %v", strings.Join(res.DebugLog, "\n"), res.EventsSeen)
	}
	wantPassed(t, res)
}
