// Command audio-sound-effects plays a sound effect at each change of turn.
//
// One processor, right after the user aggregator, plays a ding when the user's
// turn is handed to the LLM; another, after the TTS, plays a different ding
// when the bot's response ends. Frame loggers show what goes into the LLM and
// what comes out of it. The sounds are WAV files embedded in the binary.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/audio/audio-sound-effects
//
// Then open http://localhost:7860 and connect.
package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/binary"
	"errors"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// The sound effects.
var (
	//go:embed ding1.wav
	ding1WAV []byte
	//go:embed ding2.wav
	ding2WAV []byte
)

// errNotWAV is returned for a sound that is not a PCM WAV file.
var errNotWAV = errors.New("not a PCM WAV file")

// sound is a decoded sound effect.
type sound struct {
	pcm         []byte
	sampleRate  int
	numChannels int
}

// frame returns a new audio frame playing the sound.
func (s sound) frame() *frames.OutputAudioRawFrame {
	return frames.NewOutputAudioRawFrame(s.pcm, s.sampleRate, s.numChannels)
}

// decodeWAV reads the PCM audio of a WAV file with its sample rate and channel
// count.
func decodeWAV(data []byte) (sound, error) {
	if len(data) < 12 || !bytes.Equal(data[0:4], []byte("RIFF")) || !bytes.Equal(data[8:12], []byte("WAVE")) {
		return sound{}, errNotWAV
	}
	var s sound
	for rest := data[12:]; len(rest) >= 8; {
		id := string(rest[0:4])
		size := int(binary.LittleEndian.Uint32(rest[4:8]))
		body := rest[8:]
		if size > len(body) {
			size = len(body)
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return sound{}, errNotWAV
			}
			s.numChannels = int(binary.LittleEndian.Uint16(body[2:4]))
			s.sampleRate = int(binary.LittleEndian.Uint32(body[4:8]))
		case "data":
			s.pcm = body[:size]
		}
		// Chunks are padded to an even size.
		next := size + size%2
		if next > len(body) {
			break
		}
		rest = body[next:]
	}
	if s.pcm == nil || s.sampleRate == 0 || s.numChannels == 0 {
		return sound{}, errNotWAV
	}
	return s, nil
}

// outboundSoundEffect plays a ding when the bot's response ends.
type outboundSoundEffect struct {
	*processor.Base
	ding sound
}

func newOutboundSoundEffect(ding sound) *outboundSoundEffect {
	p := &outboundSoundEffect{ding: ding}
	p.Base = processor.New("OutboundSoundEffectWrapper", p)
	return p
}

// ProcessFrame plays the ding ahead of the end of each response.
func (p *outboundSoundEffect) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.LLMFullResponseEndFrame); ok {
		if err := p.PushFrame(ctx, p.ding.frame(), processor.Downstream); err != nil {
			return err
		}
		// In case anything else downstream needs it.
	}
	return p.PushFrame(ctx, f, dir)
}

// inboundSoundEffect plays a ding when the user's turn goes to the LLM.
type inboundSoundEffect struct {
	*processor.Base
	ding sound
}

func newInboundSoundEffect(ding sound) *inboundSoundEffect {
	p := &inboundSoundEffect{ding: ding}
	p.Base = processor.New("InboundSoundEffectWrapper", p)
	return p
}

// ProcessFrame plays the ding ahead of each context sent to the LLM.
func (p *inboundSoundEffect) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if _, ok := f.(*frames.LLMContextFrame); ok {
		if err := p.PushFrame(ctx, p.ding.frame(), processor.Downstream); err != nil {
			return err
		}
		// In case anything else downstream needs it.
	}
	return p.PushFrame(ctx, f, dir)
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	Twilio:  func() runner.WebSocketParams { return runner.WebSocketParams{Params: wsserver.DefaultParams()} },
	WebRTC:  transport.DefaultParams,
}

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will be spoken " +
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
	" to what the user said in a creative, helpful, and brief way."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	ding1, err := decodeWAV(ding1WAV)
	if err != nil {
		return err
	}
	ding2, err := decodeWAV(ding2WAV)
	if err != nil {
		return err
	}

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	outSound := newOutboundSoundEffect(ding1)
	inSound := newInboundSoundEffect(ding2)
	fl := processor.NewFrameLogger("LLM Out")
	fl2 := processor.NewFrameLogger("Transcription In")

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		inSound,
		fl2,
		llm,
		fl,
		tts,
		outSound,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation. The ding follows the greeting through the
		// pipeline, so it plays once the greeting finishes.
		worker.QueueFrames([]frames.Frame{frames.NewTTSSpeakFrame("Hi, I'm listening!"), ding1.frame()})
	})

	events.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
		slog.Info("Client disconnected")
		r.Cancel(ctx, "client disconnected")
	})

	return r.Run(ctx, workers.RunOptions{})
}

func bot(ctx context.Context, args runner.Arguments) error {
	t, err := runner.CreateTransport(ctx, args, transportParams)
	if err != nil {
		return err
	}
	return runBot(ctx, t, args)
}

func main() {
	runner.Main(bot)
}
