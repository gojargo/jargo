// Command features-pattern-pair-voice-switching tells a story in several
// voices, switching the TTS voice on tags the LLM writes.
//
// The LLM marks each stretch of the story with <voice>name</voice>. A
// PatternPairAggregator in an LLMText processor between the LLM and the TTS
// recognizes the tags; on each one it pushes a TTSUpdateSettingsFrame
// switching the Cartesia voice, and the TTS skips the tag itself instead of
// speaking it.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-pattern-pair-voice-switching
//
// Then open http://localhost:7860, connect, and ask for a story.
package main

import (
	"context"
	"log/slog"
	"os"
	"strings"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/text"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// voiceIDs are the Cartesia voices the story is told in.
//
//nolint:gochecknoglobals // fixed lookup table
var voiceIDs = map[string]string{
	"narrator": "c45bc5ec-dc68-4feb-8829-6e6b2748095d", // Narrator voice
	"female":   "f6ff7c0c-e396-40a9-a70b-f7607edb6937", // Female character voice
	"male":     "7cf0e2b1-8daf-4fe4-89ad-f6039398f359", // Male character voice
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

// systemPrompt is the prompt for storytelling with voice switching.
//
//nolint:lll,misspell // the prompt is the model's text, kept as written
const systemPrompt = `You are an engaging storyteller that uses different voices to bring stories to life.

You have three voices to use, but each has a specific purpose:

<voice>narrator</voice>
This is the default narrator voice. Use this for all narration, descriptions, and non-dialogue text.

<voice>female</voice>
Use this ONLY for direct speech by female characters (just the quoted text).

<voice>male</voice>
Use this ONLY for direct speech by male characters (just the quoted text).

IMPORTANT: Switch back to narrator voice immediately after character dialogue.

Here's an EXAMPLE of correct voice usage:

<voice>narrator</voice>
Sarah spotted her old friend across the café. She couldn't believe her eyes.
<voice>female</voice>
"Jacob! It's been so long!"
<voice>narrator</voice>
Sarah exclaimed, jumping up from her seat with a radiant smile.
<voice>male</voice>
"Sarah, is it really you? I can't believe it!"
<voice>narrator</voice>
Jacob replied, grinning widely as he walked over to her. The two friends embraced warmly, as if trying to make up for all the years spent apart.
<voice>female</voice>
"What are you doing in town? Last I heard you were in Seattle."
<voice>narrator</voice>
She asked, gesturing for him to join her at the table.

FOLLOW THESE RULES:
1. Always begin with the narrator voice
2. Only use character voices for the EXACT words they speak (in quotes)
3. SWITCH BACK to narrator voice for speech tags and all other text
4. Begin by asking what kind of story the user would like to hear
5. Create engaging dialogue with distinct characters

Remember: Use narrator voice for EVERYTHING except the actual quoted dialogue.`

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	// The pattern pair aggregator for voice switching.
	llmTextAggregator := text.NewPatternPairAggregator(frames.TextTypeSentence, "")

	// The pattern for voice switching.
	if err := llmTextAggregator.AddPattern("voice", "<voice>", "</voice>", text.MatchAggregate); err != nil {
		return err
	}

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	// LLM text goes through the pattern aggregator before the TTS.
	llmTextProcessor := aggregators.NewLLMTextWith("LLMTextProcessor", llmTextAggregator)

	// The handler for voice switching. It runs as the processor handles the
	// text, so the switch goes out ahead of the text that follows the tag.
	llmTextAggregator.OnPatternMatch("voice", func(match text.Aggregation) {
		voiceName := strings.ToLower(strings.TrimSpace(match.Text))
		voiceID, ok := voiceIDs[voiceName]
		if !ok {
			slog.Warn("Unknown voice", "voice", voiceName)
			return
		}
		delta := &cartesia.TTSSettings{Voice: settings.Set(voiceID)}
		_ = llmTextProcessor.PushFrame(context.Background(), frames.NewTTSUpdateSettingsFrame(delta), processor.Downstream)
		slog.Info("Switched voice", "voice", voiceName)
	})

	// The TTS starts with the narrator voice.
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: voiceIDs["narrator"],
	})
	tts.SetSkipTextTypes("voice") // Skip voice tags in TTS speech

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemPrompt)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),        // Transport user input
		stt,              // STT
		agg.User(),       // User responses
		llm,              // LLM
		llmTextProcessor, // Voice tags
		tts,              // TTS
		t.Output(),       // Transport bot output
		agg.Assistant(),  // Assistant spoken responses
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})
	r.AddWorkers(ctx, worker)

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Start the conversation with no prompt, letting the LLM follow its
		// system instruction.
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
	})

	event.On(t.Events(), transport.EventClientDisconnected, func(ctx context.Context, _ any) {
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
