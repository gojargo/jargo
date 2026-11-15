// Command features-text-transforms composes individual built-in text
// transforms instead of using the VoiceFormatter bundle, for fine-grained
// control over which transforms apply and in what order.
//
// Each transform rewrites the text of a unit. They are registered on the TTS
// service as TextTransformers, each pairing a transform with the text type it
// applies to (frames.AnyTextType for all of them).
//
// The same transformers registered on an LLMText processor rewrite the text
// for everyone downstream (the voice, the context and the client) rather than
// for the voice alone. This example strips emoji there, so the LLM never sees
// its own emoji in the context and keeps adding more.
//
// It is a billing assistant composing several transforms:
//
//	StripMarkdown      remove bold, italic and headers the LLM might add
//	NormalizeAcronyms  "API" → "A P I"
//	EmailToSpeech      "user@example.com" → "user at example dot com"
//	ExpandCurrency     "$42.50" → "forty-two dollars and fifty cents"
//	ExpandPercentages  "3.5%" → "three point five percent"
//	ReplaceText        custom substitutions (e.g. "Dr." → "Doctor")
//
//	CARTESIA_API_KEY=… ELEVENLABS_API_KEY=… ELEVENLABS_VOICE_ID=… \
//	OPENAI_API_KEY=… go run ./examples/features/features-text-transforms
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"log/slog"
	"os"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/elevenlabs"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	ttsservice "github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/utils/text"
	"github.com/gojargo/jargo/workers"
)

const systemInstruction = "You are Siobhan, a billing support assistant for Nexora, a telecom " +
	"company. Your responses are spoken aloud. Use natural formatting in " +
	"your answers: currency amounts like $42.50, percentages like 3.5%, " +
	"email addresses like support@example.com, and abbreviations like " +
	"Dr. or St. as you normally would in writing; the voice formatter " +
	"will convert them to natural speech before synthesis."

// transformer registers a text transform for every unit, whatever its type.
func transformer(t text.Transform) ttsservice.TextTransformer {
	return ttsservice.TextTransformer{
		TextType: frames.AnyTextType,
		Transform: func(_ context.Context, s string, _ frames.TextType) (string, error) {
			return t(s), nil
		},
	}
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

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := cartesia.NewSTT(cartesia.STTConfig{APIKey: os.Getenv("CARTESIA_API_KEY")})

	// Custom substitution rules, applied after every other transform. The
	// patterns are regular expressions; use regexp.QuoteMeta for literal
	// strings.
	//
	// The last rule respells a name so the TTS pronounces it correctly, which
	// works on every model.
	customSubs, err := text.ReplaceText([][2]string{
		{`\bDr\.`, "Doctor"},
		{`\bSt\.`, "Street"},
		{`\bApt\.`, "Apartment"},
		{`\bvs\b`, "versus"},
		{`(?i)\bSiobhan\b`, "shi-VAWN"},
	})
	if err != nil {
		return err
	}

	// The transform chain for a billing assistant. Order matters: strip the
	// markdown first, then expand the special patterns.
	billingTransforms := []ttsservice.TextTransformer{
		transformer(text.StripMarkdown),
		transformer(text.NormalizeAcronyms),
		transformer(text.EmailToSpeech),
		transformer(text.ExpandCurrency),
		transformer(text.ExpandPercentages),
		transformer(customSubs),
	}

	// Transforms on the LLM text processor rewrite what everyone sees: the
	// voice, the context and the client. Emoji removed here never reach the
	// context, so the LLM does not see its own emoji and keep adding more.
	stripEmoji, err := text.ReplaceText([][2]string{{`[\x{1F300}-\x{1FAFF}\x{2600}-\x{27BF}]\x{FE0F}?`, ""}})
	if err != nil {
		return err
	}
	llmTextProcessor := aggregators.NewLLMText("LLMTextProcessor")
	llmTextProcessor.SetTextTransformers(transformer(stripEmoji))

	tts := elevenlabs.NewTTS(elevenlabs.Config{
		APIKey:  os.Getenv("ELEVENLABS_API_KEY"),
		VoiceID: os.Getenv("ELEVENLABS_VOICE_ID"),
	})
	tts.SetTextTransformers(billingTransforms...)

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		llm,
		llmTextProcessor,
		tts,
		t.Output(),
		agg.Assistant(),
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

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Introduce yourself by name and let the caller know they've " +
				"reached billing support. Offer to help with their account " +
				"balance, recent charges, or payment options. Give a sample " +
				"balance such as $127.50 due on 3/15/2025 and a support email " +
				"like billing@telecom.example.com.",
		})
		worker.QueueFrames([]frames.Frame{frames.NewLLMRunFrame()})
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
