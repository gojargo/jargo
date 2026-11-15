// Command persistent-context-openai saves the conversation to disk and loads it back:
// OpenAI.
//
// The LLM has tools to save the conversation, list the saved conversations,
// and load one into the session, alongside a weather tool to have something
// to talk about. Conversations are saved as JSON files named
// /tmp/jargo_conversation_<timestamp>.json.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/persistent-context/persistent-context-openai
//
// Then open http://localhost:7860, connect, chat, and ask the bot to save the
// conversation; in a later session, ask it to list and load one.
package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// baseFilename is where the conversations are saved, with a timestamp and
// ".json" added.
const baseFilename = "/tmp/jargo_conversation_"

// getCurrentWeather gets the current weather.
func getCurrentWeather(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Location string `json:"location"`
		Format   string `json:"format"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	temperature := 24
	if args.Format == "fahrenheit" {
		temperature = 75
	}
	result, err := json.Marshal(map[string]any{
		"conditions":  "nice",
		"temperature": temperature,
		"format":      args.Format,
		"timestamp":   time.Now().Format("20060102_150405"),
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// getSavedConversationFilenames gets a list of saved conversation histories.
func getSavedConversationFilenames(ctx context.Context, params llm.FunctionCallParams) error {
	matchingFiles, err := filepath.Glob(baseFilename + "*.json")
	if err != nil {
		return err
	}
	slog.Debug("matching files", "files", matchingFiles)
	result, err := json.Marshal(map[string]any{"filenames": matchingFiles})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
}

// failure reports a failed save or load to the LLM.
func failure(ctx context.Context, params llm.FunctionCallParams, err error) error {
	result, merr := json.Marshal(map[string]any{"success": false, "error": err.Error()})
	if merr != nil {
		return merr
	}
	return params.Result(ctx, string(result), nil)
}

// saveConversation saves the current conversation to external storage.
func saveConversation(ctx context.Context, params llm.FunctionCallParams) error {
	filename := baseFilename + time.Now().Format("2006-01-02_15:04:05") + ".json"
	messages := params.Context.Messages()
	// Remove the last message, which is the instruction just given to save
	// the conversation.
	if len(messages) > 0 {
		messages = messages[:len(messages)-1]
	}
	data, err := json.MarshalIndent(messages, "", "  ")
	if err != nil {
		return failure(ctx, params, err)
	}
	slog.Debug("writing conversation to "+filename, "messages", string(data))
	if err := os.WriteFile(filename, data, 0o600); err != nil {
		return failure(ctx, params, err)
	}
	return params.Result(ctx, `{"success": true}`, nil)
}

// loadConversation loads a conversation history into the current session.
func loadConversation(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Debug("loading conversation from " + args.Filename)
	data, err := os.ReadFile(args.Filename)
	if err != nil {
		return failure(ctx, params, err)
	}
	var messages []frames.Message
	if err := json.Unmarshal(data, &messages); err != nil {
		return failure(ctx, params, err)
	}
	params.Context.SetMessages(messages)
	slog.Debug("loaded conversation from " + args.Filename)
	return params.LLM.QueueFrame(ctx, frames.NewTTSSpeakFrame("Ok, I've loaded that conversation."), processor.Downstream)
}

// tools describes the functions to the LLM, each carrying its own handler.
func tools() []frames.Tool {
	return []frames.Tool{
		{
			Name:        "get_current_weather",
			Description: "Get the current weather.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"location": {
						"type": "string",
						"description": "The city and state, e.g. \"San Francisco, CA\"."
					},
					"format": {
						"type": "string",
						"description": "The temperature unit to use. Must be either \"celsius\" or \"fahrenheit\". ` +
				`Infer this from the user's location."
					}
				},
				"required": ["location", "format"]
			}`),
			Handler: llm.FunctionCallHandler(getCurrentWeather),
		},
		{
			Name: "save_conversation",
			Description: "Save the current conversation. Use this function to persist the current " +
				"conversation to external storage.",
			Parameters: json.RawMessage(`{"type": "object", "properties": {}, "required": []}`),
			Handler:    llm.FunctionCallHandler(saveConversation),
		},
		{
			Name: "get_saved_conversation_filenames",
			Description: "Get a list of saved conversation histories. Returns a list of filenames. Each " +
				"filename includes a date and timestamp. Each file is conversation history that can be " +
				"loaded into this session.",
			Parameters: json.RawMessage(`{"type": "object", "properties": {}, "required": []}`),
			Handler:    llm.FunctionCallHandler(getSavedConversationFilenames),
		},
		{
			Name: "load_conversation",
			Description: "Load a conversation history. Use this function to load a conversation history " +
				"into the current session.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"filename": {
						"type": "string",
						"description": "The filename of the conversation history to load."
					}
				},
				"required": ["filename"]
			}`),
			Handler: llm.FunctionCallHandler(loadConversation),
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

const systemInstruction = "You are a helpful assistant in a voice conversation. Your responses will be spoken " +
	"aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Respond" +
	" to what the user said in a creative, helpful, and brief way."

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(tools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(), // Transport user input
		stt,       // STT
		agg.User(),
		model, // LLM
		tts,
		t.Output(), // Transport bot output
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
		// Kick off the conversation.
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
