// Command persistent-context-gemini saves the conversation to disk and loads it back:
// Google Gemini.
//
// The LLM has tools to save the conversation, list the saved conversations,
// and load one into the session, alongside a weather tool to have something
// to talk about, and a tool to look at the user's camera. Conversations are saved as JSON files named
// /tmp/jargo_conversation_<timestamp>.json.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… GOOGLE_API_KEY=… \
//	  go run ./examples/persistent-context/persistent-context-gemini
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
	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/deepgram"
	"github.com/gojargo/jargo/provider/google/gemini"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/service/llm"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/workers"
)

// baseFilename is where the conversations are saved, with a timestamp and
// ".json" added.
const baseFilename = "/tmp/jargo_conversation_"

// getImage is called when the user asks for a description of their camera
// feed.
func getImage(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		UserID   string `json:"user_id"`
		Question string `json:"question"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	slog.Debug("Requesting image", "user_id", args.UserID, "question", args.Question)

	// Request a user image, to be added to the context and associated with
	// the function call. The result callback goes with it, to be called once
	// the image is actually retrieved.
	request := frames.NewUserImageRequestFrame(args.UserID)
	request.Text = args.Question
	request.AppendToContext = true
	request.FunctionName = params.FunctionName
	request.ToolCallID = params.ToolCallID
	request.ResultCallback = params.Result
	return params.LLM.PushFrame(ctx, request, processor.Upstream)
}

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
	// Remove the last message (the instruction to save the context).
	if len(messages) > 0 {
		messages = messages[:len(messages)-1]
	}
	data, err := json.MarshalIndent(messages, "", "  ")
	if err != nil {
		slog.Debug("error saving conversation", "err", err)
		return failure(ctx, params, err)
	}
	slog.Debug("writing conversation to "+filename, "messages", string(data))
	if err := os.WriteFile(filename, data, 0o600); err != nil {
		slog.Debug("error saving conversation", "err", err)
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
	if err = json.Unmarshal(data, &messages); err != nil {
		return failure(ctx, params, err)
	}
	params.Context.SetMessages(messages)
	result, err := json.Marshal(map[string]any{
		"success": true,
		"message": "The most recent conversation has been loaded. Awaiting further instructions.",
	})
	if err != nil {
		return err
	}
	return params.Result(ctx, string(result), nil)
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
		{
			Name:        "get_image",
			Description: "Called when the user requests a description of their camera feed.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"user_id": {
						"type": "string",
						"description": "The ID of the user to grab the image from."
					},
					"question": {
						"type": "string",
						"description": "The question that the user is asking about the image."
					}
				},
				"required": ["user_id", "question"]
			}`),
			Handler: llm.FunctionCallHandler(getImage),
		},
	}
}

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: func() transport.Params { return cameraParams(transport.DefaultParams()) },
	WebRTC:  func() transport.Params { return cameraParams(transport.DefaultParams()) },
}

// cameraParams captures the user's camera, only on request.
func cameraParams(p transport.Params) transport.Params {
	p.VideoInEnabled = true
	p.VideoInSources = map[string]transport.VideoInSourceParams{"camera": {OnRequestOnly: true}}
	return p
}

//nolint:lll // the prompt is the model's text, kept as written
const systemInstruction = `You are a helpful assistant in a voice conversation. Your goal is to demonstrate your
capabilities in a succinct way. Your responses will be spoken aloud, so avoid emojis, bullet points, or other formatting that can't be spoken. Keep responses concise. Respond to what the user said in a creative
can't easily be spoken, such as emojis or bullet points. Respond to what the user said in a creative
and helpful way.

You have several tools you can use to help you.

You can respond to questions about the weather using the get_current_weather tool.

You can save the current conversation using the save_conversation tool. This tool allows you to save
the current conversation to external storage. If the user asks you to save the conversation, use this
save_conversation too.

You can load a saved conversation using the load_conversation tool. This tool allows you to load a
conversation from external storage. You can get a list of conversations that have been saved using the
get_saved_conversation_filenames tool.

You can answer questions about the user's video stream using the get_image tool. Some examples of phrases that ` +
	`indicate you should use the get_image tool are:
  - What do you see?
  - What's in the video?
  - Can you describe the video?
  - Tell me about what you see.
  - Tell me something interesting about what you see.
  - What's happening in the video?
`

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	model := gemini.NewLLM(gemini.Config{APIKey: os.Getenv("GOOGLE_API_KEY")})
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

	event.On(t.Events(), transport.EventClientConnected, func(_ context.Context, client any) {
		slog.Info("Client connected")

		clientID := t.GetClientID(client)

		// Kick off the conversation.
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Please introduce yourself to the user. Use '" + clientID + "' as the user ID during function calls.",
		})
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
