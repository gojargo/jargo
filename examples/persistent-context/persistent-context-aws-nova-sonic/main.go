// Command persistent-context-aws-nova-sonic saves the conversation to disk and
// loads it back: Amazon Nova Sonic, a speech-to-speech model.
//
// The model has tools to save the conversation, list the saved conversations,
// and load one into the session, alongside a weather tool to have something
// to talk about. Conversations are saved as JSON files named
// /tmp/jargo_conversation_<timestamp>.json. Loading one resets the Nova Sonic
// session, which opens a new one seeded with the loaded conversation.
//
//	AWS_ACCESS_KEY_ID=… AWS_SECRET_ACCESS_KEY=… AWS_REGION=us-east-1 \
//	  go run ./examples/persistent-context/persistent-context-aws-nova-sonic
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

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/aws/novasonic"
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

// failure reports a failed save or load to the model.
func failure(ctx context.Context, params llm.FunctionCallParams, err error) error {
	result, merr := json.Marshal(map[string]any{"success": false, "error": err.Error()})
	if merr != nil {
		return merr
	}
	return params.Result(ctx, string(result), nil)
}

// isAssistantResponse reports whether a message is a response the assistant
// spoke, rather than its request for a tool call.
func isAssistantResponse(m frames.Message) bool {
	return m.Role == frames.RoleAssistant && (m.Text != "" || len(m.Content) > 0)
}

// saveConversation saves the current conversation to external storage.
func saveConversation(ctx context.Context, params llm.FunctionCallParams) error {
	filename := baseFilename + time.Now().Format("2006-01-02_15:04:05") + ".json"
	messages := params.Context.Messages()
	// Remove the last few messages. In reverse order, they are:
	//   - the in progress save tool call
	//   - the invocation of the save tool call
	//   - the user ask to save (which may encompass one or more messages)
	// The simplest thing to do is to drop messages until the last one is an
	// assistant response.
	for len(messages) > 0 && !isAssistantResponse(messages[len(messages)-1]) {
		messages = messages[:len(messages)-1]
	}
	if len(messages) > 0 { // we never expect this to be empty
		data, err := json.MarshalIndent(messages, "", "  ")
		if err != nil {
			return failure(ctx, params, err)
		}
		slog.Debug("writing conversation to "+filename, "messages", string(data))
		if err := os.WriteFile(filename, data, 0o600); err != nil {
			return failure(ctx, params, err)
		}
	}
	return params.Result(ctx, `{"success": true}`, nil)
}

// loadConversation loads a conversation history into the current session of
// the Nova Sonic service s2s.
func loadConversation(s2s *novasonic.Service) llm.FunctionCallHandler {
	return func(ctx context.Context, params llm.FunctionCallParams) error {
		var args struct {
			Filename string `json:"filename"`
		}
		if err := json.Unmarshal(params.Arguments, &args); err != nil {
			return err
		}
		reset := func(ctx context.Context) {
			slog.Debug("loading conversation from " + args.Filename)
			data, err := os.ReadFile(args.Filename)
			if err != nil {
				_ = failure(ctx, params, err)
				return
			}
			var messages []frames.Message
			if err := json.Unmarshal(data, &messages); err != nil {
				_ = failure(ctx, params, err)
				return
			}
			// If the last message isn't from the user, add a message asking for
			// a recap.
			if len(messages) > 0 && messages[len(messages)-1].Role != frames.RoleUser {
				messages = append(messages, frames.Message{
					Role: frames.RoleUser,
					Text: "Can you catch me up on what we were talking about?",
				})
			}
			params.Context.SetMessages(messages)
			s2s.ResetConversation(ctx)
			// s2s.TriggerAssistantResponse(ctx)
		}
		// The reset closes the session the call came from, so it runs on its
		// own, outliving the call.
		go reset(context.WithoutCancel(ctx))
		return nil
	}
}

// tools describes the functions to the model, each carrying its own handler.
// load_conversation resets the session of s2s.
func tools(s2s *novasonic.Service) []frames.Tool {
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
			Handler: loadConversation(s2s),
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

// systemInstruction is the initial system instruction.
const systemInstruction = "You are a friendly assistant. The user and you will engage in a spoken dialog " +
	"exchanging the transcripts of a natural real-time conversation. Keep your responses short, generally " +
	"two or three sentences for chatty scenarios. "

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting bot")

	s2s := novasonic.New(novasonic.Config{
		SecretAccessKey: os.Getenv("AWS_SECRET_ACCESS_KEY"),
		AccessKeyID:     os.Getenv("AWS_ACCESS_KEY_ID"),
		// As of 2026-09-28, the supported regions are us-east-1, us-west-2,
		// eu-north-1 and ap-northeast-1.
		Region:       os.Getenv("AWS_REGION"),
		Voice:        "tiffany", // matthew, tiffany, amy
		Instructions: systemInstruction,
		// The tools could be given here rather than through the conversation:
		// Tools: frames.ToolsSchema{Standard: tools(s2s)},
	})

	convo := frames.NewLLMContext("")
	convo.SetTools(tools(s2s))
	// Nova Sonic doesn't emit user-turn frames. To get them (for RTVI speech
	// events, turn observers, etc.) give the user aggregator local turn
	// detection, aggregators.WithVAD with a Silero analyzer. See
	// realtime-aws-nova-sonic for the full discussion.
	agg := aggregators.New(convo)

	p := pipeline.New(
		t.Input(), // Transport user input
		agg.User(),
		s2s,        // LLM
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

	event.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
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
