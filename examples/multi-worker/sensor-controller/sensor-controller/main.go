// Command sensor-controller is a voice agent and a sensor-controller worker,
// both plain pipeline workers.
//
// Two pipeline workers run side by side:
//
//   - The voice agent is built inline in runBot: a standard transport, STT, LLM
//     and TTS pipeline. Its LLM has a single tool, ask_controller(question),
//     which forwards the user's request to the controller over the bus and
//     speaks back the response.
//   - The sensor controller (buildSensorController) is a pipeline worker whose
//     pipeline runs a simulated temperature sensor (see sensor.go) alongside an
//     LLM of its own. That LLM has tools to read the current reading, inspect
//     the rolling statistics, and change the simulated sensor's target
//     temperature and response rate.
//
// The controller is not an LLM worker and is not bridged. The voice agent and
// the controller communicate only through job requests and job responses. The
// controller collects its responses by listening to the assistant aggregator's
// turn-stopped event and pairing each LLM completion with the job in flight.
//
// An example voice exchange:
//
//	User: What's the temperature?
//	Controller: 22.1°C, holding steady.
//
//	User: Make it warmer.
//	Controller: I set the target to 26°C. Give it about 20 seconds.
//
//	User: Is it stable yet?
//	Controller: It's at 25.4°C and still climbing, almost there.
//
//	User: Why is it slow?
//	Controller: The response rate is 5%. I sped it up to 20%; it'll settle faster now.
//
// Run it with:
//
//	OPENAI_API_KEY=… DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… \
//	  go run ./examples/multi-worker/sensor-controller/sensor-controller
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sync"
	"time"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/bus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/pipeline/jobcontext"
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

// transportParams builds each transport's parameters when that transport is
// the one a session arrives on.
//
//nolint:gochecknoglobals // the transports' parameters, built per session
var transportParams = runner.TransportParams{
	Eval:    func() evalws.TransportParams { return evalws.TransportParams{Params: wsserver.DefaultParams()} },
	LiveKit: transport.DefaultParams,
	WebRTC:  transport.DefaultParams,
}

// noParameters is the schema of a tool that takes no arguments.
const noParameters = `{"type": "object", "properties": {}, "required": []}`

// round2 rounds v to two decimals.
func round2(v float64) float64 { return math.Round(v*100) / 100 }

// result reports v, encoded as JSON, as a tool call's result.
func result(ctx context.Context, params llm.FunctionCallParams, v any) error {
	encoded, err := json.Marshal(v)
	if err != nil {
		return err
	}
	return params.Result(ctx, string(encoded), nil)
}

// controllerTools are the tools of the controller's LLM, which read or change
// the sensor.
func controllerTools(sensor *sensorReader, stats *sensorStats) []frames.Tool {
	// getCurrentReading reads the sensor's current temperature in degrees
	// Celsius.
	getCurrentReading := func(ctx context.Context, params llm.FunctionCallParams) error {
		return result(ctx, params, struct {
			Temperature float64 `json:"temperature"`
		}{round2(sensor.Current())})
	}

	// getStats reports the rolling minimum, maximum, average, and trend of the
	// temperature.
	getStats := func(ctx context.Context, params llm.FunctionCallParams) error {
		return result(ctx, params, struct {
			Min   float64 `json:"min"`
			Max   float64 `json:"max"`
			Avg   float64 `json:"avg"`
			Trend string  `json:"trend"`
		}{round2(stats.Min()), round2(stats.Max()), round2(stats.Avg()), stats.Trend()})
	}

	// setTargetTemperature adjusts the target temperature; the sensor drifts
	// toward it.
	setTargetTemperature := func(ctx context.Context, params llm.FunctionCallParams) error {
		var args struct {
			TargetCelsius float64 `json:"target_celsius"`
		}
		if err := json.Unmarshal(params.Arguments, &args); err != nil {
			return err
		}
		sensor.SetTarget(args.TargetCelsius)
		return result(ctx, params, struct {
			OK        bool    `json:"ok"`
			NewTarget float64 `json:"new_target"`
		}{true, args.TargetCelsius})
	}

	// setResponseRate sets how aggressively the sensor approaches the target.
	setResponseRate := func(ctx context.Context, params llm.FunctionCallParams) error {
		var args struct {
			Rate float64 `json:"rate"`
		}
		if err := json.Unmarshal(params.Arguments, &args); err != nil {
			return err
		}
		sensor.SetResponseRate(args.Rate)
		return result(ctx, params, struct {
			OK      bool    `json:"ok"`
			NewRate float64 `json:"new_rate"`
		}{true, args.Rate})
	}

	return []frames.Tool{
		{
			Name:        "get_current_reading",
			Description: "Read the sensor's current temperature in degrees Celsius.",
			Parameters:  json.RawMessage(noParameters),
			Handler:     llm.FunctionCallHandler(getCurrentReading),
		},
		{
			Name:        "get_stats",
			Description: "Rolling minimum, maximum, average, and trend of the temperature.",
			Parameters:  json.RawMessage(noParameters),
			Handler:     llm.FunctionCallHandler(getStats),
		},
		{
			Name:        "set_target_temperature",
			Description: "Adjust the target temperature; the sensor will drift toward it.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"target_celsius": {
						"type": "number",
						"description": "The new target temperature in degrees Celsius."
					}
				},
				"required": ["target_celsius"]
			}`),
			Handler: llm.FunctionCallHandler(setTargetTemperature),
		},
		{
			Name:        "set_response_rate",
			Description: "Set how aggressively the sensor approaches the target.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"rate": {
						"type": "number",
						"description": "Response rate between 0.01 (slow) and 0.3 (fast)."
					}
				},
				"required": ["rate"]
			}`),
			Handler: llm.FunctionCallHandler(setResponseRate),
		},
	}
}

// buildSensorController builds the controller worker as a plain pipeline
// worker.
//
// The pipeline's shape is:
//
//	sensorReader → sensorStats → user_agg → llm → assistant_agg
//
// sensorReader runs an autonomous tick loop that emits a sensorReadingFrame
// every second; sensorStats consumes those readings and exposes rolling
// statistics. The LLM has four direct tools that read or change the sensor.
//
// Jobs arrive through the worker's job-request event. The handler stores the
// active job's id, then queues an LLMMessagesAppendFrame with the user's
// question and runs the LLM. When the assistant turn finishes (signaled by the
// assistant aggregator's turn-stopped event), the handler sends a job response
// carrying the LLM's answer back to the voice agent.
func buildSensorController() *pipeline.Worker {
	sensor := newSensorReader(defaultSensorReaderConfig())
	stats := newSensorStats(30)

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction("You are a temperature sensor controller. You manage a single " +
		"thermometer and answer the user's questions about it. Use the " +
		"provided tools to read the current temperature, inspect rolling " +
		"statistics, change the target temperature, or change how fast " +
		"the sensor responds. When the user asks for a vague change " +
		"('make it warmer', 'cooler'), pick a sensible target and call " +
		"set_target_temperature. Always answer in one or two short " +
		"sentences: your reply is spoken aloud.")

	convo := frames.NewLLMContext("")
	convo.SetTools(controllerTools(sensor, stats))
	agg := aggregators.New(convo)

	p := pipeline.New(
		sensor,
		stats,
		agg.User(),
		model,
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Name:                    "sensor-controller",
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	// The controller handles one job at a time (the LLM pipeline can only run
	// one turn at a time). jobID pairs the job in flight with the next
	// turn-stopped event.
	var (
		mu    sync.Mutex
		jobID string
	)

	events.On(worker.Events(), workers.EventJobRequest, func(_ context.Context, m *bus.JobRequestMessage) {
		question := fmt.Sprint(m.Payload["question"])
		slog.Info(fmt.Sprintf("Controller: received question '%s'", question))
		mu.Lock()
		jobID = m.JobID
		mu.Unlock()
		f := frames.NewLLMMessagesAppendFrame([]frames.Message{{Role: frames.RoleUser, Text: question}})
		f.RunLLM = true
		worker.QueueFrame(f)
	})

	events.On(agg.Assistant().Events(), aggregators.EventAssistantTurnStopped,
		func(ctx context.Context, message aggregators.AssistantTurnStopped) {
			// The aggregator raises this event at the end of every LLM response,
			// including the tool-call round that comes before the tool result and
			// has no spoken text. Skip those, so only the LLM's final response is
			// forwarded to the voice agent.
			if message.Content == "" {
				return
			}
			mu.Lock()
			id := jobID
			jobID = ""
			mu.Unlock()
			if id == "" {
				return
			}
			slog.Info(fmt.Sprintf("Controller: answering job %s", id[:min(8, len(id))]))
			if err := worker.SendJobResponse(ctx, id, map[string]any{"answer": message.Content},
				workers.JobResponseOptions{}); err != nil {
				slog.Warn("Controller: could not answer the job", "job", id, "err", err)
			}
		})

	return worker
}

// errNoPipelineWorker is reported when the tool runs outside a pipeline worker,
// which is what sends the job.
var errNoPipelineWorker = errors.New("the tool runs outside a pipeline worker")

// errNoAnswer is reported when the controller's response carries no answer.
var errNoAnswer = errors.New("the controller's response carries no answer")

// askController asks the temperature sensor controller anything about the
// sensor.
func askController(ctx context.Context, params llm.FunctionCallParams) error {
	var args struct {
		Question string `json:"question"`
	}
	if err := json.Unmarshal(params.Arguments, &args); err != nil {
		return err
	}
	worker, ok := params.PipelineWorker.(*pipeline.Worker)
	if !ok {
		return errNoPipelineWorker
	}
	slog.Info(fmt.Sprintf("Voice agent: forwarding to controller: '%s'", args.Question))
	job, err := worker.Job(ctx, "sensor-controller", jobcontext.Request{
		Payload: map[string]any{"question": args.Question},
		Timeout: 30 * time.Second,
	}, nil)
	if err != nil {
		return err
	}
	answer, ok := job.Response()["answer"].(string)
	if !ok {
		return errNoAnswer
	}
	return params.Result(ctx, answer, nil)
}

// voiceTools describes the voice agent's one tool to the LLM, carrying its
// handler.
func voiceTools() []frames.Tool {
	return []frames.Tool{{
		Name: "ask_controller",
		Description: "Ask the temperature sensor controller anything about the sensor.\n\n" +
			"Forward the user's request verbatim and speak back the answer.",
		Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"question": {
					"type": "string",
					"description": "The user's question or instruction to forward to the controller."
				}
			},
			"required": ["question"]
		}`),
		Handler:     llm.FunctionCallHandler(askController),
		TimeoutSecs: new(60.0),
	}}
}

func runBot(ctx context.Context, t transport.Transport, args runner.Arguments) error {
	slog.Info("Starting sensor-controller bot")

	// The voice agent: a standard transport, STT, LLM and TTS pipeline. The only
	// tool the voice LLM has is ask_controller: it knows nothing about
	// temperatures, trends or response rates.
	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})
	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc", // Jacqueline
	})

	model := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	model.SetSystemInstruction("You are a friendly voice assistant with access to a temperature " +
		"sensor controller. For ANY request about the temperature (" +
		"reading it, adjusting it, checking trends, changing how fast it " +
		"responds), call the ask_controller tool. Forward the user's " +
		"request verbatim. Then speak the controller's answer back. " +
		"Keep responses brief; do not add extra commentary.")

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	convo.SetTools(voiceTools())
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),
		stt,
		agg.User(),
		model,
		tts,
		t.Output(),
		agg.Assistant(),
	)

	worker := pipeline.NewWorker(p, pipeline.WorkerConfig{
		Name: "voice-agent",
		Params: pipeline.Params{
			EnableMetrics:      true,
			EnableUsageMetrics: true,
		},
		IdleTimeout:             args.Base().PipelineIdleTimeout,
		ProcessorUnusablePolicy: pipeline.UnusableEnd,
	})

	handleSigint := args.Base().HandleSigint
	r := workers.NewRunner(workers.RunnerConfig{HandleInterrupt: &handleSigint})

	r.AddWorkers(ctx, buildSensorController(), worker)

	events.On(t.Events(), transport.EventClientConnected, func(context.Context, any) {
		slog.Info("Client connected")
		convo.AddMessage(frames.Message{
			Role: frames.RoleDeveloper,
			Text: "Greet the user and let them know you can read or adjust a " +
				"temperature sensor on their behalf.",
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
