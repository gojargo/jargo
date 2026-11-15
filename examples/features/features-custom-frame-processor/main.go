// Command features-custom-frame-processor writes a frame processor of its own:
// one that pretty-prints every MetricsFrame passing through it.
//
// The processor sits at the end of the pipeline and pushes every frame on,
// metrics or not, which a processor must always do.
//
//	DEEPGRAM_API_KEY=… CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-custom-frame-processor
//
// Then open http://localhost:7860 and connect.
package main

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"strings"

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

// formatMetrics lists each measurement's type and fields, one per line, with
// a struct-valued field's own fields indented under it.
func formatMetrics(metrics []frames.MetricsData, indent int) string {
	tab := strings.Repeat("\t", indent)
	var lines []string
	for _, metric := range metrics {
		v := reflect.Indirect(reflect.ValueOf(metric))
		lines = append(lines, tab+v.Type().Name())
		for name, value := range fields(v) {
			if value.Kind() == reflect.Struct {
				lines = append(lines, fmt.Sprintf("%s\t%s=%s", tab, name, value.Type().Name()))
				for k, sub := range fields(value) {
					lines = append(lines, fmt.Sprintf("%s\t\t%s=%s", tab, k, repr(sub)))
				}
			} else {
				lines = append(lines, fmt.Sprintf("%s\t%s=%s", tab, name, repr(value)))
			}
		}
	}
	return strings.Join(lines, "\n")
}

// fields yields a struct's exported fields in order, with the fields of an
// embedded struct in place of the struct itself.
func fields(v reflect.Value) func(yield func(string, reflect.Value) bool) {
	return func(yield func(string, reflect.Value) bool) {
		for i := range v.NumField() {
			f := v.Type().Field(i)
			if !f.IsExported() {
				continue
			}
			if f.Anonymous && v.Field(i).Kind() == reflect.Struct {
				for name, value := range fields(v.Field(i)) {
					if !yield(name, value) {
						return
					}
				}
				continue
			}
			if !yield(f.Name, v.Field(i)) {
				return
			}
		}
	}
}

// repr renders a field value, quoting strings.
func repr(v reflect.Value) string {
	if v.Kind() == reflect.String {
		return fmt.Sprintf("%q", v.String())
	}
	return fmt.Sprintf("%v", v.Interface())
}

// metricsFrameLogger formats and logs every MetricsFrame.
type metricsFrameLogger struct {
	*processor.Base
}

func newMetricsFrameLogger() *metricsFrameLogger {
	p := &metricsFrameLogger{}
	p.Base = processor.New("MetricsFrameLogger", p)
	return p
}

// ProcessFrame logs metrics frames and pushes every frame on.
func (p *metricsFrameLogger) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := p.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	if m, ok := f.(*frames.MetricsFrame); ok {
		slog.Info(m.Name() + "\n    " + formatMetrics(m.Data, 0))
	}

	// Always push every frame.
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

	stt := deepgram.NewSTT(deepgram.Config{APIKey: os.Getenv("DEEPGRAM_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey:  os.Getenv("CARTESIA_API_KEY"),
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction(systemInstruction)

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	metricsFrameProcessor := newMetricsFrameLogger()

	p := pipeline.New(
		t.Input(),             // Transport user input
		stt,                   // STT
		agg.User(),            // User responses
		llm,                   // LLM
		tts,                   // TTS
		t.Output(),            // Transport bot output
		agg.Assistant(),       // Assistant spoken responses
		metricsFrameProcessor, // Pretty-print metrics frames
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

	events.On(t.Events(), transport.EventClientConnected, func(_ context.Context, client any) {
		slog.Info("Client connected", "client", client)
		// Kick off the conversation.
		convo.AddMessage(frames.Message{Role: frames.RoleDeveloper, Text: "Please introduce yourself to the user."})
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
