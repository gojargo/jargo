// Command features-pronunciation-dictionary says medication names right with
// a pronunciation dictionary.
//
// It loads a pronunciation dictionary and gives it to the TTS service through
// its pronunciation text transform. The file holds, for each evaluated
// service, the model the pronunciations were measured on and the words to
// hint:
//
//	{
//	  "services": {
//	    "cartesia": {
//	      "model": "sonic-3.6",
//	      "ipa": {"Adalimumab": "ˌædəˈlɪmumæb"},
//	      "plain": ["Abilify"],
//	      "unresolved": ["Carisoprodol"]
//	    }
//	  }
//	}
//
// PronunciationTransformIPA is a method of every TTS service, and each service
// writes the pronunciation in its own markup: Cartesia turns "ˌædəˈlɪmumæb"
// into <<ˌ|æ|d|ə|l|ˈ|ɪ|m|u|m|æ|b>>. The "plain" words are already said right
// without a hint, and the "unresolved" ones were never said right, so neither
// is passed on.
//
// The bot is a pharmacy assistant that talks about the medications in the
// file.
//
//	CARTESIA_API_KEY=… OPENAI_API_KEY=… \
//	  go run ./examples/features/features-pronunciation-dictionary
//
// Then open http://localhost:7860 and connect.
package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"

	"github.com/gojargo/jargo/audio/vad"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/provider/cartesia"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/runner"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/wsserver"
	"github.com/gojargo/jargo/transport/wsserver/evalws"
	"github.com/gojargo/jargo/utils/events"
	"github.com/gojargo/jargo/workers"
)

// pronunciations is the pronunciation dictionary.
//
//go:embed pronunciations.json
var pronunciations []byte

// errNoService is returned for a service the dictionary has no entry for.
var errNoService = errors.New("no pronunciations for the service")

// dictionary is one service's entry in the pronunciation dictionary.
type dictionary struct {
	// Model is the model the pronunciations were measured on.
	Model string `json:"model"`
	// IPA maps each word to hint to its pronunciation.
	IPA map[string]string `json:"ipa"`
	// hinted are the words of IPA, in the file's order.
	hinted []string
}

// loadDictionary reads the named service's entry from the dictionary.
func loadDictionary(data []byte, service string) (dictionary, error) {
	var file struct {
		Services map[string]json.RawMessage `json:"services"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		return dictionary{}, err
	}
	raw, ok := file.Services[service]
	if !ok {
		return dictionary{}, fmt.Errorf("%w: %q", errNoService, service)
	}
	var d dictionary
	if err := json.Unmarshal(raw, &d); err != nil {
		return dictionary{}, err
	}
	var entry struct {
		IPA json.RawMessage `json:"ipa"`
	}
	if err := json.Unmarshal(raw, &entry); err != nil {
		return dictionary{}, err
	}
	hinted, err := objectKeys(entry.IPA)
	if err != nil {
		return dictionary{}, err
	}
	d.hinted = hinted
	return d, nil
}

// objectKeys returns the keys of a JSON object in the order they are written.
func objectKeys(object json.RawMessage) ([]string, error) {
	dec := json.NewDecoder(bytes.NewReader(object))
	if _, err := dec.Token(); err != nil { // the opening brace
		return nil, err
	}
	var keys []string
	for dec.More() {
		tok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, _ := tok.(string)
		keys = append(keys, key)
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return nil, err
		}
	}
	return keys, nil
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

	dict, err := loadDictionary(pronunciations, "cartesia")
	if err != nil {
		return err
	}
	hinted := dict.hinted

	stt := cartesia.NewSTT(cartesia.STTConfig{APIKey: os.Getenv("CARTESIA_API_KEY")})

	tts := cartesia.NewTTS(cartesia.Config{
		APIKey: os.Getenv("CARTESIA_API_KEY"),
		// The pronunciations were measured on this model and voice.
		Model:   dict.Model,
		VoiceID: "86e30c1d-714b-4074-a1f2-1cb6b552fb49",
	})
	// Each word is matched whole and case-insensitively, wherever the LLM says
	// it.
	tts.SetTextTransformers(tts.PronunciationTransformIPA(dict.IPA))

	llm := chat.NewLLM(chat.LLMConfig{APIKey: os.Getenv("OPENAI_API_KEY")})
	llm.SetSystemInstruction("You are a pharmacy assistant in a voice conversation. Your responses are " +
		"spoken aloud, so keep them short and avoid formatting that can't be " +
		"spoken. You can talk about these medications, spelled exactly as written " +
		"here: " + strings.Join(hinted, ", ") + ".")

	silero, err := vad.NewSilero()
	if err != nil {
		return err
	}
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, aggregators.WithVAD(aggregators.VADConfig{Analyzer: silero}))

	p := pipeline.New(
		t.Input(),       // Transport user input
		stt,             // STT
		agg.User(),      // User responses
		llm,             // LLM
		tts,             // TTS
		t.Output(),      // Transport bot output
		agg.Assistant(), // Assistant spoken responses
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
			Text: "Greet the caller and offer to help with their prescriptions, naming a " +
				"few of the medications you know, such as " + strings.Join(hinted[:min(3, len(hinted))], ", ") + ".",
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
