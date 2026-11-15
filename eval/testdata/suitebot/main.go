// Command suitebot is the tiny bot the suite's tests build and spawn: it serves
// the harness on /ws at --port and echoes each user turn, recording the
// --runner-body it was given in its reply.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/gojargo/jargo/eval"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/processor/aggregators"
)

func main() {
	port := flag.String("port", "7860", "port to serve the harness on")
	body := flag.String("runner-body", "", "runner body file")
	flag.Parse()
	prefix := "echo: "
	if *body != "" {
		if data, err := os.ReadFile(*body); err == nil {
			prefix = "echo (" + string(data) + "): "
		}
	}
	mux := http.NewServeMux()
	mux.Handle("/ws", eval.Handler(func(in, out processor.Processor) *pipeline.Worker {
		agg := aggregators.New(frames.NewLLMContext("test"))
		return pipeline.NewWorker(pipeline.New(in, agg.User(), newEcho(prefix), out, agg.Assistant()),
			pipeline.WorkerConfig{})
	}))
	srv := &http.Server{Addr: "localhost:" + *port, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(srv.ListenAndServe())
}

// echo answers each turn with the user's last message.
type echo struct {
	*processor.Base
	prefix string
}

func newEcho(prefix string) *echo {
	e := &echo{prefix: prefix}
	e.Base = processor.New("Echo", e)
	return e
}

func (e *echo) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	cf, ok := f.(*frames.LLMContextFrame)
	if !ok {
		return e.PushFrame(ctx, f, dir)
	}
	last := ""
	if msgs := cf.Context.Messages(); len(msgs) > 0 {
		last = msgs[len(msgs)-1].Text
	}
	if err := e.PushFrame(ctx, frames.NewLLMFullResponseStartFrame(), processor.Downstream); err != nil {
		return err
	}
	if err := e.PushFrame(ctx, frames.NewLLMTextFrame(e.prefix+last), processor.Downstream); err != nil {
		return err
	}
	return e.PushFrame(ctx, frames.NewLLMFullResponseEndFrame(), processor.Downstream)
}
