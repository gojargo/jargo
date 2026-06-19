// Command echo is a WebRTC echo bot built on jargo: it receives microphone
// audio from a browser, runs it through a pipeline, and sends it back so you
// hear yourself.
//
// Run it, open http://localhost:8080, click start, and allow the microphone.
package main

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"io/fs"
	"log"
	"log/slog"
	"net/http"

	"github.com/gojargo/jargo/audio/opus"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/rtvi"
	"github.com/gojargo/jargo/transport"
	"github.com/gojargo/jargo/transport/pionrtc"
	"github.com/pion/webrtc/v4"
)

//go:embed static
var staticFiles embed.FS

func main() {
	const addr = ":8080"

	static, err := fs.Sub(staticFiles, "static")
	if err != nil {
		log.Fatal(err)
	}
	http.Handle("/", http.FileServer(http.FS(static)))
	http.HandleFunc("/offer", handleOffer)

	slog.Info("jargo echo bot listening", "url", "http://localhost"+addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

// handleOffer completes WebRTC signaling and starts an echo pipeline for the
// new connection.
func handleOffer(w http.ResponseWriter, r *http.Request) {
	var offer webrtc.SessionDescription
	if err := json.NewDecoder(r.Body).Decode(&offer); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	conn, err := pionrtc.NewConnection()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	answer, err := conn.Answer(offer)
	if err != nil {
		_ = conn.Close()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	go runEcho(conn)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(answer); err != nil {
		slog.Error("write answer", "err", err)
	}
}

// runEcho runs a pipeline that echoes received audio back over the connection
// until the client disconnects.
func runEcho(conn *pionrtc.Connection) {
	params := transport.DefaultParams()
	params.AudioInSampleRate = opus.SampleRate
	params.AudioOutSampleRate = opus.SampleRate

	t := pionrtc.NewTransport(conn, params)
	// The RTVI processor handles the client handshake (client-ready -> bot-ready)
	// and reports pipeline events to the client over the data channel.
	pipe := pipeline.New(t.Input(), rtvi.NewProcessor(), newEcho(), t.Output())
	task := pipeline.NewTask(pipe, pipeline.TaskParams{
		AudioInSampleRate:  opus.SampleRate,
		AudioOutSampleRate: opus.SampleRate,
	})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-conn.Done()
		cancel()
	}()

	slog.Info("echo pipeline started")
	if err := task.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
		slog.Error("echo pipeline ended", "err", err)
	}
	_ = conn.Close()
	slog.Info("echo pipeline stopped")
}

// echoProcessor turns each received audio frame into an outgoing audio frame,
// so the pipeline sends the caller's voice straight back.
type echoProcessor struct {
	*processor.Base
}

func newEcho() *echoProcessor {
	e := &echoProcessor{}
	e.Base = processor.New("Echo", e)
	return e
}

func (e *echoProcessor) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := e.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}
	if in, ok := f.(*frames.InputAudioRawFrame); ok {
		out := frames.NewOutputAudioRawFrame(in.Audio, in.SampleRate, in.NumChannels)
		return e.PushFrame(ctx, out, processor.Downstream)
	}
	return e.PushFrame(ctx, f, dir)
}
