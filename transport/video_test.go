package transport_test

import (
	"context"
	"errors"
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/event"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/transport"
)

func TestVideoInSourcesDefaultToNone(t *testing.T) {
	if p := (transport.Params{}); len(p.VideoInSources) != 0 {
		t.Errorf("sources %v, want none", p.VideoInSources)
	}
}

func TestVideoInSourcesKeyedBySource(t *testing.T) {
	p := transport.Params{
		VideoInEnabled: true,
		VideoInSources: map[string]transport.VideoInSourceParams{
			transport.CameraVideoSource: {OnRequestOnly: true},
			transport.ScreenVideoSource: {Framerate: 1},
		},
	}
	if err := p.Validate(); err != nil {
		t.Fatal(err)
	}
	if !p.VideoInSources["camera"].OnRequestOnly || p.VideoInSources["screenVideo"].Rate() != 1 {
		t.Errorf("sources %+v", p.VideoInSources)
	}
}

func TestVideoInSourceDefaultFramerate(t *testing.T) {
	if got := (transport.VideoInSourceParams{}).Rate(); got != 30 {
		t.Errorf("rate %d, want 30", got)
	}
}

func TestVideoInSourceParamsRejected(t *testing.T) {
	for name, sp := range map[string]transport.VideoInSourceParams{
		"negative framerate":             {Framerate: -1},
		"framerate with on request only": {Framerate: 5, OnRequestOnly: true},
	} {
		t.Run(name, func(t *testing.T) {
			p := transport.Params{
				VideoInEnabled: true,
				VideoInSources: map[string]transport.VideoInSourceParams{"camera": sp},
			}
			if err := p.Validate(); err == nil {
				t.Error("accepted")
			}
		})
	}
}

func TestVideoInSourcesRequireVideoInEnabled(t *testing.T) {
	p := transport.Params{VideoInSources: map[string]transport.VideoInSourceParams{"camera": {}}}
	if err := p.Validate(); err == nil {
		t.Error("sources without video input were accepted")
	}
}

// sourcesInput is an input transport that captures the camera and screen share
// from VideoInSources.
type sourcesInput struct {
	*transport.BaseInput
}

func newSourcesInput(p transport.Params) *sourcesInput {
	in := &sourcesInput{}
	in.BaseInput = transport.NewBaseInput("SourcesInput", p, in)
	return in
}

func (*sourcesInput) SupportsVideoInSource(source string) bool {
	return source == transport.CameraVideoSource || source == transport.ScreenVideoSource
}

func sourcesTransport(p transport.Params) (*sourcesInput, *fakeOutput) {
	return newSourcesInput(p), newFakeOutput(p)
}

func TestScreenInWhenSourcesIncludeScreenVideo(t *testing.T) {
	in, out := sourcesTransport(transport.Params{
		VideoInEnabled: true,
		VideoInSources: map[string]transport.VideoInSourceParams{"camera": {}, "screenVideo": {}},
	})
	caps := workerOf(t, pipeline.WorkerConfig{}, in, out).Capabilities()
	if caps.ScreenIn == nil || !*caps.ScreenIn {
		t.Errorf("screen_in %v, want true", caps.ScreenIn)
	}
}

func TestNoScreenInWhenSourcesExcludeScreenVideo(t *testing.T) {
	in, out := sourcesTransport(transport.Params{
		VideoInEnabled: true,
		VideoInSources: map[string]transport.VideoInSourceParams{"camera": {}},
	})
	caps := workerOf(t, pipeline.WorkerConfig{}, in, out).Capabilities()
	if caps.ScreenIn == nil || *caps.ScreenIn {
		t.Errorf("screen_in %v, want false", caps.ScreenIn)
	}
}

func TestScreenInUnknownWithoutSources(t *testing.T) {
	in, out := sourcesTransport(transport.Params{VideoInEnabled: true})
	caps := workerOf(t, pipeline.WorkerConfig{}, in, out).Capabilities()
	if caps.VideoIn == nil || !*caps.VideoIn {
		t.Errorf("video_in %v, want true", caps.VideoIn)
	}
	if caps.ScreenIn != nil {
		t.Errorf("screen_in %v, want unknown", *caps.ScreenIn)
	}
}

// warnings records the warnings logged while fn runs.
func warnings(fn func()) []string {
	rec := &recordingHandler{}
	prev := slog.Default()
	slog.SetDefault(slog.New(rec))
	defer slog.SetDefault(prev)
	fn()
	return rec.messages()
}

type recordingHandler struct {
	mu   sync.Mutex
	msgs []string
}

func (*recordingHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *recordingHandler) Handle(_ context.Context, r slog.Record) error {
	if r.Level < slog.LevelWarn {
		return nil
	}
	var b strings.Builder
	b.WriteString(r.Message)
	r.Attrs(func(a slog.Attr) bool {
		b.WriteString(" " + a.String())
		return true
	})
	h.mu.Lock()
	h.msgs = append(h.msgs, b.String())
	h.mu.Unlock()
	return nil
}

func (h *recordingHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *recordingHandler) WithGroup(string) slog.Handler      { return h }

func (h *recordingHandler) messages() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.msgs)
}

func TestUnsupportedSourcesWarnedAndScreenInUnknown(t *testing.T) {
	p := transport.Params{
		VideoInEnabled: true,
		VideoInSources: map[string]transport.VideoInSourceParams{"screenVideo": {}},
	}
	var in *fakeInput
	got := warnings(func() { in = newFakeInput(p, 0) })
	if len(got) != 1 || !strings.Contains(got[0], "screenVideo") {
		t.Fatalf("warnings %q, want one naming screenVideo", got)
	}
	caps := workerOf(t, pipeline.WorkerConfig{}, in, newFakeOutput(p)).Capabilities()
	if caps.ScreenIn != nil {
		t.Errorf("screen_in %v, want unknown", *caps.ScreenIn)
	}
}

func TestSupportedSourcesNotWarned(t *testing.T) {
	p := transport.Params{
		VideoInEnabled: true,
		VideoInSources: map[string]transport.VideoInSourceParams{"camera": {}},
	}
	if got := warnings(func() { sourcesTransport(p) }); len(got) != 0 {
		t.Errorf("warnings %q, want none", got)
	}
}

func TestTransportsNestedInParallelPipelineReportVideo(t *testing.T) {
	in, out := sourcesTransport(transport.Params{VideoInEnabled: true, VideoOutEnabled: true})
	par, err := pipeline.NewParallel([]processor.Processor{in}, []processor.Processor{out})
	if err != nil {
		t.Fatal(err)
	}
	caps := workerOf(t, pipeline.WorkerConfig{}, par).Capabilities()
	if caps.VideoIn == nil || !*caps.VideoIn || caps.VideoOut == nil || !*caps.VideoOut {
		t.Errorf("capabilities %+v, want video both ways", caps)
	}
}

func TestInvalidParamsFailTheStart(t *testing.T) {
	p := transport.Params{VideoInSources: map[string]transport.VideoInSourceParams{"camera": {}}}
	worker := workerOf(t, pipeline.WorkerConfig{}, newFakeInput(p, 0))
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := worker.Run(ctx); err == nil {
		t.Error("a transport with invalid params started")
	}
}

func TestAnswerImageRequestsReportsTheError(t *testing.T) {
	in := newFakeInput(transport.Params{}, 0)
	var results []string
	request := frames.NewUserImageRequestFrame("p1")
	request.ResultCallback = func(_ context.Context, result string, _ *frames.FunctionCallResultProperties) error {
		results = append(results, result)
		return nil
	}
	silent := frames.NewUserImageRequestFrame("p1")

	in.AnswerImageRequests(context.Background(),
		[]*frames.UserImageRequestFrame{request, silent}, "no video from p1")

	if len(results) != 1 || results[0] != `{"error":"no video from p1"}` {
		t.Errorf("results %q, want the error", results)
	}
}

// errVideoBufferFull is what videoOutput reports once a test stops reading.
var errVideoBufferFull = errors.New("video buffer full")

// videoOutput records the video frames written to it.
type videoOutput struct {
	*transport.BaseOutput
	videos chan *frames.OutputImageRawFrame
}

func newVideoOutput(p transport.Params) *videoOutput {
	o := &videoOutput{videos: make(chan *frames.OutputImageRawFrame, 256)}
	o.BaseOutput = transport.NewBaseOutput("VideoOutput", p, o)
	return o
}

func (o *videoOutput) WriteVideoFrame(ctx context.Context, f *frames.OutputImageRawFrame) (bool, error) {
	select {
	case o.videos <- f:
		return true, nil
	case <-ctx.Done():
		return false, ctx.Err()
	default:
		return false, errVideoBufferFull
	}
}

// runVideoOutput runs o in a pipeline, reporting the frames that reach downstream.
func runVideoOutput(t *testing.T, o *videoOutput) (*pipeline.Worker, chan frames.Frame, func()) {
	t.Helper()
	down := make(chan frames.Frame, 256)
	task := pipeline.NewWorker(pipeline.New(o), pipeline.WorkerConfig{
		ReachedDownstreamFilter: pipeline.AnyFrame,
	})
	event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
		select {
		case down <- f:
		default:
		}
	})
	runDone := make(chan error, 1)
	go func() { runDone <- task.Run(context.Background()) }()
	return task, down, func() {
		task.StopWhenDone()
		<-runDone
	}
}

func videoParams() transport.Params {
	p := transport.DefaultParams()
	p.AudioOutSampleRate = 16000
	p.VideoOutEnabled = true
	p.VideoOutWidth, p.VideoOutHeight = 4, 2
	p.VideoOutFramerate = 100
	return p
}

func nextVideo(t *testing.T, o *videoOutput) *frames.OutputImageRawFrame {
	t.Helper()
	select {
	case f := <-o.videos:
		return f
	case <-time.After(3 * time.Second):
		t.Fatal("no video frame was written")
		return nil
	}
}

func TestBaseOutputRepeatsAnImageResizedToTheVideoSize(t *testing.T) {
	o := newVideoOutput(videoParams())
	task, _, stop := runVideoOutput(t, o)
	defer stop()

	task.QueueFrame(frames.NewOutputImageRawFrame(make([]byte, 2*1*3), 2, 1, "RGB"))

	for range 3 {
		f := nextVideo(t, o)
		if f.Width != 4 || f.Height != 2 || len(f.Image) != 4*2*3 || f.Format != "RGB" {
			t.Fatalf("frame %v with %d bytes, want 4x2 RGB", f, len(f.Image))
		}
	}
}

func TestBaseOutputCyclesASprite(t *testing.T) {
	o := newVideoOutput(videoParams())
	task, _, stop := runVideoOutput(t, o)
	defer stop()

	first := frames.NewOutputImageRawFrame(make([]byte, 4*2*3), 4, 2, "RGB")
	second := frames.NewOutputImageRawFrame(make([]byte, 4*2*3), 4, 2, "RGB")
	task.QueueFrame(frames.NewSpriteFrame([]*frames.OutputImageRawFrame{first, second}))

	got := []*frames.OutputImageRawFrame{nextVideo(t, o), nextVideo(t, o), nextVideo(t, o)}
	if got[0] == got[1] || got[0] != got[2] {
		t.Errorf("the sprite was not cycled in turn")
	}
}

func TestBaseOutputSendsLiveImagesInOrder(t *testing.T) {
	p := videoParams()
	p.VideoOutIsLive = true
	o := newVideoOutput(p)
	task, _, stop := runVideoOutput(t, o)
	defer stop()

	var sent []*frames.OutputImageRawFrame
	for range 3 {
		img := frames.NewOutputImageRawFrame(make([]byte, 4*2*3), 4, 2, "RGB")
		sent = append(sent, img)
		task.QueueFrame(img)
	}
	for i, want := range sent {
		if got := nextVideo(t, o); got != want {
			t.Errorf("frame %d out of order", i)
		}
	}
	// Live video sends each image once.
	select {
	case <-o.videos:
		t.Error("a live image was sent twice")
	case <-time.After(100 * time.Millisecond):
	}
}

func TestBaseOutputPassesAssistantImagesOn(t *testing.T) {
	o := newVideoOutput(videoParams())
	task, down, stop := runVideoOutput(t, o)
	defer stop()

	img := frames.NewAssistantImageRawFrame(make([]byte, 4*2*3), 4, 2, "RGB")
	task.QueueFrame(img)

	nextVideo(t, o)
	deadline := time.After(3 * time.Second)
	for {
		select {
		case f := <-down:
			if f == img {
				return
			}
		case <-deadline:
			t.Fatal("the assistant image did not reach downstream")
		}
	}
}

func TestBaseOutputDropsImagesWithoutVideoOut(t *testing.T) {
	p := videoParams()
	p.VideoOutEnabled = false
	o := newVideoOutput(p)
	task, down, stop := runVideoOutput(t, o)

	img := frames.NewOutputImageRawFrame(make([]byte, 4*2*3), 4, 2, "RGB")
	task.QueueFrame(img)
	stop()

	select {
	case <-o.videos:
		t.Error("video was written by a transport that sends none")
	default:
	}
	for len(down) > 0 {
		if <-down == img {
			t.Error("the image went on past the output")
		}
	}
}

func TestBaseOutputReportsVideoOut(t *testing.T) {
	caps := newVideoOutput(videoParams()).Capabilities()
	if caps.VideoOut == nil || !*caps.VideoOut {
		t.Errorf("video_out %v, want true", caps.VideoOut)
	}
}

func TestBaseInputPushesVideoOnlyWhenEnabled(t *testing.T) {
	for _, enabled := range []bool{true, false} {
		p := transport.DefaultParams()
		p.VideoInEnabled = enabled
		in := newFakeInput(p, 0)
		down := make(chan frames.Frame, 16)
		task := pipeline.NewWorker(pipeline.New(in), pipeline.WorkerConfig{
			ReachedDownstreamFilter: pipeline.AnyFrame,
		})
		event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
			if _, ok := f.(*frames.UserImageRawFrame); ok {
				down <- f
			}
		})
		started := make(chan struct{})
		event.On(&task.Registry, pipeline.EventFrameReachedDownstream, func(_ context.Context, f frames.Frame) {
			if _, ok := f.(*frames.StartFrame); ok {
				close(started)
			}
		})
		runDone := make(chan error, 1)
		go func() { runDone <- task.Run(context.Background()) }()
		<-started

		in.PushVideoFrame(context.Background(), frames.NewUserImageRawFrame("u", make([]byte, 3), 1, 1, "RGB"))

		select {
		case <-down:
			if !enabled {
				t.Error("video was pushed with video input off")
			}
		case <-time.After(300 * time.Millisecond):
			if enabled {
				t.Error("video was not pushed with video input on")
			}
		}
		task.StopWhenDone()
		<-runDone
	}
}

func TestBaseTransportDefaults(t *testing.T) {
	b := transport.BaseTransport{Name: "Test"}
	var id string
	got := warnings(func() { id = b.GetClientID(struct{}{}) })
	if id != "" || len(got) != 1 {
		t.Errorf("id %q with warnings %q, want none with a warning", id, got)
	}
	got = warnings(func() {
		if err := b.CaptureParticipantVideo(context.Background(), "p1", 30, "camera", false); err != nil {
			t.Error(err)
		}
	})
	if len(got) != 1 {
		t.Errorf("warnings %q, want one", got)
	}
}
