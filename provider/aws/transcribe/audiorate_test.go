package transcribe

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/stt"
)

// The rate Transcribe is sent the user's audio at. It takes 8 or 16 kHz only,
// so audio at any other rate has to be resampled before it goes out, not just
// declared at a rate it is not at.

// tone is a 220 Hz sine of n samples, as 16-bit mono PCM at rate.
func tone(rate, n int) []byte {
	pcm := make([]byte, 2*n)
	for i := range n {
		v := int16(math.Sin(2*math.Pi*220*float64(i)/float64(rate)) * 8000)
		binary.LittleEndian.PutUint16(pcm[2*i:], uint16(v))
	}
	return pcm
}

// recordingConnector is the Transcribe connector with the session swapped for
// one that records what it is sent, and the rate it was opened for.
type recordingConnector struct {
	*connector
	sent chan []byte

	mu    sync.Mutex
	rates []int
}

func (c *recordingConnector) Connect(ctx context.Context, sampleRate int) (stt.Stream, error) {
	c.mu.Lock()
	c.rates = append(c.rates, sampleRate)
	c.mu.Unlock()
	return &recordingStream{ctx: ctx, sent: c.sent}, nil
}

func (c *recordingConnector) opened() []int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]int(nil), c.rates...)
}

type recordingStream struct {
	ctx  context.Context //nolint:containedctx // the session context, set on dial
	sent chan []byte
}

func (s *recordingStream) Send(audio []byte) error {
	s.sent <- append([]byte(nil), audio...)
	return nil
}

func (s *recordingStream) Recv() ([]stt.Result, error) {
	<-s.ctx.Done()
	return nil, s.ctx.Err()
}

func (s *recordingStream) Close() error { return nil }

// transcribeSends streams one clip at rate through a Transcribe service running
// at that rate, and returns what reached the session and the rate it was
// opened for.
func transcribeSends(t *testing.T, rate int, clip []byte) (sent []byte, opened []int) {
	t.Helper()
	conn := &recordingConnector{
		connector: &connector{cfg: Config{Language: defaultLanguage}},
		sent:      make(chan []byte, 4),
	}
	svc := stt.NewStream("TranscribeSTT", conn, rate)
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()
	task.QueueFrame(frames.NewInputAudioRawFrame(clip, rate, 1))
	select {
	case sent = <-conn.sent:
	case <-time.After(3 * time.Second):
		t.Fatal("no audio reached the session")
	}
	task.StopWhenDone()
	select {
	case <-done:
	case <-time.After(3 * time.Second):
		t.Fatal("the pipeline did not shut down")
	}
	return sent, conn.opened()
}

// Audio at a rate Transcribe has no format for goes out resampled to 16 kHz,
// on a session opened for 16 kHz.
func TestTranscribeResamplesARateItDoesNotTakeTo16kHz(t *testing.T) {
	t.Parallel()
	clip := tone(24000, 4800)

	sent, opened := transcribeSends(t, 24000, clip)

	r, err := resample.New(24000, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if want := r.Process(clip); !bytes.Equal(sent, want) {
		t.Errorf("sent %d bytes, want the clip resampled to 16 kHz (%d bytes)", len(sent), len(want))
	}
	if len(opened) != 1 || opened[0] != 16000 {
		t.Errorf("sessions opened for %v Hz, want one for 16000 Hz", opened)
	}
}

// Audio at a rate Transcribe takes goes out untouched.
func TestTranscribeSends8And16kHzAsIs(t *testing.T) {
	t.Parallel()
	for _, rate := range []int{8000, 16000} {
		clip := tone(rate, rate/10)

		sent, opened := transcribeSends(t, rate, clip)

		if !bytes.Equal(sent, clip) {
			t.Errorf("at %d Hz the clip was altered on its way out, want it sent as it is", rate)
		}
		if len(opened) != 1 || opened[0] != rate {
			t.Errorf("at %d Hz sessions opened for %v Hz, want one for %d Hz", rate, opened, rate)
		}
	}
}

// With no keys given and none in the environment, shared files or instance
// metadata, the session cannot be signed: connecting fails with an error that
// says credentials are what is missing, rather than opening anything.
func TestTranscribeWithoutAnyCredentialsFailsToConnect(t *testing.T) {
	none := filepath.Join(t.TempDir(), "none")
	for k, v := range map[string]string{
		"AWS_ACCESS_KEY_ID":                      "",
		"AWS_SECRET_ACCESS_KEY":                  "",
		"AWS_SESSION_TOKEN":                      "",
		"AWS_PROFILE":                            "",
		"AWS_SHARED_CREDENTIALS_FILE":            none,
		"AWS_CONFIG_FILE":                        none,
		"AWS_EC2_METADATA_DISABLED":              "true",
		"AWS_CONTAINER_CREDENTIALS_FULL_URI":     "",
		"AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "",
		"AWS_WEB_IDENTITY_TOKEN_FILE":            "",
	} {
		t.Setenv(k, v)
	}
	c := &connector{cfg: Config{Region: "us-east-1", Language: defaultLanguage}}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	stream, err := c.Connect(ctx, 16000)
	if err == nil {
		_ = stream.Close()
		t.Fatal("connected with no credentials anywhere, want an error")
	}
	if !strings.Contains(strings.ToLower(err.Error()), "credentials") {
		t.Errorf("error = %v, want one naming the missing credentials", err)
	}
}
