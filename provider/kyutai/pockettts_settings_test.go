package kyutai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/settings"
)

// Ported from upstream (test_pocket_tts_voice_update). A voice set while the
// pipeline runs is the one the next sentence is spoken in.
func TestPocketTTSVoiceUpdate(t *testing.T) {
	var mu sync.Mutex
	var voices []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		voices = append(voices, r.FormValue("voice_url"))
		mu.Unlock()
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write(append(wavHeader(24000), make([]byte, 64)...))
	}))
	t.Cleanup(srv.Close)

	svc := NewPocketTTS(PocketTTSConfig{BaseURL: srv.URL, Voice: "alba", SampleRate: 24000})
	task := pipeline.NewWorker(pipeline.New(svc), pipeline.WorkerConfig{})
	done := make(chan error, 1)
	go func() { done <- task.Run(context.Background()) }()

	task.QueueFrame(frames.NewTTSSpeakFrame("First voice."))
	time.Sleep(200 * time.Millisecond)
	task.QueueFrame(frames.NewTTSUpdateSettingsFrame(&settings.TTS{Voice: settings.Set("jane")}))
	time.Sleep(100 * time.Millisecond)
	task.QueueFrame(frames.NewTTSSpeakFrame("Second voice."))
	time.Sleep(300 * time.Millisecond)
	task.StopWhenDone()
	<-done

	mu.Lock()
	defer mu.Unlock()
	if want := []string{"alba", "jane"}; !slices.Equal(voices, want) {
		t.Errorf("voices asked for = %v, want %v", voices, want)
	}
}
