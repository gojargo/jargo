package eval

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/service/tts"
	"github.com/gojargo/jargo/transport/wsserver/rtviws"
)

const (
	// audioChunkMS is the size of each streamed mic frame — the granularity a live
	// transport delivers, and what VAD and turn models consume.
	audioChunkMS = 20
	// silenceTailMS is how much trailing silence follows a user turn's speech, so
	// the bot's VAD registers the end of the turn.
	silenceTailMS = 1000
)

// errNoUserAudio is returned when the user TTS produced no audio for a turn.
//
//nolint:gochecknoglobals // sentinel error
var errNoUserAudio = errors.New("eval: user TTS produced no audio")

// sendUserAudio synthesizes text and streams it to the bot as microphone audio:
// ~20ms frames at real-time cadence, then a tail of silence so the bot's VAD
// ends the turn. Real-time pacing matters — flooding the whole utterance at once
// breaks VAD windows and turn-detecting STTs.
func (s *session) sendUserAudio(ctx context.Context, text string) error {
	pcm, rate, err := synthesize(ctx, s.userTTS, text)
	if err != nil {
		return err
	}
	if rate == 0 || len(pcm) == 0 {
		return errNoUserAudio
	}

	bytesPerChunk := (rate * audioChunkMS / 1000) * 2 // 16-bit mono
	ticker := time.NewTicker(audioChunkMS * time.Millisecond)
	defer ticker.Stop()

	send := func(chunk []byte) error {
		if err := s.client.send(ctx, rtviws.RawAudio(chunk, rate, 1)); err != nil {
			return err
		}
		select {
		case <-ticker.C:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	for off := 0; off < len(pcm); off += bytesPerChunk {
		end := min(off+bytesPerChunk, len(pcm))
		if err := send(pcm[off:end]); err != nil {
			return err
		}
	}

	silence := make([]byte, bytesPerChunk)
	for sent := 0; sent < silenceTailMS; sent += audioChunkMS {
		if err := send(silence); err != nil {
			return err
		}
	}
	return nil
}

// synthesize runs ttsService one-shot to render text to PCM, returning the audio
// and its sample rate. It drives the service as a single-node pipeline and
// collects the emitted TTS audio.
func synthesize(ctx context.Context, ttsService *tts.Base, text string) ([]byte, int, error) {
	var (
		pcm  []byte
		rate int
		once sync.Once
	)
	done := make(chan struct{})
	task := pipeline.NewTask(pipeline.New(ttsService), pipeline.TaskParams{
		OnReachedDownstream: func(f frames.Frame) {
			switch fr := f.(type) {
			case *frames.TTSAudioRawFrame:
				pcm = append(pcm, fr.Audio...)
				rate = fr.SampleRate
			case *frames.TTSStoppedFrame:
				once.Do(func() { close(done) })
			}
		},
	})

	runErr := make(chan error, 1)
	go func() { runErr <- task.Run(ctx) }()

	task.QueueFrame(frames.NewTTSSpeakFrame(text))

	select {
	case <-done:
	case <-ctx.Done():
		task.Cancel()
		<-runErr
		return nil, 0, ctx.Err()
	}
	task.StopWhenDone()
	if err := <-runErr; err != nil {
		return nil, 0, err
	}
	return pcm, rate, nil
}
