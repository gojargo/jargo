package eval

import (
	"context"
	"encoding/base64"
	"errors"
	"time"

	"github.com/gojargo/jargo/processor/rtvi"
)

const (
	// audioChunkMS is the size of each streamed frame of the user's audio: the
	// granularity a live transport delivers, and what VAD and turn models
	// consume.
	audioChunkMS = 20
	// silenceTailMS is how much silence follows a user turn's speech, so the
	// bot's VAD registers the end of the turn.
	silenceTailMS = 1000
)

// errNoUserAudio is a user turn the user TTS produced no audio for.
var errNoUserAudio = errors.New("eval: user TTS produced no audio")

// streamUserAudio streams a user turn's audio to the bot as raw-audio messages,
// 16-bit mono PCM at sampleRate: frames of audioChunkMS at a real-time cadence,
// then a tail of silence so the bot's VAD ends the turn. Real-time pacing
// matters: flooding the whole utterance at once breaks VAD windows and
// turn-detecting STTs.
//
// It returns once the turn, silence included, has gone out.
func (c *Client) streamUserAudio(ctx context.Context, pcm []byte, sampleRate int) error {
	if sampleRate == 0 || len(pcm) == 0 {
		return errNoUserAudio
	}
	bytesPerChunk := (sampleRate * audioChunkMS / 1000) * 2
	ticker := time.NewTicker(audioChunkMS * time.Millisecond)
	defer ticker.Stop()

	send := func(chunk []byte) error {
		c.Send(c.message(rtvi.TypeRawAudio, rtvi.RawAudioData{
			Base64Audio: base64.StdEncoding.EncodeToString(chunk),
			SampleRate:  sampleRate,
			NumChannels: 1,
		}))
		select {
		case <-ticker.C:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}
	}

	for off := 0; off < len(pcm); off += bytesPerChunk {
		if err := send(pcm[off:min(off+bytesPerChunk, len(pcm))]); err != nil {
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
