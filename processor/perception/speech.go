package perception

import (
	"bytes"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/classifier"
)

// recentSpeech is the user's recent speech, kept to cut a clip from each time
// they stop speaking.
//
// Each stretch of speech is kept with the silence around it: up to a second of
// the audio heard before the VAD reported it, and the silence the VAD waited
// for after it. A classifier's clip is the latest stretch whole, topped up from
// the end of the speech before it until it is as long as the classifier answers
// best. Speech from before the bot stopped speaking or before a mute is never
// joined to speech after it, so a reply is heard on its own: speech from before
// or during the bot's turn says how the user sounded then, not how they reply.
// Audio heard while the user is muted is not kept.
//
// It is not safe for concurrent use: the reader holds its lock around every
// call.
type recentSpeech struct {
	// reach is how much speech the classifiers' clips reach back for between
	// them.
	reach      time.Duration
	segment    *audio.SpeechSegmentBuffer
	sampleRate int
	// stretches is the user's recent stretches of speech, the latest last.
	stretches [][]byte
	muted     bool
	// speaking is whether the user is speaking; ended is closed while they are
	// silent, and endedAt is when they stopped, or zero while they speak and
	// before they ever have.
	speaking bool
	ended    chan struct{}
	endedAt  time.Time
}

func newRecentSpeech(classifiers []classifier.Classifier) *recentSpeech {
	var reach, longest time.Duration
	bounded := len(classifiers) > 0
	for _, c := range classifiers {
		reach = max(reach, c.PreferredAudio())
		if c.MaxAudio() <= 0 {
			bounded = false
		}
		longest = max(longest, c.MaxAudio())
	}
	// A stretch is kept whole unless every classifier bounds its clips.
	if !bounded {
		longest = 0
	}
	ended := make(chan struct{})
	close(ended)
	return &recentSpeech{
		reach:   reach,
		segment: audio.NewSpeechSegmentBuffer(audio.DefaultPreSpeech, longest),
		ended:   ended,
	}
}

// setSampleRate sets the sample rate of the user's audio.
func (s *recentSpeech) setSampleRate(sampleRate int) {
	s.sampleRate = sampleRate
	s.segment.SetSampleRate(sampleRate)
}

// append adds the user's audio, unless they are muted.
func (s *recentSpeech) append(pcm []byte) {
	if !s.muted {
		s.segment.Append(pcm)
	}
}

// start marks the start of the user's speech.
func (s *recentSpeech) start() {
	s.segment.Start()
	if !s.speaking {
		s.ended = make(chan struct{})
	}
	s.speaking = true
	s.endedAt = time.Time{}
}

// stop keeps the user's speech, which ended stopSecs before the VAD said so.
func (s *recentSpeech) stop(stopSecs float64) {
	if s.speaking {
		s.keep(s.segment.Stop())
	}
	// The stop strategies count from when the user stopped, too.
	s.endedAt = time.Now().Add(-time.Duration(stopSecs * float64(time.Second)))
	s.end()
}

// end marks the user silent.
func (s *recentSpeech) end() {
	if s.speaking {
		s.speaking = false
		close(s.ended)
	}
}

// clip is a classifier's clip of the user's latest speech, and false if it is
// too short for it.
func (s *recentSpeech) clip(c classifier.Classifier) (classifier.Audio, bool) {
	pcm := s.latest(c.PreferredAudio())
	if most := c.MaxAudio(); most > 0 {
		if n := s.bytes(most); n > 0 && len(pcm) > n {
			pcm = pcm[len(pcm)-n:]
		}
	}
	if len(pcm) == 0 || len(pcm) < s.bytes(c.MinAudio()) {
		return classifier.Audio{}, false
	}
	return classifier.Audio{Audio: pcm, SampleRate: s.sampleRate}, true
}

// botStoppedSpeaking keeps the speech from before and during the bot's turn
// out of the clips after it. Speech the user is in the middle of is kept whole.
func (s *recentSpeech) botStoppedSpeaking() {
	s.stretches = nil
}

// mute drops the speech kept when the user is muted, and ends any speech they
// are in.
func (s *recentSpeech) mute(muted bool) {
	s.muted = muted
	if !muted {
		return
	}
	s.segment.Stop()
	s.stretches = nil
	if s.speaking {
		s.endedAt = time.Now()
		s.end()
	}
}

// keep keeps a stretch of speech, and the ones before it a clip can still
// reach.
func (s *recentSpeech) keep(stretch []byte) {
	if len(stretch) > 0 {
		s.stretches = append(s.stretches, stretch)
	}
	reach := s.bytes(s.reach)
	for len(s.stretches) > 1 && total(s.stretches[1:]) >= reach {
		s.stretches = s.stretches[1:]
	}
}

// latest is the latest stretch of speech, topped up from the end of the ones
// before it to last d.
func (s *recentSpeech) latest(d time.Duration) []byte {
	if len(s.stretches) == 0 {
		return nil
	}
	earlier, latest := s.stretches[:len(s.stretches)-1], s.stretches[len(s.stretches)-1]
	missing := max(0, s.bytes(d)-len(latest))
	topUp := bytes.Join(earlier, nil)
	if missing < len(topUp) {
		topUp = topUp[len(topUp)-missing:]
	}
	out := make([]byte, 0, len(topUp)+len(latest))
	return append(append(out, topUp...), latest...)
}

// bytes is how many bytes of 16-bit mono audio make d.
func (s *recentSpeech) bytes(d time.Duration) int {
	return int(d.Seconds()*float64(s.sampleRate)) * 2
}

// total is how many bytes the stretches hold between them.
func total(stretches [][]byte) int {
	n := 0
	for _, s := range stretches {
		n += len(s)
	}
	return n
}
