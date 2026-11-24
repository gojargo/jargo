package rime

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"slices"
	"testing"
	"time"

	"github.com/gojargo/jargo/audio/resample"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor/aggregators"
	"github.com/gojargo/jargo/processor/turns"
	"github.com/gojargo/jargo/service/settings"
	errs "github.com/gojargo/jargo/utils/errors"
	"github.com/gojargo/jargo/utils/events"
)

// Ported from upstream (test_stream_starts_with_audio_from_before_speech).
func TestSTTStreamStartsWithAudioFromBeforeSpeech(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	before, speech := pcm(1), pcm(2)
	r.send(audioFrame(before, 16000), vadStart())
	stream := nextStream(t, client)
	r.send(audioFrame(speech, 16000), vadStop(0))
	waitClosed(t, "the stream's audio never ended", stream.exhausted)
	if got, want := stream.audio(), append(slices.Clone(before), speech...); !bytes.Equal(got, want) {
		t.Errorf("the stream was sent %d bytes, want the %d before and during speech", len(got), len(want))
	}
	stream.finish("hello", "en")
	if !r.waitTranscripts(t, 1)[0].Finalized {
		t.Error("the transcript was not finalized")
	}
	if n := client.count(); n != 1 {
		t.Errorf("streams = %d, want one", n)
	}
}

// Ported from upstream (test_audio_without_vad_is_not_streamed).
func TestSTTAudioWithoutVADIsNotStreamed(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	for range 10 {
		r.send(audioFrame(pcm(1), 16000))
	}
	time.Sleep(50 * time.Millisecond)
	if n := client.count(); n != 0 {
		t.Errorf("streams = %d, want none", n)
	}
}

// Ported from upstream (test_interims_are_cumulative_for_both_contracts).
func TestSTTInterimsAreCumulativeForBothContracts(t *testing.T) {
	cases := []struct {
		contract string
		updates  []transcriptUpdate
		want     []string
	}{
		{contractRevised, []transcriptUpdate{partial("a red"), partial("a blue")}, []string{"a red", "a blue"}},
		{
			contractAppendFragments,
			[]transcriptUpdate{delta("hello "), delta("world\nnext")},
			[]string{"hello ", "hello world\nnext"},
		},
	}
	for _, c := range cases {
		t.Run(c.contract, func(t *testing.T) {
			client := newFakeRime()
			r := runSTT(t, STTConfig{OutputContract: c.contract}, client)
			r.waitConnected(t)
			r.send(vadStart(), audioFrame(pcm(1), 16000))
			stream := nextStream(t, client)
			if stream.opts.outputContract != c.contract {
				t.Errorf("output contract = %q, want %q", stream.opts.outputContract, c.contract)
			}
			for i, u := range c.updates {
				stream.updates <- u
				r.waitInterims(t, i+1)
			}
			r.send(vadStop(0))
			waitClosed(t, "the stream's audio never ended", stream.exhausted)
			stream.finish(c.want[len(c.want)-1], "en")
			if got := texts(r.waitTranscripts(t, 1)); !slices.Equal(got, c.want[len(c.want)-1:]) {
				t.Errorf("transcripts = %q, want %q", got, c.want[len(c.want)-1:])
			}
			if got := r.interims(); !slices.Equal(got, c.want) {
				t.Errorf("interims = %q, want %q", got, c.want)
			}
		})
	}
}

// Ported from upstream
// (test_finals_keep_speech_order_and_each_segment_keeps_its_settings).
func TestSTTFinalsKeepSpeechOrderAndEachSegmentKeepsItsSettings(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	r.speak(audioFrame(pcm(1), 16000))
	first := nextStream(t, client)
	r.send(frames.NewSTTUpdateSettingsFrame(&settings.STT{Language: settings.Set("es-MX")}))
	waitFor(t, "the language was never updated", func() bool {
		return r.c.segmentSettings().Language.Or("") == "es-MX"
	})
	r.speak(audioFrame(pcm(1), 16000))
	second := nextStream(t, client)
	if first.opts.language != "en" || second.opts.language != "es-MX" {
		t.Errorf("languages = %q, %q, want en then es-MX", first.opts.language, second.opts.language)
	}
	waitClosed(t, "the second stream's audio never ended", second.exhausted)
	second.finish("segundo", "es")
	waitClosed(t, "the second stream never closed", second.closed)
	time.Sleep(20 * time.Millisecond)
	if n := len(r.transcripts()); n != 0 {
		t.Fatalf("transcripts = %d, want the later final held back", n)
	}
	first.finish("first", "en")
	finals := r.waitTranscripts(t, 2)
	if got := texts(finals); !slices.Equal(got, []string{"first", "segundo"}) {
		t.Errorf("transcripts = %q, want them in speech order", got)
	}
	if got := finalized(finals); !slices.Equal(got, []bool{false, true}) {
		t.Errorf("finalized = %v, want only the latest", got)
	}
}

// Ported from upstream (test_final_is_not_finalized_once_the_user_speaks_again).
func TestSTTFinalIsNotFinalizedOnceTheUserSpeaksAgain(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	r.speak(audioFrame(pcm(1), 16000))
	first := nextStream(t, client)
	r.send(vadStart(), audioFrame(pcm(1), 16000))
	nextStream(t, client)
	waitClosed(t, "the first stream's audio never ended", first.exhausted)
	first.finish("first", "en")
	if r.waitTranscripts(t, 1)[0].Finalized {
		t.Error("the final was finalized though the user spoke again")
	}
}

// Ported from upstream (test_stream_error_category_and_usability).
func TestSTTStreamErrorCategoryAndUsability(t *testing.T) {
	cases := []struct {
		kind     errorKind
		category errs.Category
		usable   bool
	}{
		{kindAuthentication, errs.Authentication, false},
		{kindPermission, errs.Authorization, false},
		{kindBilling, errs.Quota, true},
		{kindInput, errs.InvalidRequest, false},
		{kindTimeout, errs.Connectivity, true},
		{kindUnavailable, errs.Connectivity, true},
		{kindResourceLimit, errs.Unknown, true},
		{kindStream, errs.Unknown, true},
	}
	for _, c := range cases {
		t.Run(c.kind.String(), func(t *testing.T) {
			client := newFakeRime()
			r := runSTT(t, STTConfig{}, client)
			r.waitConnected(t)
			r.send(vadStart(), audioFrame(pcm(1), 16000))
			stream := nextStream(t, client)
			stream.updates <- partial("unfinished")
			r.waitInterims(t, 1)
			stream.updates <- &rimeError{kind: c.kind, message: "echoed recognition hints", requestID: "request-7"}
			got := r.waitErrors(t, 1)
			time.Sleep(20 * time.Millisecond)
			if n := len(r.errors()); n != 1 {
				t.Fatalf("errors = %d, want one", n)
			}
			ef := got[0]
			if ef.Category != c.category {
				t.Errorf("category = %v, want %v", ef.Category, c.category)
			}
			if re := asRimeError(ef.Err); re == nil || re.kind != c.kind {
				t.Errorf("error = %v, want the stream's own", ef.Err)
			}
			if !bytes.Contains([]byte(ef.Error), []byte("request-7")) ||
				bytes.Contains([]byte(ef.Error), []byte("recognition hints")) {
				t.Errorf("message = %q, want the request named and the details left out", ef.Error)
			}
			if n := len(r.transcripts()); n != 0 {
				t.Errorf("transcripts = %d, want none", n)
			}
			if r.c.svc.Usable() != c.usable {
				t.Errorf("usable = %t, want %t", r.c.svc.Usable(), c.usable)
			}
		})
	}
}

// Ported from upstream (test_failed_segment_does_not_block_the_next).
func TestSTTFailedSegmentDoesNotBlockTheNext(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	r.speak(audioFrame(pcm(1), 16000))
	first := nextStream(t, client)
	r.speak(audioFrame(pcm(1), 16000))
	second := nextStream(t, client)
	waitClosed(t, "the second stream's audio never ended", second.exhausted)
	second.finish("second", "en")
	first.updates <- newRimeError(kindUnavailable, "down")
	r.waitErrors(t, 1)
	finals := r.waitTranscripts(t, 1)
	if got := texts(finals); !slices.Equal(got, []string{"second"}) || !finals[0].Finalized {
		t.Errorf("transcripts = %q (finalized %v), want the second finalized", got, finalized(finals))
	}
}

// Ported from upstream (test_credential_failure_recovers_when_marked_usable).
func TestSTTCredentialFailureRecoversWhenMarkedUsable(t *testing.T) {
	for _, kind := range []errorKind{kindAuthentication, kindPermission} {
		t.Run(kind.String(), func(t *testing.T) {
			client := newFakeRime()
			r := runSTT(t, STTConfig{}, client)
			r.waitConnected(t)
			r.send(vadStart(), audioFrame(pcm(1), 16000))
			first := nextStream(t, client)
			first.updates <- newRimeError(kind, "rejected")
			r.waitErrors(t, 1)
			waitClosed(t, "the stream never closed", first.closed)
			r.send(vadStop(0))
			if r.c.svc.Usable() {
				t.Fatal("the service stayed usable")
			}
			r.speak(audioFrame(pcm(1), 16000))
			time.Sleep(30 * time.Millisecond)
			if n := client.count(); n != 1 {
				t.Fatalf("streams = %d, want no new one while unusable", n)
			}

			r.c.svc.SetUsable(t.Context(), true)
			r.speak(audioFrame(pcm(1), 16000))
			second := nextStream(t, client)
			waitClosed(t, "the second stream's audio never ended", second.exhausted)
			second.finish("hello again", "en")
			if got := texts(r.waitTranscripts(t, 1)); !slices.Equal(got, []string{"hello again"}) {
				t.Errorf("transcripts = %q, want the recovered one", got)
			}
		})
	}
}

// Ported from upstream (test_startup_credential_failure_recovers_when_marked_usable,
// in part): a client that cannot be created for want of a key is an
// authentication failure, which leaves the service unusable.
func TestSTTMissingKeyIsAnAuthenticationFailure(t *testing.T) {
	_, err := newSTTClient("", nil)
	re := asRimeError(err)
	if re == nil || re.kind != kindAuthentication {
		t.Fatalf("error = %v, want an authentication failure", err)
	}
	c := newSTT(STTConfig{}, nil)
	if got := c.ClassifyError(err); got != errs.Authentication {
		t.Errorf("category = %v, want authentication", got)
	}
}

// Ported from upstream (test_supported_rate_is_sent_as_is).
func TestSTTSupportedRateIsSentAsIs(t *testing.T) {
	for _, rate := range sttSupportedSampleRates {
		t.Run("", func(t *testing.T) {
			client := newFakeRime()
			r := runSTT(t, STTConfig{SampleRate: rate}, client)
			r.waitConnected(t)
			speech := make([]byte, rate*2/10)
			r.speak(audioFrame(speech, rate))
			stream := nextStream(t, client)
			waitClosed(t, "the stream's audio never ended", stream.exhausted)
			if stream.opts.sampleRate != rate {
				t.Errorf("sample rate = %d, want %d", stream.opts.sampleRate, rate)
			}
			if got := stream.audio(); !bytes.Equal(got, speech) {
				t.Errorf("the stream was sent %d bytes, want the %d spoken", len(got), len(speech))
			}
		})
	}
}

// Ported from upstream (test_unsupported_rate_is_resampled).
func TestSTTUnsupportedRateIsResampled(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{SampleRate: 44100}, client)
	r.waitConnected(t)
	chunk := bytes.Repeat([]byte{1, 0}, 4410)
	r.speak(audioFrame(chunk, 44100), audioFrame(chunk, 44100), audioFrame(chunk, 44100))
	stream := nextStream(t, client)
	waitClosed(t, "the stream's audio never ended", stream.exhausted)
	if stream.opts.sampleRate != 16000 {
		t.Errorf("sample rate = %d, want 16000", stream.opts.sampleRate)
	}
	if n := len(stream.audio()); n == 0 || n > 3*3200 {
		t.Errorf("the stream was sent %d bytes, want the speech at 16 kHz", n)
	}
}

// Ported from upstream (test_pre_speech_and_speech_are_resampled_as_one_stream).
func TestSTTPreSpeechAndSpeechAreResampledAsOneStream(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{SampleRate: 44100}, client)
	r.waitConnected(t)
	var up, down []byte
	for range 34 {
		for b := range 256 {
			up = append(up, byte(b))
			down = append(down, byte(255-b))
		}
	}
	r.send(audioFrame(up, 44100))
	r.speak(audioFrame(down, 44100))
	stream := nextStream(t, client)
	waitClosed(t, "the stream's audio never ended", stream.exhausted)

	resampler, err := resample.New(44100, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := resampler.Process(append(slices.Clone(up), down...))
	if got := stream.audio(); !bytes.Equal(got, want) {
		t.Errorf("the stream was sent %d bytes, want the %d of one resampled stream", len(got), len(want))
	}
}

// Ported from upstream
// (test_a_segment_does_not_start_with_the_previous_segments_audio).
func TestSTTASegmentDoesNotStartWithThePreviousSegmentsAudio(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{SampleRate: 44100}, client)
	r.waitConnected(t)
	tone := make([]byte, 0, 2*4410)
	for n := range 4410 {
		v := int16(8000 * math.Sin(2*math.Pi*440*float64(n)/44100))
		tone = binary.LittleEndian.AppendUint16(tone, uint16(v))
	}
	r.speak(audioFrame(tone, 44100))
	waitClosed(t, "the first stream's audio never ended", nextStream(t, client).exhausted)
	// More than the pre-speech window of silence between the segments.
	for range 12 {
		r.send(audioFrame(make([]byte, 8820), 44100))
	}
	r.speak(audioFrame(make([]byte, 8820), 44100))
	stream := nextStream(t, client)
	waitClosed(t, "the second stream's audio never ended", stream.exhausted)

	sent := stream.audio()
	if len(sent) == 0 {
		t.Fatal("the second stream was sent nothing")
	}
	for i := 0; i+1 < len(sent); i += 2 {
		if v := int16(binary.LittleEndian.Uint16(sent[i:])); v > 100 || v < -100 {
			t.Fatalf("sample %d = %d, want silence", i/2, v)
		}
	}
}

// Ported from upstream (test_usage_and_ttfb_are_reported_with_the_final).
func TestSTTUsageAndTTFBAreReportedWithTheFinal(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	r.send(audioFrame(make([]byte, 3200), 16000), vadStart())
	r.send(audioFrame(make([]byte, 3200), 16000), vadStop(0.1))
	stream := nextStream(t, client)
	waitClosed(t, "the stream's audio never ended", stream.exhausted)
	stream.finish("done", "en")
	r.waitTranscripts(t, 1)

	var seconds float64
	ttfb := 0
	for _, m := range downstreamOf[*frames.MetricsFrame](r) {
		for _, d := range m.Data {
			switch d := d.(type) {
			case frames.STTUsageMetricsData:
				seconds += d.Value.AudioSeconds
			case frames.TTFBMetricsData:
				ttfb++
			}
		}
	}
	if math.Abs(seconds-0.2) > 1e-6 {
		t.Errorf("usage = %v seconds, want 0.2", seconds)
	}
	if ttfb != 1 {
		t.Errorf("TTFB reports = %d, want one", ttfb)
	}
}

// Ported from upstream (test_mute_stops_audio_and_new_segments).
func TestSTTMuteStopsAudioAndNewSegments(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	speech := pcm(1)
	r.send(vadStart(), audioFrame(speech, 16000))
	stream := nextStream(t, client)
	r.send(frames.NewSTTMuteFrame(true), audioFrame(pcm(2), 16000), vadStop(0))
	waitClosed(t, "the stream's audio never ended", stream.exhausted)
	if got := stream.audio(); !bytes.Equal(got, speech) {
		t.Errorf("the stream was sent %d bytes, want only the %d before the mute", len(got), len(speech))
	}
	stream.finish("before mute", "en")
	r.waitTranscripts(t, 1)
	r.speak(audioFrame(pcm(1), 16000))
	time.Sleep(30 * time.Millisecond)
	if n := client.count(); n != 1 {
		t.Errorf("streams = %d, want no new one while muted", n)
	}
}

// Ported from upstream (test_interruption_keeps_the_segment_streaming).
func TestSTTInterruptionKeepsTheSegmentStreaming(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	r.send(vadStart(), audioFrame(pcm(1), 16000))
	stream := nextStream(t, client)
	r.send(frames.NewInterruptionFrame(), audioFrame(pcm(1), 16000), vadStop(0))
	waitClosed(t, "the stream's audio never ended", stream.exhausted)
	stream.finish("barge in", "en")
	if got := texts(r.waitTranscripts(t, 1)); !slices.Equal(got, []string{"barge in"}) {
		t.Errorf("transcripts = %q, want the barge in", got)
	}
}

// Ported from upstream (test_end_frame_finishes_the_segment_in_progress).
func TestSTTEndFrameFinishesTheSegmentInProgress(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	r.send(vadStart(), audioFrame(pcm(1), 16000))
	stream := nextStream(t, client)
	r.task.StopWhenDone()
	waitClosed(t, "the stream's audio never ended", stream.exhausted)
	stream.finish("last words", "en")
	select {
	case <-r.done:
	case <-time.After(waitTimeout):
		t.Fatal("the pipeline did not end")
	}
	if got := texts(r.transcripts()); !slices.Equal(got, []string{"last words"}) {
		t.Errorf("transcripts = %q, want the last words", got)
	}
	if !client.closed.Load() {
		t.Error("the client was not closed")
	}
	r.done <- nil
}

// Ported from upstream (test_cancel_drops_segments_without_errors).
func TestSTTCancelDropsSegmentsWithoutErrors(t *testing.T) {
	client := newFakeRime()
	r := runSTT(t, STTConfig{}, client)
	r.waitConnected(t)
	r.send(vadStart(), audioFrame(pcm(1), 16000))
	stream := nextStream(t, client)
	r.task.Cancel(t.Context(), "cancel")
	select {
	case <-r.done:
	case <-time.After(waitTimeout):
		t.Fatal("the pipeline did not cancel")
	}
	r.done <- nil
	if !client.closed.Load() {
		t.Error("the client was not closed")
	}
	waitClosed(t, "the stream was not closed", stream.closed)
	if n := len(r.transcripts()); n != 0 {
		t.Errorf("transcripts = %d, want none", n)
	}
	if n := len(r.errors()); n != 0 {
		t.Errorf("errors = %v, want none", r.errors())
	}
}

// userTurns runs a service with a user aggregator after it, recording each
// user turn's text as it ends.
func userTurns(t *testing.T, client *fakeRime, cfg STTConfig, opts ...aggregators.Option) (
	*sttRun, *frames.LLMContext, func() []string,
) {
	t.Helper()
	convo := frames.NewLLMContext("")
	agg := aggregators.New(convo, opts...)
	turnsCh := make(chan string, 4)
	events.On(agg.User().Events(), aggregators.EventUserTurnStopped,
		func(_ context.Context, stopped aggregators.UserTurnStopped) { turnsCh <- stopped.Content })
	r := runSTT(t, cfg, client, agg.User())
	var got []string
	return r, convo, func() []string {
		for {
			select {
			case s := <-turnsCh:
				got = append(got, s)
			default:
				return got
			}
		}
	}
}

// Ported from upstream (test_user_turn_waits_for_the_latest_final).
func TestSTTUserTurnWaitsForTheLatestFinal(t *testing.T) {
	for _, secondText := range []string{"second", ""} {
		t.Run(secondText, func(t *testing.T) {
			client := newFakeRime()
			ttfs := 60 * time.Second
			r, _, stopped := userTurns(t, client, STTConfig{TTFSP99: &ttfs}, aggregators.WithTurns(turns.Config{
				Strategies: turns.UserTurnStrategies{
					Start: []turns.StartStrategy{turns.NewVADStart()},
					Stop: []turns.StopStrategy{
						turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{UserSpeechTimeout: 10 * time.Millisecond}),
					},
				},
			}))
			r.waitConnected(t)
			r.speak(audioFrame(pcm(1), 16000))
			first := nextStream(t, client)
			r.speak(audioFrame(pcm(1), 16000))
			second := nextStream(t, client)
			time.Sleep(30 * time.Millisecond)
			first.finish("first", "en")
			r.waitTranscripts(t, 1)
			if got := stopped(); len(got) != 0 {
				t.Fatalf("the turn ended with %q before the latest final", got)
			}
			waitClosed(t, "the second stream's audio never ended", second.exhausted)
			second.finish(secondText, "en")
			want := "first second"
			if secondText == "" {
				want = "first"
			}
			waitFor(t, "the user turn never ended", func() bool { return len(stopped()) > 0 })
			if got := stopped(); !slices.Equal(got, []string{want}) {
				t.Errorf("turns = %q, want %q", got, want)
			}
		})
	}
}

// Ported from upstream (test_empty_final_adds_no_user_message).
func TestSTTEmptyFinalAddsNoUserMessage(t *testing.T) {
	client := newFakeRime()
	r, convo, _ := userTurns(t, client, STTConfig{}, aggregators.WithTurns(turns.Config{
		Strategies: turns.UserTurnStrategies{
			Start: []turns.StartStrategy{turns.NewVADStart()},
			Stop:  []turns.StopStrategy{turns.NewSpeechTimeoutStop(turns.SpeechTimeoutConfig{})},
		},
	}))
	r.waitConnected(t)
	r.speak(audioFrame(pcm(1), 16000))
	stream := nextStream(t, client)
	waitClosed(t, "the stream's audio never ended", stream.exhausted)
	stream.finish("", "en")
	if !r.waitTranscripts(t, 1)[0].Finalized {
		t.Error("the empty final was not finalized")
	}
	time.Sleep(30 * time.Millisecond)
	if n := len(convo.Messages()); n != 0 {
		t.Errorf("messages = %d, want none", n)
	}
}
