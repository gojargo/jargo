package eval

import (
	"context"
	"fmt"
	"time"
)

// The base driver: what the user says next, and how the run is scored.
//
// A driver runs the conversation over the session's client and event stream
// and assembles the result. The scripted driver plays a file's turns and
// matches their expectations.

// turnSender is what a driver sends the user's turns through: the Client.
type turnSender interface {
	HasUserTTS() bool
	SendText(text string)
	SendDTMF(keys string)
	SendImage(path string) error
	Say(ctx context.Context, text string) error
	SayWithPauses(ctx context.Context, parts []TurnPart) error
	Play(ctx context.Context, path string) error
}

// driver is what a session runs: the conversation, and how it is scored.
type driver interface {
	// run drives the conversation to its end and returns the failures. The
	// error is the harness failing to send, or ctx ending.
	run(ctx context.Context) ([]AssertionFailure, error)
	// recordFailure notes a run-level failure the session raised while the
	// driver was running, scored against the trace's current turn.
	recordFailure(f AssertionFailure)
	// close closes the judge, once the run has ended.
	close(ctx context.Context)
}

// baseDriver is the part every driver shares: the client the user's turns go
// through, the stream it reads, the judge the turns are recorded for, and the
// progress it reports.
type baseDriver struct {
	client   turnSender
	stream   *EventStream
	judge    Judge
	trace    *Trace
	progress func(Progress)
}

// botMetrics is the metrics reports the bot sent during the run, for the
// result: each one's time and data.
func (d *baseDriver) botMetrics() []BotMetricsReport { return botMetricsOf(d.stream) }

// close closes the judge, once the run has ended.
func (d *baseDriver) close(ctx context.Context) {
	if d.judge != nil {
		closeJudge(ctx, d.judge)
	}
}

// say sends one user utterance to the bot.
//
// It goes out as the recording in audioFile when given, spoken by the user TTS
// when the client has one, else as text. Bot output still queued from before is
// dropped first, so it cannot be matched as the reply. text is also recorded in
// the judge's conversation so a later reply is judged in context (a terse
// "That's four" answering this question).
func (d *baseDriver) say(ctx context.Context, text, audioFile string, parts []TurnPart) error {
	d.stream.DropPendingBotOutput("before send")
	how := audioFile
	if how == "" {
		how = ModalityText
		if d.client.HasUserTTS() {
			how = ModalityAudio
		}
	}
	d.trace.Log(fmt.Sprintf("send: %q (%s)", text, how))
	// Only what the bot says in reply to this input is matched from here on.
	// It is marked before the send rather than after it: the send is handed to
	// the pipeline on another goroutine, and the bot's reply can start arriving
	// before the send returns.
	d.stream.InputSent()
	var err error
	switch {
	case audioFile != "":
		err = d.client.Play(ctx, audioFile)
	case parts != nil && d.client.HasUserTTS():
		err = d.client.SayWithPauses(ctx, parts)
	case d.client.HasUserTTS():
		err = d.client.Say(ctx, text)
	default:
		d.client.SendText(text)
	}
	if err != nil {
		return err
	}
	if d.judge != nil {
		d.judge.AddUserMessage(text)
	}
	return nil
}

// press sends DTMF keypresses as the user's turn. The keys are recorded for the
// judge, so the bot's reply is judged knowing what was pressed.
func (d *baseDriver) press(keys string) {
	d.stream.DropPendingBotOutput("before send")
	d.trace.Log(fmt.Sprintf("send: dtmf %q", keys))
	d.stream.InputSent()
	d.client.SendDTMF(keys)
	if d.judge != nil {
		d.judge.AddUserMessage("(DTMF keypad input: " + keys + ")")
	}
}

// sleep waits out d, or returns early if ctx is canceled. A non-positive d
// returns at once.
func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
