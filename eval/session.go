package eval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/service/tts"
)

// defaultTimeout is the latency budget for an expectation that sets no within_ms.
const defaultTimeout = 15 * time.Second

// errBotNotReady is returned when the bot never completes the RTVI handshake.
//
//nolint:gochecknoglobals // sentinel error
var errBotNotReady = errors.New("eval: bot did not become ready")

// event is a friendly, translated view of an RTVI server message — the level a
// scenario asserts on.
type event struct {
	kind     string // one of the Event* constants
	text     string // llm_response: the bot's reply text, joined
	function string // function_call: the tool name
}

// session drives one scenario against a connected bot.
type session struct {
	client   *client
	scenario *Scenario
	judge    Judge
	userTTS  *tts.Base // non-nil in audio mode: synthesizes each user turn

	// llmBuf accumulates bot-llm-text between bot-llm-started and bot-llm-stopped.
	llmBuf strings.Builder
}

// run completes the handshake, then plays every turn.
func (s *session) run(ctx context.Context) (Result, error) {
	res := Result{Scenario: s.scenario.Name}
	if err := s.handshake(ctx); err != nil {
		return res, err
	}
	for i, turn := range s.scenario.Turns {
		failures, err := s.runTurn(ctx, turn, i+1)
		if err != nil {
			return res, err
		}
		res.Failures = append(res.Failures, failures...)
	}
	return res, nil
}

// handshake sends client-ready and waits for the bot's bot-ready reply.
func (s *session) handshake(ctx context.Context) error {
	ready := rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeClientReady, ID: "eval",
		Data: map[string]any{"version": rtvi.ProtocolVersion},
	}
	if err := s.client.send(ctx, ready); err != nil {
		return err
	}
	deadline := time.Now().Add(defaultTimeout)
	for {
		in, ok := s.next(ctx, deadline)
		if !ok {
			return fmt.Errorf("%w within %s", errBotNotReady, defaultTimeout)
		}
		if in.Type == rtvi.TypeBotReady {
			return nil
		}
	}
}

// runTurn sends the user's text and matches the turn's expectations in order.
// Every expectation's budget is anchored at the send, so a stalled turn fails
// within a single budget rather than one per expectation.
func (s *session) runTurn(ctx context.Context, turn Turn, turnNum int) ([]Failure, error) {
	if s.userTTS != nil {
		if err := s.sendUserAudio(ctx, turn.User); err != nil {
			return nil, err
		}
	} else if err := s.client.send(ctx, sendText(turn.User)); err != nil {
		return nil, err
	}
	anchor := time.Now()

	var failures []Failure
	for j, exp := range turn.Expect {
		budget := defaultTimeout
		if exp.WithinMS > 0 {
			budget = time.Duration(exp.WithinMS) * time.Millisecond
		}
		fail, timedOut := s.matchExpectation(ctx, exp, anchor.Add(budget), turnNum, j+1)
		if fail != nil {
			failures = append(failures, *fail)
		}
		if timedOut {
			// No more events will match this turn's remaining expectations.
			break
		}
	}
	return failures, nil
}

// matchExpectation waits for an event matching exp, skipping unrelated events,
// until deadline. It returns a failure (and whether it timed out) or nil on a
// match.
func (s *session) matchExpectation(
	ctx context.Context, exp Expectation, deadline time.Time, turnNum, expNum int,
) (*Failure, bool) {
	for {
		in, ok := s.next(ctx, deadline)
		if !ok {
			return &Failure{
				Turn: turnNum, Expectation: expNum, Event: exp.Event,
				Reason: fmt.Sprintf("no matching %s event arrived within budget", exp.Event),
			}, true
		}
		ev := s.translate(in)
		if ev == nil || ev.kind != exp.Event {
			continue // not this expectation's event; skip it
		}
		// A named function_call skips calls to other tools and keeps waiting.
		if exp.Event == EventFunctionCall && exp.Function != "" && ev.function != exp.Function {
			continue
		}
		if reason := s.verify(ctx, exp, *ev); reason != "" {
			return &Failure{Turn: turnNum, Expectation: expNum, Event: exp.Event, Reason: reason}, false
		}
		return nil, false
	}
}

// verify checks an event that matched by kind against the expectation's content
// assertions. It returns an empty string on success.
func (s *session) verify(ctx context.Context, exp Expectation, ev event) string {
	if exp.TextContains != "" &&
		!strings.Contains(strings.ToLower(ev.text), strings.ToLower(exp.TextContains)) {
		return fmt.Sprintf("reply %q does not contain %q", ev.text, exp.TextContains)
	}
	if exp.Judge != "" {
		if s.judge == nil {
			return fmt.Sprintf("judge criterion %q set but no judge is configured", exp.Judge)
		}
		pass, reason, err := s.judge.Evaluate(ctx, exp.Judge, ev.text)
		if err != nil {
			return fmt.Sprintf("judge error: %v", err)
		}
		if !pass {
			return fmt.Sprintf("judge rejected %q: %s", exp.Judge, reason)
		}
	}
	return ""
}

// translate converts an RTVI server message into a friendly event, or nil if it
// carries no event on its own (bot-llm-text is accumulated until the response
// ends).
func (s *session) translate(in rtvi.Incoming) *event {
	switch in.Type {
	case rtvi.TypeUserStartedSpeaking:
		return &event{kind: EventUserStartedSpeaking}
	case rtvi.TypeUserStoppedSpeaking:
		return &event{kind: EventUserStoppedSpeaking}
	case rtvi.TypeUserTranscription:
		var d rtvi.UserTranscriptionData
		if json.Unmarshal(in.Data, &d) != nil || !d.Final {
			return nil // only final transcriptions are turn-level events
		}
		return &event{kind: EventUserTranscription, text: d.Text}
	case rtvi.TypeBotLLMStarted:
		s.llmBuf.Reset()
		return &event{kind: EventLLMStarted}
	case rtvi.TypeBotLLMText:
		var d rtvi.TextData
		if json.Unmarshal(in.Data, &d) == nil {
			s.llmBuf.WriteString(d.Text)
		}
		return nil
	case rtvi.TypeBotLLMStopped:
		text := s.llmBuf.String()
		s.llmBuf.Reset()
		return &event{kind: EventLLMResponse, text: text}
	case rtvi.TypeLLMFunctionCall:
		var d rtvi.LLMFunctionCallData
		_ = json.Unmarshal(in.Data, &d)
		return &event{kind: EventFunctionCall, function: d.FunctionName}
	default:
		return nil
	}
}

// next reads the next server message, blocking until one arrives, the deadline
// passes, or ctx is canceled.
func (s *session) next(ctx context.Context, deadline time.Time) (rtvi.Incoming, bool) {
	remaining := time.Until(deadline)
	if remaining <= 0 {
		return rtvi.Incoming{}, false
	}
	timer := time.NewTimer(remaining)
	defer timer.Stop()
	select {
	case in, ok := <-s.client.incoming:
		return in, ok
	case <-timer.C:
		return rtvi.Incoming{}, false
	case <-ctx.Done():
		return rtvi.Incoming{}, false
	}
}

// sendText builds a send-text message that appends the user's turn and runs the
// LLM immediately, without a spoken audio response.
func sendText(content string) rtvi.Message {
	runNow, noAudio := true, false
	return rtvi.Message{
		Label: rtvi.MessageLabel, Type: rtvi.TypeSendText, ID: "eval",
		Data: rtvi.SendTextData{
			Content: content,
			Options: &rtvi.SendTextOptions{RunImmediately: &runNow, AudioResponse: &noAudio},
		},
	}
}
