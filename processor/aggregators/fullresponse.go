package aggregators

import (
	"context"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// EventCompletion fires with a whole LLM response once it has been gathered.
// It carries a Completion.
//
//	events.On(agg.Events(), aggregators.EventCompletion,
//	    func(ctx context.Context, c aggregators.Completion) { … })
const EventCompletion = "on_completion"

// Completion is a gathered LLM response.
type Completion struct {
	// Text is everything the model said between the start and end of the
	// response.
	Text string
	// Completed reports whether the response finished. It is false for a
	// response an interruption cut short, where Text is what had been said by
	// then.
	Completed bool
}

// FullResponse gathers a whole LLM response and reports it, leaving the frames
// it gathered from untouched.
//
// It collects the text between an LLMFullResponseStartFrame and its matching end
// frame, and raises EventCompletion with the result. An interruption reports
// what had been gathered by then, marked as unfinished.
//
// It is for something that wants each reply whole and off the frame path: a
// transcript, a moderation check, an evaluation. Nothing it sees is consumed.
type FullResponse struct {
	*processor.Base
	aggregation string
	started     bool
}

// NewFullResponse builds a full-response aggregator.
func NewFullResponse(name string) *FullResponse {
	a := &FullResponse{}
	a.Base = processor.New(name, a)
	a.Events().Register(EventCompletion, false)
	return a
}

// ProcessFrame gathers the response and forwards every frame untouched.
func (a *FullResponse) ProcessFrame(ctx context.Context, f frames.Frame, dir processor.Direction) error {
	if err := a.Base.ProcessFrame(ctx, f, dir); err != nil {
		return err
	}

	switch fr := f.(type) {
	case *frames.InterruptionFrame:
		a.report(ctx, false)
	case *frames.LLMFullResponseStartFrame:
		a.started = true
	case *frames.LLMFullResponseEndFrame:
		a.report(ctx, true)
	case *frames.LLMTextFrame:
		// Text outside a response belongs to nothing, so it is not gathered.
		if a.started {
			a.aggregation += fr.Text
		}
	}

	return a.PushFrame(ctx, f, dir)
}

// report raises the completion and starts the next one afresh.
func (a *FullResponse) report(ctx context.Context, completed bool) {
	said := a.aggregation
	a.aggregation = ""
	a.started = false
	a.Events().Call(ctx, EventCompletion, a, Completion{Text: said, Completed: completed})
}
