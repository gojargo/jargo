package llm

import (
	"context"
	"log/slog"
	"strings"
	"sync"

	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// TurnCompletionExtension reads the turn-completion marker that opens
// each of the model's responses.
//
// Each response starts with a marker: complete (the turn is complete),
// incomplete short (the user will continue shortly) or incomplete long (the
// user needs more time). The extension strips the marker from the spoken text
// and reports it:
//
//   - upstream, once per inference, as a UserTurnCompletionFrame for the turns
//     package's LLMTurnCompletionStop to decide on. A tool call before any
//     marker reports a complete turn, and a response without a marker reports
//     no kind;
//   - in both directions as a UserTurnInferenceCompletedFrame when the turn is
//     complete;
//   - downstream as an LLMMarkerFrame, which the assistant aggregator writes to
//     the conversation.
//
// Text after an incomplete marker is not spoken: the marker is the whole
// response. When the response ends, its raw text and marker are reported
// downstream as an LLMMarkerResponseFrame.
//
// The extension contributes the turn-completion instructions to the system
// instruction, with a note that a reading of how the user spoke does not decide
// the turn when the model is taught one (expressive conversation).
//
// It is not for an application to attach: on its own, the extension is half the
// feature. The stop strategy installs it, acts on its reports and sends the
// re-prompts, and turns.FilterIncompleteUserTurnStrategies on the user
// aggregator is how an application turns turn completion on.
//
//nolint:revive // the name is the one the extension design uses throughout
type TurnCompletionExtension struct {
	BaseExtension

	mu     sync.Mutex
	config UserTurnCompletionConfig
	// expressive is whether the pipeline has expressive conversation on, and
	// perception the signals perceived of how the user spoke, from the latest
	// PerceptionMetadataFrame, with it on.
	expressive bool
	perception *expressive.PerceptionCapabilities
	// contextFrameID is the LLMContextFrame that started the current inference,
	// and reported whether its turn-completion judgment has been reported
	// upstream yet.
	contextFrameID uint64
	reported       bool
	// buffer holds the response text seen so far, until a marker appears in it.
	buffer string
	// raw is the current response as the model produced it, and foundMarker
	// and foundKind the marker read from it and what it meant, reported
	// together when the response ends. foundKind is empty until a marker is
	// read.
	raw         strings.Builder
	foundMarker string
	foundKind   string
	// marker is the verdict for the current response. turnMarkerNone means no
	// marker yet, so text keeps buffering until one appears. TurnMarkerIncomplete
	// doubles as a latch: the prompt tells the model to emit only the marker,
	// but if it disobeys and streams more text, it stays suppressed.
	marker TurnMarker
}

// NewTurnCompletionExtension builds the turn-completion extension with
// cfg: the markers, the instructions, the re-prompts and their timeouts. The
// zero value is the defaults.
func NewTurnCompletionExtension(cfg UserTurnCompletionConfig) *TurnCompletionExtension {
	return &TurnCompletionExtension{config: cfg}
}

// Order implements Extension. The extension sits "closer" to the service
// than any other reader of the model's output, so it filters the output first:
// the marker opens the response, and what follows it reads differently once it
// is stripped.
func (e *TurnCompletionExtension) Order() int { return -1 }

// Config is the turn completion configuration.
func (e *TurnCompletionExtension) Config() UserTurnCompletionConfig {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.config
}

// SetConfig replaces the turn completion configuration. The service composes
// its instruction anew when its settings change it.
func (e *TurnCompletionExtension) SetConfig(cfg UserTurnCompletionConfig) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.config = cfg
}

// Setup implements Extension, recording whether the pipeline has expressive
// conversation on.
func (e *TurnCompletionExtension) Setup(_ context.Context, s processor.Setup) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.expressive = s.EnableExpressiveConversation
	return nil
}

// SystemInstruction implements Extension: the turn-completion instructions,
// with the perception note when the model is taught a reading of how the user
// spoke.
func (e *TurnCompletionExtension) SystemInstruction() string {
	e.mu.Lock()
	defer e.mu.Unlock()
	parts := []string{e.config.CompletionInstructions()}
	if e.perception != nil && len(e.perception.Signals) > 0 {
		parts = append(parts, expressive.PerceptionTurnCompletionInstructions(e.config.Markers()))
	}
	return strings.Join(parts, "\n\n")
}

// ProcessFrame implements Extension: it follows the inference the service
// runs, and resets on an interruption.
func (e *TurnCompletionExtension) ProcessFrame(
	ctx context.Context, f frames.Frame, _ processor.Direction,
) error {
	switch fr := f.(type) {
	case *frames.InterruptionFrame:
		for _, text := range e.resetTurn() {
			if err := e.PushFrame(ctx, text, processor.Downstream); err != nil {
				return err
			}
		}
	case *frames.LLMContextFrame:
		e.mu.Lock()
		e.contextFrameID = fr.ID()
		e.reported = false
		e.mu.Unlock()
	case *frames.PerceptionMetadataFrame:
		e.mu.Lock()
		if e.expressive {
			perception := fr.Perception
			e.perception = &perception
		}
		e.mu.Unlock()
	}
	return nil
}

// FilterPushFrame implements Extension: it reads the marker from the
// response's text, and reports it. The frames it returns are the text held back
// until the marker is read, the marker frame, and the response's report at its
// end.
func (e *TurnCompletionExtension) FilterPushFrame(
	ctx context.Context, f frames.Frame, _ processor.Direction,
) ([]frames.Frame, error) {
	switch fr := f.(type) {
	case *frames.FunctionCallsStartedFrame:
		// A tool call is the model judging the turn complete. Reported before
		// the function dispatches, so the turn can end before its result
		// reaches the assistant aggregator.
		if err := e.reportTurnCompletion(ctx, completionKindComplete, ""); err != nil {
			return nil, err
		}
	case *frames.LLMFullResponseStartFrame:
		e.mu.Lock()
		e.raw.Reset()
		e.foundMarker, e.foundKind = "", ""
		e.mu.Unlock()
	case *frames.LLMFullResponseEndFrame:
		// A response that ends without a marker or a tool call carries no
		// judgment.
		if err := e.reportTurnCompletion(ctx, completionKindNone, ""); err != nil {
			return nil, err
		}
		var out []frames.Frame
		if report := e.responseReport(); report != nil {
			out = append(out, report)
		}
		out = append(out, e.resetTurn()...)
		return append(out, f), nil
	case *frames.LLMTextFrame:
		return e.filterText(ctx, fr)
	}
	return []frames.Frame{f}, nil
}

// filterText holds the response's text until its marker is read, then lets it
// through:
//
//   - until a marker appears, the text is buffered;
//   - an incomplete marker is the whole response: the marker is reported and
//     pushed, and the rest of the response is dropped;
//   - after the complete marker, the text after it is pushed, and all later
//     text flows through as it comes.
func (e *TurnCompletionExtension) filterText(
	ctx context.Context, f *frames.LLMTextFrame,
) ([]frames.Frame, error) {
	e.mu.Lock()
	e.raw.WriteString(f.Text)

	// The prompt tells the model to emit only the marker; if it disobeys and
	// streams more text, it stays suppressed.
	switch e.marker {
	case turnMarkerNone:
		// Still looking for the marker, below.
	case TurnMarkerIncomplete:
		e.mu.Unlock()
		return nil, nil
	case TurnMarkerComplete:
		e.mu.Unlock()
		return []frames.Frame{f}, nil
	}

	e.buffer += f.Text
	buffer := e.buffer
	completeMarker, shortMarker, longMarker := e.config.Markers()

	// The short marker is looked for first, matching the order the protocol
	// presents them in: the user was cut off or needs time, and the marker is
	// the whole response.
	incomplete, found := IncompleteShort, ""
	switch {
	case strings.Contains(buffer, shortMarker):
		incomplete, found = IncompleteShort, shortMarker
	case strings.Contains(buffer, longMarker):
		incomplete, found = IncompleteLong, longMarker
	}
	if found != "" {
		e.marker = TurnMarkerIncomplete
		e.buffer = ""
		e.mu.Unlock()
		slog.Debug("an incomplete turn was reported, suppressing the response",
			"kind", incomplete.String(), "marker", found)
		// The bot produces no spoken output for an incomplete turn, so the
		// marker is the whole context entry: a stand-alone assistant message.
		marker, err := e.mark(ctx, found, incomplete.String())
		if err != nil {
			return nil, err
		}
		return []frames.Frame{marker}, nil
	}

	_, rest, ok := strings.Cut(buffer, completeMarker)
	if !ok {
		e.mu.Unlock()
		return nil, nil
	}
	e.buffer = ""
	e.marker = TurnMarkerComplete
	e.mu.Unlock()
	slog.Debug("a complete turn was reported, pushing the buffered response")

	// The assistant aggregator prepends the marker to the aggregated text that
	// follows, so the context message reads as the marker followed by the
	// response.
	marker, err := e.mark(ctx, completeMarker, completionKindComplete)
	if err != nil {
		return nil, err
	}
	out := []frames.Frame{marker}
	// The marker and text can arrive in one chunk, so whatever followed the
	// marker in this chunk is pushed as speech.
	if rest = strings.TrimPrefix(rest, " "); rest != "" {
		out = append(out, frames.NewLLMTextFrame(rest))
	}
	return out, nil
}

// mark reports the marker read from the response, and returns its frame for
// the conversation. Only a complete turn's marker belongs to the response that
// follows.
func (e *TurnCompletionExtension) mark(ctx context.Context, marker, kind string) (*frames.LLMMarkerFrame, error) {
	e.mu.Lock()
	e.foundMarker, e.foundKind = marker, kind
	e.mu.Unlock()
	if err := e.reportTurnCompletion(ctx, kind, marker); err != nil {
		return nil, err
	}
	mf := frames.NewLLMMarkerFrame(marker)
	mf.AppendToContextImmediately = kind != completionKindComplete
	return mf, nil
}

// reportTurnCompletion pushes the turn-completion judgment upstream, once per
// inference, as a UserTurnCompletionFrame naming the inference it judges. A
// complete turn is also broadcast as a UserTurnInferenceCompletedFrame.
func (e *TurnCompletionExtension) reportTurnCompletion(ctx context.Context, kind, marker string) error {
	e.mu.Lock()
	if e.reported {
		e.mu.Unlock()
		return nil
	}
	e.reported = true
	id := e.contextFrameID
	e.mu.Unlock()

	report := frames.NewUserTurnCompletionFrame(kind, marker, id)
	if err := e.PushFrame(ctx, report, processor.Upstream); err != nil {
		return err
	}
	if kind != completionKindComplete {
		return nil
	}
	return e.BroadcastFrame(ctx, func() frames.Frame { return frames.NewUserTurnInferenceCompletedFrame() })
}

// responseReport is the response's raw text and marker, once there was a
// response to read, or nil. It is a diagnostic for a consumer checking the
// model against the protocol it was given, and plays no part in the
// conversation.
func (e *TurnCompletionExtension) responseReport() *frames.LLMMarkerResponseFrame {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.raw.Len() == 0 && e.foundKind == "" {
		return nil
	}
	complete, short, long := e.config.Markers()
	report := frames.NewLLMMarkerResponseFrame(e.raw.String(), e.foundMarker, e.foundKind,
		[]string{complete, short, long})
	e.raw.Reset()
	e.foundMarker, e.foundKind = "", ""
	return report
}

// resetTurn resets the turn's state between responses. It returns the text
// buffered without a marker ever appearing, as a frame to push so it is not
// lost, so a model that ignored the protocol still says something.
func (e *TurnCompletionExtension) resetTurn() []frames.Frame {
	e.mu.Lock()
	var out []frames.Frame
	orphaned := e.marker == turnMarkerNone && e.buffer != ""
	if orphaned {
		out = append(out, frames.NewLLMTextFrame(e.buffer))
	}
	markers := e.config.Markers
	e.buffer = ""
	e.marker = turnMarkerNone
	e.mu.Unlock()
	if orphaned {
		complete, short, long := markers()
		slog.Warn("the model's response did not contain a turn completion marker (" +
			complete + "/" + short + "/" + long + "); pushing its text anyway, as the system prompt may be " +
			"missing the turn completion instructions")
	}
	return out
}
