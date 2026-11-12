package nvidia

import (
	"context"
	"log/slog"
	"strings"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
	"github.com/gojargo/jargo/provider/openai/chat"
	"github.com/gojargo/jargo/utils/text"
)

// noKey is the key sent to an endpoint that takes none, a local NIM.
const noKey = "not-needed"

// NewLLM builds an NVIDIA NIM LLM service.
//
// Some models served by NIM reason before they answer, and the service keeps
// that reasoning off the answer, pushing it as thoughts so it is never spoken.
// A model that writes it into the answer text ahead of the answer, wrapped in
// <think> tags (DeepSeek-R1, some Nemotron models), has that leading block
// parsed out; one that streams it in a field of its own, reasoning_content
// (Nemotron Nano models), has the field read.
//
// A local NIM takes no API key, so none is needed when BaseURL points at one.
// The cloud endpoint requires one, and a warning is logged without it.
func NewLLM(cfg chat.LLMConfig) *chat.LLMService {
	if cfg.BaseURL == "" {
		cfg.BaseURL = baseURL
	}
	if cfg.APIKey == "" {
		if strings.Contains(cfg.BaseURL, "api.nvidia.com") {
			slog.Warn("NvidiaLLM: using the cloud endpoint but no API key was provided. " +
				"An API key is required for the cloud endpoint. " +
				"Set BaseURL to your local NIM endpoint for local deployments.")
		}
		cfg.APIKey = noKey
	}
	return chat.NewCompatLLM(chat.Compat{
		Name:         "NvidiaLLM",
		BaseURL:      baseURL,
		DefaultModel: defaultModel,
		StreamFilter: newReasoningFilter,
	}, cfg)
}

// reasoningFilter pushes a response's reasoning as thoughts: a leading
// <think> block in the answer text, and the reasoning_content field.
type reasoningFilter struct {
	s     *chat.LLMService
	think *text.TaggedLLMTextParser
	// hasReasoningField is set while a thought read from the reasoning field is
	// open.
	hasReasoningField bool
}

func newReasoningFilter(s *chat.LLMService) chat.StreamFilter {
	return &reasoningFilter{s: s, think: text.NewTaggedLLMTextParser("<think>", "</think>")}
}

// Delta pushes the delta's reasoning as thoughts, and strips a leading
// <think> block from its content, so the service reads only the answer.
func (f *reasoningFilter) Delta(ctx context.Context, d *chat.StreamDelta) error {
	switch {
	case d.ReasoningContent != "":
		if !f.hasReasoningField {
			f.s.StopTTFBMetrics(ctx)
			f.hasReasoningField = true
			if err := f.push(ctx, frames.NewLLMThoughtStartFrame()); err != nil {
				return err
			}
		}
		if err := f.push(ctx, frames.NewLLMThoughtTextFrame(d.ReasoningContent)); err != nil {
			return err
		}
	case f.hasReasoningField && d.Content != "":
		f.hasReasoningField = false
		if err := f.push(ctx, frames.NewLLMThoughtEndFrame()); err != nil {
			return err
		}
	}
	if d.Content == "" {
		return nil
	}
	var visible strings.Builder
	for _, c := range f.think.ParseNextChunk(d.Content) {
		if c.Kind == text.TaggedText {
			visible.WriteString(c.Text)
			continue
		}
		if err := f.pushThought(ctx, c); err != nil {
			return err
		}
	}
	d.Content = visible.String()
	return nil
}

// End releases the text held back while deciding whether the response opened
// with <think>, when it completed, and discards it when it was cut short. A
// thought left open is ended either way.
func (f *reasoningFilter) End(ctx context.Context, completed bool, emit func(string) error) error {
	var chunks []text.TaggedChunk
	if completed {
		chunks = f.think.Flush()
	} else {
		chunks = f.think.Reset()
	}
	for _, c := range chunks {
		var err error
		if c.Kind == text.TaggedText {
			err = emit(c.Text)
		} else {
			err = f.pushThought(ctx, c)
		}
		if err != nil {
			return err
		}
	}
	if f.hasReasoningField {
		f.hasReasoningField = false
		return f.push(ctx, frames.NewLLMThoughtEndFrame())
	}
	return nil
}

// pushThought pushes a piece of a <think> block as a thought frame.
func (f *reasoningFilter) pushThought(ctx context.Context, c text.TaggedChunk) error {
	switch c.Kind {
	case text.TaggedThoughtStart:
		f.s.StopTTFBMetrics(ctx)
		return f.push(ctx, frames.NewLLMThoughtStartFrame())
	case text.TaggedThoughtText:
		return f.push(ctx, frames.NewLLMThoughtTextFrame(c.Text))
	case text.TaggedThoughtEnd:
		return f.push(ctx, frames.NewLLMThoughtEndFrame())
	case text.TaggedText:
	}
	return nil
}

func (f *reasoningFilter) push(ctx context.Context, fr frames.Frame) error {
	return f.s.PushFrame(ctx, fr, processor.Downstream)
}
