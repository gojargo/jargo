package observers

import (
	"context"
	"fmt"
	"log/slog"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/processor"
)

// llmService is implemented by a language-model service, a realtime one
// included. The LLM logger uses it to report only what actually passed through
// a model, rather than every frame of that kind wherever it came from.
type llmService interface{ LLMService() }

// sttService is implemented by a speech-to-text service. The transcription
// logger uses it to report only the transcripts a transcriber produced, rather
// than any text frame carrying the same shape.
type sttService interface{ STTService() }

// logger returns the logger to write to, defaulting to the process logger.
func logger(l *slog.Logger) *slog.Logger {
	if l == nil {
		return slog.Default()
	}
	return l
}

// seconds renders a pipeline-clock timestamp the way the log lines report it.
func seconds(d time.Duration) float64 { return d.Seconds() }

// FrameEndpoint selects which end of a handover a filter decides on.
type FrameEndpoint int

const (
	// SourceEndpoint decides on the processor pushing the frame.
	SourceEndpoint FrameEndpoint = iota
	// DestinationEndpoint decides on the processor receiving it.
	DestinationEndpoint
)

// DebugFrameFilter narrows what a DebugLog observer reports to one kind of
// frame, optionally only where one end of the handover is a particular kind of
// processor.
type DebugFrameFilter struct {
	// Frame is an instance of the frame type to log; its type is what is
	// matched, not its contents.
	Frame frames.Frame
	// Match, when set, logs the frame only where the processor at Endpoint
	// satisfies it. Leave it nil to log the frame wherever it travels.
	Match func(processor.Processor) bool
	// Endpoint is the end of the handover Match decides on.
	Endpoint FrameEndpoint
}

// defaultExcludedFields are the frame fields a debug log leaves out unless told
// otherwise: the binary payloads, which say nothing a log reader wants and bury
// what does.
//
//nolint:gochecknoglobals // a constant set, kept in one place
var defaultExcludedFields = []string{"Audio", "Image", "Images"}

// DebugLogConfig configures a DebugLog observer.
type DebugLogConfig struct {
	// Logger is the destination; slog.Default() when nil.
	Logger *slog.Logger
	// Frames selects what to log. An empty list logs every frame, which on a
	// real pipeline is a great many.
	Frames []DebugFrameFilter
	// ExcludeFields names the frame fields left out of the log. A nil slice
	// leaves out the binary payloads (Audio, Image, Images); an empty non-nil
	// slice leaves out nothing.
	ExcludeFields []string
}

// DebugLog logs the frames going by with their contents, for working out what a
// pipeline is actually doing. Every exported field of a frame is rendered, so it
// serves any frame type without knowing anything about it.
type DebugLog struct {
	log     *slog.Logger
	filters []DebugFrameFilter
	exclude map[string]struct{}
}

// NewDebugLog builds a DebugLog observer.
func NewDebugLog(cfg DebugLogConfig) *DebugLog {
	excluded := cfg.ExcludeFields
	if excluded == nil {
		excluded = defaultExcludedFields
	}
	set := make(map[string]struct{}, len(excluded))
	for _, name := range excluded {
		set[name] = struct{}{}
	}
	return &DebugLog{log: logger(cfg.Logger), filters: cfg.Frames, exclude: set}
}

// OnPushFrame implements processor.Observer.
func (o *DebugLog) OnPushFrame(data processor.FramePushed) {
	if !o.shouldLog(data) {
		return
	}
	o.log.Log(context.Background(), slog.LevelDebug, "frame",
		"source", name(data.Source),
		"destination", name(data.Destination),
		"direction", data.Direction.String(),
		"frame", data.Frame.Name(),
		"fields", fields(data.Frame, o.exclude),
		"at", seconds(data.Timestamp),
	)
}

// shouldLog reports whether a handover passes the configured filters. No filters
// means every frame passes.
func (o *DebugLog) shouldLog(data processor.FramePushed) bool {
	if len(o.filters) == 0 {
		return true
	}
	want := reflect.TypeOf(data.Frame)
	for _, f := range o.filters {
		if reflect.TypeOf(f.Frame) != want {
			continue
		}
		if f.Match == nil {
			return true
		}
		if f.Endpoint == DestinationEndpoint {
			return f.Match(data.Destination)
		}
		return f.Match(data.Source)
	}
	return false
}

// fields renders the exported fields of a frame, leaving out the excluded ones
// and anything unset.
func fields(f frames.Frame, exclude map[string]struct{}) string {
	return strings.Join(structFields(reflect.Indirect(reflect.ValueOf(f)), exclude), ", ")
}

// structFields collects one struct's exported fields, descending into whatever
// it embeds. A frame that builds on another (a transcript on a text frame)
// carries what it inherited there, and the frame bases contribute nothing: what
// they hold is unexported.
func structFields(v reflect.Value, exclude map[string]struct{}) []string {
	if v.Kind() != reflect.Struct {
		return nil
	}
	t := v.Type()
	var parts []string
	for i := range t.NumField() {
		sf := t.Field(i)
		if !sf.IsExported() {
			continue
		}
		if sf.Anonymous {
			parts = append(parts, structFields(reflect.Indirect(v.Field(i)), exclude)...)
			continue
		}
		if _, skip := exclude[sf.Name]; skip {
			continue
		}
		rendered, ok := formatValue(v.Field(i))
		if !ok {
			continue
		}
		parts = append(parts, sf.Name+": "+rendered)
	}
	return parts
}

// formatValue renders one field, reporting false for one that was never set and
// so has nothing to say.
func formatValue(v reflect.Value) (string, bool) {
	switch v.Kind() {
	case reflect.Pointer, reflect.Interface, reflect.Map, reflect.Func:
		if v.IsNil() {
			return "", false
		}
	case reflect.Slice:
		if v.IsNil() {
			return "", false
		}
		if v.Type().Elem().Kind() == reflect.Uint8 {
			return fmt.Sprintf("%d bytes", v.Len()), true
		}
		if v.Len() == 0 {
			return "[]", true
		}
		// A long list of structured values is unreadable in a log line and is
		// usually a conversation, so it is reported by the count instead.
		if v.Len() > 3 && composite(v.Type().Elem()) {
			return fmt.Sprintf("%d items", v.Len()), true
		}
	case reflect.String:
		return fmt.Sprintf("%q", v.String()), true
	default:
		// Everything else renders as itself.
	}
	if named, ok := selfNaming(v); ok {
		return named, true
	}
	return fmt.Sprintf("%v", v.Interface()), true
}

// selfNaming renders a value that identifies itself by name, which is what a
// frame carries when it names a service: the switcher frames, a settings update
// aimed at one service, a summarizer given its own model.
//
// The value is never formatted. Its fields are the machinery it runs on rather
// than anything a log reader wants, and a service holds the configuration it was
// built with, so writing them out would put its API key in the log.
func selfNaming(v reflect.Value) (string, bool) {
	named, ok := v.Interface().(interface{ Name() string })
	if !ok {
		return "", false
	}
	return named.Name(), true
}

// composite reports whether a type is one whose values are too big to render
// inline.
func composite(t reflect.Type) bool {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	switch t.Kind() {
	case reflect.Struct, reflect.Map, reflect.Slice, reflect.Interface:
		return true
	default:
		return false
	}
}

// name is a processor's name, or "" when there is no processor (a frame reported
// with only one end of the handover filled in).
func name(p processor.Processor) string {
	if p == nil {
		return ""
	}
	return p.Name()
}

// LLMLogConfig configures an LLMLog observer.
type LLMLogConfig struct {
	// Logger is the destination; slog.Default() when nil.
	Logger *slog.Logger
}

// LLMLog logs what a language-model service was given and what it produced: the
// context it was asked to answer, the tokens it generated, the bounds of each
// response, and the tool calls it made along with their results.
//
// Only frames pushed to or from a model service are reported, so the same frame
// types traveling elsewhere in the pipeline are left alone.
type LLMLog struct {
	log *slog.Logger
}

// NewLLMLog builds an LLMLog observer.
func NewLLMLog(cfg LLMLogConfig) *LLMLog { return &LLMLog{log: logger(cfg.Logger)} }

// OnPushFrame implements processor.Observer.
func (o *LLMLog) OnPushFrame(data processor.FramePushed) {
	_, fromLLM := data.Source.(llmService)
	_, toLLM := data.Destination.(llmService)
	if !fromLLM && !toLLM {
		return
	}

	ctx := context.Background()
	at := seconds(data.Timestamp)
	switch f := data.Frame.(type) {
	case *frames.LLMFullResponseStartFrame:
		o.log.DebugContext(ctx, "llm response started", "service", name(data.Source), "at", at)
	case *frames.LLMFullResponseEndFrame:
		o.log.DebugContext(ctx, "llm response ended", "service", name(data.Source), "at", at)
	case *frames.LLMTextFrame:
		o.log.DebugContext(ctx, "llm generating", "service", name(data.Source), "text", f.Text, "at", at)
	case *frames.FunctionCallInProgressFrame:
		// Only the upstream half is reported: the call goes out both ways, and
		// logging both would report every call twice.
		if data.Direction != processor.Downstream {
			o.log.DebugContext(ctx, "llm function call",
				"service", name(data.Source),
				"tool_call_id", f.ToolCallID,
				"function", f.ToolName,
				"arguments", string(f.Args),
				"at", at,
			)
		}
	case *frames.LLMContextFrame:
		// The whole conversation, because what the model was asked is the point
		// of the line: a reply that reads wrong is usually a context that was.
		var messages []frames.Message
		if f.Context != nil {
			messages = f.Context.Messages()
		}
		o.log.DebugContext(ctx, "llm context",
			"service", name(data.Destination),
			"messages", messages,
			"at", at,
		)
	case *frames.FunctionCallResultFrame:
		o.log.DebugContext(ctx, "llm function call result",
			"service", name(data.Source),
			"tool_call_id", f.ToolCallID,
			"function", f.ToolName,
			"result", f.Result,
			"at", at,
		)
	}
}

// TranscriptionLogConfig configures a TranscriptionLog observer.
type TranscriptionLogConfig struct {
	// Logger is the destination; slog.Default() when nil.
	Logger *slog.Logger
}

// TranscriptionLog logs what a speech-to-text service heard, final transcripts
// and interim ones alike. Only what a transcriber produced is reported, so a
// transcript arriving from somewhere else is left alone.
type TranscriptionLog struct {
	log *slog.Logger
}

// NewTranscriptionLog builds a TranscriptionLog observer.
func NewTranscriptionLog(cfg TranscriptionLogConfig) *TranscriptionLog {
	return &TranscriptionLog{log: logger(cfg.Logger)}
}

// OnPushFrame implements processor.Observer.
func (o *TranscriptionLog) OnPushFrame(data processor.FramePushed) {
	if _, ok := data.Source.(sttService); !ok {
		return
	}
	ctx := context.Background()
	at := seconds(data.Timestamp)
	switch f := data.Frame.(type) {
	case *frames.TranscriptionFrame:
		o.log.DebugContext(ctx, "transcription",
			"service", name(data.Source), "text", f.Text, "user", f.UserID, "at", at)
	case *frames.InterimTranscriptionFrame:
		o.log.DebugContext(ctx, "interim transcription",
			"service", name(data.Source), "text", f.Text, "user", f.UserID, "at", at)
	}
}

// MetricsLogConfig configures a MetricsLog observer.
type MetricsLogConfig struct {
	// Logger is the destination; slog.Default() when nil.
	Logger *slog.Logger
	// Include selects the kinds of measurement to log, given as instances of
	// the frames.MetricsData types wanted. An empty list logs every kind.
	Include []frames.MetricsData
}

// MetricsLog logs the measurements the pipeline reports: what each service took
// to answer, how much it was given, and what it billed.
type MetricsLog struct {
	log     *slog.Logger
	include map[reflect.Type]struct{}

	mu sync.Mutex
	dd deduper
}

// NewMetricsLog builds a MetricsLog observer.
func NewMetricsLog(cfg MetricsLogConfig) *MetricsLog {
	var include map[reflect.Type]struct{}
	if len(cfg.Include) > 0 {
		include = make(map[reflect.Type]struct{}, len(cfg.Include))
		for _, d := range cfg.Include {
			include[reflect.TypeOf(d)] = struct{}{}
		}
	}
	return &MetricsLog{log: logger(cfg.Logger), include: include, dd: newDeduper(0)}
}

// OnPushFrame implements processor.Observer.
func (o *MetricsLog) OnPushFrame(data processor.FramePushed) {
	f, ok := data.Frame.(*frames.MetricsFrame)
	if !ok {
		return
	}
	// A metrics frame is reported at every handover it makes, and the
	// measurements it carries are the same each time.
	o.mu.Lock()
	seen := o.dd.seenBefore(f.ID())
	o.mu.Unlock()
	if seen {
		return
	}
	for _, d := range f.Data {
		if !o.shouldLog(d) {
			continue
		}
		o.logMetric(d, seconds(data.Timestamp))
	}
}

// shouldLog reports whether one measurement is one of the kinds asked for.
func (o *MetricsLog) shouldLog(d frames.MetricsData) bool {
	if o.include == nil {
		return true
	}
	_, ok := o.include[reflect.TypeOf(d)]
	return ok
}

// logMetric writes one measurement.
func (o *MetricsLog) logMetric(d frames.MetricsData, at float64) {
	ctx := context.Background()
	base := []any{"processor", d.MetricsProcessor()}
	if m := d.MetricsModel(); m != "" {
		base = append(base, "model", m)
	}
	with := func(rest ...any) []any { return append(append(base[:len(base):len(base)], rest...), "at", at) }

	switch m := d.(type) {
	case frames.TTFBMetricsData:
		o.log.DebugContext(ctx, "ttfb", with("value", m.Value.Seconds())...)
	case frames.TTFAMetricsData:
		o.log.DebugContext(ctx, "ttfa", with(
			"value", m.TTFA.Seconds(), "leading_silence", m.LeadingSilence.Seconds())...)
	case frames.ProcessingMetricsData:
		o.log.DebugContext(ctx, "processing time", with("value", m.Value.Seconds())...)
	case frames.LLMUsageMetricsData:
		o.log.DebugContext(ctx, "llm token usage", with(tokenUsage(m.Value)...)...)
	case frames.STTUsageMetricsData:
		o.log.DebugContext(ctx, "stt usage", with("audio_seconds", m.Value.AudioSeconds)...)
	case frames.TTSUsageMetricsData:
		o.log.DebugContext(ctx, "tts usage", with("characters", m.Value)...)
	case frames.TextAggregationMetricsData:
		o.log.DebugContext(ctx, "text aggregation", with("value", m.Value.Seconds())...)
	case frames.TurnMetricsData:
		o.log.DebugContext(ctx, "turn", with(
			"complete", m.Complete,
			"probability", m.Probability,
			"e2e_ms", float64(m.E2EProcessing.Microseconds())/1000,
		)...)
	default:
		o.log.DebugContext(ctx, "metrics", with("value", fmt.Sprintf("%v", d))...)
	}
}

// tokenUsage renders a token count as log attributes, reporting only the counts
// the service accounted for: an absent count is not a measured zero.
func tokenUsage(u frames.LLMTokenUsage) []any {
	attrs := []any{
		"prompt", u.PromptTokens,
		"completion", u.CompletionTokens,
		"total", u.TotalTokens,
	}
	optional := []struct {
		key   string
		count *int64
	}{
		{"cache_read", u.CacheReadTokens},
		{"cache_creation", u.CacheCreationTokens},
		{"reasoning", u.ReasoningTokens},
		{"input_audio", u.InputAudioTokens},
		{"output_audio", u.OutputAudioTokens},
		{"cache_read_audio", u.CacheReadAudioTokens},
		{"input_text", u.InputTextTokens},
		{"output_text", u.OutputTextTokens},
	}
	for _, o := range optional {
		if n, reported := frames.TokenCount(o.count); reported {
			attrs = append(attrs, o.key, n)
		}
	}
	return attrs
}

// Compile-time interface checks.
var (
	_ processor.Observer = (*DebugLog)(nil)
	_ processor.Observer = (*LLMLog)(nil)
	_ processor.Observer = (*TranscriptionLog)(nil)
	_ processor.Observer = (*MetricsLog)(nil)
)
