package observers

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

// Bounds on what is worth naming in a timeline.
const (
	// defaultMinContribution is the shortest stretch reported in its own right.
	// Anything shorter is a frame hop rather than work worth naming, so it is
	// rolled into the single pipeline contribution.
	defaultMinContribution = 5 * time.Millisecond
	// printsAsZero is half of the last digit TurnContributionLines prints, below
	// which a stretch would show as zero and so is not part of the timeline at
	// all.
	printsAsZero = 500 * time.Microsecond
)

// momentKind is a point in a user-to-bot cycle that a contribution can start or
// end at.
//
// Most are observed as frames. momentLLMChunk and momentSentence are worked back
// from a service's own metric, because no frame marks them.
type momentKind string

// The points a cycle is divided at.
const (
	momentSilence          momentKind = "silence"
	momentReady            momentKind = "ready"
	momentVADStop          momentKind = "vad stop"
	momentTranscript       momentKind = "transcript"
	momentLLMRequest       momentKind = "llm request"
	momentLLMChunk         momentKind = "llm chunk"
	momentMarkerComplete   momentKind = "marker complete"
	momentMarkerIncomplete momentKind = "marker incomplete"
	momentHandlersStart    momentKind = "handlers start"
	momentHandlersEnd      momentKind = "handlers end"
	momentFirstText        momentKind = "first text"
	momentSentence         momentKind = "sentence"
	momentFirstAudio       momentKind = "first audio"
	momentBotSpeaking      momentKind = "bot speaking"
)

// moment is one moment in a cycle.
type moment struct {
	// at is when the moment happened.
	at time.Time
	// kind is what happened.
	kind momentKind
	// source is the name of whatever produced it, used when a span takes its
	// owner from one end.
	source string
	// derived reports that it was worked back from a metric rather than observed
	// as a frame. A derived moment sits at or before the frame that reported it,
	// which decides the order when the two share a time.
	derived bool
}

// ownerSource is which end of a span supplies its owner, when the table does not
// name one outright.
type ownerSource int

const (
	// ownerLiteral takes the owner from the span's own owner field.
	ownerLiteral ownerSource = iota
	ownerOpener
	ownerCloser
)

// The settings that govern time nobody's code is spending.
const (
	ownerTurnCompletion = "config: filter_incomplete_user_turns"
	ownerVADStopSecs    = "config: VAD StopSecs"
	ownerTurnStrategies = "config: user turn strategies"
	ownerTextAggregator = "config: text aggregator"
	// ownerBotCode is the application's own code rather than any setting.
	ownerBotCode = "bot"
	// ownerPipeline is the framework itself, which is what unnamed time is.
	ownerPipeline = "jargo"
)

// The labels the table uses more than once, where one stretch is named by
// several pairs of moments.
const (
	keyTurnCompletion    = "turn_completion"
	labelTurnCompletion  = "turn completion"
	labelSpeechSynthesis = "speech synthesis"
	labelLLMInference    = "LLM inference"
)

// span is one stretch of a cycle, named by the moments at its ends.
type span struct {
	// start is the moment the stretch runs from, end the moment it runs to.
	start, end momentKind
	// key is a stable identifier, which survives a label being reworded.
	key string
	// label is how the stretch reads in a timeline.
	label string
	// owner is the setting that governs the time, or the bot whose code spends
	// it, when ownerFrom is ownerLiteral.
	owner string
	// ownerFrom says which end of the span supplies the processor that spends
	// the time, when the owner is not named outright.
	ownerFrom ownerSource
	// core reports a core stage, which stays listed however brief so the
	// timeline keeps its shape from one turn to the next.
	core bool
}

// spans names the stretch between each pair of adjacent moments. A pair that is
// absent leaves that stretch unnamed, to be reported as pipeline time.
//
//nolint:gochecknoglobals // the timeline's shape, fixed at build time
var spans = []span{
	// Silence the VAD had to hear before it would call the turn over. It is a
	// setting rather than a service, and shortening it trades latency for
	// transcription accuracy.
	{
		start: momentSilence, end: momentVADStop,
		key: "endpointing_wait", label: "endpointing wait",
		owner: ownerVADStopSecs, core: true,
	},
	// From a pipeline that is ready with a client on it, to the bot asking for
	// something to say: the application's connected handler, and the frame it
	// queues reaching the LLM.
	{
		start: momentReady, end: momentLLMRequest,
		key: "first_request", label: "first request",
		owner: ownerBotCode, core: true,
	},
	{
		start: momentVADStop, end: momentTranscript,
		key: "transcription", label: "transcription",
		ownerFrom: ownerCloser, core: true,
	},
	// Whatever decides the user is done: a fixed speech timeout, an end-of-turn
	// model's inference, or an external signal.
	{
		start: momentTranscript, end: momentLLMRequest,
		key: "turn_detection", label: "turn detection",
		owner: ownerTurnStrategies, core: true,
	},
	{
		start: momentLLMRequest, end: momentLLMChunk,
		key: "llm_inference", label: labelLLMInference,
		ownerFrom: ownerOpener, core: true,
	},
	// The gate reading the verdict, whichever way it went.
	{
		start: momentLLMChunk, end: momentMarkerComplete,
		key: keyTurnCompletion, label: labelTurnCompletion,
		owner: ownerTurnCompletion,
	},
	{
		start: momentLLMChunk, end: momentMarkerIncomplete,
		key: keyTurnCompletion, label: labelTurnCompletion,
		owner: ownerTurnCompletion,
	},
	// The token the marker occupies before anything can be spoken.
	{
		start: momentMarkerComplete, end: momentFirstText,
		key: keyTurnCompletion, label: labelTurnCompletion,
		owner: ownerTurnCompletion,
	},
	{
		start: momentMarkerIncomplete, end: momentLLMRequest,
		key: "waiting_for_user", label: "waiting for user",
		owner: ownerTurnCompletion,
	},
	// Between the LLM's first chunk and the call being dispatched, the LLM is
	// still writing the call.
	{
		start: momentLLMChunk, end: momentHandlersStart,
		key: "llm_tool_call", label: "LLM tool call",
		ownerFrom: ownerOpener,
	},
	{
		start: momentHandlersStart, end: momentHandlersEnd,
		key: "function_handler", label: "function handler",
		ownerFrom: ownerCloser,
	},
	// Waiting for a full sentence before speaking any of it.
	{
		start: momentFirstText, end: momentSentence,
		key: "sentence_aggregation", label: "sentence aggregation",
		owner: ownerTextAggregator,
	},
	{
		start: momentSentence, end: momentFirstAudio,
		key: "speech_synthesis", label: labelSpeechSynthesis,
		ownerFrom: ownerCloser, core: true,
	},
	{
		start: momentFirstText, end: momentFirstAudio,
		key: "speech_synthesis", label: labelSpeechSynthesis,
		ownerFrom: ownerCloser, core: true,
	},
	{
		start: momentFirstAudio, end: momentBotSpeaking,
		key: "output_transport", label: "output transport",
		ownerFrom: ownerCloser,
	},
}

// spanKey identifies a stretch by the moments at its ends.
type spanKey struct{ start, end momentKind }

// spansByMoments indexes the table for lookup.
//
//nolint:gochecknoglobals // derived from spans, fixed at build time
var spansByMoments = func() map[spanKey]span {
	m := make(map[spanKey]span, len(spans))
	for _, s := range spans {
		m[spanKey{s.start, s.end}] = s
	}
	return m
}()

// One stretch is two spans rather than one, so it is chosen rather than looked
// up: where turn completion is in use, a response that carried no marker is
// buffered whole before anything can be spoken; where it is not, the same wait
// is the LLM still streaming, so its inference covers it.
//
//nolint:gochecknoglobals // the timeline's shape, fixed at build time
var (
	awaitingSpeakableText = span{
		start: momentLLMChunk, end: momentFirstText,
		key: "awaiting_speakable_text", label: "awaiting speakable text",
		ownerFrom: ownerCloser,
	}
	llmInferenceUntilText = span{
		start: momentLLMChunk, end: momentFirstText,
		key: "llm_inference", label: labelLLMInference,
		ownerFrom: ownerOpener, core: true,
	}
)

// MeasuredFrom is where a measured interval was anchored.
//
// A greeting is timed from the client connecting, a turn from the user falling
// silent, so a partial interval is never mistaken for a whole one.
type MeasuredFrom string

// The anchors an interval can be measured from.
const (
	// MeasuredFromNothing is the zero value, on a breakdown carrying no
	// contributions.
	MeasuredFromNothing         MeasuredFrom = ""
	MeasuredFromUserSilence     MeasuredFrom = "user_silence"
	MeasuredFromClientConnected MeasuredFrom = "client_connected"
)

// LatencyOwnerKind is what kind of thing a contribution's owner is.
//
// Grouping on this answers where a turn's time went without matching on the
// owner's name: a service that could be swapped, a setting the bot chose, the
// bot's own code, or the pipeline between them.
type LatencyOwnerKind string

// The kinds of thing that spend a turn's time.
const (
	OwnerService  LatencyOwnerKind = "service"
	OwnerSetting  LatencyOwnerKind = "setting"
	OwnerBot      LatencyOwnerKind = "bot"
	OwnerPipeline LatencyOwnerKind = "pipeline"
)

// LatencyContribution is one named part of the user-to-bot interval.
//
// Contributions account for the whole interval, so their durations sum to the
// measured latency. Unlike a TTFB, which reports what a service spent once it
// was asked, a contribution can also name time no service is measuring: the
// silence a VAD waits out, the tokens a turn-completion marker occupies before
// any speakable text, or a hold while the pipeline waits for the user to finish
// a sentence.
type LatencyContribution struct {
	// Key is a stable identifier for this part of a turn. Unlike Label, it is
	// safe to group on: it survives the label being reworded.
	Key string
	// Label is what the time was spent on.
	Label string
	// Owner is what spent it: a processor name, or a "config:" tag naming the
	// setting that governs it.
	Owner string
	// OwnerKind is what kind of thing the owner is.
	OwnerKind LatencyOwnerKind
	// StartTime is when this part started.
	StartTime time.Time
	// Duration is how long it took.
	Duration time.Duration
}

// TurnContributionLines formats the contributions for logging, one per line plus
// a total.
//
// One breakdown covers one user-to-bot cycle, so these lines describe a single
// turn, or, for the first thing the bot says, the wait before it from the
// pipeline being ready with a client on it, which the total line names.
//
// byCost orders by duration, largest first, rather than in the order things
// happened. It is what to ask for when the question is what to optimize rather
// than what the turn did.
//
// It returns nothing when no contributions were collected.
func (b LatencyBreakdown) TurnContributionLines(byCost bool) []string {
	if len(b.Contributions) == 0 {
		return nil
	}

	ordered := append([]LatencyContribution(nil), b.Contributions...)
	if byCost {
		sort.SliceStable(ordered, func(i, j int) bool {
			return ordered[i].Duration > ordered[j].Duration
		})
	}

	lines := make([]string, 0, len(ordered)+1)
	var total time.Duration
	for _, c := range ordered {
		lines = append(lines, fmt.Sprintf("%6.3fs  %-20s [%s]", c.Duration.Seconds(), c.Label, c.Owner))
		total += c.Duration
	}

	anchor := ""
	if b.MeasuredFrom == MeasuredFromClientConnected {
		anchor = " (from client connected)"
	}
	return append(lines, fmt.Sprintf("%6.3fs  TOTAL%s", total.Seconds(), anchor))
}

// mark records a moment, if a cycle is being measured. It returns the recorded
// moment, or nil when it was not recorded.
//
// once keeps only the first of a kind, for frames that repeat within a cycle
// such as text and audio. The caller holds o.mu.
func (o *UserBotLatency) mark(m moment, once bool) *moment {
	if !o.measuring() {
		return nil
	}
	if once && o.seen(m.kind) {
		return nil
	}
	if m.at.IsZero() {
		m.at = o.now()
	}
	o.moments = append(o.moments, m)
	// A copy rather than a pointer into the slice: appending to it later can
	// move the backing array, and a moment never changes once recorded.
	recorded := m
	return &recorded
}

// seen reports whether a moment of this kind has been recorded this cycle. The
// caller holds o.mu.
func (o *UserBotLatency) seen(kind momentKind) bool {
	for _, m := range o.moments {
		if m.kind == kind {
			return true
		}
	}
	return false
}

// measuring reports whether a cycle is open: a user turn, or the wait for the
// greeting. The caller holds o.mu.
func (o *UserBotLatency) measuring() bool {
	if !o.stopped.IsZero() {
		return true
	}
	return !o.clientConnected.IsZero() && !o.firstSpeechDone
}

// derivedMoments are the moments that come from metrics rather than from frames.
//
// Handlers dispatched together run concurrently, so they become one pair of
// moments named for the one the wait actually depends on: listing each would
// count the same stretch of wall clock more than once. The caller holds o.mu.
func (o *UserBotLatency) derivedMoments() []moment {
	var out []moment

	if o.textAgg != nil && o.textAgg.Duration > 0 {
		for _, m := range o.moments {
			if m.kind != momentFirstText {
				continue
			}
			out = append(out, moment{
				at:      m.at.Add(o.textAgg.Duration),
				kind:    momentSentence,
				derived: true,
			})
			break
		}
	}

	for _, group := range o.concurrentHandlers() {
		slowest, first, last := group[0], group[0].StartTime, group[0].StartTime.Add(group[0].Duration)
		for _, c := range group[1:] {
			if c.Duration > slowest.Duration {
				slowest = c
			}
			if c.StartTime.Before(first) {
				first = c.StartTime
			}
			if end := c.StartTime.Add(c.Duration); end.After(last) {
				last = end
			}
		}
		owner := slowest.FunctionName
		if len(group) > 1 {
			owner = fmt.Sprintf("%s +%d", owner, len(group)-1)
		}
		out = append(out,
			moment{at: first, kind: momentHandlersStart, derived: true},
			moment{at: last, kind: momentHandlersEnd, source: owner, derived: true},
		)
	}
	return out
}

// concurrentHandlers groups function handlers whose executions overlap in time,
// in the order they started. The caller holds o.mu.
func (o *UserBotLatency) concurrentHandlers() [][]FunctionCallMetrics {
	byStart := append([]FunctionCallMetrics(nil), o.calls...)
	sort.SliceStable(byStart, func(i, j int) bool {
		return byStart[i].StartTime.Before(byStart[j].StartTime)
	})

	var groups [][]FunctionCallMetrics
	for _, call := range byStart {
		if len(groups) > 0 {
			last := groups[len(groups)-1]
			end := last[0].StartTime.Add(last[0].Duration)
			for _, c := range last[1:] {
				if e := c.StartTime.Add(c.Duration); e.After(end) {
					end = e
				}
			}
			if !call.StartTime.After(end) {
				groups[len(groups)-1] = append(last, call)
				continue
			}
		}
		groups = append(groups, []FunctionCallMetrics{call})
	}
	return groups
}

// buildContributions names each stretch between one moment and the next.
//
// Spans run from one moment to the following one, so they tile the cycle and
// cannot overlap. A stretch whose pair of moments has no entry in the table is
// left unnamed and reported as pipeline time, which is what makes a gap in the
// accounting visible rather than silently absorbed.
//
// It returns nothing when the cycle was never measured. The caller holds o.mu.
func (o *UserBotLatency) buildContributions() []LatencyContribution {
	start := o.turnStart
	if start.IsZero() {
		// A greeting begins once the pipeline is ready and someone is there to
		// hear it, whichever came second: before the pipeline started, the
		// startup report has the time; before a client connected, the bot was
		// waiting on a person rather than working.
		for _, at := range []time.Time{o.clientConnected, o.pipelineStarted} {
			if at.After(start) {
				start = at
			}
		}
	}
	if start.IsZero() || !o.seen(momentBotSpeaking) {
		return nil
	}

	moments := o.timeline(start)

	var named []LatencyContribution
	core := map[string]struct{}{}
	for i := 0; i+1 < len(moments); i++ {
		c, isCore, ok := o.contributionFor(moments[i], moments[i+1])
		if !ok {
			continue
		}
		if isCore {
			core[c.Label] = struct{}{}
		}
		named = append(named, c)
	}

	named = mergeAdjacent(named)

	// A core stage stays listed however brief, but nothing is listed that would
	// print as zero: a stage that took no measurable time is not part of the
	// timeline.
	kept := named[:0]
	for _, c := range named {
		_, isCore := core[c.Label]
		if c.Duration >= o.minContribution() || (isCore && c.Duration >= printsAsZero) {
			kept = append(kept, c)
		}
	}

	// Time the named stages did not cover, which includes the spans the
	// threshold filtered out. It is spread across the cycle rather than sitting
	// anywhere in it, so it is listed last, and it is listed whenever there is
	// any of it, so the contributions always sum to the interval they describe.
	pipeline := moments[len(moments)-1].at.Sub(start)
	for _, c := range kept {
		pipeline -= c.Duration
	}
	if pipeline >= printsAsZero {
		kept = append(kept, LatencyContribution{
			Key:       "pipeline",
			Label:     "pipeline",
			Owner:     ownerPipeline,
			OwnerKind: OwnerPipeline,
			StartTime: start,
			Duration:  pipeline,
		})
	}
	return kept
}

// timeline is every moment of the cycle from start onwards, in the order they
// happened, opened by the moment the interval is anchored at. The caller holds
// o.mu.
func (o *UserBotLatency) timeline(start time.Time) []moment {
	anchor := moment{at: start, kind: momentReady}
	if !o.turnStart.IsZero() {
		anchor.kind = momentSilence
	}
	moments := []moment{anchor}
	for _, m := range o.moments {
		if !m.at.Before(start) {
			moments = append(moments, m)
		}
	}
	moments = append(moments, o.derivedMoments()...)
	sort.SliceStable(moments, func(i, j int) bool {
		if !moments[i].at.Equal(moments[j].at) {
			return moments[i].at.Before(moments[j].at)
		}
		// A derived moment sits at or before the frame that reported it.
		return moments[i].derived && !moments[j].derived
	})
	return moments
}

// contributionFor names the stretch between two adjacent moments, reporting
// false for a pair the table leaves unnamed or one with no duration. The second
// result reports a core stage, which stays listed however brief.
func (o *UserBotLatency) contributionFor(opener, closer moment) (LatencyContribution, bool, bool) {
	if !closer.at.After(opener.at) {
		return LatencyContribution{}, false, false
	}

	var s span
	var ok bool
	if opener.kind == momentLLMChunk && closer.kind == momentFirstText {
		s, ok = llmInferenceUntilText, true
		if o.markersSeen {
			s = awaitingSpeakableText
		}
	} else {
		s, ok = spansByMoments[spanKey{opener.kind, closer.kind}]
	}
	if !ok {
		return LatencyContribution{}, false, false
	}

	owner, ownerKind := s.owner, OwnerService
	switch s.ownerFrom {
	case ownerOpener:
		owner = opener.source
	case ownerCloser:
		owner = closer.source
	case ownerLiteral:
		// A literal owner is not a processor: either the setting that governs
		// the time, or the bot's own code.
		ownerKind = OwnerBot
		if strings.HasPrefix(owner, "config: ") {
			ownerKind = OwnerSetting
		}
	}

	return LatencyContribution{
		Key:       s.key,
		Label:     s.label,
		Owner:     owner,
		OwnerKind: ownerKind,
		StartTime: opener.at,
		Duration:  closer.at.Sub(opener.at),
	}, s.core, true
}

// mergeAdjacent joins neighboring spans that say the same thing.
//
// A marker sits in the middle of what a reader thinks of as one stretch, so the
// table names both halves the same way and they are joined here. Only spans that
// meet are joined: two of the same name either side of an unnamed gap stay
// apart, so the gap is reported rather than absorbed.
func mergeAdjacent(in []LatencyContribution) []LatencyContribution {
	var merged []LatencyContribution
	for _, c := range in {
		if len(merged) > 0 {
			prev := &merged[len(merged)-1]
			gap := c.StartTime.Sub(prev.StartTime.Add(prev.Duration))
			if gap < printsAsZero && prev.Label == c.Label && prev.Owner == c.Owner {
				prev.Duration = c.StartTime.Add(c.Duration).Sub(prev.StartTime)
				continue
			}
		}
		merged = append(merged, c)
	}
	return merged
}
