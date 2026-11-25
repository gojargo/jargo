package eval

import (
	"slices"
	"strconv"
	"strings"
)

// A simulation's defaults: the caps on the persona's turns, on the run's wall
// clock, and on a lull in which neither side does anything.
const (
	DefaultMaxTurns     = 20
	DefaultMaxDurationS = 300.0
	DefaultMaxSilenceS  = 30.0
)

// The measures a measured metric can name, computed by the harness from the
// run.
const (
	// MeasureTurns is the persona's turns.
	MeasureTurns = "turns"
	// MeasureDuration is the conversation's seconds, from its first line to the
	// hang-up.
	MeasureDuration = "duration"
	// MeasureWords is the longest bot reply in words, each Chinese or Japanese
	// character counting as one.
	MeasureWords = "words"
	// MeasureLatency is the slowest reply, in seconds.
	MeasureLatency = "latency"
	// MeasureFunctionCalls compares the calls the bot made with a list of the
	// calls it should make.
	MeasureFunctionCalls = "function_calls"
)

// SimulationMeasures are what a measured metric can measure.
//
//nolint:gochecknoglobals // fixed list, read only
var SimulationMeasures = []string{MeasureTurns, MeasureDuration, MeasureWords, MeasureLatency, MeasureFunctionCalls}

// SimulationMetric is a quality metric: a judged criterion, or a measure with a
// range or a call list.
type SimulationMetric struct {
	// Name is the metric's name in the results.
	Name string
	// Criterion is what the judge decides on each bot turn; empty for a
	// measured metric.
	Criterion string
	// MinScore is the share of the bot's turns the judge must answer yes for,
	// in 0..1, below which a judged metric fails the run; nil reports the score
	// without gating.
	MinScore *float64
	// Measure is one of SimulationMeasures; empty for a judged metric.
	Measure string
	// MinValue is the measured value's lower bound, inclusive, or nil.
	MinValue *float64
	// MaxValue is the measured value's upper bound, inclusive, or nil.
	MaxValue *float64
	// Calls, for function_calls, is the calls the bot should make: each a name,
	// or a name with Args as a subset of the call's arguments; an empty, non-nil
	// list means none. Nil for every other metric.
	Calls []FunctionCall
}

// Rubric is what a conversation is judged on, without the caller a simulation
// plays. It judges a conversation the harness didn't drive, such as a
// production call. In a scenario file, an entry with a success: and neither
// persona: nor turns: is a rubric. Its metrics are judged ones: a measured
// metric needs a live run.
type Rubric struct {
	// Name is the rubric's name.
	Name string
	// Success is what counts as the bot having done its job, judged over the
	// whole conversation.
	Success string
	// Metrics are the criteria every reply is judged on.
	Metrics []SimulationMetric
	// Judge is the judge.eval: block, as a simulation's.
	Judge JudgeEvalConfig
}

// NewRubric builds a rubric over the default judge, refusing a measured metric.
func NewRubric(name, success string, metrics []SimulationMetric) (*Rubric, error) {
	r := &Rubric{Name: name, Success: success, Metrics: metrics, Judge: DefaultJudgeEvalConfig()}
	return r, r.validate()
}

// validate refuses a measured metric: a measure needs a live run.
func (r *Rubric) validate() error {
	for _, m := range r.Metrics {
		if m.Measure != "" {
			return invalidf("rubric %s: metric %s is measured; a rubric's metrics are judged "+
				"('criterion:'), since a measure needs a live run", pyRepr(r.Name), pyRepr(m.Name))
		}
	}
	return nil
}

// ScenarioName is the rubric's name.
func (r *Rubric) ScenarioName() string { return r.Name }

func (*Rubric) loadedScenario() {}

// RubricFromMap builds a rubric from a mapping with a rubric entry's keys:
// name, success, and the optional metrics and judge block, each as a scenario
// file writes it; other keys are ignored. source is where the mapping came
// from, for error messages; empty is "rubric".
func RubricFromMap(data map[string]any, source string) (*Rubric, error) {
	where := source
	if where == "" {
		where = "rubric"
	}
	name, ok := data[keyName].(string)
	if !ok || name == "" {
		return nil, invalidf("%s: missing or invalid 'name:' field", where)
	}
	success, ok := data["success"].(string)
	if !ok || strings.TrimSpace(success) == "" {
		return nil, invalidf("%s: missing or invalid 'success:' field (a non-empty string)", where)
	}
	_, _, judge, err := parseJudgeBlock(data["judge"], where)
	if err != nil {
		return nil, err
	}
	metrics, err := parseMetrics(data["metrics"], where)
	if err != nil {
		return nil, invalidf("%s: %v", where, err)
	}
	r := &Rubric{Name: name, Success: strings.TrimSpace(success), Metrics: metrics, Judge: judge}
	if err := r.validate(); err != nil {
		return nil, invalidf("%s: %v", where, err)
	}
	return r, nil
}

// SimulationScenario is a simulated scenario: a caller rather than a script.
// A persona LLM holds the conversation with the bot on its own, and a judge
// reads the whole conversation.
type SimulationScenario struct {
	// Name is the simulation's name.
	Name string
	// Persona is who the caller is, as free text for the persona LLM.
	Persona string
	// Goal is what the caller wants from the call.
	Goal string
	// Success is what counts as the bot having done its job, for the judge.
	Success string
	// Simulator is the persona LLM, the same shape as judge.eval:; its zero
	// value is the default local model.
	Simulator LLMServiceConfig
	// Metrics are the quality metrics.
	Metrics []SimulationMetric
	// Configured is the simulation's user: and judge: blocks.
	Configured
	// MaxTurns caps the persona's turns.
	MaxTurns int
	// MaxDurationS caps the run's wall clock, in seconds.
	MaxDurationS float64
	// MaxSilenceS caps a lull with no event from either side, in seconds.
	MaxSilenceS float64
	// Runs is how many times the suite runs the simulation; every run must
	// pass.
	Runs int
	// SourcePath is the file the simulation was loaded from.
	SourcePath string
}

// ScenarioName is the simulation's name.
func (s *SimulationScenario) ScenarioName() string { return s.Name }

func (*SimulationScenario) loadedScenario() {}

// parseSimulation parses a simulation's mapping, as read from path.
func parseSimulation(data map[string]any, path string) (*SimulationScenario, error) {
	text := func(key string) (string, error) {
		v, ok := data[key].(string)
		if !ok || strings.TrimSpace(v) == "" {
			return "", invalidf("%s: missing or invalid '%s:' field (a non-empty string)", path, key)
		}
		return strings.TrimSpace(v), nil
	}

	simulator, err := parseSimulator(data["simulator"], path)
	if err != nil {
		return nil, err
	}
	userAudio, userSpeech, err := parseUserBlock(data["user"], path)
	if err != nil {
		return nil, err
	}
	if userAudio && userSpeech == nil {
		return nil, invalidf("%s: 'user.modality: audio' requires a 'user.speech:' block "+
			"(TTS service + voice) to synthesize the persona's turns", path)
	}
	botAudio, transcriber, judge, err := parseJudgeBlock(data["judge"], path)
	if err != nil {
		return nil, err
	}

	s := &SimulationScenario{
		Simulator:   simulator,
		UserAudio:   userAudio,
		UserSpeech:  userSpeech,
		BotAudio:    botAudio,
		Transcriber: transcriber,
		Judge:       judge,
		SourcePath:  path,
	}
	for _, f := range []stringField{
		{keyName, &s.Name}, {"persona", &s.Persona}, {"goal", &s.Goal}, {"success", &s.Success},
	} {
		if *f.dst, err = text(f.key); err != nil {
			return nil, err
		}
	}
	if s.Metrics, err = parseMetrics(data["metrics"], path); err != nil {
		return nil, err
	}
	if err = parseCaps(s, data, path); err != nil {
		return nil, err
	}
	return s, nil
}

// parseSimulator parses the simulator: block, the persona LLM; omitted, it is
// the default local model.
func parseSimulator(raw any, path string) (LLMServiceConfig, error) {
	if !truthy(raw) {
		return LLMServiceConfig{}, nil
	}
	block, ok := raw.(map[string]any)
	if !ok {
		return LLMServiceConfig{}, invalidf("%s: 'simulator:' must be a mapping naming the persona LLM", path)
	}
	return parseLLMConfig(block, "simulator", path)
}

// parseCaps reads a simulation's backstops and its run count.
func parseCaps(s *SimulationScenario, data map[string]any, path string) error {
	var err error
	if s.MaxTurns, err = positiveInt(data, "max_turns", DefaultMaxTurns, path); err != nil {
		return err
	}
	if s.MaxDurationS, err = positiveNumber(data, "max_duration_s", DefaultMaxDurationS, path); err != nil {
		return err
	}
	if s.MaxSilenceS, err = positiveNumber(data, "max_silence_s", DefaultMaxSilenceS, path); err != nil {
		return err
	}
	s.Runs, err = positiveInt(data, "runs", 1, path)
	return err
}

// parseMetrics parses the metrics: list.
func parseMetrics(raw any, path string) ([]SimulationMetric, error) {
	if raw == nil {
		return nil, nil
	}
	list, ok := raw.([]any)
	if !ok {
		return nil, invalidf("%s: 'metrics:' must be a list", path)
	}
	var metrics []SimulationMetric
	for idx, entry := range list {
		item, ok := entry.(map[string]any)
		if !ok {
			return nil, invalidf("%s: metric #%d must be a mapping", path, idx)
		}
		criterion, measure := item["criterion"], item["measure"]
		if (criterion == nil) == (measure == nil) {
			return nil, invalidf("%s: metric #%d is judged ('criterion:') or measured ('measure:'), "+
				"one of the two", path, idx)
		}
		rawName, named := item[keyName]
		if !named {
			rawName = measure
		}
		name, ok := rawName.(string)
		if !ok || name == "" {
			return nil, invalidf("%s: metric #%d needs a 'name:'", path, idx)
		}
		if slices.ContainsFunc(metrics, func(m SimulationMetric) bool { return m.Name == name }) {
			return nil, invalidf("%s: metric %s is listed twice", path, pyRepr(name))
		}
		metric, err := parseMetric(item, name, criterion, measure, path)
		if err != nil {
			return nil, err
		}
		metrics = append(metrics, metric)
	}
	return metrics, nil
}

// parseMetric parses one metric, judged or measured, once its name is known.
func parseMetric(item map[string]any, name string, criterion, measure any, path string) (SimulationMetric, error) {
	if measure != nil {
		return parseMeasure(item, name, measure, path)
	}
	text, ok := criterion.(string)
	if !ok || text == "" {
		return SimulationMetric{}, invalidf("%s: metric %s needs a 'criterion:' for the judge", path, pyRepr(name))
	}
	metric := SimulationMetric{Name: name, Criterion: text}
	if v := item["min_score"]; v != nil {
		score, ok := isNumber(v)
		if !ok || !(score >= 0 && score <= 1) {
			return SimulationMetric{}, invalidf("%s: metric %s 'min_score:' must be a number in 0..1", path, pyRepr(name))
		}
		metric.MinScore = &score
	}
	return metric, nil
}

// parseMeasure parses a measured metric: a known measure and at least one
// bound.
func parseMeasure(item map[string]any, name string, measure any, path string) (SimulationMetric, error) {
	m, ok := measure.(string)
	if !ok || !slices.Contains(SimulationMeasures, m) {
		return SimulationMetric{}, invalidf("%s: metric %s 'measure:' must be one of %s",
			path, pyRepr(name), strings.Join(SimulationMeasures, ", "))
	}
	if _, set := item["min_score"]; set {
		return SimulationMetric{}, invalidf("%s: metric %s is measured; it takes a range, not 'min_score:'",
			path, pyRepr(name))
	}
	if m == MeasureFunctionCalls {
		return parseCallsMeasure(item, name, path)
	}
	if _, set := item["calls"]; set {
		return SimulationMetric{}, invalidf("%s: metric %s takes a range; 'calls:' belongs to "+
			"'measure: function_calls'", path, pyRepr(name))
	}
	metric := SimulationMetric{Name: name, Measure: m}
	for _, b := range []struct {
		key string
		dst **float64
	}{{"min_value", &metric.MinValue}, {"max_value", &metric.MaxValue}} {
		v := item[b.key]
		if v == nil {
			continue
		}
		f, ok := isNumber(v)
		if !ok {
			return SimulationMetric{}, invalidf("%s: metric %s '%s:' must be a number", path, pyRepr(name), b.key)
		}
		*b.dst = &f
	}
	if metric.MinValue == nil && metric.MaxValue == nil {
		return SimulationMetric{}, invalidf("%s: metric %s needs a 'min_value:' or a 'max_value:'", path, pyRepr(name))
	}
	return metric, nil
}

// parseCallsMeasure parses a function_calls metric: the list of calls the bot
// should make.
func parseCallsMeasure(item map[string]any, name, path string) (SimulationMetric, error) {
	_, hasMin := item["min_value"]
	_, hasMax := item["max_value"]
	if hasMin || hasMax {
		return SimulationMetric{}, invalidf("%s: metric %s compares the bot's calls with 'calls:', not a range",
			path, pyRepr(name))
	}
	raw, ok := item["calls"].([]any)
	if !ok {
		return SimulationMetric{}, invalidf("%s: metric %s needs a 'calls:' list, the calls the bot should make "+
			"([] for none)", path, pyRepr(name))
	}
	calls := make([]FunctionCall, 0, len(raw))
	for idx, entry := range raw {
		call, err := metricCall(entry, name, path, idx)
		if err != nil {
			return SimulationMetric{}, err
		}
		calls = append(calls, call)
	}
	return SimulationMetric{Name: name, Measure: MeasureFunctionCalls, Calls: calls}, nil
}

// positiveInt reads a key that must be a positive integer, or its default.
func positiveInt(data map[string]any, key string, def int, path string) (int, error) {
	v, set := data[key]
	if !set {
		return def, nil
	}
	n, ok := isInt(v)
	if !ok || n < 1 {
		return 0, invalidf("%s: '%s:' must be a positive integer", path, key)
	}
	return n, nil
}

// positiveNumber reads a key that must be a positive number, or its default.
func positiveNumber(data map[string]any, key string, def float64, path string) (float64, error) {
	v, set := data[key]
	if !set {
		return def, nil
	}
	f, ok := isNumber(v)
	if !ok || f <= 0 {
		return 0, invalidf("%s: '%s:' must be a positive number", path, key)
	}
	return f, nil
}

// DescribeSimulation is a three-line summary of a simulation's user config,
// judge config, and goal, for pre-run logs: the lines of DescribeConfig, the
// user line also naming the persona LLM that plays the user and the run's caps,
// then the caller's goal:
//
//	user  -> modality: text | persona: ollama/gemma4:12b | max_turns: 8 | max_duration_s: 120 | max_silence_s: 30
//	judge -> modality: text | eval: ollama/gemma4:12b
//	goal  -> Book a table for two at 6 PM, then end the call.
//
// With color, the keywords are ANSI-colored as DescribeConfig colors them.
func DescribeSimulation(s *SimulationScenario, color bool) string {
	user := append(userSegments(&s.Configured),
		configSegment{"persona", llmIdentity(s.Simulator), cfgEval},
		configSegment{"max_turns", strconv.Itoa(s.MaxTurns), cfgLimit},
		configSegment{"max_duration_s", formatG(s.MaxDurationS), cfgLimit},
		configSegment{"max_silence_s", formatG(s.MaxSilenceS), cfgLimit},
	)
	return renderConfigLines([]configLine{
		{label: "user", segs: user},
		{label: "judge", segs: judgeSegments(&s.Configured)},
		{label: "goal", text: strings.Join(strings.Fields(s.Goal), " ")},
	}, color)
}

// formatG renders a number in its shortest form: 300, 7.5.
func formatG(f float64) string {
	return strconv.FormatFloat(f, 'g', 6, 64)
}

// metricCall parses one entry of a function_calls metric's calls: list: a name,
// or a mapping with a name and optional args.
func metricCall(entry any, metric, path string, idx int) (FunctionCall, error) {
	if s, ok := entry.(string); ok && s != "" {
		return FunctionCall{Name: s}, nil
	}
	m, _ := entry.(map[string]any)
	callName, ok := m[keyName].(string)
	if !ok || callName == "" {
		return FunctionCall{}, invalidf("%s: metric %s 'calls:' entry #%d must be a name or a mapping with 'name:'",
			path, pyRepr(metric), idx)
	}
	call := FunctionCall{Name: callName}
	if v := m["args"]; v != nil {
		args, ok := v.(map[string]any)
		if !ok {
			return FunctionCall{}, invalidf("%s: metric %s 'calls:' entry #%d 'args:' must be a mapping",
				path, pyRepr(metric), idx)
		}
		call.Args = args
	}
	return call, nil
}
