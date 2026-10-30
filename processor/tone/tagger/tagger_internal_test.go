package tagger

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gojargo/jargo/adapter/tts/cartesia"
	"github.com/gojargo/jargo/adapter/tts/elevenlabs"
	"github.com/gojargo/jargo/expressive"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/pipeline"
	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/tone/internal/shared"
	"github.com/gojargo/jargo/processor/tone/internal/tonetest"
)

var (
	errBoom      = errors.New("boom")
	errNoNetwork = errors.New("no network")
)

//nolint:gochecknoglobals // test fixtures
var expressiveParams = pipeline.Params{EnableExpressiveConversation: true}

func cartesiaMetadata() *frames.TTSMetadataFrame {
	f := frames.NewTTSMetadataFrame("tts")
	f.Expression = cartesia.CartesiaExpressionAdapter{}.Capabilities()
	return f
}

func elevenLabsMetadata() *frames.TTSMetadataFrame {
	f := frames.NewTTSMetadataFrame("tts")
	f.Expression = elevenlabs.ElevenLabsAudioTagExpressionAdapter{}.Capabilities()
	return f
}

type call struct {
	turn     TurnContext
	earlier  []string
	sentence string
}

// fakeModel labels each sentence from a script, keyed by the sentence.
type fakeModel struct {
	labels          map[string]SentenceLabels
	breathsPerReply *int
	err             error
	kinds           []BreathKind
	mu              sync.Mutex
	calls           []call
}

func (m *fakeModel) Tag(turn TurnContext, earlier []string, sentence string) (SentenceLabels, error) {
	m.mu.Lock()
	m.calls = append(m.calls, call{turn, earlier, sentence})
	m.mu.Unlock()
	if m.err != nil {
		return SentenceLabels{}, m.err
	}
	return m.labels[sentence], nil
}

func (m *fakeModel) BreathsPerReply() (int, bool) {
	if m.breathsPerReply == nil {
		return 0, false
	}
	return *m.breathsPerReply, true
}

func (m *fakeModel) BreathKinds() []BreathKind { return m.kinds }

func (m *fakeModel) made() []call {
	m.mu.Lock()
	defer m.mu.Unlock()
	return slices.Clone(m.calls)
}

func (m *fakeModel) turns() []TurnContext {
	var out []TurnContext
	for _, c := range m.made() {
		out = append(out, c.turn)
	}
	return out
}

// standIn makes each test load its own model, and opens models with download
// and load in place of the real ones.
func standIn(
	t *testing.T,
	download func(size Size, revision string, threads int) (model, error),
	load func(dir string, threads int) (model, error),
) {
	t.Helper()
	oldModels, oldDownload, oldLoad := models, downloadRelease, loadDir
	t.Cleanup(func() { models, downloadRelease, loadDir = oldModels, oldDownload, oldLoad })
	models = &shared.Cache[modelKey, model]{}
	downloadRelease = func(_ context.Context, size Size, revision string, threads int, _ string) (model, error) {
		return download(size, revision, threads)
	}
	loadDir = func(dir string, threads int) (model, error) {
		if load == nil {
			t.Fatal("loaded a folder")
		}
		return load(dir, threads)
	}
}

// unshared makes the next tagger load its own model.
func unshared() { models = &shared.Cache[modelKey, model]{} }

// tagger is a tagger whose model downloads as m.
func tagger(t *testing.T, m model) *Tagger {
	t.Helper()
	standIn(t, func(Size, string, int) (model, error) { return m, nil }, nil)
	return newTagger(t, Config{})
}

func newTagger(t *testing.T, cfg Config) *Tagger {
	t.Helper()
	tg, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	return tg
}

func reply(chunks ...string) []any {
	out := []any{frames.NewLLMFullResponseStartFrame()}
	for _, c := range chunks {
		out = append(out, frames.NewLLMTextFrame(c))
	}
	return append(out, frames.NewLLMFullResponseEndFrame())
}

func withPolicy(policy expressive.ExpressionPolicy) pipeline.Params {
	return pipeline.Params{EnableExpressiveConversation: true, ExpressionPolicy: &policy}
}

func texts(down []frames.Frame) []string {
	var out []string
	for _, f := range tonetest.Of[*frames.AggregatedTextFrame](down) {
		out = append(out, f.Text)
	}
	return out
}

func sends(parts ...any) []any {
	var out []any
	for _, p := range parts {
		if list, ok := p.([]any); ok {
			out = append(out, list...)
		} else {
			out = append(out, p)
		}
	}
	return out
}

// tagged runs a tagger whose model is m, with params, and returns the text it
// sent on.
func tagged(t *testing.T, m model, params pipeline.Params, send ...any) []string {
	t.Helper()
	unshared()
	down, _ := tonetest.Run(t, tagger(t, m), params, sends(send...)...)
	return texts(down)
}

func expectTexts(t *testing.T, got []string, want ...string) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Errorf("tagged %q, want %q", got, want)
	}
}

func roles(messages []Message) [][2]string {
	var out [][2]string
	for _, m := range messages {
		out = append(out, [2]string{m.Role, m.Text})
	}
	return out
}

func breathe() expressive.ExpressionPolicy {
	return expressive.ExpressionPolicy{Guidance: []expressive.Guidance{expressive.GuidanceBreathe}}
}

func breathBefore() SentenceLabels { return SentenceLabels{SoundBefore: "breath"} }

func TestTheTaggerAnnouncesItselfBothWaysAtStart(t *testing.T) {
	down, up := tonetest.Run(t, tagger(t, &fakeModel{}), expressiveParams)

	if len(tonetest.Of[*frames.ExpressionMetadataFrame](down)) == 0 {
		t.Error("not announced downstream")
	}
	if len(tonetest.Of[*frames.ExpressionMetadataFrame](up)) == 0 {
		t.Error("not announced upstream")
	}
}

func TestLabelsAreWrittenByTheirUniversalNames(t *testing.T) {
	m := &fakeModel{labels: map[string]SentenceLabels{
		"Congratulations!":    {Delivery: "warm", SoundAfter: "laugh"},
		"Your code is A B 7.": {Spans: []Span{{13, 18, "slow"}}},
	}}

	got := tagged(t, m, expressiveParams, cartesiaMetadata(), reply("Congratulations! Your code ", "is A B 7."))

	expectTexts(t, got, "<<warm>> Congratulations! [[laugh]]", "<<neutral>> Your code is <<slow: A B 7>>.")
}

func TestADeliveryIsWrittenOnlyWhereItChanges(t *testing.T) {
	calm := SentenceLabels{Delivery: "calm"}
	m := &fakeModel{labels: map[string]SentenceLabels{"I'm sorry.": calm, "I'll fix it.": calm}}

	got := tagged(t, m, expressiveParams, elevenLabsMetadata(), reply("I'm sorry. I'll fix it."))

	expectTexts(t, got, "<<calm>> I'm sorry.", "I'll fix it.")
}

func TestEachSentenceIsTaggedAfterTheOnesBeforeIt(t *testing.T) {
	m := &fakeModel{}

	tagged(t, m, expressiveParams, elevenLabsMetadata(), reply("One. Two. Three."))

	type seen struct {
		earlier  string
		sentence string
	}
	var got []seen
	for _, c := range m.made() {
		got = append(got, seen{strings.Join(c.earlier, "|"), c.sentence})
	}
	want := []seen{{"", "One."}, {"One.", "Two."}, {"One.|Two.", "Three."}}
	if !slices.Equal(got, want) {
		t.Errorf("calls %q, want %q", got, want)
	}
}

func TestThePolicyNarrowsTheNamesWritten(t *testing.T) {
	m := &fakeModel{labels: map[string]SentenceLabels{"Great!": {Delivery: "warm", SoundBefore: "laugh"}}}
	policy := expressive.ExpressionPolicy{
		Delivery: []expressive.ExpressionDelivery{expressive.DeliveryCalm},
		Sounds:   []expressive.ExpressionSound{},
	}

	got := tagged(t, m, withPolicy(policy), cartesiaMetadata(), reply("Great!"))

	expectTexts(t, got, "Great!")
}

func TestASentenceThatCantBeTaggedIsSpokenUntaggedWithAnError(t *testing.T) {
	m := &fakeModel{err: errBoom}

	down, up := tonetest.Run(t, tagger(t, m), expressiveParams, sends(elevenLabsMetadata(), reply("Hello there."))...)

	expectTexts(t, texts(down), "Hello there.")
	errs := tonetest.Of[*frames.ErrorFrame](up)
	if len(errs) == 0 || errs[0].Fatal {
		t.Errorf("errors %v, want a non-fatal one", errs)
	}
}

func TestABreathIsOfferedOnlyWithTheBreathePreset(t *testing.T) {
	m := &fakeModel{}

	tagged(t, m, expressiveParams, elevenLabsMetadata(), reply("Hi."))
	tagged(t, m, withPolicy(breathe()), elevenLabsMetadata(), reply("Hi."))

	turns := m.turns()
	without, with := turns[0].Space, turns[1].Space
	if without.Sounds.Has("breath") || without.Breaths != nil {
		t.Errorf("without the preset, sounds %v and breaths %v", without.Sounds, without.Breaths)
	}
	if name, _ := with.Sounds.Name("breath"); name != "breath" {
		t.Errorf("breath is written %q", name)
	}
	want := Names{{"light breath", "light breath"}, {"deep breath", "deep breath"}, {"exhale", "exhale"}}
	if !reflect.DeepEqual(with.Breaths, want) {
		t.Errorf("breaths %v, want %v", with.Breaths, want)
	}
}

func TestTheCallIsKeptFromTheTurnFrames(t *testing.T) {
	m := &fakeModel{}
	user := frames.NewLLMContextUserTurnFrame("My order is late.", "t1")
	user.Perception = []expressive.PerceptionSignal{{Name: "emotion", Labels: []expressive.PerceptionLabel{
		{Label: "angry", Confidence: 0.62}, {Label: "disgusted", Confidence: 0.2},
	}}}

	tagged(t, m, expressiveParams, elevenLabsMetadata(),
		frames.NewLLMContextAssistantTurnFrame("How can I help?", "t0"), user, reply("Sorry."))

	turn := m.made()[0].turn
	want := [][2]string{{"assistant", "How can I help?"}, {"user", "My order is late."}}
	if got := roles(turn.Messages); !slices.Equal(got, want) {
		t.Errorf("messages %q, want %q", got, want)
	}
	// The reader's emotions, folded into the tones the model reads.
	heard := []expressive.PerceptionSignal{{Name: "tone", Labels: []expressive.PerceptionLabel{
		{Label: "frustrated", Confidence: 0.82},
	}}}
	if got := turn.Heard(); !reflect.DeepEqual(got, heard) {
		t.Errorf("heard %+v, want %+v", got, heard)
	}
}

func TestEachReplyIsTaggedKnowingTheLabelsOfTheOneBefore(t *testing.T) {
	calm := SentenceLabels{Delivery: "calm"}
	m := &fakeModel{labels: map[string]SentenceLabels{"I'm sorry.": calm}}

	tagged(t, m, expressiveParams, elevenLabsMetadata(), reply("I'm sorry. Let me look."), reply("Done."))

	turns := m.turns()
	if turns[0].Previous != nil {
		t.Errorf("first reply read %+v", turns[0].Previous)
	}
	if want := []SentenceLabels{calm, {}}; !reflect.DeepEqual(turns[2].Previous, want) {
		t.Errorf("second reply read %+v, want %+v", turns[2].Previous, want)
	}
}

func TestBreathsPastTheModelsLimitForAReplyAreDropped(t *testing.T) {
	one := 1
	m := &fakeModel{
		labels:          map[string]SentenceLabels{"One.": breathBefore(), "Two.": breathBefore()},
		breathsPerReply: &one,
	}

	got := tagged(t, m, withPolicy(breathe()), elevenLabsMetadata(), reply("One. Two."))

	expectTexts(t, got, "[[breath]] One.", "Two.")
}

func TestEachBreathIsDrawnAsAKindAtTheModelsShares(t *testing.T) {
	for _, kind := range []string{"light breath", "deep breath", "exhale"} {
		t.Run(kind, func(t *testing.T) {
			m := &fakeModel{labels: map[string]SentenceLabels{"One.": breathBefore()}, kinds: []BreathKind{{kind, 1.0}}}

			got := tagged(t, m, withPolicy(breathe()), elevenLabsMetadata(), reply("One."))

			expectTexts(t, got, "[["+kind+"]] One.")
		})
	}
}

func TestACallDrawsItsBreathsAlikeEachTime(t *testing.T) {
	kinds := []BreathKind{{"light breath", 0.59}, {"deep breath", 0.36}, {"exhale", 0.05}}
	var replies []any
	labels := map[string]SentenceLabels{}
	for n := range 8 {
		replies = append(replies, reply(fmt.Sprintf("Reply %d.", n))...)
		labels[fmt.Sprintf("Reply %d.", n)] = breathBefore()
	}

	var runs [2][]string
	for i := range runs {
		runs[i] = tagged(t, &fakeModel{labels: labels, kinds: kinds}, withPolicy(breathe()),
			elevenLabsMetadata(), replies)
	}

	if !slices.Equal(runs[0], runs[1]) {
		t.Errorf("drawn %q, then %q", runs[0], runs[1])
	}
	for _, s := range runs[0] {
		kind, _, _ := strings.Cut(s, "]]")
		if !slices.Contains([]string{"[[light breath", "[[deep breath", "[[exhale"}, kind) {
			t.Errorf("drawn %q", s)
		}
	}
}

func TestTheNextReplyReadsADrawnBreathAsABreath(t *testing.T) {
	m := &fakeModel{labels: map[string]SentenceLabels{"One.": breathBefore()}, kinds: []BreathKind{{"exhale", 1.0}}}

	tagged(t, m, withPolicy(breathe()), elevenLabsMetadata(), reply("One."), reply("Two."))

	if got := m.made()[1].turn.Previous; !reflect.DeepEqual(got, []SentenceLabels{breathBefore()}) {
		t.Errorf("the next reply read %+v", got)
	}
}

func TestAPolicyThatNamesOneBreathDrawsOnlyThatKind(t *testing.T) {
	labels := map[string]SentenceLabels{}
	var replies []any
	var want []string
	for n := range 6 {
		labels[fmt.Sprintf("Reply %d.", n)] = breathBefore()
		replies = append(replies, reply(fmt.Sprintf("Reply %d.", n))...)
		want = append(want, fmt.Sprintf("[[light breath]] Reply %d.", n))
	}
	m := &fakeModel{labels: labels, kinds: []BreathKind{{"light breath", 0.5}, {"exhale", 0.5}}}
	policy := breathe()
	policy.Sounds = []expressive.ExpressionSound{expressive.SoundLightBreath}

	got := tagged(t, m, withPolicy(policy), elevenLabsMetadata(), replies)

	expectTexts(t, got, want...)
}

func TestAPolicyThatNamesOnlyAKindStillBreathes(t *testing.T) {
	m := &fakeModel{
		labels: map[string]SentenceLabels{"One.": breathBefore()},
		kinds:  []BreathKind{{"light breath", 0.5}, {"deep breath", 0.5}},
	}
	policy := breathe()
	policy.Sounds = []expressive.ExpressionSound{expressive.SoundDeepBreath}

	got := tagged(t, m, withPolicy(policy), elevenLabsMetadata(), reply("One."))

	expectTexts(t, got, "[[deep breath]] One.")
}

func TestANameForBreathIsWrittenForEveryBreath(t *testing.T) {
	m := &fakeModel{labels: map[string]SentenceLabels{"One.": breathBefore()}, kinds: []BreathKind{{"exhale", 1.0}}}
	standIn(t, func(Size, string, int) (model, error) { return m, nil }, nil)
	tg := newTagger(t, Config{Vocabulary: map[expressive.UniversalName][]string{
		expressive.SoundBreath: {"inhales deeply"},
	}})

	down, _ := tonetest.Run(t, tg, withPolicy(breathe()), sends(elevenLabsMetadata(), reply("One."))...)

	expectTexts(t, texts(down), "[[inhales deeply]] One.")
}

func TestAModelsBreathKindsComeFromItsConfig(t *testing.T) {
	m := modelWithConfig(t, `{"breath_kinds": {"light breath": 0.6, "exhale": 0.4}}`)
	if want := []BreathKind{{"light breath", 0.6}, {"exhale", 0.4}}; !slices.Equal(m.BreathKinds(), want) {
		t.Errorf("breath kinds %v, want %v", m.BreathKinds(), want)
	}
	// A model without them writes a plain breath.
	if got := modelWithConfig(t, `{}`).BreathKinds(); len(got) != 0 {
		t.Errorf("breath kinds %v, want none", got)
	}
}

// decoded is a sentence's labels from a model's six heads: the sound before's
// logits and the breath head's.
func decoded(t *testing.T, before []float32, breathLogit float32, sentence string) SentenceLabels {
	t.Helper()
	m := modelWithConfig(t, `{
		"deliveries": ["none", "calm"],
		"sounds": ["none", "laugh", "chuckle", "sigh", "breath"],
		"sound_bias": {"breath": [[5, -1.0], [null, 0.0]]}
	}`)
	space := LabelSpace{
		Deliveries: Names{{"calm", "calm"}},
		Sounds:     Names{{"laugh", "laughs"}, {"breath", "quick breath"}},
	}
	quiet := []float32{9, 0, 0, 0, 0}
	tokens := [][]float32{{9, 0, 0}}
	h := heads{
		delivery: []float32{9, 0}, before: before, after: quiet,
		spans: tokens, emphasis: tokens, breath: []float32{breathLogit},
	}
	labels, err := m.decode(h, []offset{{0, 2, true}}, sentence, space)
	if err != nil {
		t.Fatal(err)
	}
	return labels
}

func TestABreathComesFromItsOwnHeadShiftedForTheSentencesLength(t *testing.T) {
	quiet := []float32{9, 0, 0, 0, 0}

	// Up to five words, the breath head's logit is shifted down by one.
	if got := decoded(t, quiet, 0.5, "Hi.").SoundBefore; got != "" {
		t.Errorf("a short sentence breathes: %q", got)
	}
	if got := decoded(t, quiet, 0.5, strings.Repeat("Hi ", 6)).SoundBefore; got != "breath" {
		t.Errorf("a long sentence has %q before it, want a breath", got)
	}
}

func TestTheSoundHeadsNeverChooseABreath(t *testing.T) {
	breathClassWins := []float32{0, 0, 0, 0, 9}

	if got := decoded(t, breathClassWins, -1, strings.Repeat("Hi ", 6)).SoundBefore; got != "" {
		t.Errorf("sound before %q, want none", got)
	}
}

func TestABreathGoesOnlyWhereNoOtherSoundStartsTheSentence(t *testing.T) {
	laugh := []float32{0, 9, 0, 0, 0}

	if got := decoded(t, laugh, 9, strings.Repeat("Hi ", 6)).SoundBefore; got != "laugh" {
		t.Errorf("sound before %q, want laugh", got)
	}
}

func TestASentenceIsReadAsTheTextTheModelWasTrainedOn(t *testing.T) {
	policy := expressive.ExpressionPolicy{Guidance: []expressive.Guidance{
		expressive.GuidanceMatchStyle, expressive.GuidanceSlowForDetails, expressive.GuidanceBreathe,
	}}
	heard := []expressive.PerceptionSignal{{Name: "tone", Labels: []expressive.PerceptionLabel{
		{Label: "frustrated", Confidence: 0.62}, {Label: "neutral", Confidence: 0.3},
	}}}
	space := ResolveLabelSpace(elevenlabs.ElevenLabsAudioTagExpressionAdapter{}.Capabilities(), policy,
		"ElevenLabsTTSService#0", nil)
	turn := TurnContext{
		Messages: []Message{
			{Role: "assistant", Text: "Thanks for calling. How can I help?"},
			{Role: "user", Text: "I need my new booking code.", Heard: heard},
		},
		Space:    space,
		Policy:   policy,
		Previous: []SentenceLabels{{Delivery: "calm"}, {SoundBefore: "breath"}},
	}
	// tagger-rc1's layout.
	l := defaultLayout()
	l.PreviousTags = true

	call, sentence, start := encode(turn, []string{"Sure, one moment."}, "Your new code is AB12CD.", l)

	// What training wrote for this turn.
	if want := "caller (tone frustrated likely, neutral guess): I need my new booking code."; call != want {
		t.Errorf("call %q, want %q", call, want)
	}
	want := "tts elevenlabs | can calm sympathetic apologetic warm excited surprised sad grateful " +
		"curious confident slow fast | sounds laugh chuckle sigh breath | stress yes | policy " +
		"match style, slow for details, breathe | caller sounded tone frustrated likely, neutral " +
		"guess | last reply calm breath | state new detail | reply so far: Sure, one moment. | " +
		"sentence: Your new code is AB12CD."
	if sentence != want {
		t.Errorf("sentence\n %q, want\n %q", sentence, want)
	}
	if start != 354 {
		t.Errorf("start %d, want 354", start)
	}
}

func TestAServiceTheModelWasntTrainedWithIsReadAsOther(t *testing.T) {
	capabilities := elevenlabs.ElevenLabsAudioTagExpressionAdapter{}.Capabilities()
	policy := expressive.ExpressionPolicy{}

	if got := ResolveLabelSpace(capabilities, policy, "ElevenLabsTTSService#0", nil).TTS; got != "elevenlabs" {
		t.Errorf("read as %q", got)
	}
	if got := ResolveLabelSpace(capabilities, policy, "MyTTSService#0", nil).TTS; got != "other" {
		t.Errorf("read as %q", got)
	}
}

// The names the services the model was trained with take in this module, as
// their TTSMetadataFrame carries them, are read as those services.
func TestTheTrainedServicesAreReadByTheirNamesHere(t *testing.T) {
	for name, want := range map[string]string{
		"CartesiaTTS#0": "cartesia", "CartesiaHTTPTTS#1": "cartesia",
		"ElevenLabsTTS#0": "elevenlabs", "ElevenLabsRealtimeTTS#2": "elevenlabs",
		"FishTTS#0": "fish", "MiniMaxTTS#0": "minimax", "RimeTTS#0": "other",
		// Python lower-cases U+0130 to two code points, so it is no I here.
		"MİNİMAX": "other", "Kartesia": "other", "FISHK": "fish",
	} {
		if got := trainedService(name); got != want {
			t.Errorf("%s is read as %q, want %q", name, got, want)
		}
	}
}

func TestALayoutThatReadsNoTurnsReadsNoneOfTheCall(t *testing.T) {
	turn := TurnContext{Messages: []Message{{Role: "user", Text: "My order is late."}}}
	l := defaultLayout()
	l.ContextTurns = 0

	if call, _, _ := encode(turn, nil, "Sorry.", l); call != "" {
		t.Errorf("call %q, want none", call)
	}
}

func TestTextTheUserTypedIsPartOfTheCall(t *testing.T) {
	m := &fakeModel{}

	tagged(t, m, expressiveParams, elevenLabsMetadata(),
		frames.NewLLMContextAssistantTurnFrame("How can I help?", "t0"),
		// RTVI drains the pipeline before it appends text that runs at once.
		50*time.Millisecond,
		rtvi.NewSendTextFrame("1", "The spots have resolved!", ""),
		reply("That's encouraging news."))

	turn := m.made()[0].turn
	want := [][2]string{{"assistant", "How can I help?"}, {"user", "The spots have resolved!"}}
	if got := roles(turn.Messages); !slices.Equal(got, want) {
		t.Errorf("messages %q, want %q", got, want)
	}
	if turn.Heard() != nil {
		t.Errorf("heard %+v, want nothing", turn.Heard())
	}
}

func TestAConfirmedTurnReplacesItsSpeculationWhereItStands(t *testing.T) {
	m := &fakeModel{}
	early := frames.NewLLMContextUserTurnFrame("My order", "t1")
	early.Speculation = true

	tagged(t, m, expressiveParams, elevenLabsMetadata(),
		early, reply("Sorry."),
		frames.NewLLMContextAssistantTurnFrame("Sorry.", "t2"),
		frames.NewLLMContextUserTurnFrame("My order is late.", "t1"),
		reply("Next."))

	turns := m.turns()
	if got, want := roles(turns[0].Messages), [][2]string{{"user", "My order"}}; !slices.Equal(got, want) {
		t.Errorf("early reply read %q, want %q", got, want)
	}
	want := [][2]string{{"user", "My order is late."}, {"assistant", "Sorry."}}
	if got := roles(turns[1].Messages); !slices.Equal(got, want) {
		t.Errorf("later reply read %q, want %q", got, want)
	}
}

func TestAWithdrawnSpeculationLeavesTheCall(t *testing.T) {
	m := &fakeModel{}
	early := frames.NewLLMContextUserTurnFrame("My order", "t1")
	early.Speculation = true

	tagged(t, m, expressiveParams, elevenLabsMetadata(),
		frames.NewLLMContextAssistantTurnFrame("How can I help?", "t0"),
		early,
		50*time.Millisecond,
		frames.NewUserTurnResponseCancelFrame(),
		frames.NewLLMContextAssistantTurnFrame("Take your time.", "t2"),
		frames.NewLLMContextUserTurnFrame("My order is late.", "t3"),
		reply("Sorry."))

	want := [][2]string{
		{"assistant", "How can I help?"}, {"assistant", "Take your time."}, {"user", "My order is late."},
	}
	if got := roles(m.made()[0].turn.Messages); !slices.Equal(got, want) {
		t.Errorf("messages %q, want %q", got, want)
	}
}

func TestRepliesPassThroughWhenTheTTSPerformsNothing(t *testing.T) {
	m := &fakeModel{}

	got := tagged(t, m, expressiveParams, frames.NewTTSMetadataFrame("tts"), reply("Hello there."))

	expectTexts(t, got, "Hello there.")
	if len(m.made()) != 0 {
		t.Errorf("the model was called %d times", len(m.made()))
	}
}

func TestWithoutExpressiveConversationNothingIsTaggedOrAnnounced(t *testing.T) {
	m := &fakeModel{}

	down, up := tonetest.Run(t, tagger(t, m), pipeline.Params{}, sends(elevenLabsMetadata(), reply("Hello there."))...)

	expectTexts(t, texts(down), "Hello there.")
	if len(tonetest.Of[*frames.ExpressionMetadataFrame](slices.Concat(down, up))) != 0 {
		t.Error("announced without expressive conversation")
	}
	if len(m.made()) != 0 {
		t.Errorf("the model was called %d times", len(m.made()))
	}
}

func TestTheTTSMetadataGoesOnUnchanged(t *testing.T) {
	sent := elevenLabsMetadata()

	_, up := tonetest.RunUpstream(t, tagger(t, &fakeModel{}), expressiveParams, sent)

	var got []*expressive.ExpressionCapabilities
	for _, f := range tonetest.Of[*frames.TTSMetadataFrame](up) {
		got = append(got, f.Expression)
	}
	if len(got) != 1 || !reflect.DeepEqual(got[0], sent.Expression) {
		t.Errorf("expression %+v, want %+v", got, sent.Expression)
	}
}

func TestAModelThatCantLoadLeavesTheTaggerUnusable(t *testing.T) {
	standIn(t, func(Size, string, int) (model, error) { return nil, errNoNetwork }, nil)
	tg := newTagger(t, Config{})

	down, up := tonetest.Run(t, tg, expressiveParams, sends(elevenLabsMetadata(), reply("<<calm>> Hello there."))...)

	if tg.Usable() {
		t.Error("the tagger is still usable")
	}
	if len(tonetest.Of[*frames.ErrorFrame](up)) == 0 {
		t.Error("the failed load was not reported")
	}
	// The LLM is still taught the tags, and its reply is spoken as it wrote it.
	if len(tonetest.Of[*frames.ExpressionMetadataFrame](slices.Concat(down, up))) != 0 {
		t.Error("announced without a model")
	}
	expectTexts(t, texts(down), "<<calm>> Hello there.")
}

func TestThePipelineStartsOnceTheModelHasLoaded(t *testing.T) {
	standIn(t, func(Size, string, int) (model, error) {
		time.Sleep(200 * time.Millisecond)
		return &fakeModel{labels: map[string]SentenceLabels{"Hello there.": {Delivery: "calm"}}}, nil
	}, nil)

	down, _ := tonetest.Run(t, newTagger(t, Config{}), expressiveParams,
		sends(elevenLabsMetadata(), reply("Hello there."))...)

	announced := slices.IndexFunc(down, func(f frames.Frame) bool {
		_, ok := f.(*frames.ExpressionMetadataFrame)
		return ok
	})
	text := slices.IndexFunc(down, func(f frames.Frame) bool {
		_, ok := f.(*frames.AggregatedTextFrame)
		return ok
	})
	if announced < 0 || text < 0 || announced > text {
		t.Fatalf("announced at %d, reply at %d", announced, text)
	}
	if got, _ := down[text].(*frames.AggregatedTextFrame); got.Text != "<<calm>> Hello there." {
		t.Errorf("reply %q", got.Text)
	}
}

func TestEveryTaggerSharesOneModel(t *testing.T) {
	m := &fakeModel{}
	loads := 0
	standIn(t, func(Size, string, int) (model, error) {
		loads++
		return m, nil
	}, nil)

	for range 2 {
		tonetest.Run(t, newTagger(t, Config{}), expressiveParams)
	}

	if loads != 1 {
		t.Errorf("loaded %d times", loads)
	}
}

func TestTheModelDownloadsAtTheSizeAndReleaseAskedFor(t *testing.T) {
	type asked struct {
		size     Size
		revision string
		threads  int
	}
	var got []asked
	standIn(t, func(size Size, revision string, threads int) (model, error) {
		got = append(got, asked{size, revision, threads})
		return &fakeModel{}, nil
	}, nil)

	for _, cfg := range []Config{{}, {Size: Size32M, Revision: "tagger-rc9"}} {
		tonetest.Run(t, newTagger(t, cfg), expressiveParams)
	}

	if want := []asked{{"17m", "tagger-rc1", 1}, {"32m", "tagger-rc9", 1}}; !slices.Equal(got, want) {
		t.Errorf("downloaded %+v, want %+v", got, want)
	}
}

func TestAFolderOfYourOwnLoadsInPlaceOfADownload(t *testing.T) {
	dir := t.TempDir()
	type loaded struct {
		dir     string
		threads int
	}
	var got []loaded
	standIn(t,
		func(Size, string, int) (model, error) { return nil, errNoNetwork },
		func(d string, threads int) (model, error) {
			got = append(got, loaded{d, threads})
			return &fakeModel{}, nil
		})

	var down []frames.Frame
	for range 2 {
		down, _ = tonetest.Run(t, newTagger(t, Config{ModelDir: dir, Threads: 2}), expressiveParams)
	}

	// Loaded once a process, and shared.
	if want := []loaded{{dir, 2}}; !slices.Equal(got, want) {
		t.Errorf("loaded %+v, want %+v", got, want)
	}
	if len(tonetest.Of[*frames.ExpressionMetadataFrame](down)) == 0 {
		t.Error("not announced")
	}
}

func TestAVocabularyNamesALabelFirstWhereTheServicePerformsIt(t *testing.T) {
	m := &fakeModel{}
	standIn(t, func(Size, string, int) (model, error) { return m, nil }, nil)
	vocabulary := map[expressive.UniversalName][]string{
		expressive.DeliveryWarm: {"cheerful"}, expressive.SoundLaugh: {"giggles"},
	}

	for _, metadata := range []*frames.TTSMetadataFrame{elevenLabsMetadata(), cartesiaMetadata()} {
		unshared()
		tonetest.Run(t, newTagger(t, Config{Vocabulary: vocabulary}), expressiveParams, sends(metadata, reply("Hi."))...)
	}

	turns := m.turns()
	name := func(n Names, label string) string { s, _ := n.Name(label); return s }
	// ElevenLabs takes free text, so it performs both names.
	if got := name(turns[0].Space.Deliveries, "warm"); got != "cheerful" {
		t.Errorf("ElevenLabs writes warm as %q", got)
	}
	if got := name(turns[0].Space.Sounds, "laugh"); got != "giggles" {
		t.Errorf("ElevenLabs writes laugh as %q", got)
	}
	// Cartesia performs neither, so the universal names stand.
	if got := name(turns[1].Space.Deliveries, "warm"); got != "warm" {
		t.Errorf("Cartesia writes warm as %q", got)
	}
	if got := name(turns[1].Space.Sounds, "laugh"); got != "laugh" {
		t.Errorf("Cartesia writes laugh as %q", got)
	}
}

func TestAVocabularyForALabelTheTaggerDoesntWriteIsRefused(t *testing.T) {
	_, err := New(Config{Vocabulary: map[expressive.UniversalName][]string{
		expressive.ExpressionDelivery("cheerful"): {"happy"},
	}})

	if !errors.Is(err, ErrVocabulary) || !strings.Contains(fmt.Sprint(err), "cheerful") {
		t.Errorf("err = %v, want one naming cheerful", err)
	}
}

// A label is read by its name, so two keys of one name are one label named
// twice, which Go's map can hold and Python's dict cannot.
func TestAVocabularyNamingALabelTwiceIsRefused(t *testing.T) {
	_, err := New(Config{Vocabulary: map[expressive.UniversalName][]string{
		expressive.SoundLaugh: {"giggles"}, expressive.ExpressionDelivery("laugh"): {"cackles"},
	}})

	if !errors.Is(err, ErrVocabulary) {
		t.Errorf("err = %v, want ErrVocabulary", err)
	}
}

func TestASizeTheModelIsNotReleasedInIsRefused(t *testing.T) {
	if _, err := New(Config{Size: "64m"}); err == nil {
		t.Error("a 64m model was accepted")
	}
}
