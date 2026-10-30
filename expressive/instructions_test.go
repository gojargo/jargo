package expressive_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/expressive"
)

// How the instruction that teaches an LLM to write expression tags is built.

func fixed() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds: []expressive.ExpressionTerm{{Name: "laughter"}},
		Delivery: []expressive.ExpressionTerm{
			{Name: "calm"}, {Name: "happy"}, {Name: "sad"},
		},
		PauseRangeMS: &[2]int{100, 5000},
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryWarm: {"happy"},
			expressive.SoundLaugh:   {"laughter"},
		},
	}
}

func freeText() *expressive.ExpressionCapabilities {
	return &expressive.ExpressionCapabilities{
		Sounds: []expressive.ExpressionTerm{{Name: "laughs"}, {Name: "woo", Description: "an excited whoop"}},
		Delivery: []expressive.ExpressionTerm{
			{Name: "excited"}, {Name: "whispers", Description: "speaking in a whisper"},
		},
		AllowsFreeTextSounds:   true,
		AllowsFreeTextDelivery: true,
		Emphasis:               true,
		ServiceVocabulary: map[expressive.UniversalName][]string{
			expressive.DeliveryWhispering: {"whispers"},
		},
	}
}

func contains(t *testing.T, instructions, want string) {
	t.Helper()
	if !strings.Contains(instructions, want) {
		t.Errorf("the instruction lacks %q:\n%s", want, instructions)
	}
}

func lacks(t *testing.T, instructions, unwanted string) {
	t.Helper()
	if strings.Contains(instructions, unwanted) {
		t.Errorf("the instruction has %q:\n%s", unwanted, instructions)
	}
}

func TestNothingIsTaughtForAServiceThatPerformsNothing(t *testing.T) {
	if got := expressive.ExpressionInstructions(nil, nil, nil); got != "" {
		t.Errorf("nil capabilities taught %q", got)
	}
	if got := expressive.ExpressionInstructions(&expressive.ExpressionCapabilities{}, nil, nil); got != "" {
		t.Errorf("empty capabilities taught %q", got)
	}
}

func TestAServiceIsOfferedTheCoreUniversalNamesItPerforms(t *testing.T) {
	instructions := expressive.ExpressionInstructions(fixed(), nil, nil)

	contains(t, instructions, "Use only these deliveries, written exactly as listed: <<calm>>, <<warm>>, <<sad>>.")
	contains(t, instructions, "Use only these sounds, written exactly as listed: [[laugh]].")
}

func TestAServiceThatPerformsDeliveryIsTaughtSpans(t *testing.T) {
	contains(t, expressive.ExpressionInstructions(fixed(), nil, nil),
		"- <<calm: A B 7>> speaks just the words after the colon, A B 7")
	lacks(t, expressive.ExpressionInstructions(&expressive.ExpressionCapabilities{
		Sounds:            []expressive.ExpressionTerm{{Name: "laughter"}},
		ServiceVocabulary: map[expressive.UniversalName][]string{expressive.SoundLaugh: {"laughter"}},
	}, nil, nil),

		"speaks just the words after the colon")
}

func TestAFreeTextServiceIsOfferedUniversalNamesAsTheOnlyChoices(t *testing.T) {
	instructions := expressive.ExpressionInstructions(freeText(), nil, nil)

	contains(t, instructions, "Use only these deliveries, written exactly as listed: <<calm>>, <<sympathetic>>, "+
		"<<apologetic>>, <<warm>>, <<excited>>, <<surprised>>, <<sad>>, <<grateful>>, "+
		"<<curious>>, <<confident>>, <<slow>>, <<fast>>.")
	contains(t, instructions, "Use only these sounds, written exactly as listed: [[laugh]], [[chuckle]], [[sigh]].")
	lacks(t, instructions, "Others work too")
}

func TestABreathIsOfferedOnlyToAPolicyThatAsksForBreaths(t *testing.T) {
	breathes := &expressive.ExpressionPolicy{Guidance: []expressive.Guidance{expressive.GuidanceBreathe}}

	lacks(t, expressive.ExpressionInstructions(freeText(), nil, nil), "[[breath]]")
	contains(t, expressive.ExpressionInstructions(freeText(), breathes, nil), "[[sigh]], [[breath]].")
}

func TestOnlyTheControlsAServicePerformsAreTaught(t *testing.T) {
	fixedText := expressive.ExpressionInstructions(fixed(), nil, nil)
	free := expressive.ExpressionInstructions(freeText(), nil, nil)

	lacks(t, fixedText, "^^word^^")
	contains(t, fixedText, "[[pause 500]]")
	contains(t, free, "^^word^^ stresses a word.")
	lacks(t, free, "[[pause")
}

func TestAPolicyNarrowsAFixedListInItsOwnOrder(t *testing.T) {
	policy := &expressive.ExpressionPolicy{
		Delivery: []expressive.ExpressionDelivery{"sad", "relieved", "calm"},
		Sounds:   []expressive.ExpressionSound{},
	}

	instructions := expressive.ExpressionInstructions(fixed(), policy, nil)

	contains(t, instructions, "exactly as listed: <<sad>>, <<calm>>.")
	lacks(t, instructions, "[[sound]]")
}

func TestAPolicyNamesAnyDeliveryForAFreeTextService(t *testing.T) {
	policy := &expressive.ExpressionPolicy{Delivery: []expressive.ExpressionDelivery{"whispering", "relieved"}}

	// A universal name is described as the service describes its own name for it.
	contains(t, expressive.ExpressionInstructions(freeText(), policy, nil),
		"exactly as listed: <<whispering>> (speaking in a whisper), <<relieved>>.")
}

func TestNothingIsTaughtWhenAPolicyOffersNothingPerformable(t *testing.T) {
	capabilities := &expressive.ExpressionCapabilities{Delivery: []expressive.ExpressionTerm{{Name: "calm"}}}
	policy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle},
		Delivery: []expressive.ExpressionDelivery{"relieved"},
	}

	if got := expressive.ExpressionInstructions(capabilities, policy, nil); got != "" {
		t.Errorf("taught %q, want nothing", got)
	}
}

func TestGuidanceFollowsTheTagsInOrder(t *testing.T) {
	policy := &expressive.ExpressionPolicy{Guidance: []expressive.Guidance{
		expressive.GuidanceMatchStyle, expressive.GuidanceLine("Never laugh while discussing billing."),
	}}

	instructions := expressive.ExpressionInstructions(fixed(), policy, nil)

	want := "\n\n" + string(expressive.GuidanceMatchStyle) + "\nNever laugh while discussing billing."
	if !strings.HasSuffix(instructions, want) {
		t.Errorf("the instruction ends %q, want it to end %q", instructions, want)
	}
}

func TestNeutralInAPolicyIsTaughtOnItsOwnLine(t *testing.T) {
	instructions := expressive.ExpressionInstructions(fixed(),
		&expressive.ExpressionPolicy{Delivery: []expressive.ExpressionDelivery{"neutral", "calm"}}, nil)

	contains(t, instructions, "exactly as listed: <<calm>>.")
	contains(t, instructions, "<<neutral>> returns to your normal voice.")
}

func TestAServiceThatPerformsSpeedIsOfferedSlowWithWhenToUseIt(t *testing.T) {
	capabilities := &expressive.ExpressionCapabilities{
		Delivery: append([]expressive.ExpressionTerm{{Name: "calm"}}, expressive.SpeedDelivery()...),
	}
	policy := &expressive.ExpressionPolicy{
		Guidance: []expressive.Guidance{expressive.GuidanceSlowForDetails},
		Delivery: []expressive.ExpressionDelivery{"calm", "slow"},
	}

	instructions := expressive.ExpressionInstructions(capabilities, policy, nil)

	contains(t, instructions, "<<slow>> (to say something carefully, such as a number, a code or an address)")
	// A span is shown slowing a detail, as details are said.
	contains(t, instructions, "- <<slow: A B 7>> speaks just the words after the colon")
	if !strings.HasSuffix(instructions, string(expressive.GuidanceSlowForDetails)) {
		t.Errorf("the instruction does not end with the preset:\n%s", instructions)
	}
}

func TestAFreeTextServiceWithNoExamplesIsOfferedTheCoreNames(t *testing.T) {
	capabilities := &expressive.ExpressionCapabilities{AllowsFreeTextSounds: true, AllowsFreeTextDelivery: true}

	instructions := expressive.ExpressionInstructions(capabilities, nil, nil)

	contains(t, instructions, "written exactly as listed: <<calm>>, <<sympathetic>>,")
	contains(t, instructions, "- <<slow: A B 7>> speaks just the words after the colon")
	contains(t, instructions, "written exactly as listed: [[laugh]], [[chuckle]], [[sigh]].")
}

func TestAFreeTextServiceIsTaughtNoneOfAKindThePolicyNamesNoneOf(t *testing.T) {
	capabilities := &expressive.ExpressionCapabilities{
		AllowsFreeTextSounds: true, AllowsFreeTextDelivery: true, Emphasis: true,
	}

	got := expressive.ExpressionInstructions(capabilities, &expressive.ExpressionPolicy{
		Delivery: []expressive.ExpressionDelivery{}, Sounds: []expressive.ExpressionSound{},
	}, nil)

	if want := "You can direct how your words are spoken with tags:\n- ^^word^^ stresses a word."; got != want {
		t.Errorf("taught %q, want %q", got, want)
	}
}

func TestAPolicyKeepsPresetsApartFromLinesWithTheSameText(t *testing.T) {
	policy := expressive.ExpressionPolicy{Guidance: []expressive.Guidance{
		expressive.GuidanceMatchStyle, expressive.GuidanceLine(expressive.GuidanceMatchStyle),
	}}

	preset, line := policy.Guidance[0], policy.Guidance[1]
	if _, ok := preset.(expressive.ExpressionGuidance); !ok {
		t.Error("the preset is not an ExpressionGuidance")
	}
	if _, ok := line.(expressive.ExpressionGuidance); ok {
		t.Error("the developer's line is an ExpressionGuidance")
	}
	if preset.String() != line.String() {
		t.Error("the two lines do not read the same")
	}
}

func TestNoWarningWithoutAPolicyOrWhenEveryTagIsPerformable(t *testing.T) {
	for name, warning := range map[string]string{
		"no policy":         expressive.ExpressionPolicyWarning(fixed(), nil),
		"no policy, no tts": expressive.ExpressionPolicyWarning(nil, nil),
		"performable": expressive.ExpressionPolicyWarning(fixed(),
			&expressive.ExpressionPolicy{Delivery: []expressive.ExpressionDelivery{"calm"}}),
		// A free-text service performs any delivery or sound it is given.
		"free text performs": expressive.ExpressionPolicyWarning(freeText(),
			&expressive.ExpressionPolicy{Delivery: []expressive.ExpressionDelivery{"relieved"}}),
	} {
		if warning != "" {
			t.Errorf("%s: warned %q", name, warning)
		}
	}
}

func TestAWarningNamesThePolicyTagsTheServiceCantPerform(t *testing.T) {
	policy := &expressive.ExpressionPolicy{
		Delivery: []expressive.ExpressionDelivery{"neutral", "calm", "relieved"},
		Sounds:   []expressive.ExpressionSound{"sighs"},
	}

	// Every service returns to its baseline, so neutral is never left out.
	want := "the TTS service can't perform <<relieved>>, [[sighs]] from the expression " +
		"policy, so they aren't offered"
	if got := expressive.ExpressionPolicyWarning(fixed(), policy); got != want {
		t.Errorf("warned %q, want %q", got, want)
	}
}

func TestAWarningSaysWhenTheServicePerformsNoneOfThePolicy(t *testing.T) {
	capabilities := &expressive.ExpressionCapabilities{Delivery: []expressive.ExpressionTerm{{Name: "calm"}}}
	want := "the TTS service performs none of the expression policy's tags, so they " +
		"and its guidance presets aren't taught"

	if got := expressive.ExpressionPolicyWarning(capabilities,
		&expressive.ExpressionPolicy{Delivery: []expressive.ExpressionDelivery{"relieved"}}); got != want {
		t.Errorf("warned %q, want %q", got, want)
	}
	if got := expressive.ExpressionPolicyWarning(nil, &expressive.ExpressionPolicy{}); got != want {
		t.Errorf("warned %q, want %q", got, want)
	}
}

func tonePerception() *expressive.PerceptionCapabilities {
	return &expressive.PerceptionCapabilities{Signals: []expressive.PerceptionTerm{
		{Name: "tone", Labels: []string{"positive", "neutral", "sad", "frustrated"}},
		{Name: "speech_rate", Description: "Words per second."},
	}}
}

func TestPerceptionIsTaughtWithAnExampleAndItsSignals(t *testing.T) {
	instructions := expressive.ExpressionInstructions(nil, nil, tonePerception())

	contains(t, instructions, "in parentheses, such as ([tone positive=0.86, neutral=0.10]), each")
	contains(t, instructions, "A reading below 0.5 is a guess: don't act on it.")
	want := "- tone: positive, neutral, sad, frustrated.\n- speech_rate: Words per second."
	if !strings.HasSuffix(instructions, want) {
		t.Errorf("the instruction ends %q, want it to end %q", instructions, want)
	}
}

func TestPerceptionComesBeforeTheTagsAndTheGuidanceLast(t *testing.T) {
	policy := &expressive.ExpressionPolicy{Guidance: []expressive.Guidance{expressive.GuidanceMatchStyle}}

	instructions := expressive.ExpressionInstructions(fixed(), policy, tonePerception())

	reading := strings.Index(instructions, "([tone")
	tags := strings.Index(instructions, "You can direct how your words are spoken")
	guidance := strings.Index(instructions, string(expressive.GuidanceMatchStyle))
	if reading < 0 || reading >= tags || tags >= guidance {
		t.Errorf("reading at %d, tags at %d, guidance at %d, want them in that order", reading, tags, guidance)
	}
}

func TestOnlyTheDeveloperGuidanceIsKeptForPerceptionWithoutTags(t *testing.T) {
	policy := &expressive.ExpressionPolicy{Guidance: []expressive.Guidance{
		expressive.GuidanceMatchStyle,
		expressive.GuidanceLine("If the user sounds frustrated, apologize once."),
		expressive.GuidanceSlowForDetails,
	}}

	instructions := expressive.ExpressionInstructions(nil, policy, tonePerception())

	if !strings.HasSuffix(instructions, "\n\nIf the user sounds frustrated, apologize once.") {
		t.Errorf("the instruction does not end with the developer's line:\n%s", instructions)
	}
	lacks(t, instructions, string(expressive.GuidanceMatchStyle))
	lacks(t, instructions, "<<slow")
}

func TestADeveloperLineWithAPresetsTextIsKeptForPerceptionWithoutTags(t *testing.T) {
	line := expressive.GuidanceLine(expressive.GuidanceMatchStyle)

	instructions := expressive.ExpressionInstructions(nil,
		&expressive.ExpressionPolicy{Guidance: []expressive.Guidance{line}}, tonePerception())

	if !strings.HasSuffix(instructions, "\n\n"+string(line)) {
		t.Errorf("the instruction does not end with the developer's line:\n%s", instructions)
	}
}
