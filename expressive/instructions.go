package expressive

import (
	"fmt"
	"strings"
)

// The instruction that teaches an LLM to take part in an expressive
// conversation (alpha). It has two parts: the tags the active TTS service can
// perform, by their universal names, built from its ExpressionCapabilities;
// and the developer's guidance on how to answer, from an ExpressionPolicy.

// ExpressionInstructions builds the instruction for an LLM in an expressive
// conversation.
//
// capabilities is what the TTS service that speaks the LLM's replies can
// perform, or nil for nothing; policy is which tags to offer, and guidance on
// how to answer, or nil for the default policy. It returns the empty string
// when the TTS service can perform none of the tags offered.
func ExpressionInstructions(capabilities *ExpressionCapabilities, policy *ExpressionPolicy) string {
	if policy == nil {
		policy = &ExpressionPolicy{}
	}
	tags := tagsSection(capabilities, policy)
	if tags == "" {
		return ""
	}
	parts := []string{tags}
	var guidance []string
	for _, line := range policy.Guidance {
		if line != nil {
			guidance = append(guidance, line.String())
		}
	}
	if len(guidance) > 0 {
		parts = append(parts, strings.Join(guidance, "\n"))
	}
	return strings.Join(parts, "\n\n")
}

// ExpressionPolicyWarning tells what an expression policy asks for that a TTS
// service cannot perform.
//
// capabilities is what the TTS service can perform, or nil for nothing. It
// returns the empty string when there is no policy or the service can perform
// every tag it offers.
func ExpressionPolicyWarning(capabilities *ExpressionCapabilities, policy *ExpressionPolicy) string {
	if policy == nil {
		return ""
	}
	if capabilities == nil || tagsSection(capabilities, policy) == "" {
		return "the TTS service performs none of the expression policy's tags, so they " +
			"and its guidance presets aren't taught"
	}
	if leftOut := policyLeftOut(capabilities, policy); len(leftOut) > 0 {
		return fmt.Sprintf("the TTS service can't perform %s from the expression "+
			"policy, so they aren't offered", strings.Join(leftOut, ", "))
	}
	return ""
}

func tagsSection(capabilities *ExpressionCapabilities, policy *ExpressionPolicy) string {
	if capabilities == nil {
		return ""
	}
	var tags []string
	var deliveryNames []string
	if policy.Delivery == nil {
		deliveryNames = namesOf(CoreDelivery())
	} else {
		deliveryNames = namesOf(policy.Delivery)
	}
	var delivery []ExpressionTerm
	for _, term := range offered(deliveryNames, capabilities.DeliveryName, capabilities.Delivery) {
		// Taught on its own, as the return to the voice's baseline.
		if term.Name != NeutralDelivery {
			delivery = append(delivery, term)
		}
	}
	if len(delivery) > 0 {
		tags = append(tags,
			"- <<delivery>> at the start of a sentence speaks it, and the sentences after it, "+
				"with that delivery, until the next delivery tag. <<neutral>> returns to your "+
				"normal voice. "+choices("deliveries", delivery, "<<%s>>"))
		// A real span shows that its delivery comes before the colon and the
		// words to speak after it. Written with placeholders, a span reads as a
		// label and its value, as in <<delivery: sympathetic>>.
		example := spanExample(delivery)
		tags = append(tags, fmt.Sprintf(
			"- <<%s: A B 7>> speaks just the words after the colon, A B 7, with the "+
				"delivery before it, %s, on top of the delivery around them, and then goes "+
				"back to it. Keep it to a few words.", example, example))
	}
	var soundNames []string
	if policy.Sounds == nil {
		// A breath is offered by default only to a policy that asks for breaths.
		for _, s := range CoreSounds() {
			if s != SoundBreath || policy.Breathes() {
				soundNames = append(soundNames, string(s))
			}
		}
	} else {
		soundNames = namesOf(policy.Sounds)
	}
	if sounds := offered(soundNames, capabilities.SoundName, capabilities.Sounds); len(sounds) > 0 {
		tags = append(tags, "- [[sound]] makes a sound at that point. "+choices("sounds", sounds, "[[%s]]"))
	}
	if capabilities.Emphasis {
		tags = append(tags, "- ^^word^^ stresses a word.")
	}
	if capabilities.PauseRangeMS != nil {
		tags = append(tags, "- [[pause 500]] pauses for about that many milliseconds.")
	}
	if len(tags) == 0 {
		return ""
	}
	return "You can direct how your words are spoken with tags:\n" + strings.Join(tags, "\n")
}

// policyLeftOut is the policy's tags the service cannot perform, each as it
// would be written.
func policyLeftOut(capabilities *ExpressionCapabilities, policy *ExpressionPolicy) []string {
	var tags []string
	for _, name := range policy.Delivery {
		// Every service returns to its baseline: neutral is taught on its own.
		if _, ok := capabilities.DeliveryName(string(name)); string(name) != NeutralDelivery && !ok {
			tags = append(tags, "<<"+string(name)+">>")
		}
	}
	for _, name := range policy.Sounds {
		if _, ok := capabilities.SoundName(string(name)); !ok {
			tags = append(tags, "[["+string(name)+"]]")
		}
	}
	return tags
}

// offered is the names to offer, the policy's or the core ones, that the
// service performs. Each is offered with the description the service gives the
// name it performs it by.
func offered(
	names []string, performedName func(string) (string, bool), declared []ExpressionTerm,
) []ExpressionTerm {
	descriptions := make(map[string]string, len(declared))
	for _, term := range declared {
		descriptions[term.Name] = term.Description
	}
	var out []ExpressionTerm
	for _, name := range names {
		if performed, ok := performedName(name); ok {
			out = append(out, ExpressionTerm{Name: name, Description: descriptions[performed]})
		}
	}
	return out
}

// spanExample is the delivery a span is shown with: slow, for details, unless
// only others are listed.
func spanExample(delivery []ExpressionTerm) string {
	for _, term := range delivery {
		if term.Name == SlowDelivery {
			return SlowDelivery
		}
	}
	if len(delivery) > 0 {
		return delivery[0].Name
	}
	return SlowDelivery
}

// choices is the terms offered, each written as its tag, as in <<calm>> or
// [[laugh]].
func choices(kind string, terms []ExpressionTerm, tag string) string {
	listed := make([]string, 0, len(terms))
	for _, term := range terms {
		name := fmt.Sprintf(tag, term.Name)
		if term.Description != "" {
			name += " (" + term.Description + ")"
		}
		listed = append(listed, name)
	}
	return fmt.Sprintf("Use only these %s, written exactly as listed: %s. "+
		"Any other word in a tag is ignored, so if none fits, write no tag.", kind, strings.Join(listed, ", "))
}

func namesOf[T ~string](values []T) []string {
	out := make([]string, len(values))
	for i, v := range values {
		out[i] = string(v)
	}
	return out
}
