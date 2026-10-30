package expressive

// The developer's policy for expressive conversation (alpha). An LLM is taught
// expression tags by their universal names, those the active TTS service can
// perform. The policy says which it is offered and when to use them.

// Guidance is one line of an [ExpressionPolicy]'s guidance: a preset, an
// [ExpressionGuidance], or a line of the developer's own, a [GuidanceLine].
//
// The two are told apart by their type, so a line of the developer's that has
// a preset's text stays the developer's line.
type Guidance interface {
	// String is the line as written into the LLM's instruction.
	String() string
	guidance()
}

// ExpressionGuidance is a guidance preset for an [ExpressionPolicy].
//
// Each preset is a short paragraph on one aspect of delivery, written to
// combine with other presets and with the developer's own lines. Presets are
// about writing tags, so an LLM taught none, such as one whose replies a tagger
// marks up, is not given them.
type ExpressionGuidance string

// String implements [Guidance].
func (g ExpressionGuidance) String() string { return string(g) }

func (ExpressionGuidance) guidance() {}

// The guidance presets.
const (
	// GuidanceMatchStyle answers in the user's mood and energy, but stays calm
	// with hostility.
	GuidanceMatchStyle ExpressionGuidance = "Let your voice match the user's mood and energy: " +
		"be excited for their good news, sympathetic with their bad news, and laugh when they " +
		"say something funny. If they are hostile, stay calm instead of matching it. Most " +
		"sentences need no tag."
	// GuidanceSlowForDetails says just what the user may need to write down
	// slowly, in a span. It is for a service that performs the slow delivery.
	GuidanceSlowForDetails ExpressionGuidance = "When you give the user something to write " +
		"down, such as a code, a phone number or an email address, say just that slowly: " +
		"Your code is <<slow: A B 7 4 9>>. Anything else?"
	// GuidanceBreathe takes an audible breath before a long or difficult
	// sentence now and then, at most once a reply. It is for a service that
	// performs a breath. A tagger reads it as a switch for breaths of its own
	// choosing.
	GuidanceBreathe ExpressionGuidance = "Now and then, take an audible breath before a " +
		"sentence that is long, carries hard news or starts a careful explanation, as a person " +
		"on the phone does. At most one breath a reply, and most replies need none: rarely " +
		"before a short or routine sentence, never by habit at the start of every reply, and " +
		"never right after another sound."
)

// GuidanceLine is a line of guidance of the developer's own, such as "Never
// laugh while discussing billing.".
type GuidanceLine string

// String implements [Guidance].
func (g GuidanceLine) String() string { return string(g) }

func (GuidanceLine) guidance() {}

// ExpressionPolicy is which expression tags an LLM is offered, and when to use
// them.
//
//	expressive.ExpressionPolicy{
//		Guidance: []expressive.Guidance{
//			expressive.GuidanceMatchStyle,
//			expressive.GuidanceLine("Never laugh while discussing billing."),
//		},
//		Delivery: []expressive.ExpressionDelivery{
//			expressive.DeliveryCalm, expressive.DeliveryHappy, expressive.DeliverySympathetic,
//		},
//	}
type ExpressionPolicy struct {
	// Guidance is the lines on when and how much to express, such as
	// GuidanceMatchStyle or the developer's own. They are joined in order, so a
	// later line can refine an earlier one. The developer's lines reach the LLM
	// whenever it has a part to play in expressive conversation: tags to write,
	// or a reading of the user to act on. The presets are about writing tags, so
	// they go to whoever writes them: the LLM, or a tagger in its place.
	Guidance []Guidance
	// Delivery is the deliveries to offer, by their universal names, or nil for
	// the core ones ([CoreDelivery]). A name that is not universal, such as an
	// accent, is offered as written. Those the TTS service cannot perform are
	// left out, with a warning. An empty, non-nil list offers none.
	Delivery []ExpressionDelivery
	// Sounds is the sounds to offer, as Delivery offers deliveries. Nil offers
	// the core ones ([CoreSounds]), and a breath only with GuidanceBreathe.
	Sounds []ExpressionSound
}

// Breathes reports whether the policy's guidance asks for breaths, with
// GuidanceBreathe. A line of the developer's with that preset's text asks for
// them too.
func (p *ExpressionPolicy) Breathes() bool {
	if p == nil {
		return false
	}
	for _, g := range p.Guidance {
		if g != nil && g.String() == string(GuidanceBreathe) {
			return true
		}
	}
	return false
}
