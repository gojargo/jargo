// Package expressive holds what a TTS service can perform in an expressive
// conversation, which lets an agent hear how something was said and answer in
// kind.
//
// An LLM writes expression into its reply as inline tags: [[name]] for a sound
// or silence at one point, <<name>> for the delivery of the words after it,
// <<name: words>> for the delivery of just those words, and ^^word^^ to stress
// a word. A delivery lasts until the next delivery tag or the end of the
// response, and <<neutral>> returns to the voice's baseline.
//
// For example:
//
//	<<warm>> It's [[laugh]] ^^back^^ up. [[pause 300]] Your code is <<slow: A B 7>>.
//
// A TTS service declares which of these it can perform with
// [ExpressionCapabilities], and broadcasts them at start in a
// frames.TTSMetadataFrame, so the LLM can be told which tags to write.
//
// Every delivery and sound the services perform is named the same way on each,
// the universal names: [ExpressionDelivery] and [ExpressionSound]. A universal
// name in a tag, such as <<warm>> or [[laugh]], is sent to each service by its
// own name for it, such as ElevenLabs' "warmly" or Cartesia's "laughter",
// while everything reported keeps the name as written.
//
// The tags are read by the formatter package and reported by the spoken
// package, below this one.
//
// Alpha: the tag syntax and these types are likely to change between releases.
package expressive

// NeutralDelivery is the delivery that returns to the voice's baseline. It is
// part of the protocol, so it is a delivery whatever a service declares.
const NeutralDelivery = "neutral"

// EmphasisDelivery is the delivery of stressed words, written ^^words^^.
const EmphasisDelivery = "emphasis"

// SlowDelivery and FastDelivery slow the words down and speed them up. They
// are part of the protocol, so every service that performs speed names them
// the same.
const (
	SlowDelivery = "slow"
	FastDelivery = "fast"
)

// UniversalName is a delivery or a sound by its universal name: an
// [ExpressionDelivery] or an [ExpressionSound]. It keys
// [ExpressionCapabilities.ServiceVocabulary].
type UniversalName interface {
	// String is the name as written in a tag.
	String() string
	universal()
}

// ExpressionDelivery is a delivery by its universal name, the same on every TTS
// service.
//
// With [ExpressionSound], these name everything the services with an
// expression adapter perform, each in one spelling, an adjective for a
// delivery. A tag that names one is sent by each service's own name for it
// ([ExpressionCapabilities.ServiceVocabulary]), and reported as written.
type ExpressionDelivery string

// String implements [UniversalName].
func (d ExpressionDelivery) String() string { return string(d) }

func (ExpressionDelivery) universal() {}

// The universal deliveries.
const (
	DeliveryAffectionate  ExpressionDelivery = "affectionate"
	DeliveryAgitated      ExpressionDelivery = "agitated"
	DeliveryAlarmed       ExpressionDelivery = "alarmed"
	DeliveryAmazed        ExpressionDelivery = "amazed"
	DeliveryAmused        ExpressionDelivery = "amused"
	DeliveryAngry         ExpressionDelivery = "angry"
	DeliveryAnnoyed       ExpressionDelivery = "annoyed"
	DeliveryAnticipating  ExpressionDelivery = "anticipating"
	DeliveryAnxious       ExpressionDelivery = "anxious"
	DeliveryApologetic    ExpressionDelivery = "apologetic"
	DeliveryAppalled      ExpressionDelivery = "appalled"
	DeliveryAshamed       ExpressionDelivery = "ashamed"
	DeliveryBored         ExpressionDelivery = "bored"
	DeliveryCalm          ExpressionDelivery = "calm"
	DeliveryCautious      ExpressionDelivery = "cautious"
	DeliveryCompassionate ExpressionDelivery = "compassionate"
	DeliveryConfident     ExpressionDelivery = "confident"
	DeliveryConfused      ExpressionDelivery = "confused"
	DeliveryContemplative ExpressionDelivery = "contemplative"
	DeliveryContemptuous  ExpressionDelivery = "contemptuous"
	DeliveryContent       ExpressionDelivery = "content"
	DeliveryCrying        ExpressionDelivery = "crying"
	DeliveryCurious       ExpressionDelivery = "curious"
	DeliveryCute          ExpressionDelivery = "cute"
	DeliveryDeadpan       ExpressionDelivery = "deadpan"
	DeliveryDejected      ExpressionDelivery = "dejected"
	DeliveryDelighted     ExpressionDelivery = "delighted"
	DeliveryDepressed     ExpressionDelivery = "depressed"
	DeliveryDesperate     ExpressionDelivery = "desperate"
	DeliveryDetermined    ExpressionDelivery = "determined"
	DeliveryDisappointed  ExpressionDelivery = "disappointed"
	DeliveryDisdainful    ExpressionDelivery = "disdainful"
	DeliveryDisgusted     ExpressionDelivery = "disgusted"
	DeliveryDismissive    ExpressionDelivery = "dismissive"
	DeliveryDistant       ExpressionDelivery = "distant"
	DeliveryDoubtful      ExpressionDelivery = "doubtful"
	DeliveryDramatic      ExpressionDelivery = "dramatic"
	DeliveryElated        ExpressionDelivery = "elated"
	DeliveryEmbarrassed   ExpressionDelivery = "embarrassed"
	DeliveryEmpathetic    ExpressionDelivery = "empathetic"
	DeliveryEncouraging   ExpressionDelivery = "encouraging"
	DeliveryEnthusiastic  ExpressionDelivery = "enthusiastic"
	DeliveryEnvious       ExpressionDelivery = "envious"
	DeliveryEuphoric      ExpressionDelivery = "euphoric"
	DeliveryExcited       ExpressionDelivery = "excited"
	DeliveryFast          ExpressionDelivery = "fast"
	DeliveryFearful       ExpressionDelivery = "fearful"
	DeliveryFlirtatious   ExpressionDelivery = "flirtatious"
	DeliveryFrustrated    ExpressionDelivery = "frustrated"
	DeliveryGrateful      ExpressionDelivery = "grateful"
	DeliveryGuilty        ExpressionDelivery = "guilty"
	DeliveryHappy         ExpressionDelivery = "happy"
	DeliveryHesitant      ExpressionDelivery = "hesitant"
	DeliveryHopeful       ExpressionDelivery = "hopeful"
	DeliveryHurt          ExpressionDelivery = "hurt"
	DeliveryHysterical    ExpressionDelivery = "hysterical"
	DeliveryImpressed     ExpressionDelivery = "impressed"
	DeliveryIndifferent   ExpressionDelivery = "indifferent"
	DeliveryInsecure      ExpressionDelivery = "insecure"
	DeliveryIronic        ExpressionDelivery = "ironic"
	DeliveryJealous       ExpressionDelivery = "jealous"
	DeliveryLonely        ExpressionDelivery = "lonely"
	DeliveryLoud          ExpressionDelivery = "loud"
	DeliveryMad           ExpressionDelivery = "mad"
	DeliveryMelancholic   ExpressionDelivery = "melancholic"
	DeliveryMischievous   ExpressionDelivery = "mischievous"
	DeliveryMocking       ExpressionDelivery = "mocking"
	DeliveryMoved         ExpressionDelivery = "moved"
	DeliveryMuttering     ExpressionDelivery = "muttering"
	DeliveryMysterious    ExpressionDelivery = "mysterious"
	DeliveryNervous       ExpressionDelivery = "nervous"
	DeliveryNostalgic     ExpressionDelivery = "nostalgic"
	DeliveryOptimistic    ExpressionDelivery = "optimistic"
	DeliveryOutraged      ExpressionDelivery = "outraged"
	DeliveryPanicked      ExpressionDelivery = "panicked"
	DeliveryPeaceful      ExpressionDelivery = "peaceful"
	DeliveryPessimistic   ExpressionDelivery = "pessimistic"
	DeliveryPlayful       ExpressionDelivery = "playful"
	DeliveryProfessional  ExpressionDelivery = "professional"
	DeliveryProud         ExpressionDelivery = "proud"
	DeliveryQuestioning   ExpressionDelivery = "questioning"
	DeliveryReassuring    ExpressionDelivery = "reassuring"
	DeliveryRegretful     ExpressionDelivery = "regretful"
	DeliveryRejected      ExpressionDelivery = "rejected"
	DeliveryRelaxed       ExpressionDelivery = "relaxed"
	DeliveryRelieved      ExpressionDelivery = "relieved"
	DeliveryResigned      ExpressionDelivery = "resigned"
	DeliverySad           ExpressionDelivery = "sad"
	DeliverySarcastic     ExpressionDelivery = "sarcastic"
	DeliverySatisfied     ExpressionDelivery = "satisfied"
	DeliveryScared        ExpressionDelivery = "scared"
	DeliveryScreaming     ExpressionDelivery = "screaming"
	DeliverySerene        ExpressionDelivery = "serene"
	DeliverySerious       ExpressionDelivery = "serious"
	DeliverySheepish      ExpressionDelivery = "sheepish"
	DeliveryShouting      ExpressionDelivery = "shouting"
	DeliverySincere       ExpressionDelivery = "sincere"
	DeliverySinging       ExpressionDelivery = "singing"
	DeliverySkeptical     ExpressionDelivery = "skeptical"
	DeliverySlow          ExpressionDelivery = "slow"
	DeliverySoft          ExpressionDelivery = "soft"
	DeliverySomber        ExpressionDelivery = "somber"
	DeliveryStern         ExpressionDelivery = "stern"
	DeliverySurprised     ExpressionDelivery = "surprised"
	DeliverySympathetic   ExpressionDelivery = "sympathetic"
	DeliveryThoughtful    ExpressionDelivery = "thoughtful"
	DeliveryThreatened    ExpressionDelivery = "threatened"
	DeliveryTired         ExpressionDelivery = "tired"
	DeliveryTriumphant    ExpressionDelivery = "triumphant"
	DeliveryTrusting      ExpressionDelivery = "trusting"
	DeliveryUncertain     ExpressionDelivery = "uncertain"
	DeliveryUnhappy       ExpressionDelivery = "unhappy"
	DeliveryUpset         ExpressionDelivery = "upset"
	DeliveryUrgent        ExpressionDelivery = "urgent"
	DeliveryWarm          ExpressionDelivery = "warm"
	DeliveryWhispering    ExpressionDelivery = "whispering"
	DeliveryWistful       ExpressionDelivery = "wistful"
	DeliveryWorried       ExpressionDelivery = "worried"
)

// ExpressionSound is a sound by its universal name, the same on every TTS
// service. It is named as [ExpressionDelivery] names a delivery, in a base
// word, such as "laugh".
type ExpressionSound string

// String implements [UniversalName].
func (s ExpressionSound) String() string { return string(s) }

func (ExpressionSound) universal() {}

// The universal sounds.
const (
	SoundBreath        ExpressionSound = "breath"
	SoundBurp          ExpressionSound = "burp"
	SoundCackle        ExpressionSound = "cackle"
	SoundChuckle       ExpressionSound = "chuckle"
	SoundClearThroat   ExpressionSound = "clear throat"
	SoundCough         ExpressionSound = "cough"
	SoundCry           ExpressionSound = "cry"
	SoundDeepBreath    ExpressionSound = "deep breath"
	SoundExhale        ExpressionSound = "exhale"
	SoundGasp          ExpressionSound = "gasp"
	SoundGiggle        ExpressionSound = "giggle"
	SoundGroan         ExpressionSound = "groan"
	SoundGrowl         ExpressionSound = "growl"
	SoundGrunt         ExpressionSound = "grunt"
	SoundGulp          ExpressionSound = "gulp"
	SoundHiccup        ExpressionSound = "hiccup"
	SoundHiss          ExpressionSound = "hiss"
	SoundHuff          ExpressionSound = "huff"
	SoundHum           ExpressionSound = "hum"
	SoundInhale        ExpressionSound = "inhale"
	SoundLaugh         ExpressionSound = "laugh"
	SoundLaughHarder   ExpressionSound = "laugh harder"
	SoundLightBreath   ExpressionSound = "light breath"
	SoundLipSmack      ExpressionSound = "lip smack"
	SoundMoan          ExpressionSound = "moan"
	SoundPant          ExpressionSound = "pant"
	SoundScoff         ExpressionSound = "scoff"
	SoundSharpExhale   ExpressionSound = "sharp exhale"
	SoundShriek        ExpressionSound = "shriek"
	SoundShush         ExpressionSound = "shush"
	SoundSigh          ExpressionSound = "sigh"
	SoundSneeze        ExpressionSound = "sneeze"
	SoundSniff         ExpressionSound = "sniff"
	SoundSniffle       ExpressionSound = "sniffle"
	SoundSnore         ExpressionSound = "snore"
	SoundSnort         ExpressionSound = "snort"
	SoundSob           ExpressionSound = "sob"
	SoundStartLaughing ExpressionSound = "start laughing"
	SoundSwallow       ExpressionSound = "swallow"
	SoundTongueClick   ExpressionSound = "tongue click"
	SoundUm            ExpressionSound = "um"
	SoundWheeze        ExpressionSound = "wheeze"
	SoundWhimper       ExpressionSound = "whimper"
	SoundWhistle       ExpressionSound = "whistle"
	SoundWoo           ExpressionSound = "woo"
	SoundYawn          ExpressionSound = "yawn"
)

// Deliveries is every universal delivery, in order.
func Deliveries() []ExpressionDelivery {
	return []ExpressionDelivery{
		DeliveryAffectionate, DeliveryAgitated, DeliveryAlarmed, DeliveryAmazed, DeliveryAmused,
		DeliveryAngry, DeliveryAnnoyed, DeliveryAnticipating, DeliveryAnxious, DeliveryApologetic,
		DeliveryAppalled, DeliveryAshamed, DeliveryBored, DeliveryCalm, DeliveryCautious,
		DeliveryCompassionate, DeliveryConfident, DeliveryConfused, DeliveryContemplative,
		DeliveryContemptuous, DeliveryContent, DeliveryCrying, DeliveryCurious, DeliveryCute,
		DeliveryDeadpan, DeliveryDejected, DeliveryDelighted, DeliveryDepressed, DeliveryDesperate,
		DeliveryDetermined, DeliveryDisappointed, DeliveryDisdainful, DeliveryDisgusted,
		DeliveryDismissive, DeliveryDistant, DeliveryDoubtful, DeliveryDramatic, DeliveryElated,
		DeliveryEmbarrassed, DeliveryEmpathetic, DeliveryEncouraging, DeliveryEnthusiastic,
		DeliveryEnvious, DeliveryEuphoric, DeliveryExcited, DeliveryFast, DeliveryFearful,
		DeliveryFlirtatious, DeliveryFrustrated, DeliveryGrateful, DeliveryGuilty, DeliveryHappy,
		DeliveryHesitant, DeliveryHopeful, DeliveryHurt, DeliveryHysterical, DeliveryImpressed,
		DeliveryIndifferent, DeliveryInsecure, DeliveryIronic, DeliveryJealous, DeliveryLonely,
		DeliveryLoud, DeliveryMad, DeliveryMelancholic, DeliveryMischievous, DeliveryMocking,
		DeliveryMoved, DeliveryMuttering, DeliveryMysterious, DeliveryNervous, DeliveryNostalgic,
		DeliveryOptimistic, DeliveryOutraged, DeliveryPanicked, DeliveryPeaceful, DeliveryPessimistic,
		DeliveryPlayful, DeliveryProfessional, DeliveryProud, DeliveryQuestioning, DeliveryReassuring,
		DeliveryRegretful, DeliveryRejected, DeliveryRelaxed, DeliveryRelieved, DeliveryResigned,
		DeliverySad, DeliverySarcastic, DeliverySatisfied, DeliveryScared, DeliveryScreaming,
		DeliverySerene, DeliverySerious, DeliverySheepish, DeliveryShouting, DeliverySincere,
		DeliverySinging, DeliverySkeptical, DeliverySlow, DeliverySoft, DeliverySomber, DeliveryStern,
		DeliverySurprised, DeliverySympathetic, DeliveryThoughtful, DeliveryThreatened, DeliveryTired,
		DeliveryTriumphant, DeliveryTrusting, DeliveryUncertain, DeliveryUnhappy, DeliveryUpset,
		DeliveryUrgent, DeliveryWarm, DeliveryWhispering, DeliveryWistful, DeliveryWorried,
	}
}

// Sounds is every universal sound, in order.
func Sounds() []ExpressionSound {
	return []ExpressionSound{
		SoundBreath, SoundBurp, SoundCackle, SoundChuckle, SoundClearThroat, SoundCough, SoundCry,
		SoundDeepBreath, SoundExhale, SoundGasp, SoundGiggle, SoundGroan, SoundGrowl, SoundGrunt,
		SoundGulp, SoundHiccup, SoundHiss, SoundHuff, SoundHum, SoundInhale, SoundLaugh,
		SoundLaughHarder, SoundLightBreath, SoundLipSmack, SoundMoan, SoundPant, SoundScoff,
		SoundSharpExhale, SoundShriek, SoundShush, SoundSigh, SoundSneeze, SoundSniff, SoundSniffle,
		SoundSnore, SoundSnort, SoundSob, SoundStartLaughing, SoundSwallow, SoundTongueClick, SoundUm,
		SoundWheeze, SoundWhimper, SoundWhistle, SoundWoo, SoundYawn,
	}
}

// CoreDelivery is the universal deliveries an LLM is taught when its policy
// names none, and that a tone tagger writes.
func CoreDelivery() []ExpressionDelivery {
	return []ExpressionDelivery{
		DeliveryCalm, DeliverySympathetic, DeliveryApologetic, DeliveryWarm, DeliveryExcited,
		DeliverySurprised, DeliverySad, DeliveryGrateful, DeliveryCurious, DeliveryConfident,
		DeliverySlow, DeliveryFast,
	}
}

// CoreSounds is the universal sounds an LLM is taught when its policy names
// none, and that a tone tagger writes.
func CoreSounds() []ExpressionSound {
	return []ExpressionSound{SoundLaugh, SoundChuckle, SoundSigh, SoundBreath}
}

// IsDelivery reports whether name is a universal delivery.
func IsDelivery(name string) bool {
	_, ok := universalDeliveries[ExpressionDelivery(name)]
	return ok
}

// IsSound reports whether name is a universal sound.
func IsSound(name string) bool {
	_, ok := universalSounds[ExpressionSound(name)]
	return ok
}

// universalDeliveries and universalSounds look the universal names up.
//
//nolint:gochecknoglobals // read-only lookup tables built once from the lists above
var (
	universalDeliveries = setOf(Deliveries())
	universalSounds     = setOf(Sounds())
)

func setOf[T comparable](values []T) map[T]struct{} {
	out := make(map[T]struct{}, len(values))
	for _, v := range values {
		out[v] = struct{}{}
	}
	return out
}

// DeliveryScope is where in a response a TTS service can change delivery.
//
// In "<<calm>> Sure, <<excited>> it worked!", a service with
// [DeliveryScopeAnywhere] speaks "it worked!" excited. One with
// [DeliveryScopeSentence] speaks the whole sentence calm, the delivery its
// words start in, and the next sentence excited. A span's delivery is the
// exception: [DeliveryScopeSentence] speaks the whole of
// "Your code is <<slow: A B 7>>." slowly, so the span's words still are.
type DeliveryScope string

const (
	// DeliveryScopeAnywhere is a delivery that takes effect where its tag
	// appears. It is the default: the empty scope reads as it.
	DeliveryScopeAnywhere DeliveryScope = "anywhere"
	// DeliveryScopeSentence is a delivery that can change only between
	// requests, for a service that takes delivery as a setting of each request,
	// such as MiniMax, or that changes it reliably only between generations,
	// such as Cartesia. A sentence is one request; when the service streams
	// tokens, a request is a token, so a change can take effect within a
	// sentence.
	DeliveryScopeSentence DeliveryScope = "sentence"
)

// ExpressionTerm is one sound or delivery a TTS service can perform.
type ExpressionTerm struct {
	// Name is the name the service performs it by, such as "laughter" or
	// "cheerful".
	Name string
	// Description is a short line on when to use it, or empty.
	Description string
}

// ExpressionCapabilities is the expression tags a TTS service can perform.
//
// The LLM is taught these by their universal names, and tags outside them are
// left out of the service's speech. A service that can laugh, speak in two
// moods, pause and stress words declares:
//
//	expressive.ExpressionCapabilities{
//		Sounds: []expressive.ExpressionTerm{{Name: "laughs"}},
//		Delivery: []expressive.ExpressionTerm{
//			{Name: "warmly"},
//			{Name: "calm", Description: "to reassure a worried caller"},
//		},
//		PauseRangeMS: &[2]int{100, 3000},
//		Emphasis:     true,
//		ServiceVocabulary: map[expressive.UniversalName][]string{
//			expressive.SoundLaugh:   {"laughs"},
//			expressive.DeliveryWarm: {"warmly"},
//		},
//	}
type ExpressionCapabilities struct {
	// Sounds are the sounds the service can make at one point, by its own
	// names.
	Sounds []ExpressionTerm
	// Delivery is the deliveries the service can apply to the words that
	// follow, by its own names.
	Delivery []ExpressionTerm
	// AllowsFreeTextSounds is whether the service also accepts sounds outside
	// Sounds, described in a few words.
	AllowsFreeTextSounds bool
	// AllowsFreeTextDelivery is whether the service also accepts deliveries
	// outside Delivery, described in a few words.
	AllowsFreeTextDelivery bool
	// PauseRangeMS is the shortest and longest pause the service can make, in
	// milliseconds, or nil when it can't pause.
	PauseRangeMS *[2]int
	// Emphasis is whether the service can stress words marked ^^word^^.
	Emphasis bool
	// DeliveryScope is where in a response the service can change delivery.
	// Empty is [DeliveryScopeAnywhere].
	DeliveryScope DeliveryScope
	// ServiceVocabulary is the names the service performs universal deliveries
	// and sounds by, best first, where they differ from the universal names,
	// such as {DeliveryWarm: {"content", "happy"}}. A universal name written in
	// a tag is sent by the first of these the service performs. A universal
	// name left out is performed by its own, when the service declares it or
	// accepts free text.
	ServiceVocabulary map[UniversalName][]string
}

// DeliveryName is the name the service performs a delivery by, and false when
// it can't.
//
// A universal delivery is performed by its first name in ServiceVocabulary the
// service performs, else by its own; any other name by itself. The service
// performs a name it declares, or any name when it accepts free text. name is
// the delivery as written in its tag.
func (c *ExpressionCapabilities) DeliveryName(name string) (string, bool) {
	var names []string
	if IsDelivery(name) {
		names = c.ServiceVocabulary[ExpressionDelivery(name)]
	}
	return performedName(name, c.Delivery, names, c.AllowsFreeTextDelivery)
}

// SoundName is the name the service performs a sound by, and false when it
// can't. It is read as [ExpressionCapabilities.DeliveryName] reads a delivery.
func (c *ExpressionCapabilities) SoundName(name string) (string, bool) {
	var names []string
	if IsSound(name) {
		names = c.ServiceVocabulary[ExpressionSound(name)]
	}
	return performedName(name, c.Sounds, names, c.AllowsFreeTextSounds)
}

func performedName(name string, terms []ExpressionTerm, names []string, freeText bool) (string, bool) {
	declared := make(map[string]struct{}, len(terms))
	for _, t := range terms {
		declared[t.Name] = struct{}{}
	}
	for _, candidate := range append(append([]string(nil), names...), name) {
		if _, ok := declared[candidate]; freeText || ok {
			return candidate, true
		}
	}
	return "", false
}

// SpeedDelivery is slow and fast, as a service that performs them declares
// them.
func SpeedDelivery() []ExpressionTerm {
	return []ExpressionTerm{
		{Name: SlowDelivery, Description: "to say something carefully, such as a number, a code or an address"},
		{Name: FastDelivery, Description: "to say something in passing"},
	}
}
