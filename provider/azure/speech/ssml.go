package speech

import (
	"strings"

	"github.com/gojargo/jargo/expressive"
)

const (
	// defaultTTSVoice is the voice synthesis uses when none is configured.
	defaultTTSVoice = "en-US-SaraNeural"
	// defaultTTSLanguage is the SSML language when none is configured.
	defaultTTSLanguage = "en-US"
)

// ssmlEscaper escapes the characters SSML reserves. The ampersand goes first,
// so the entities the others become are not escaped again.
//
//nolint:gochecknoglobals // stateless and safe for concurrent use
var ssmlEscaper = strings.NewReplacer(
	"&", "&amp;",
	"<", "&lt;",
	">", "&gt;",
	`"`, "&quot;",
	"'", "&apos;",
)

// ssmlParams is what the SSML document a sentence is sent in is built from: the
// voice and language, and how the voice is to speak. An empty field is left out
// of the document.
type ssmlParams struct {
	Voice    string
	Language string
	// Effect is the audio effect processor on <voice>, such as "eq_car".
	Effect string
	// VoiceParameters is the semicolon-separated parameters attribute on <voice>,
	// for HD voices.
	VoiceParameters string
	// ForceLocale wraps the text in <lang>, so a multilingual voice speaks in
	// Language.
	ForceLocale bool
	// Style, StyleDegree and Role go on <mstts:express-as>.
	Style       string
	StyleDegree string
	Role        string
	// Rate, Pitch and Volume go on <prosody>.
	Rate   string
	Pitch  string
	Volume string
	// Emphasis is the level of <emphasis>.
	Emphasis string
}

// constructSSML wraps text in the SSML document it is synthesized from,
// escaping the text so it cannot break the markup.
func constructSSML(text string, p ssmlParams) string {
	escaped := ssmlEscaper.Replace(text)

	voiceAttrs := "name='" + p.Voice + "'"
	if p.Effect != "" {
		voiceAttrs += " effect='" + p.Effect + "'"
	}
	if p.VoiceParameters != "" {
		voiceAttrs += " parameters='" + p.VoiceParameters + "'"
	}

	var b strings.Builder
	b.WriteString("<speak version='1.0' xml:lang='" + p.Language + "' " +
		"xmlns='http://www.w3.org/2001/10/synthesis' " +
		"xmlns:mstts='http://www.w3.org/2001/mstts'>" +
		"<voice " + voiceAttrs + ">" +
		"<mstts:silence type='Sentenceboundary' value='20ms' />")

	if p.ForceLocale {
		b.WriteString("<lang xml:lang='" + p.Language + "'>")
	}

	if p.Style != "" {
		b.WriteString(expressAsTag(p))
	}

	prosodyAttrs := prosodyAttributes(p)
	// Only wrapped in prosody when there is something to set.
	if len(prosodyAttrs) > 0 {
		b.WriteString("<prosody " + strings.Join(prosodyAttrs, " ") + ">")
	}

	if p.Emphasis != "" {
		b.WriteString("<emphasis level='" + p.Emphasis + "'>")
	}

	b.WriteString(escaped)

	if p.Emphasis != "" {
		b.WriteString("</emphasis>")
	}
	if len(prosodyAttrs) > 0 {
		b.WriteString("</prosody>")
	}
	if p.Style != "" {
		b.WriteString("</mstts:express-as>")
	}
	if p.ForceLocale {
		b.WriteString("</lang>")
	}
	b.WriteString("</voice></speak>")
	return b.String()
}

// expressAsTag is the opening <mstts:express-as> tag for the style.
func expressAsTag(p ssmlParams) string {
	tag := "<mstts:express-as style='" + p.Style + "'"
	if p.StyleDegree != "" {
		tag += " styledegree='" + p.StyleDegree + "'"
	}
	if p.Role != "" {
		tag += " role='" + p.Role + "'"
	}
	return tag + ">"
}

// prosodyAttributes are the attributes of <prosody>, or none when nothing is
// set.
func prosodyAttributes(p ssmlParams) []string {
	var attrs []string
	if p.Rate != "" {
		attrs = append(attrs, "rate='"+p.Rate+"'")
	}
	if p.Pitch != "" {
		attrs = append(attrs, "pitch='"+p.Pitch+"'")
	}
	if p.Volume != "" {
		attrs = append(attrs, "volume='"+p.Volume+"'")
	}
	return attrs
}

// hdVoice reports whether a voice performs style markers: the DragonHD and
// DragonHD Omni voices do. DragonHD Flash voices each have styles of their own,
// so they perform none.
func hdVoice(voice string) bool {
	return strings.Contains(voice, ":DragonHD") && !strings.Contains(voice, ":DragonHDFlash")
}

// voiceExpressionAdapter is the expression adapter for a voice: style markers on
// HD voices, and none on the others.
func voiceExpressionAdapter(voice string) expressive.ExpressionAdapter {
	if hdVoice(voice) {
		return HDExpressionAdapter{}
	}
	return expressive.BaseExpressionAdapter{}
}
