package security

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// Redacted is what RedactSecrets puts in place of a credential.
const Redacted = "[REDACTED]"

// credentialWords are the words that mark a credential's name. A name marks one
// when one of its words is one of these, so x-api-key and X-Amz-Security-Token
// do while keyterms and max_tokens don't.
//
//nolint:gochecknoglobals // a fixed set, never written
var credentialWords = map[string]bool{
	"apikey":        true,
	"auth":          true,
	"authorization": true,
	"cookie":        true,
	"credential":    true,
	"credentials":   true,
	"creds":         true,
	"jwt":           true,
	"key":           true,
	"passwd":        true,
	"password":      true,
	"pwd":           true,
	"secret":        true,
	"sig":           true,
	"signature":     true,
	"token":         true,
	// Google's resumable upload session ID, which authorizes the upload by itself.
	"uploadid": true,
}

// authSchemes are the schemes that may come ahead of a credential's value. The
// scheme is kept and the credential after it masked.
//
//nolint:gochecknoglobals // a fixed list, never written
var authSchemes = []string{"Bearer", "Basic", "Token", "Api-Token", "Key", "Digest", "AWS4-HMAC-SHA256"}

// spaceClass is the body of a character class matching every character that
// counts as whitespace in a credential's surroundings: the ASCII ones, the
// information separators, and the Unicode spaces.
const spaceClass = `\t\n\v\f\r \x1c-\x1f\x{85}\x{a0}\x{1680}\x{2000}-\x{200a}\x{2028}\x{2029}\x{202f}\x{205f}\x{3000}`

//nolint:gochecknoglobals // compiled once, immutable
var (
	// urlPassword matches a URL's password with the scheme, user and colon
	// ahead of it in the first group and the @ after it. Whether the scheme
	// starts a word is checked by hand.
	urlPassword = regexp.MustCompile(`(?i)([a-z][a-z0-9+.-]*://[^` + spaceClass + `:/@]+:)[^` + spaceClass + `/@]+@`)

	// knownTokens matches the credentials recognizable by their format alone,
	// all but a Slack webhook's secret path, which slackWebhookPrefix finds.
	// The formats are after gitleaks' rules for the same credentials
	// (github.com/gitleaks/gitleaks, config/gitleaks.toml), loosened where a
	// looser match costs nothing: a log line is masked, not scanned for
	// findings. Every token but a PEM key starts a word, which is checked by
	// hand. The first group holds the formats that come after a Slack webhook's
	// in the order they are tried, and the second a PEM private key, through
	// its END line or, cut short, the end of the text.
	knownTokens = regexp.MustCompile(`(?:` +
		`sk-[A-Za-z0-9_-]{20,}` + // OpenAI, Anthropic
		`|(?:A3T[A-Z0-9]|AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16}` + // AWS access key IDs
		`|ABSK[A-Za-z0-9+/]{109,269}={0,2}` + // Amazon Bedrock API keys, long-lived
		`|bedrock-api-key-[A-Za-z0-9+/=_-]+` + // Amazon Bedrock API keys, short-lived
		`|AIza[0-9A-Za-z_-]{35}` + // Google API keys
		`|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[\p{L}\p{N}_]{82,}` + // GitHub
		`|xox[abeoprs]-[A-Za-z0-9-]{10,}|xapp-\p{Nd}-[A-Za-z0-9-]{10,}` + // Slack
		`|(` +
		`(?:hf|api_org)_[A-Za-z]{34}` + // Hugging Face
		`|pplx-[A-Za-z0-9]{48}` + // Perplexity
		`|SK[0-9a-fA-F]{32}` + // Twilio API keys
		`|eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*` + // JWTs, signed or not
		`))` +
		`|(-----BEGIN[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----` +
		`(?:[\s\S]*?-----END[ A-Z0-9_-]{0,100}PRIVATE KEY(?: BLOCK)?-----|[\s\S]*))`)

	// slackWebhookPrefix matches the part of a Slack webhook's URL ahead of its
	// path, which is the secret.
	slackWebhookPrefix = regexp.MustCompile(`hooks\.slack\.com/(?:services|workflows|triggers)/`)

	// exceptionName matches an exception or warning class name, optionally
	// qualified by its package (KeyError, jwt.exceptions.InvalidTokenError). It
	// names what went wrong, so the text after it is the explanation, not a
	// credential.
	exceptionName = regexp.MustCompile(`^(?:[a-z_][a-z0-9_]*\.)*[A-Z][A-Za-z0-9]*(?:Error|Exception|Warning)$`)

	camelBoundary = regexp.MustCompile(`([a-z0-9])([A-Z])`)
	nameSeparator = regexp.MustCompile(`[-_.]`)
)

// RedactSecrets masks the credentials in text, leaving the rest as written.
//
// Error messages often quote whatever a failure carried: an SDK error that
// prints the headers it sent, a URL signed for one download, a provider body
// echoing the key it rejected. RedactSecrets masks the credential and keeps the
// rest, so a message still says what went wrong. It masks three things:
//
//   - The value of any name=value or name: value pair (or :=, =>), quoted or
//     not, whose name marks a credential (api_key, Authorization,
//     X-Amz-Signature, apiKey...). This covers query strings, printed header
//     maps and plain prose alike. An auth scheme (Bearer, Basic...) ahead of the
//     value is kept. An exception class name followed by its message
//     (KeyError: ...) is not a credential name.
//   - A URL's password (scheme://user:password@host).
//   - Tokens recognizable by their format alone: provider key prefixes, AWS
//     access key IDs, JWTs and PEM private keys.
//
// Each is replaced by Redacted. Redacting text twice reads the same as
// redacting it once.
func RedactSecrets(text string) string {
	// Known formats go first: a PEM key spans lines, and masking its first line
	// as a pair's value would leave the rest unrecognizable.
	text = redactKnownTokens(text)
	text = redactPairs(text)
	return redactURLPasswords(text)
}

// redactKnownTokens masks every token knownTokens finds that starts a word,
// and every Slack webhook's secret path.
func redactKnownTokens(text string) string {
	// A webhook's path is told by the URL ahead of it, which may itself be part
	// of a token masked before it, so the prefixes are found in the text as
	// written.
	prefixes := slackWebhookPrefix.FindAllStringIndex(text, -1)

	var out strings.Builder
	last, pos := 0, 0
	for pos <= len(text) {
		start, end, after, found := nextKnownToken(text, pos)
		hookStart, hookEnd, hookFound := nextSlackWebhook(text, prefixes, pos)
		// The token starting first wins, and at the same start the format tried
		// first: a webhook's path is tried after the formats that come before it.
		if hookFound && (!found || hookStart < start || (hookStart == start && after)) {
			start, end, found = hookStart, hookEnd, true
		}
		if !found {
			break
		}
		out.WriteString(text[last:start])
		out.WriteString(Redacted)
		last, pos = end, end
	}
	out.WriteString(text[last:])
	return out.String()
}

// nextKnownToken finds the first token knownTokens matches at or after pos
// that starts a word, or a PEM key, which need not. after reports whether its
// format is one tried after a Slack webhook's.
func nextKnownToken(text string, pos int) (start, end int, after, found bool) {
	for pos <= len(text) {
		m := knownTokens.FindStringSubmatchIndex(text[pos:])
		if m == nil {
			return 0, 0, false, false
		}
		start, end = m[0]+pos, m[1]+pos
		if m[4] >= 0 || wordBoundary(text, start) {
			return start, end, m[2] >= 0, true
		}
		// A token that does not start a word is no token, and the search goes
		// on from the next character.
		pos = start + runeLen(text, start)
	}
	return 0, 0, false, false
}

// nextSlackWebhook finds the first Slack webhook's secret path starting at or
// after pos: twenty or more of its characters, right after one of prefixes,
// starting a word.
func nextSlackWebhook(text string, prefixes [][]int, pos int) (start, end int, found bool) {
	for _, p := range prefixes {
		start = p[1]
		if start < pos || !wordBoundary(text, start) {
			continue
		}
		end = start
		for end < len(text) && isWebhookPathByte(text[end]) {
			end++
		}
		if end-start >= 20 {
			return start, end, true
		}
	}
	return 0, 0, false
}

// isWebhookPathByte reports whether c can be part of a Slack webhook's secret
// path.
func isWebhookPathByte(c byte) bool {
	return isASCIILetter(c) || ('0' <= c && c <= '9') || c == '+' || c == '/'
}

// redactURLPasswords masks the password of every URL whose scheme starts a
// word.
func redactURLPasswords(text string) string {
	var out strings.Builder
	last, pos := 0, 0
	for pos <= len(text) {
		m := urlPassword.FindStringSubmatchIndex(text[pos:])
		if m == nil {
			break
		}
		for i := range m {
			m[i] += pos
		}
		if !wordBoundary(text, m[0]) {
			pos = m[0] + runeLen(text, m[0])
			continue
		}
		out.WriteString(text[last:m[3]])
		out.WriteString(Redacted)
		// The @ ending the match is not part of the password.
		last, pos = m[1]-1, m[1]-1
	}
	out.WriteString(text[last:])
	return out.String()
}

// redactPairs masks the value of every pair whose name marks a credential.
//
// Only a credential's value is taken whole. Any other pair's value is read on
// as ordinary text, so the pairs it holds (URL: https://...?token=..., error:
// Authorization: Bearer ...) are found in place.
func redactPairs(text string) string {
	var out strings.Builder
	pos := 0
	for {
		start, ok := searchPairStart(text, pos)
		if !ok {
			break
		}
		var value pairValue
		found := false
		if start.masksValue() {
			value, found = matchValue(text, start.end)
		}
		if found {
			// A credential's value that is only the next credential's name, as
			// token: is in api_key: token: abc, gives way to that credential.
			if following, ok := matchPairStart(text, start.end); ok &&
				following.masksValue() && following.end > value.end {
				found = false
			}
		}
		if !found {
			out.WriteString(text[pos:start.end])
			pos = start.end
			continue
		}
		out.WriteString(text[pos:start.end])
		if value.closingQuote != "" {
			out.WriteString(value.closingQuote)
		} else {
			out.WriteString(value.openingQuote)
		}
		out.WriteString(Redacted)
		out.WriteString(value.closingQuote)
		pos = value.end
	}
	out.WriteString(text[pos:])
	return out.String()
}

// pairStart is the start of a name=value or name: value pair (or :=, =>), up
// to where its value begins.
type pairStart struct {
	// end is where the value begins.
	end int
	// name is the pair's name, with its quotes when it is quoted as in a
	// printed map.
	name string
	// quoted reports whether the name is quoted.
	quoted bool
	// separator is what joins the name to the value, with the whitespace
	// around it.
	separator string
	// scheme reports whether an auth scheme (Bearer, Basic...) comes ahead of
	// the value.
	scheme bool
}

// masksValue reports whether the pair's value is a credential to mask.
func (p pairStart) masksValue() bool {
	return isCredentialName(p.name) && !p.isExceptionName()
}

// isExceptionName reports whether the pair is an exception class name followed
// by its message, which explains what went wrong rather than quoting a
// credential.
func (p pairStart) isExceptionName() bool {
	return !p.quoted && !p.scheme &&
		strings.TrimFunc(p.separator, isSpace) == ":" &&
		exceptionName.MatchString(p.name)
}

// isCredentialName reports whether name marks a credential: either its words
// run together, or one of them on its own, is a credential word.
func isCredentialName(name string) bool {
	words := strings.ToLower(camelBoundary.ReplaceAllString(strings.Trim(name, `"'`), "${1}_${2}"))
	var parts []string
	for _, p := range nameSeparator.Split(words, -1) {
		if p != "" {
			parts = append(parts, p)
		}
	}
	if credentialWords[strings.Join(parts, "")] {
		return true
	}
	for _, p := range parts {
		if credentialWords[p] {
			return true
		}
	}
	return false
}

// searchPairStart finds the first pair start at or after pos.
func searchPairStart(text string, pos int) (pairStart, bool) {
	for i := pos; i < len(text); i += runeLen(text, i) {
		if p, ok := matchPairStart(text, i); ok {
			return p, true
		}
	}
	return pairStart{}, false
}

// matchPairStart reads a pair start beginning at i. The name begins a word,
// starts with a letter and runs on through letters, digits, underscores, dots
// and hyphens, optionally quoted. The separator is :=, =>, : or =, with any
// whitespace around it.
func matchPairStart(text string, i int) (pairStart, bool) {
	if i > 0 {
		if prev, _ := utf8.DecodeLastRuneInString(text[:i]); isNameRune(prev) {
			return pairStart{}, false
		}
	}
	nameEnd, quoted, ok := readName(text, i)
	if !ok {
		return pairStart{}, false
	}
	sepEnd, ok := readSeparator(text, nameEnd)
	if !ok {
		return pairStart{}, false
	}
	end, scheme := readScheme(text, sepEnd)
	return pairStart{
		end:       end,
		name:      text[i:nameEnd],
		quoted:    quoted,
		separator: text[nameEnd:sepEnd],
		scheme:    scheme,
	}, true
}

// readName reads a pair's name beginning at i and returns where it ends.
func readName(text string, i int) (end int, quoted, ok bool) {
	j := i
	var quote byte
	if j < len(text) && (text[j] == '"' || text[j] == '\'') {
		quote = text[j]
		j++
	}
	if j >= len(text) || !isASCIILetter(text[j]) {
		return 0, false, false
	}
	j++
	for j < len(text) {
		r, size := utf8.DecodeRuneInString(text[j:])
		if !isNameRune(r) {
			break
		}
		j += size
	}
	if quote != 0 {
		if j >= len(text) || text[j] != quote {
			return 0, false, false
		}
		j++
	}
	return j, quote != 0, true
}

// readSeparator reads the separator beginning at i, with the whitespace around
// it, and returns where it ends.
func readSeparator(text string, i int) (int, bool) {
	j := skipSpace(text, i)
	switch {
	case strings.HasPrefix(text[j:], ":="), strings.HasPrefix(text[j:], "=>"):
		j += 2
	case j < len(text) && (text[j] == ':' || text[j] == '='):
		j++
	default:
		return 0, false
	}
	return skipSpace(text, j), true
}

// readScheme reads an auth scheme and the whitespace after it beginning at i,
// and returns where the value begins: past the scheme when there is one, at i
// when there is none.
func readScheme(text string, i int) (int, bool) {
	for _, s := range authSchemes {
		if !strings.HasPrefix(text[i:], s) {
			continue
		}
		if after := skipSpace(text, i+len(s)); after > i+len(s) {
			return after, true
		}
	}
	return i, false
}

// pairValue is a pair's value.
type pairValue struct {
	// end is where the value ends.
	end int
	// closingQuote is the quote around a value quoted at both ends.
	closingQuote string
	// openingQuote is the quote ahead of a value cut short before its closing
	// quote, as a truncated print of it leaves it.
	openingQuote string
}

// matchValue reads a pair's value beginning at p: quoted on one line, or
// running to the next whitespace or delimiter. A quoted value cut short before
// its closing quote runs to the next delimiter instead. A value already
// redacted is read whole, so text redacted twice reads the same as text
// redacted once.
func matchValue(text string, p int) (pairValue, bool) {
	if p < len(text) && (text[p] == '"' || text[p] == '\'') {
		q := text[p]
		rest := text[p+1:]
		if at := strings.IndexByte(rest, q); at >= 0 && !strings.Contains(rest[:at], "\n") {
			return pairValue{end: p + 1 + at + 1, closingQuote: string(q)}, true
		}
	}

	j := p
	opening := ""
	if j < len(text) && (text[j] == '"' || text[j] == '\'') {
		opening = text[j : j+1]
		j++
	}
	bareStart := j
	for j < len(text) {
		if strings.HasPrefix(text[j:], Redacted) {
			j += len(Redacted)
			continue
		}
		r, size := utf8.DecodeRuneInString(text[j:])
		if isSpace(r) || strings.ContainsRune(`,;&"'}])[`, r) {
			break
		}
		j += size
	}
	if j == bareStart {
		return pairValue{}, false
	}
	return pairValue{end: j, openingQuote: opening}, true
}

// skipSpace returns the first position at or after i that is not whitespace.
func skipSpace(text string, i int) int {
	for i < len(text) {
		r, size := utf8.DecodeRuneInString(text[i:])
		if !isSpace(r) {
			break
		}
		i += size
	}
	return i
}

// wordBoundary reports whether i lies between a word character and anything
// else, either way round.
func wordBoundary(text string, i int) bool {
	before, after := false, false
	if i > 0 {
		r, _ := utf8.DecodeLastRuneInString(text[:i])
		before = isWordRune(r)
	}
	if i < len(text) {
		r, _ := utf8.DecodeRuneInString(text[i:])
		after = isWordRune(r)
	}
	return before != after
}

// runeLen is the length in bytes of the character at i, one past the end of
// the text.
func runeLen(text string, i int) int {
	if i >= len(text) {
		return 1
	}
	_, size := utf8.DecodeRuneInString(text[i:])
	return size
}

// isWordRune reports whether r is a letter, a digit or an underscore, in any
// script.
func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsNumber(r)
}

// isNameRune reports whether r can continue a pair's name.
func isNameRune(r rune) bool {
	return isWordRune(r) || r == '.' || r == '-'
}

func isASCIILetter(c byte) bool {
	return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z')
}

// isSpace reports whether r is whitespace, the information separators
// included.
func isSpace(r rune) bool {
	return unicode.IsSpace(r) || (0x1c <= r && r <= 0x1f)
}
