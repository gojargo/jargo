// Package spec is the machine-readable specification of the RTVI protocol: an
// AsyncAPI 3.0 document for each protocol version, which lists every message a
// client sends and the server sends, with each field's schema and description,
// an example, the setting that turns the message on, the lowest client version
// that receives it, and the rules the server follows across messages.
//
// The documents ship with the package, at <version>/rtvi.asyncapi.json next to
// this file, so a client library can generate its protocol code from them at a
// pinned version, and anyone can build an RTVI client or server from them.
// Read them with Document. README.md next to this file describes the documents
// and how they are kept.
//
// released.json records each protocol version a release has shipped, with the
// hash of its wire format, which never changes after the release. Read it with
// Released, and compute a document's hash with ContractSHA256.
package spec

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"path"
	"slices"
	"sort"
	"strconv"
	"strings"
	"unicode/utf16"
)

// documentName is the name of a protocol version's document, in the folder
// named after the version.
const documentName = "rtvi.asyncapi.json"

// releasedName is the name of the record of released protocol versions.
const releasedName = "released.json"

//go:embed released.json */rtvi.asyncapi.json
var files embed.FS

// Versions returns every protocol version that has a document, oldest first.
func Versions() []string {
	// The only error is a malformed pattern, and this one is fixed.
	paths, _ := fs.Glob(files, "*/"+documentName)
	versions := make([]string, 0, len(paths))
	for _, p := range paths {
		versions = append(versions, path.Dir(p))
	}
	slices.SortFunc(versions, compareVersions)
	return versions
}

// compareVersions orders two protocol versions of the form
// "major.minor.patch" by their numbers, a part that is not a number counting as
// zero.
func compareVersions(a, b string) int {
	return slices.Compare(versionParts(a), versionParts(b))
}

// versionParts reads a version's numbers, a part that is not a number reading
// as zero.
func versionParts(version string) []int {
	parts := strings.Split(version, ".")
	out := make([]int, len(parts))
	for i, part := range parts {
		out[i], _ = strconv.Atoi(part)
	}
	return out
}

// Document returns a protocol version's AsyncAPI document, as shipped.
func Document(version string) ([]byte, error) {
	doc, err := fs.ReadFile(files, path.Join(version, documentName))
	if err != nil {
		return nil, fmt.Errorf("rtvi spec: no document for protocol %s: %w", version, err)
	}
	return doc, nil
}

// Release is one protocol version a release has shipped, as released.json
// records it.
type Release struct {
	// ContractSHA256 is the SHA-256 of the version's contract, as hex. See
	// ContractSHA256.
	ContractSHA256 string `json:"contract_sha256"`
	// Jargo is the jargo release that first shipped the version.
	Jargo string `json:"jargo"`
}

// Released returns the protocol versions a release has shipped, by version.
func Released() (map[string]Release, error) {
	raw, err := fs.ReadFile(files, releasedName)
	if err != nil {
		return nil, fmt.Errorf("rtvi spec: %w", err)
	}
	var released map[string]Release
	if err := json.Unmarshal(raw, &released); err != nil {
		return nil, fmt.Errorf("rtvi spec: reading %s: %w", releasedName, err)
	}
	return released, nil
}

// ContractSHA256 returns the SHA-256 of a protocol version's contract, as hex.
//
// The contract is the document without its descriptions, summaries, titles,
// examples, tags and deprecations, so clearer wording doesn't change it, and
// any change to the wire format does. It is hashed as compact JSON with its
// keys sorted and every character outside printable ASCII escaped.
func ContractSHA256(version string) (string, error) {
	doc, err := Document(version)
	if err != nil {
		return "", err
	}
	dec := json.NewDecoder(bytes.NewReader(doc))
	// The numbers are hashed as the document writes them.
	dec.UseNumber()
	var node any
	if err := dec.Decode(&node); err != nil {
		return "", fmt.Errorf("rtvi spec: reading the document of protocol %s: %w", version, err)
	}
	var buf bytes.Buffer
	writeCanonical(&buf, contract(node, false))
	sum := sha256.Sum256(buf.Bytes())
	return hex.EncodeToString(sum[:]), nil
}

// prose are the keys that only describe: a deprecation describes too, since a
// deprecated message or field is sent and accepted as before.
//
//nolint:gochecknoglobals // fixed table, read only
var prose = map[string]bool{
	"description": true, "summary": true, "title": true, "examples": true, "tags": true,
	"deprecated": true, "x-rtvi-deprecated": true,
}

// named are the maps whose keys are names rather than keywords, kept whatever
// they are called.
//
//nolint:gochecknoglobals // fixed table, read only
var named = map[string]bool{
	"properties": true, "patternProperties": true, "$defs": true, "schemas": true,
	"messages": true, "channels": true, "operations": true,
}

// contract is a document without its prose: what a client and a server must
// agree on. A map whose keys are names keeps them all.
func contract(node any, isNamed bool) any {
	switch n := node.(type) {
	case []any:
		out := make([]any, len(n))
		for i, item := range n {
			out[i] = contract(item, false)
		}
		return out
	case map[string]any:
		out := make(map[string]any, len(n))
		for key, value := range n {
			switch {
			case isNamed:
				out[key] = contract(value, false)
			case !prose[key]:
				out[key] = contract(value, named[key])
			}
		}
		return out
	default:
		return node
	}
}

// writeCanonical writes a decoded JSON value as compact JSON, with the keys of
// each object sorted and every character outside printable ASCII escaped.
func writeCanonical(buf *bytes.Buffer, node any) {
	switch n := node.(type) {
	case map[string]any:
		keys := make([]string, 0, len(n))
		for key := range n {
			keys = append(keys, key)
		}
		// Byte order of UTF-8 is code point order.
		sort.Strings(keys)
		buf.WriteByte('{')
		for i, key := range keys {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeString(buf, key)
			buf.WriteByte(':')
			writeCanonical(buf, n[key])
		}
		buf.WriteByte('}')
	case []any:
		buf.WriteByte('[')
		for i, item := range n {
			if i > 0 {
				buf.WriteByte(',')
			}
			writeCanonical(buf, item)
		}
		buf.WriteByte(']')
	case string:
		writeString(buf, n)
	case json.Number:
		buf.WriteString(n.String())
	case bool:
		buf.WriteString(strconv.FormatBool(n))
	case nil:
		buf.WriteString("null")
	}
}

// writeString writes a JSON string, escaping every character outside printable
// ASCII, one beyond the Basic Multilingual Plane as its UTF-16 surrogate pair.
func writeString(buf *bytes.Buffer, s string) {
	const hexDigits = "0123456789abcdef"
	escape := func(r rune) {
		buf.WriteString(`\u`)
		for shift := 12; shift >= 0; shift -= 4 {
			buf.WriteByte(hexDigits[(r>>shift)&0xf])
		}
	}
	buf.WriteByte('"')
	for _, r := range s {
		switch {
		case r == '"':
			buf.WriteString(`\"`)
		case r == '\\':
			buf.WriteString(`\\`)
		case r == '\n':
			buf.WriteString(`\n`)
		case r == '\r':
			buf.WriteString(`\r`)
		case r == '\t':
			buf.WriteString(`\t`)
		case r == '\b':
			buf.WriteString(`\b`)
		case r == '\f':
			buf.WriteString(`\f`)
		case r >= ' ' && r <= '~':
			buf.WriteRune(r)
		case r > 0xffff:
			high, low := utf16.EncodeRune(r)
			escape(high)
			escape(low)
		default:
			escape(r)
		}
	}
	buf.WriteByte('"')
}
