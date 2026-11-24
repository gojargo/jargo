package spec_test

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/gojargo/jargo/processor/rtvi"
	"github.com/gojargo/jargo/processor/rtvi/spec"
)

// The checks of the documents on their own: each is well formed, the current
// protocol version has one, and a released version's wire format is unchanged.
// The checks of the documents against the code, and of the server against the
// documents, are in the rtvi package's conformance tests.

// document reads a protocol version's document.
func document(t *testing.T, version string) map[string]any {
	t.Helper()
	raw, err := spec.Document(version)
	if err != nil {
		t.Fatal(err)
	}
	var doc map[string]any
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatalf("protocol %s: %v", version, err)
	}
	return doc
}

// object reads a JSON object out of a decoded document, failing the test when
// it is anything else.
func object(t *testing.T, node any, keys ...string) map[string]any {
	t.Helper()
	for _, key := range keys {
		m, ok := node.(map[string]any)
		if !ok {
			t.Fatalf("%s: not an object", key)
		}
		node = m[key]
	}
	m, ok := node.(map[string]any)
	if !ok {
		t.Fatalf("%v: not an object", keys)
	}
	return m
}

// list reads a JSON array out of a decoded object.
func list(t *testing.T, node map[string]any, key string) []any {
	t.Helper()
	items, ok := node[key].([]any)
	if !ok {
		t.Fatalf("%s: not an array", key)
	}
	return items
}

// messages are a document's messages, by type.
func messages(t *testing.T, doc map[string]any) map[string]map[string]any {
	t.Helper()
	out := map[string]map[string]any{}
	for _, m := range object(t, doc, "components", "messages") {
		msg, ok := m.(map[string]any)
		if !ok {
			t.Fatal("a message is not an object")
		}
		name, _ := msg["name"].(string)
		out[name] = msg
	}
	return out
}

// TestCurrentProtocolVersionHasADocument checks the protocol version the
// server speaks has its document, the newest one.
func TestCurrentProtocolVersionHasADocument(t *testing.T) {
	versions := spec.Versions()
	if len(versions) == 0 || versions[len(versions)-1] != rtvi.ProtocolVersion {
		t.Fatalf("documents for %v, want the newest to be protocol %s", versions, rtvi.ProtocolVersion)
	}
}

// TestDocumentVersionIsItsProtocolVersion checks each document describes the
// protocol version its folder is named after.
func TestDocumentVersionIsItsProtocolVersion(t *testing.T) {
	for _, version := range spec.Versions() {
		if got := object(t, document(t, version), "info")["version"]; got != version {
			t.Errorf("spec/%s describes protocol %v", version, got)
		}
	}
}

// TestReleasedWireFormatsAreUnchanged checks a released version's wire format
// never changes: a client implements it from its document, so a change to the
// wire format is a new version. Descriptions and examples can still be
// improved.
func TestReleasedWireFormatsAreUnchanged(t *testing.T) {
	released, err := spec.Released()
	if err != nil {
		t.Fatal(err)
	}
	if len(released) == 0 {
		t.Fatal("no released protocol version recorded")
	}
	for version, entry := range released {
		t.Run(version, func(t *testing.T) {
			got, err := spec.ContractSHA256(version)
			if err != nil {
				t.Fatalf("spec/%s was released in jargo %s: %v", version, entry.Jargo, err)
			}
			if got != entry.ContractSHA256 {
				t.Errorf("the contract of protocol %s changed since jargo %s released it. For a change "+
					"to the wire format, bump ProtocolVersion and restore the released version. For a "+
					"schema that describes what the server already sends more precisely, update "+
					"contract_sha256 in spec/released.json (see spec/README.md)", version, entry.Jargo)
			}
		})
	}
}

// TestContractLeavesOutProse checks the contract hash ignores what only
// describes, and changes with the wire format.
func TestContractLeavesOutProse(t *testing.T) {
	version := rtvi.ProtocolVersion
	want, err := spec.ContractSHA256(version)
	if err != nil {
		t.Fatal(err)
	}
	released, err := spec.Released()
	if err != nil {
		t.Fatal(err)
	}
	if entry, ok := released[version]; ok && entry.ContractSHA256 != want {
		t.Fatalf("contract of %s = %s, want the released %s", version, want, entry.ContractSHA256)
	}
	if other := spec.Versions()[0]; other != version {
		got, err := spec.ContractSHA256(other)
		if err != nil {
			t.Fatal(err)
		}
		if got == want {
			t.Errorf("protocols %s and %s hash the same, though their wire formats differ", other, version)
		}
	}
	if _, err := spec.ContractSHA256("0.0.0"); err == nil {
		t.Error("a version without a document hashed without an error")
	}
}

// TestMessageTypesAreUnique checks no two messages share a type, and each is
// filed under its type.
func TestMessageTypesAreUnique(t *testing.T) {
	for _, version := range spec.Versions() {
		doc := document(t, version)
		seen := map[string]bool{}
		for key, m := range object(t, doc, "components", "messages") {
			name, _ := m.(map[string]any)["name"].(string)
			if seen[name] {
				t.Errorf("spec/%s: %q twice", version, name)
			}
			seen[name] = true
			if key != strings.ReplaceAll(name, "-", "_") {
				t.Errorf("spec/%s: %q filed as %q", version, name, key)
			}
		}
	}
}

// operationTypes are the types of the messages an operation lists.
func operationTypes(t *testing.T, doc map[string]any, operation string) []string {
	t.Helper()
	msgs := object(t, doc, "components", "messages")
	var types []string
	for _, ref := range list(t, object(t, doc, "operations", operation), "messages") {
		target, _ := ref.(map[string]any)["$ref"].(string)
		key := target[strings.LastIndex(target, "/")+1:]
		if _, ok := object(t, doc, "channels", "rtvi", "messages")[key]; !ok {
			t.Errorf("%s lists %s, which the channel does not carry", operation, target)
		}
		m, ok := msgs[key].(map[string]any)
		if !ok {
			t.Fatalf("%s lists %s, which the document does not describe", operation, target)
		}
		name, _ := m["name"].(string)
		types = append(types, name)
	}
	return types
}

// TestOperationsSplitMessagesByDirection checks every message is either one
// the client sends or one the server sends, never both.
func TestOperationsSplitMessagesByDirection(t *testing.T) {
	for _, version := range spec.Versions() {
		doc := document(t, version)
		received := operationTypes(t, doc, "receiveClientMessages")
		sent := operationTypes(t, doc, "sendServerMessages")
		all := append(slices.Clone(received), sent...)
		slices.Sort(all)
		var want []string
		for name := range messages(t, doc) {
			want = append(want, name)
		}
		slices.Sort(want)
		if !slices.Equal(all, want) {
			t.Errorf("spec/%s: the operations list %v, want each of %v once", version, all, want)
		}
	}
}

// TestBehaviorsAreUniqueAndNameKnownMessages checks each behavior has an id of
// its own and involves only messages the document describes.
func TestBehaviorsAreUniqueAndNameKnownMessages(t *testing.T) {
	for _, version := range spec.Versions() {
		doc := document(t, version)
		types := messages(t, doc)
		ids := map[string]bool{}
		for _, b := range list(t, doc, "x-rtvi-behaviors") {
			behavior, _ := b.(map[string]any)
			id, _ := behavior["id"].(string)
			if ids[id] {
				t.Errorf("spec/%s: behavior %q twice", version, id)
			}
			ids[id] = true
			for _, m := range list(t, behavior, "messages") {
				if _, ok := types[m.(string)]; !ok {
					t.Errorf("spec/%s: behavior %q names %v, which the document does not describe", version, id, m)
				}
			}
		}
	}
}

// TestEveryMessageIsInOneGroup checks each message is in exactly one of the
// groups the document lists, which the documentation shows together.
func TestEveryMessageIsInOneGroup(t *testing.T) {
	for _, version := range spec.Versions() {
		doc := document(t, version)
		groups := map[string]bool{}
		for _, tag := range list(t, object(t, doc, "info"), "tags") {
			name, _ := tag.(map[string]any)["name"].(string)
			groups[name] = true
		}
		for name, m := range messages(t, doc) {
			tags := list(t, m, "tags")
			if len(tags) != 1 {
				t.Errorf("spec/%s: %s is in %d groups, want one", version, name, len(tags))
				continue
			}
			if group, _ := tags[0].(map[string]any)["name"].(string); !groups[group] {
				t.Errorf("spec/%s: %s is in group %q, which the document does not list", version, name, group)
			}
		}
	}
}

// undescribed are the fields in a schema, nested ones included, that have no
// description.
func undescribed(schema any, path string) []string {
	var found []string
	switch s := schema.(type) {
	case map[string]any:
		for key, value := range s {
			if key != "properties" {
				found = append(found, undescribed(value, path)...)
				continue
			}
			props, _ := value.(map[string]any)
			for field, fieldSchema := range props {
				if _, ok := fieldSchema.(map[string]any)["description"]; !ok {
					found = append(found, path+field)
				}
				found = append(found, undescribed(fieldSchema, path+field+".")...)
			}
		}
	case []any:
		for _, value := range s {
			found = append(found, undescribed(value, path)...)
		}
	}
	return found
}

// TestEveryFieldHasADescription checks a client's author finds a description
// of each field, and of each message's data.
func TestEveryFieldHasADescription(t *testing.T) {
	for _, version := range spec.Versions() {
		doc := document(t, version)
		var missing []string
		for name, schema := range object(t, doc, "components", "schemas") {
			for _, field := range undescribed(schema, "") {
				missing = append(missing, name+"."+field)
			}
		}
		for name, m := range object(t, doc, "components", "messages") {
			data, ok := object(t, m, "payload", "properties")["data"].(map[string]any)
			if !ok {
				continue
			}
			_, ref := data["$ref"]
			_, anyOf := data["anyOf"]
			_, described := data["description"]
			if !ref && !anyOf && !described {
				missing = append(missing, name+".data")
			}
			for _, field := range undescribed(data, "") {
				missing = append(missing, name+".data."+field)
			}
		}
		slices.Sort(missing)
		if len(missing) != 0 {
			t.Errorf("spec/%s: no description for %v", version, missing)
		}
	}
}

// TestResponsesCarryAnID checks the messages that answer a client message
// repeat its id.
func TestResponsesCarryAnID(t *testing.T) {
	for _, version := range spec.Versions() {
		msgs := messages(t, document(t, version))
		for _, msgType := range []string{rtvi.TypeBotReady, rtvi.TypeServerResponse, rtvi.TypeErrorResponse} {
			if !slices.Contains(list(t, object(t, msgs[msgType], "payload"), "required"), any("id")) {
				t.Errorf("spec/%s: %s does not carry an id", version, msgType)
			}
		}
	}
}
