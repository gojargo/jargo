package gemini

import (
	"encoding/json"
	"reflect"
	"sync"
	"testing"

	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/service/llm/adapter"
)

// toolCallContent builds a model turn holding one function call, signed when
// signature is not empty.
func toolCallContent(id, signature string) map[string]any {
	part := map[string]any{keyFunctionCall: map[string]any{
		adapterKeyID: id, adapterKeyName: "tool_" + id, "args": map[string]any{"q": id},
	}}
	if signature != "" {
		part[keyThoughtSignature] = signature
	}
	return map[string]any{adapterKeyRole: roleModel, adapterKeyParts: []map[string]any{part}}
}

// toolResponseContent builds a user turn holding one function response.
func toolResponseContent(id string) map[string]any {
	return map[string]any{adapterKeyRole: roleUser, adapterKeyParts: []map[string]any{{
		keyFunctionResponse: map[string]any{
			adapterKeyID: id, adapterKeyName: "tool_" + id, "response": map[string]any{"ok": true},
		},
	}}}
}

// callSignature builds a kept signature bookmarked to a function call.
func callSignature(id string) map[string]any {
	return map[string]any{
		keyType: typeThoughtSignature, keySignature: "sig-" + id,
		keyBookmark: map[string]any{BookmarkFunctionCall: id},
	}
}

// roles lists the roles of the contents in order.
func roles(contents []map[string]any) []string {
	out := make([]string, 0, len(contents))
	for _, c := range contents {
		r, _ := c[adapterKeyRole].(string)
		out = append(out, r)
	}
	return out
}

// partIDs lists the id of the function call or response each part carries.
func partIDs(c map[string]any) []string {
	var out []string
	for _, p := range contentParts(c) {
		for _, key := range []string{keyFunctionCall, keyFunctionResponse} {
			if m, ok := p[key].(map[string]any); ok {
				id, _ := m[adapterKeyID].(string)
				out = append(out, id)
			}
		}
	}
	return out
}

func wantStrings(t *testing.T, what string, got, want []string) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("%s = %v, want %v", what, got, want)
	}
}

// Parallel calls interleaved with their responses merge into one model turn and
// one user turn.
func TestMergeParallelInterleavedCallsAndResponses(t *testing.T) {
	got := mergeParallelToolCallsForThinking([]map[string]any{callSignature("c1")}, []map[string]any{
		toolCallContent("c1", "sig-c1"), toolResponseContent("c1"),
		toolCallContent("c2", ""), toolResponseContent("c2"),
	})
	wantStrings(t, "roles", roles(got), []string{roleModel, roleUser})
	wantStrings(t, "calls", partIDs(got[0]), []string{"c1", "c2"})
	wantStrings(t, "responses", partIDs(got[1]), []string{"c1", "c2"})
}

// Calls grouped first, then their responses, still merge into matching turns.
func TestMergeBatchOrderedCallsAndResponses(t *testing.T) {
	got := mergeParallelToolCallsForThinking([]map[string]any{callSignature("c1")}, []map[string]any{
		toolCallContent("c1", "sig-c1"), toolCallContent("c2", ""),
		toolResponseContent("c1"), toolResponseContent("c2"),
	})
	wantStrings(t, "roles", roles(got), []string{roleModel, roleUser})
	wantStrings(t, "calls", partIDs(got[0]), []string{"c1", "c2"})
	wantStrings(t, "responses", partIDs(got[1]), []string{"c1", "c2"})
}

// Three parallel calls merge into a single model turn and user turn.
func TestMergeThreeParallelCalls(t *testing.T) {
	got := mergeParallelToolCallsForThinking([]map[string]any{callSignature("c1")}, []map[string]any{
		toolCallContent("c1", "sig-c1"), toolResponseContent("c1"),
		toolCallContent("c2", ""), toolResponseContent("c2"),
		toolCallContent("c3", ""), toolResponseContent("c3"),
	})
	wantStrings(t, "roles", roles(got), []string{roleModel, roleUser})
	if len(contentParts(got[0])) != 3 || len(contentParts(got[1])) != 3 {
		t.Errorf("parts = %d calls, %d responses, want 3 of each",
			len(contentParts(got[0])), len(contentParts(got[1])))
	}
}

// A lone signed call and its response pass through as they are.
func TestMergeSingleCallUnchanged(t *testing.T) {
	got := mergeParallelToolCallsForThinking([]map[string]any{callSignature("c1")}, []map[string]any{
		toolCallContent("c1", "sig-c1"), toolResponseContent("c1"),
	})
	wantStrings(t, "roles", roles(got), []string{roleModel, roleUser})
	if len(contentParts(got[0])) != 1 || len(contentParts(got[1])) != 1 {
		t.Errorf("contents = %v, want one part each", got)
	}
}

// Sequential calls, each with its own signature, stay in separate turns.
func TestMergeSequentialCallsWithSignaturesNotMerged(t *testing.T) {
	got := mergeParallelToolCallsForThinking(
		[]map[string]any{callSignature("c1"), callSignature("c2")},
		[]map[string]any{
			toolCallContent("c1", "sig-c1"), toolResponseContent("c1"),
			toolCallContent("c2", "sig-c2"), toolResponseContent("c2"),
		})
	wantStrings(t, "roles", roles(got), []string{roleModel, roleUser, roleModel, roleUser})
	for i, c := range got {
		if len(contentParts(c)) != 1 {
			t.Errorf("content %d = %v, want one part", i, c)
		}
	}
}

// A parallel group followed by a sequential call are grouped independently.
func TestMergeMixedParallelAndSequentialGroups(t *testing.T) {
	got := mergeParallelToolCallsForThinking(
		[]map[string]any{callSignature("c1"), callSignature("c3")},
		[]map[string]any{
			toolCallContent("c1", "sig-c1"), toolResponseContent("c1"),
			toolCallContent("c2", ""), toolResponseContent("c2"),
			toolCallContent("c3", "sig-c3"), toolResponseContent("c3"),
		})
	wantStrings(t, "roles", roles(got), []string{roleModel, roleUser, roleModel, roleUser})
	lens := make([]int, 0, len(got))
	for _, c := range got {
		lens = append(lens, len(contentParts(c)))
	}
	if !reflect.DeepEqual(lens, []int{2, 2, 1, 1}) {
		t.Errorf("part counts = %v, want [2 2 1 1]", lens)
	}
}

// A message that is not a tool turn inside a group is kept and put back after
// it, rather than ending the group.
func TestMergeCollectsInterleavedNonToolMessage(t *testing.T) {
	text := textContent(roleModel, "thinking out loud")
	got := mergeParallelToolCallsForThinking([]map[string]any{callSignature("c1")}, []map[string]any{
		toolCallContent("c1", "sig-c1"), toolResponseContent("c1"),
		text,
		toolCallContent("c2", ""), toolResponseContent("c2"),
	})
	wantStrings(t, "roles", roles(got), []string{roleModel, roleUser, roleModel})
	wantStrings(t, "calls", partIDs(got[0]), []string{"c1", "c2"})
	wantStrings(t, "responses", partIDs(got[1]), []string{"c1", "c2"})
	if parts := contentParts(got[2]); len(parts) != 1 || parts[0][adapterKeyText] != "thinking out loud" {
		t.Errorf("content 2 = %v, want the interleaved text", got[2])
	}
}

// Without any signature bookmarked to a function call, nothing is merged.
func TestMergeNoThoughtSignaturesUnchanged(t *testing.T) {
	got := mergeParallelToolCallsForThinking(nil, []map[string]any{
		toolCallContent("c1", ""), toolResponseContent("c1"),
	})
	if len(got) != 2 {
		t.Errorf("contents = %v, want both left as they were", got)
	}
}

// An empty conversation stays empty.
func TestMergeEmptyMessages(t *testing.T) {
	if got := mergeParallelToolCallsForThinking(nil, nil); len(got) != 0 {
		t.Errorf("contents = %v, want none", got)
	}
}

// A merged parallel batch keeps Gemini's shape: the first call signed, the rest
// left unsigned rather than given the placeholder.
func TestSignedParallelGroupLeftUnsignedAfterFirstCall(t *testing.T) {
	got := mergeParallelToolCallsForThinking([]map[string]any{callSignature("c1")}, []map[string]any{
		toolCallContent("c1", "sig-c1"), toolResponseContent("c1"),
		toolCallContent("c2", ""), toolResponseContent("c2"),
	})
	addPlaceholderThoughtSignatures(got)
	parts := contentParts(got[0])
	if parts[0][keyThoughtSignature] != "sig-c1" {
		t.Errorf("first call = %v, want its own signature", parts[0])
	}
	if sig, ok := parts[1][keyThoughtSignature]; ok {
		t.Errorf("second call carries %v, want it unsigned", sig)
	}
}

// TestThoughtSignaturesReturnToTheirParts runs the whole conversion over a
// conversation the Gemini service wrote: each kept signature goes back on the
// part it was given for, the parallel batch is regrouped, and the signature
// messages themselves are not sent.
func TestThoughtSignaturesReturnToTheirParts(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("Book a table and order a coffee")
	convo.AddMessage(NewThoughtSignature("sig-a", map[string]any{BookmarkFunctionCall: "a"}))
	convo.AddAssistantToolCall(frames.ToolCall{ID: "a", Name: "book_table"})
	convo.AddAssistantToolCall(frames.ToolCall{ID: "b", Name: "order_coffee"})
	convo.AddToolResult(frames.ToolResult{ID: "a", Content: `{"ok":true}`})
	convo.AddToolResult(frames.ToolResult{ID: "b", Content: `{"ok":true}`})
	convo.AddMessage(NewThoughtSignature("sig-text", map[string]any{BookmarkText: "All done. Enjoy"}))
	convo.AddAssistantMessage("All done.  Enjoy your coffee.")

	got := mustContents(t, convo.Messages())
	wantStrings(t, "roles", roles(got), []string{roleUser, roleModel, roleUser, roleModel})
	calls := contentParts(got[1])
	wantStrings(t, "calls", partIDs(got[1]), []string{"a", "b"})
	if calls[0][keyThoughtSignature] != "sig-a" {
		t.Errorf("first call = %v, want the signature kept for it", calls[0])
	}
	if sig, ok := calls[1][keyThoughtSignature]; ok {
		t.Errorf("second call carries %v, want it unsigned", sig)
	}
	wantStrings(t, "responses", partIDs(got[2]), []string{"a", "b"})
	if sig := contentParts(got[3])[0][keyThoughtSignature]; sig != "sig-text" {
		t.Errorf("reply signature = %v, want the one bookmarked to its text", sig)
	}
}

// A signature applies to the last part of the first model turn, at or after the
// previous match, that its bookmark identifies.
func TestApplyThoughtSignaturesInOrder(t *testing.T) {
	contents := []map[string]any{
		textContent(roleModel, "Hello"),
		toolCallContent("c1", ""),
		textContent(roleModel, "Hello"),
	}
	applyThoughtSignatures([]map[string]any{
		{keySignature: "s1", keyBookmark: map[string]any{BookmarkText: "Hello"}},
		{keySignature: "s2", keyBookmark: map[string]any{BookmarkFunctionCall: "c1"}},
		{keySignature: "s3", keyBookmark: map[string]any{BookmarkText: "Hello"}},
		{keySignature: "", keyBookmark: map[string]any{BookmarkText: "Hello"}},
	}, contents)
	for i, want := range []string{"s1", "s2", "s3"} {
		if got := contentParts(contents[i])[0][keyThoughtSignature]; got != want {
			t.Errorf("content %d signature = %v, want %q", i, got, want)
		}
	}
}

// A text bookmark matches through whitespace differences and a prefix either
// way, for a part cut short by an interruption and for a signature that arrived
// with the first chunk.
func TestTextBookmarkMatches(t *testing.T) {
	part := map[string]any{adapterKeyText: "We have  tea\nand coffee."}
	for _, bookmark := range []string{"We have tea and coffee.", "We have tea", "We have tea and coffee. More"} {
		if !textBookmarkMatches(bookmark, part) {
			t.Errorf("bookmark %q did not match %v", bookmark, part)
		}
	}
	if textBookmarkMatches("Something else", part) {
		t.Error("an unrelated bookmark matched")
	}
}

// An inline-data bookmark matches a part whose data has the same length.
func TestInlineDataBookmarkMatches(t *testing.T) {
	part := map[string]any{keyInlineData: map[string]any{keyData: []byte("abcd")}}
	if !inlineDataBookmarkMatches(map[string]any{keyData: "YWJjZA=="}, part) {
		t.Error("data of the same length did not match")
	}
	if inlineDataBookmarkMatches(map[string]any{keyData: "YWI="}, part) {
		t.Error("data of another length matched")
	}
}

// nativeCallContent builds a native model turn with an unsigned call, which the
// conversion gives the placeholder.
func nativeCallContent() frames.Message {
	return adapter.CreateLLMSpecificMessage(&Adapter{}, map[string]any{
		adapterKeyRole: roleModel,
		adapterKeyParts: []map[string]any{{
			keyFunctionCall: map[string]any{adapterKeyName: "f", "args": map[string]any{}},
		}},
	})
}

// TestConversionDoesNotMutateNativeContents checks a native content kept in the
// conversation comes out of a conversion exactly as it went in, though the
// conversion puts a signature on its parts.
func TestConversionDoesNotMutateNativeContents(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	convo.AddMessage(nativeCallContent())
	convo.AddUserMessage("next")
	before, err := json.Marshal(convo.Messages()[1].Native)
	if err != nil {
		t.Fatal(err)
	}

	got := mustContents(t, convo.Messages())
	if sig := contentParts(got[1])[0][keyThoughtSignature]; sig != wantPlaceholder {
		t.Errorf("sent call signature = %v, want the placeholder", sig)
	}
	after, err := json.Marshal(convo.Messages()[1].Native)
	if err != nil {
		t.Fatal(err)
	}
	if string(before) != string(after) {
		t.Errorf("native content changed by the conversion:\nbefore %s\nafter  %s", before, after)
	}
}

// TestConcurrentConversionsOfNativeContents runs several conversions of one
// conversation at once, as a one-shot inference beside a generation does. Run
// under -race it fails if a conversion writes into a content the conversation
// owns.
func TestConcurrentConversionsOfNativeContents(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	convo.AddMessage(nativeCallContent())
	convo.AddUserMessage("next")

	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			p, err := (&Adapter{}).LLMInvocationParams(convo, adapter.Options{})
			if err != nil {
				t.Error(err)
				return
			}
			if len(p.Contents) != 3 {
				t.Errorf("contents = %v, want three", p.Contents)
			}
		})
	}
	wg.Wait()
}

// TestLoggingElidesThoughtSignatures checks the logged form hides the opaque
// signatures rather than printing them.
func TestLoggingElidesThoughtSignatures(t *testing.T) {
	convo := frames.NewLLMContext("")
	convo.AddUserMessage("hi")
	convo.AddMessage(NewThoughtSignature("secret", map[string]any{BookmarkText: "hello"}))
	convo.AddAssistantMessage("hello")

	logged := (&Adapter{}).MessagesForLogging(convo)
	if sig := contentParts(logged[1])[0][keyThoughtSignature]; sig != "..." {
		t.Errorf("logged signature = %v, want it elided", sig)
	}
}
