package realtime

import (
	"encoding/json"
	"slices"
	"strings"
	"unicode/utf8"
)

// historyRoles are the roles of the messages a history holds.
//
//nolint:gochecknoglobals // a fixed list
var historyRoles = []string{"user", "assistant", "system"}

// policyLimit bounds the interruption policy changes kept until a reply
// selects its settings revision.
const policyLimit = 128

// speedSetting validates a speech speed.
func speedSetting(value float64) (float64, error) {
	if !(value >= 0.85 && value <= 1.3) {
		return 0, inputError("Speed must be a number from 0.85 to 1.3")
	}
	return value, nil
}

// interruptionSettings is the wire form of the interruption rules given. An
// empty list clears the server's setting.
func interruptionSettings(value TurnDetectionUpdate) (map[string]any, error) {
	if err := value.validate(); err != nil {
		return nil, err
	}
	result := map[string]any{}
	if value.InterruptMinWords != nil {
		result[keyInterruptMin] = *value.InterruptMinWords
	}
	if value.InterruptPhrases != nil {
		result[keyInterruptPhr] = value.InterruptPhrases
	}
	if value.AcknowledgmentPhrases != nil {
		result[keyAckPhrases] = value.AcknowledgmentPhrases
	}
	return result, nil
}

// turnSettings is the wire form of the turn settings a conversation opens
// with.
func turnSettings(value TurnDetection) (map[string]any, error) {
	if err := value.validate(); err != nil {
		return nil, err
	}
	result, err := interruptionSettings(value.TurnDetectionUpdate)
	if err != nil {
		return nil, err
	}
	if value.Eagerness != "" {
		result[keyEagerness] = value.Eagerness
	}
	if value.EndOfTurnTimeoutMS != nil {
		result[keyEndTimeout] = *value.EndOfTurnTimeoutMS
	}
	if value.EndOfTurnMinSilenceMS != nil {
		result[keyMinSilence] = *value.EndOfTurnMinSilenceMS
	}
	return result, nil
}

// fillerSettings is the wire form of a filler.
func fillerSettings(filler Filler) (map[string]any, error) {
	if err := filler.validate(); err != nil {
		return nil, err
	}
	result := map[string]any{keyType: filler.Type}
	if filler.Text != nil {
		result[keyText] = *filler.Text
	}
	if filler.Instructions != nil {
		result["instructions"] = *filler.Instructions
	}
	return result, nil
}

// toolSettings is the wire form of the tools a session advertises.
func toolSettings(tools []toolDefinition) ([]any, error) {
	result := make([]any, 0, len(tools))
	names := map[string]bool{}
	for _, tool := range tools {
		if strings.TrimSpace(tool.name) == "" || names[tool.name] {
			return nil, inputError("Tool names must be nonblank and unique")
		}
		names[tool.name] = true
		if tool.parameters == nil {
			return nil, inputError("Tool parameters must be an object and description must be a string")
		}
		value := map[string]any{
			keyType: "function",
			"function": map[string]any{
				keyName:       tool.name,
				"description": tool.description,
				"parameters":  tool.parameters,
			},
		}
		if tool.filler != nil {
			filler, err := fillerSettings(*tool.filler)
			if err != nil {
				return nil, err
			}
			value["prism_filler"] = filler
		}
		result = append(result, value)
	}
	if _, err := json.Marshal(result); err != nil {
		return nil, inputError("Tool schemas must contain finite JSON values")
	}
	return result, nil
}

// itemSettings is the wire form of a history item.
func itemSettings(item historyItem) (map[string]any, error) {
	value := map[string]any{keyType: item.kind}
	if item.id != "" {
		if utf8.RuneCountInString(item.id) > 32 {
			return nil, inputError("item.id must contain 1 to 32 characters")
		}
		value[keyID] = item.id
	}
	switch item.kind {
	case itemMessage:
		if !slices.Contains(historyRoles, item.role) {
			return nil, inputError("history role must be user, assistant, or system")
		}
		value[keyRole], value[keyText] = item.role, item.text
	case itemFunctionCall:
		if item.callID == "" || item.name == "" {
			return nil, inputError("past calls require a call_id and name")
		}
		value[keyCallID], value[keyName], value[keyArguments] = item.callID, item.name, item.arguments
	case itemFunctionOut:
		if item.callID == "" {
			return nil, inputError("tool results require a call_id")
		}
		value[keyCallID], value[keyOutput] = item.callID, item.output
	default:
		return nil, inputError("item must be a history message, call, or result")
	}
	for _, field := range value {
		if text, ok := field.(string); ok && !utf8.ValidString(text) {
			return nil, inputError("history item fields must be valid UTF-8")
		}
	}
	return value, nil
}

// policyChange is the interruption policy from one settings revision on. A
// nil policy is a change whose revision is not known.
type policyChange struct {
	firstRevision int
	interrupt     *bool
}

// interruptionPolicies keeps the changes to whether speech interrupts a reply
// until a reply selects its settings revision, and with it the policy it runs
// under.
type interruptionPolicies struct {
	changes      []policyChange
	current      bool
	lastRevision *int
}

func newInterruptionPolicies() *interruptionPolicies {
	return &interruptionPolicies{current: true}
}

func (p *interruptionPolicies) add(change policyChange) {
	p.changes = append(p.changes, change)
	if len(p.changes) > policyLimit {
		p.changes = p.changes[1:]
	}
}

// record takes the policy an acknowledged settings update holds.
func (p *interruptionPolicies) record(revision *int, interrupt bool) {
	switch {
	case len(p.changes) == 0:
		first := 0
		if revision != nil {
			first = *revision
		}
		p.add(policyChange{firstRevision: first, interrupt: &interrupt})
	case revision == nil:
		if interrupt != p.current && p.changes[len(p.changes)-1].interrupt != nil {
			// The change happened after the last known acknowledgment, but its
			// exact revision cannot be assigned to a later reply.
			firstUnknown := 0
			if p.lastRevision != nil {
				firstUnknown = *p.lastRevision + 1
			}
			p.add(policyChange{firstRevision: firstUnknown})
		}
	case p.changes[len(p.changes)-1].interrupt == nil || interrupt != p.current:
		p.add(policyChange{firstRevision: *revision, interrupt: &interrupt})
	}
	if revision != nil {
		last := *revision
		p.lastRevision = &last
	}
	p.current = interrupt
}

// resolve is the policy a reply at revision runs under.
func (p *interruptionPolicies) resolve(revision *int) (bool, error) {
	if revision == nil || len(p.changes) == 0 {
		return p.current, nil
	}
	for _, change := range slices.Backward(p.changes) {
		if change.firstRevision <= *revision {
			if change.interrupt == nil {
				return false, resourceLimitError("Reply interruption policy revision is unknown")
			}
			return *change.interrupt, nil
		}
	}
	return false, resourceLimitError("Reply settings revision is no longer retained")
}
