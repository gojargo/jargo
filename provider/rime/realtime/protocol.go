package realtime

import (
	"encoding/base64"
	"encoding/json"
	"math"
	"slices"
	"strings"
	"time"
)

// The kinds of request the client sends, and of the events that answer them.
const (
	kindSessionUpdate = "session.update"
	kindItemCreate    = "conversation.item.create"
	kindHistoryGet    = "prism.conversation.get"
	kindItemDelete    = "conversation.item.delete"
	kindResponseNew   = "response.create"
	kindResponseStop  = "response.cancel"
	kindAudioClear    = "input_audio_buffer.clear"
	kindItemTruncate  = "conversation.item.truncate"
)

// The keys of the wire objects read and written more than once.
const (
	keyType         = "type"
	keyEventID      = "event_id"
	keyItem         = "item"
	keyItemID       = "item_id"
	keyCallID       = "call_id"
	keyResponse     = "response"
	keyResponseID   = "response_id"
	keySession      = "session"
	keyMetadata     = "metadata"
	keyTurn         = "turn_detection"
	keyTools        = "tools"
	keyText         = "text"
	keyRole         = "role"
	keyName         = "name"
	keyArguments    = "arguments"
	keyOutput       = "output"
	keyID           = "id"
	keyDelta        = "delta"
	keyContentIndex = "content_index"
	keyAudioEndMS   = "audio_end_ms"
	keyTranscript   = "transcript"
	keyError        = "error"
	keyRevision     = "prism_settings_revision"
	keyLabels       = "prism_response_metadata"
	keyCause        = "prism_cause"
	keyInterruptMin = "interrupt_min_words"
	keyInterruptPhr = "interrupt_phrases"
	keyAckPhrases   = "acknowledgment_phrases"
	keyMinSilence   = "end_of_turn_min_silence_ms"
	keyEndTimeout   = "prism_end_of_turn_timeout_ms"
	keyEagerness    = "eagerness"
	prismKeyPrefix  = "prism_"
)

// maxSafeInteger is the largest integer a JSON number carries exactly.
const maxSafeInteger = 9007199254740991

// outputSampleRate is the rate of the audio Prism sends.
const outputSampleRate = 24000

// ackKind tells the acknowledgments a request can be answered with apart.
type ackKind int

const (
	ackSessionUpdated ackKind = iota
	ackHistoryListed
	ackItemDeleted
	ackItemCreated
	ackToolResultCreated
	ackAudioCleared
	ackItemTruncated
	ackResponseStarted
	ackResponseEnded
)

// String names the acknowledgment as the protocol does.
func (k ackKind) String() string {
	switch k {
	case ackSessionUpdated:
		return "session.updated"
	case ackHistoryListed:
		return "history.listed"
	case ackItemDeleted:
		return "item.deleted"
	case ackItemCreated:
		return "item.created"
	case ackToolResultCreated:
		return "tool.result.created"
	case ackAudioCleared:
		return "audio.cleared"
	case ackItemTruncated:
		return "item.truncated"
	case ackResponseStarted:
		return "response.started"
	case ackResponseEnded:
		return "response.ended"
	default:
		return "unknown"
	}
}

// sessionView is a session as the server describes it.
type sessionView struct {
	speed                   *float64
	version                 *string
	review                  *ReviewSettings
	truncation              *Truncation
	settingsRevision        *int
	id                      string
	requestID               string
	responseMetadata        map[string]string
	voice                   string
	interruptOnSpeech       *bool
	toolResultTimeout       time.Duration
	toolContinuationTimeout time.Duration
	turnDetection           TurnDetectionInfo
}

// The payloads of the events that are the client's own business: the opening of
// the session and the acknowledgments of its requests.
type (
	sessionCreated struct{ session sessionView }
	sessionUpdated struct {
		session sessionView
		info    *SessionInfo
	}
	itemCreated       struct{ itemID string }
	toolResultCreated struct{ itemID, callID string }
	historyListed     struct{ items []historyItem }
	itemDeleted       struct{ itemID string }
	audioCleared      struct{}
	sessionIdle       struct{ idleTimeout time.Duration }
	// responseEnded is the end of a reply as the wire carries it, with the token
	// a playback receipt names it by.
	responseEnded struct {
		ResponseEnded

		drainToken string
	}
	// reviewRequest is speech proposed for review. The client configures no
	// review, so one arriving is a protocol failure.
	reviewRequest struct{}
)

// ackKindOf is the acknowledgment a payload is, and whether it is one.
func ackKindOf(payload any) (ackKind, bool) {
	switch payload.(type) {
	case sessionUpdated:
		return ackSessionUpdated, true
	case historyListed:
		return ackHistoryListed, true
	case itemDeleted:
		return ackItemDeleted, true
	case itemCreated:
		return ackItemCreated, true
	case toolResultCreated:
		return ackToolResultCreated, true
	case audioCleared:
		return ackAudioCleared, true
	case ItemTruncated:
		return ackItemTruncated, true
	case ResponseStarted:
		return ackResponseStarted, true
	case responseEnded:
		return ackResponseEnded, true
	default:
		return 0, false
	}
}

// serverEvent is one decoded event.
type serverEvent struct {
	serverTimeMS *int
	eventID      string
	requestID    string
	payload      any
}

// request is a kind of request, which carries and checks the kind of its
// acknowledgment.
type request struct {
	kind   string
	result ackKind
	// allowsItemCreated lets an item acknowledgment come before the result, as
	// it does for a typed turn.
	allowsItemCreated bool
	// history marks a history write, whose acknowledgment has to name the item
	// written.
	history bool
}

// check reports whether ack is the acknowledgment the request expects.
func (r *request) check(ack any) error {
	kind, ok := ackKindOf(ack)
	if !ok || kind != r.result {
		return streamError("Unexpected " + kind.String() + " acknowledgment for " + r.kind)
	}
	return nil
}

// The requests the client makes.
//
//nolint:gochecknoglobals // the fixed table of request kinds
var (
	reqUpdate        = &request{kind: kindSessionUpdate, result: ackSessionUpdated}
	reqGetHistory    = &request{kind: kindHistoryGet, result: ackHistoryListed}
	reqDeleteItem    = &request{kind: kindItemDelete, result: ackItemDeleted}
	reqToolResult    = &request{kind: kindItemCreate, result: ackToolResultCreated}
	reqHistoryItem   = &request{kind: kindItemCreate, result: ackItemCreated, history: true}
	reqHistoryResult = &request{kind: kindItemCreate, result: ackToolResultCreated, history: true}
	reqCreate        = &request{kind: kindResponseNew, result: ackResponseStarted}
	reqText          = &request{kind: kindResponseNew, result: ackResponseStarted, allowsItemCreated: true}
	reqCancel        = &request{kind: kindResponseStop, result: ackResponseEnded}
	reqClear         = &request{kind: kindAudioClear, result: ackAudioCleared}
	reqTruncate      = &request{kind: kindItemTruncate, result: ackItemTruncated}
)

// responseMetadata is a copy of the labels given for every reply.
func responseMetadata(labels map[string]string) map[string]any {
	out := make(map[string]any, len(labels))
	for key, label := range labels {
		out[key] = label
	}
	return out
}

// requestMetadata is one reply's labels with the protocol fields the client
// owns added.
func requestMetadata(labels map[string]string, reserved map[string]string) (map[string]any, error) {
	out := responseMetadata(labels)
	for key := range labels {
		if strings.HasPrefix(key, prismKeyPrefix) {
			return nil, inputError("request metadata keys cannot start with prism_")
		}
	}
	for key, value := range reserved {
		out[key] = value
	}
	return out, nil
}

// truncationSettings is the wire form of a truncation policy.
func truncationSettings(value Truncation) (any, error) {
	if err := value.validate(); err != nil {
		return nil, inputError("truncation must be auto, disabled, or RetentionRatio")
	}
	if value.Type == truncationRetention {
		return map[string]any{keyType: truncationRetention, truncationRetention: value.RetentionRatio}, nil
	}
	return value.Type, nil
}

// reader reads the fields of a server event, keeping the first one that does
// not follow the protocol.
type reader struct {
	err error
	// sessionID is the session the event belongs to; nil before the session is
	// created.
	sessionID *string
}

func (r *reader) fail(name string) {
	if r.err == nil {
		r.err = invalidField(name)
	}
}

func (r *reader) object(value any, name string) map[string]any {
	object, ok := value.(map[string]any)
	if !ok {
		r.fail(name)
		return map[string]any{}
	}
	return object
}

// optObject is the object under key, or an empty one when there is none.
func (r *reader) optObject(parent map[string]any, key, name string) map[string]any {
	if parent[key] == nil {
		return map[string]any{}
	}
	return r.object(parent[key], name)
}

func (r *reader) str(value any, name string) string {
	s, ok := value.(string)
	if !ok {
		r.fail(name)
	}
	return s
}

func (r *reader) id(value any, name string) string {
	s := r.str(value, name)
	if s == "" {
		r.fail(name)
	}
	return s
}

func (r *reader) optString(value any, name string) string {
	if value == nil {
		return ""
	}
	return r.str(value, name)
}

func (r *reader) optID(value any, name string) string {
	if value == nil {
		return ""
	}
	return r.id(value, name)
}

func (r *reader) integer(value any, name string) int {
	number, ok := value.(float64)
	if !ok || number < 0 || number > maxSafeInteger || math.Trunc(number) != number {
		r.fail(name)
		return 0
	}
	return int(number)
}

// boundedInteger is the integer under key within the bounds, or nil when the
// event has none.
func (r *reader) boundedInteger(view map[string]any, key string, minimum, maximum int) *int {
	raw, ok := view[key]
	if !ok {
		return nil
	}
	value := r.integer(raw, key)
	if value < minimum || value > maximum {
		r.fail(key)
	}
	return &value
}

func (r *reader) revision(view map[string]any, name string) *int {
	raw, ok := view[keyRevision]
	if !ok {
		return nil
	}
	value := r.integer(raw, name)
	return &value
}

// duration is a positive number of seconds, or zero when there is none.
func (r *reader) duration(value any, name string) time.Duration {
	if value == nil {
		return 0
	}
	seconds, ok := value.(float64)
	if !ok || math.IsInf(seconds, 0) || math.IsNaN(seconds) || seconds <= 0 {
		r.fail(name)
		return 0
	}
	return time.Duration(seconds * float64(time.Second))
}

func (r *reader) phrases(view map[string]any, key string) []string {
	raw, ok := view[key]
	if !ok {
		return nil
	}
	list, ok := raw.([]any)
	if !ok || len(list) > 100 {
		r.fail(key)
		return nil
	}
	out := make([]string, 0, len(list))
	for _, item := range list {
		out = append(out, r.str(item, key))
	}
	return out
}

// ref names the reply value identifies, in the event's session.
func (r *reader) ref(value any) ResponseRef {
	if r.sessionID == nil {
		if r.err == nil {
			r.err = streamError("Event arrived before session.created")
		}
		return ResponseRef{}
	}
	return ResponseRef{SessionID: *r.sessionID, ResponseID: r.id(value, "response.id")}
}

func (r *reader) truncation(value any) *Truncation {
	if s, ok := value.(string); ok && (s == truncationAuto || s == truncationDisabled) {
		return &Truncation{Type: s}
	}
	if object, ok := value.(map[string]any); ok && object[keyType] == truncationRetention {
		if ratio, ok := object[truncationRetention].(float64); ok {
			truncation := TruncationRetentionRatio(ratio)
			if truncation.validate() == nil {
				return &truncation
			}
		}
	}
	r.fail("session.truncation")
	return nil
}

func (r *reader) review(value any) *ReviewSettings {
	view := r.object(value, "prism_review")
	action, _ := view["on_timeout"].(string)
	settings := &ReviewSettings{OnTimeout: action}
	switch action {
	case "stop", "speak":
	case "say":
		settings.TimeoutLine = r.str(view["timeout_line"], "timeout_line")
		line := settings.TimeoutLine
		if (Filler{Type: fillerText, Text: &line}).validate() != nil {
			r.fail("prism_review")
		}
	default:
		r.fail("prism_review.on_timeout")
	}
	if _, ok := view["timeout_line"]; ok && action != "say" {
		r.fail("prism_review.timeout_line")
	}
	settings.TimeoutMS = r.integer(view["timeout_ms"], "timeout_ms")
	if settings.TimeoutMS < 100 || settings.TimeoutMS > 5000 {
		r.fail("prism_review")
	}
	return settings
}

// clientMetadata is the client's labels in metadata: its string values under
// keys that are not Prism's.
func clientMetadata(metadata map[string]any) map[string]string {
	out := map[string]string{}
	for key, value := range metadata {
		if label, ok := value.(string); ok && !strings.HasPrefix(key, prismKeyPrefix) {
			out[key] = label
		}
	}
	return out
}

func (r *reader) turnDetection(detection map[string]any) TurnDetectionInfo {
	info := TurnDetectionInfo{}
	if raw, ok := detection[keyEagerness]; ok {
		info.Eagerness = r.str(raw, "session.turn_detection.eagerness")
		if !slices.Contains([]string{"low", "medium", "high", "auto"}, info.Eagerness) {
			info.Eagerness = scopeUnknown
		}
	}
	if raw, ok := detection[keyEndTimeout]; ok {
		name := "session.turn_detection." + keyEndTimeout
		timeout := r.integer(raw, name)
		if timeout < 1000 || timeout > 10000 {
			r.fail(name)
		}
		info.EndOfTurnTimeoutMS = &timeout
	}
	info.EndOfTurnMinSilenceMS = r.boundedInteger(detection, keyMinSilence, 0, 5000)
	info.InterruptMinWords = r.boundedInteger(detection, keyInterruptMin, 1, 10)
	info.InterruptPhrases = r.phrases(detection, keyInterruptPhr)
	info.AcknowledgmentPhrases = r.phrases(detection, keyAckPhrases)
	return info
}

func (r *reader) speed(view map[string]any) *float64 {
	raw, ok := view["speed"]
	if !ok {
		return nil
	}
	speed, ok := raw.(float64)
	if !ok || speed < 0.85 || speed > 1.3 {
		r.fail("session.speed")
		return nil
	}
	return &speed
}

func (r *reader) interrupt(view map[string]any) *bool {
	const key = "prism_effective_interrupt_response"
	raw, present := view[key]
	if interrupt, ok := raw.(bool); ok {
		return &interrupt
	}
	if present {
		r.fail("session." + key)
	}
	return nil
}

func (r *reader) session(value any) sessionView {
	view := r.object(value, keySession)
	detection := map[string]any{}
	if raw, ok := view[keyTurn]; ok {
		detection = r.object(raw, "session.turn_detection")
	}
	waits := map[string]any{}
	if raw, ok := view["prism_tool_waits"]; ok {
		waits = r.object(raw, "session.prism_tool_waits")
		_, result := waits["result_timeout_s"]
		_, continuation := waits["continuation_timeout_s"]
		if !result || !continuation {
			r.fail("session.prism_tool_waits")
		}
	}
	out := sessionView{
		speed:             r.speed(view),
		settingsRevision:  r.revision(view, "session."+keyRevision),
		id:                r.id(view[keyID], "session.id"),
		requestID:         r.optString(view["request_id"], "session.request_id"),
		interruptOnSpeech: r.interrupt(view),
		turnDetection:     r.turnDetection(detection),
		toolResultTimeout: r.duration(waits["result_timeout_s"], "result_timeout_s"),
		toolContinuationTimeout: r.duration(
			waits["continuation_timeout_s"], "continuation_timeout_s"),
	}
	if raw, ok := view["version"]; ok {
		version := r.str(raw, "session.version")
		out.version = &version
	}
	if raw, ok := view["prism_review"]; ok {
		out.review = r.review(raw)
	}
	if raw, ok := view["truncation"]; ok {
		out.truncation = r.truncation(raw)
	}
	if raw, ok := view[keyLabels]; ok {
		out.responseMetadata = clientMetadata(r.object(raw, "session."+keyLabels))
	}
	if raw, ok := view["voice"]; ok {
		out.voice = r.str(raw, "session.voice")
	}
	return out
}

func (r *reader) fault(value any) Fault {
	failure := r.object(value, keyError)
	owner := r.optObject(failure, "owner", "error.owner")
	scope := scopeUnknown
	if raw, ok := failure["scope"]; ok {
		scope = r.str(raw, "error.scope")
	}
	fault := Fault{
		Code:          scopeUnknown,
		Scope:         scopeUnknown,
		RequestID:     r.optString(owner[keyEventID], "error.owner.event_id"),
		ResponseID:    r.optString(owner[keyResponseID], "error.owner.response_id"),
		ItemID:        r.optString(owner[keyItemID], "error.owner.item_id"),
		CorrelationID: r.optString(failure["correlation_id"], "error.correlation_id"),
		Parameter:     r.optString(failure["param"], "error.param"),
	}
	if raw, ok := failure["code"]; ok {
		fault.Code = r.str(raw, "error.code")
	}
	if raw, ok := failure["message"]; ok {
		fault.Message = r.str(raw, "error.message")
	}
	if slices.Contains([]string{scopeEvent, "utterance", "response", "tool_roundtrip", "review", scopeSession}, scope) {
		fault.Scope = scope
	}
	if scope == "review" {
		fault.ReviewID = r.id(owner["review_id"], "error.owner.review_id")
	} else {
		fault.ReviewID = r.optID(owner["review_id"], "error.owner.review_id")
	}
	if calls := owner["call_ids"]; calls != nil {
		list, ok := calls.([]any)
		if !ok {
			r.fail("error.owner.call_ids")
		}
		for _, call := range list {
			fault.CallIDs = append(fault.CallIDs, r.id(call, "error.owner.call_ids"))
		}
	}
	return fault
}

// arguments reads a tool call's arguments, which arrive as the JSON text of an
// object.
func (r *reader) arguments(value any) (map[string]any, string) {
	text := r.str(value, keyArguments)
	if r.err != nil {
		return nil, ""
	}
	var parsed any
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		r.err = streamError("Invalid realtime event")
		return nil, ""
	}
	object, ok := parsed.(map[string]any)
	if !ok {
		r.fail(keyArguments)
		return nil, ""
	}
	return object, text
}

// audio reads a chunk of the audio Prism generated: base64 of whole 16-bit
// samples.
func (r *reader) audio(value any) []byte {
	encoded := r.str(value, keyDelta)
	data, err := base64.StdEncoding.Strict().DecodeString(encoded)
	if err != nil || len(data)%2 != 0 {
		r.fail("audio.delta")
		return nil
	}
	return data
}

// listedItem reads one item of the history the server lists.
func listedItem(value any) (historyItem, error) {
	object, ok := value.(map[string]any)
	id, _ := object[keyID].(string)
	if !ok || id == "" {
		return historyItem{}, streamError("Invalid history item id")
	}
	var failure error
	field := func(name string) string {
		v, ok := object[name].(string)
		switch {
		case failure != nil:
		case !ok:
			failure = streamError("Invalid history item " + name)
		case v == "" && (name == keyCallID || name == keyName):
			failure = streamError("Invalid history " + name)
		}
		return v
	}
	item := historyItem{id: id}
	switch object[keyType] {
	case itemMessage:
		item.kind, item.role = itemMessage, field(keyRole)
		if failure == nil && !slices.Contains([]string{"user", "assistant", "system"}, item.role) {
			failure = streamError("Invalid history role")
		}
		item.text = field(keyText)
	case itemFunctionCall:
		item.kind = itemFunctionCall
		item.callID, item.name, item.arguments = field(keyCallID), field(keyName), field(keyArguments)
	case itemFunctionOut:
		item.kind = itemFunctionOut
		item.callID, item.output = field(keyCallID), field(keyOutput)
	default:
		return historyItem{}, streamError("Invalid history item type")
	}
	return item, failure
}

// decode reads one message from the server. It returns nil for an event the
// client has no use for.
func decode(binary bool, raw []byte, sessionID *string) (*serverEvent, error) {
	if binary {
		return nil, streamError("Expected a JSON text event")
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		return nil, streamError("Invalid realtime event")
	}
	r := &reader{sessionID: sessionID}
	event := r.object(value, "event")
	kind := r.str(event[keyType], keyType)
	if r.err != nil {
		return nil, r.err
	}
	decoder, ok := decoders[kind]
	if !ok {
		return nil, nil //nolint:nilnil // an event the client has no use for
	}
	payload := decoder(r, event)
	out := &serverEvent{
		serverTimeMS: r.boundedInteger(event, "server_time_ms", 0, maxSafeInteger),
		requestID:    r.optString(event["prism_request_event_id"], "prism_request_event_id"),
		payload:      payload,
	}
	if raw, ok := event[keyEventID]; ok {
		out.eventID = r.str(raw, keyEventID)
	}
	if r.err != nil {
		return nil, r.err
	}
	if payload == nil {
		return nil, nil //nolint:nilnil // an output item that is not a message
	}
	return out, nil
}

// decoder reads the payload of one kind of event.
type decoder func(r *reader, event map[string]any) any

// decoders are the events the client reads, by their wire kind.
//
//nolint:gochecknoglobals // the fixed table of event kinds
var decoders = map[string]decoder{
	"prism.review.requested":                                decodeReview,
	"session.created":                                       decodeSessionCreated,
	"session.updated":                                       decodeSessionUpdated,
	"conversation.item.created":                             decodeItemCreated,
	"prism.conversation":                                    decodeHistory,
	"conversation.item.deleted":                             decodeItemDeleted,
	"input_audio_buffer.cleared":                            func(*reader, map[string]any) any { return audioCleared{} },
	"session.idle_timeout":                                  decodeIdleTimeout,
	"conversation.item.truncated":                           decodeItemTruncated,
	"prism.typed_input.ready":                               func(*reader, map[string]any) any { return InputReady{} },
	"prism.input_audio_buffer.speech_hint":                  decodeSpeechHint,
	"input_audio_buffer.speech_started":                     decodeSpeechStarted,
	"input_audio_buffer.speech_stopped":                     decodeSpeechStopped,
	"input_audio_buffer.committed":                          decodeInputCommitted,
	"conversation.item.input_audio_transcription.delta":     decodeTranscriptDelta,
	"conversation.item.input_audio_transcription.completed": decodeTranscriptFinal,
	"conversation.item.input_audio_transcription.failed":    decodeTranscriptFailed,
	"response.created":                                      decodeResponseCreated,
	"response.done":                                         decodeResponseDone,
	"response.function_call_arguments.done":                 decodeToolCall,
	"response.output_item.added":                            decodeMessageStarted,
	"response.text.delta":                                   decodeTextDelta,
	"response.text.done":                                    decodeTextDone,
	"response.audio.delta":                                  decodeAudioDelta,
	"response.audio.done":                                   decodeAudioDone,
	keyError:                                                decodeFault,
}

func decodeReview(r *reader, event map[string]any) any {
	r.id(event["review_id"], "review_id")
	r.id(event[keyResponseID], keyResponseID)
	r.id(event[keyItemID], keyItemID)
	r.integer(event["index"], "index")
	r.str(event[keyText], keyText)
	return reviewRequest{}
}

func decodeSessionCreated(r *reader, event map[string]any) any {
	return sessionCreated{session: r.session(event[keySession])}
}

func decodeSessionUpdated(r *reader, event map[string]any) any {
	return sessionUpdated{session: r.session(event[keySession])}
}

func decodeItemCreated(r *reader, event map[string]any) any {
	item := r.object(event[keyItem], keyItem)
	if item[keyType] == itemFunctionOut {
		return toolResultCreated{
			callID: r.id(item[keyCallID], "item.call_id"),
			itemID: r.optString(item[keyID], "item.id"),
		}
	}
	if kind, ok := item[keyType]; ok && kind != itemMessage && kind != itemFunctionCall {
		r.fail("item.type")
	}
	return itemCreated{itemID: r.id(item[keyID], "item.id")}
}

func decodeHistory(r *reader, event map[string]any) any {
	list, ok := event["items"].([]any)
	if !ok {
		r.fail("items")
		return nil
	}
	items := make([]historyItem, 0, len(list))
	for _, raw := range list {
		item, err := listedItem(raw)
		if err != nil {
			if r.err == nil {
				r.err = err
			}
			return nil
		}
		items = append(items, item)
	}
	return historyListed{items: items}
}

func decodeItemDeleted(r *reader, event map[string]any) any {
	return itemDeleted{itemID: r.id(event[keyItemID], keyItemID)}
}

func decodeIdleTimeout(r *reader, event map[string]any) any {
	if event["reason"] != "idle_timeout" {
		r.fail("reason")
	}
	seconds, ok := event["idle_timeout_s"].(float64)
	if !ok || math.IsInf(seconds, 0) || math.IsNaN(seconds) || seconds < 1 {
		r.fail("idle_timeout_s")
		return nil
	}
	return sessionIdle{idleTimeout: time.Duration(seconds * float64(time.Second))}
}

func decodeItemTruncated(r *reader, event map[string]any) any {
	out := ItemTruncated{
		ItemID:       r.id(event[keyItemID], keyItemID),
		ContentIndex: r.integer(event[keyContentIndex], keyContentIndex),
		AudioEndMS:   r.integer(event[keyAudioEndMS], keyAudioEndMS),
	}
	if raw, ok := event["prism_kept_text"]; ok {
		kept := r.str(raw, "prism_kept_text")
		out.KeptText = &kept
	}
	return out
}

func decodeSpeechHint(r *reader, event map[string]any) any {
	return SpeechHint{AudioStartMS: r.integer(event["audio_start_ms"], "audio_start_ms")}
}

func decodeSpeechStarted(r *reader, event map[string]any) any {
	return SpeechStarted{
		ItemID:       r.id(event[keyItemID], keyItemID),
		AudioStartMS: r.integer(event["audio_start_ms"], "audio_start_ms"),
	}
}

func decodeSpeechStopped(r *reader, event map[string]any) any {
	out := SpeechStopped{
		ItemID:     r.id(event[keyItemID], keyItemID),
		AudioEndMS: r.integer(event[keyAudioEndMS], keyAudioEndMS),
	}
	if raw, ok := event["prism_end_reason"]; ok {
		out.EndReason = r.str(raw, "prism_end_reason")
		known := []string{"semantic", "timeout", "silence", "max_duration", "detector_changed"}
		if !slices.Contains(known, out.EndReason) {
			out.EndReason = scopeUnknown
		}
	}
	return out
}

func decodeInputCommitted(r *reader, event map[string]any) any {
	return InputCommitted{ItemID: r.id(event[keyItemID], keyItemID)}
}

func decodeTranscriptDelta(r *reader, event map[string]any) any {
	out := TranscriptDelta{
		ItemID: r.id(event[keyItemID], keyItemID),
		Delta:  r.str(event[keyDelta], keyDelta),
	}
	if raw, ok := event[keyTranscript]; ok {
		transcript := r.str(raw, keyTranscript)
		out.Transcript = &transcript
	}
	return out
}

func decodeTranscriptFinal(r *reader, event map[string]any) any {
	return TranscriptFinal{
		ItemID: r.id(event[keyItemID], keyItemID),
		Text:   r.str(event[keyTranscript], keyTranscript),
	}
}

func decodeTranscriptFailed(r *reader, event map[string]any) any {
	failure := r.object(event[keyError], keyError)
	itemID := r.id(event[keyItemID], keyItemID)
	return TranscriptFailed{ItemID: itemID, Error: Fault{
		Code:    r.str(failure["code"], "error.code"),
		Message: r.str(failure["message"], "error.message"),
		Scope:   "utterance",
		ItemID:  itemID,
	}}
}

// cause is what started a reply, from the cause the server names.
func cause(named *string) string {
	if named == nil {
		return causeSpeech
	}
	switch *named {
	case "user_text":
		return causeText
	case "tool_continuation":
		return causeTools
	case causeProactive:
		return causeProactive
	default:
		return scopeUnknown
	}
}

func decodeResponseCreated(r *reader, event map[string]any) any {
	response := r.object(event[keyResponse], keyResponse)
	metadata := r.optObject(response, keyMetadata, keyMetadata)
	var named *string
	if raw, ok := metadata[keyCause]; ok {
		value := r.str(raw, keyCause)
		named = &value
	}
	out := ResponseStarted{
		SettingsRevision: r.revision(response, "response."+keyRevision),
		Response:         r.ref(response[keyID]),
		Cause:            cause(named),
		InputItemID:      r.optString(metadata["prism_input_item_id"], "prism_input_item_id"),
		InstructionItemID: r.optID(
			metadata["prism_instruction_item_id"], "prism_instruction_item_id"),
		Metadata: clientMetadata(metadata),
	}
	if parent := r.optString(metadata["prism_parent_response_id"], "prism_parent_response_id"); parent != "" ||
		metadata["prism_parent_response_id"] != nil {
		ref := r.ref(parent)
		out.Parent = &ref
	}
	switch ids := response["prism_input_item_ids"].(type) {
	case nil:
		if metadata["prism_input_item_id"] != nil {
			out.InputItemIDs = []string{out.InputItemID}
		}
	case []any:
		for _, id := range ids {
			out.InputItemIDs = append(out.InputItemIDs, r.id(id, "prism_input_item_ids"))
		}
	default:
		r.fail("prism_input_item_ids")
	}
	return out
}

func (r *reader) responseTruncation(response map[string]any) *ResponseTruncation {
	raw, ok := response["prism_truncation"]
	if !ok {
		return nil
	}
	value := r.object(raw, "response.prism_truncation")
	out := &ResponseTruncation{
		ItemsLeftOut: r.integer(value["items_left_out"], "prism_truncation.items_left_out"),
	}
	if out.ItemsLeftOut == 0 {
		r.fail("prism_truncation.items_left_out")
	}
	if kept, ok := value["first_kept_item_id"]; ok {
		out.FirstKeptItemID = r.str(kept, "prism_truncation.first_kept_item_id")
	}
	return out
}

func (r *reader) responseUsage(response map[string]any) *ResponseUsage {
	if response["usage"] == nil {
		return nil
	}
	usage := r.object(response["usage"], "response.usage")
	return &ResponseUsage{
		InputTokens:   r.integer(usage["input_tokens"], "response.usage.input_tokens"),
		OutputTokens:  r.integer(usage["output_tokens"], "response.usage.output_tokens"),
		TotalTokens:   r.integer(usage["total_tokens"], "response.usage.total_tokens"),
		ContextWindow: r.integer(usage["context_window"], "response.usage.context_window"),
	}
}

func decodeResponseDone(r *reader, event map[string]any) any {
	response := r.object(event[keyResponse], keyResponse)
	metadata := r.optObject(response, keyMetadata, keyMetadata)
	details := r.optObject(response, "status_details", "status_details")
	status := r.str(response["status"], "response.status")
	if !slices.Contains([]string{"completed", "cancelled", "failed"}, status) { //nolint:misspell // the wire value
		status = scopeUnknown
	}
	return responseEnded{
		ResponseEnded: ResponseEnded{
			SettingsRevision: r.revision(response, "response."+keyRevision),
			Truncation:       r.responseTruncation(response),
			Usage:            r.responseUsage(response),
			Response:         r.ref(response[keyID]),
			Status:           status,
			Reason:           r.optString(details["reason"], "reason"),
		},
		drainToken: r.optString(metadata["prism_drain_token"], "prism_drain_token"),
	}
}

func decodeToolCall(r *reader, event map[string]any) any {
	ref := r.ref(event[keyResponseID])
	out := ToolCall{
		Call: ToolCallRef{
			SessionID:  ref.SessionID,
			ResponseID: ref.ResponseID,
			CallID:     r.id(event[keyCallID], keyCallID),
		},
		ItemID: r.id(event[keyItemID], keyItemID),
		Name:   r.str(event[keyName], keyName),
	}
	out.Arguments, out.raw = r.arguments(event[keyArguments])
	return out
}

// output names the message an output event is about. The content index has to
// be zero: the client handles one content per message.
func (r *reader) output(event map[string]any, itemID any, contentIndex any) OutputRef {
	index := r.integer(contentIndex, keyContentIndex)
	if r.err == nil && index != 0 {
		r.err = streamError("Unsupported Prism contract: content_index must be zero")
	}
	return OutputRef{
		Response:     r.ref(event[keyResponseID]),
		ItemID:       r.id(itemID, keyItemID),
		OutputIndex:  r.integer(event["output_index"], "output_index"),
		ContentIndex: index,
	}
}

func decodeMessageStarted(r *reader, event map[string]any) any {
	item := r.object(event[keyItem], keyItem)
	if r.str(item[keyType], "item.type") != itemMessage {
		return nil
	}
	index := event[keyContentIndex]
	if _, ok := event[keyContentIndex]; !ok {
		index = float64(0)
	}
	return MessageStarted{Output: r.output(event, item[keyID], index)}
}

func decodeTextDelta(r *reader, event map[string]any) any {
	output := r.output(event, event[keyItemID], event[keyContentIndex])
	return TextDelta{Output: output, Delta: r.str(event[keyDelta], keyDelta)}
}

func decodeTextDone(r *reader, event map[string]any) any {
	output := r.output(event, event[keyItemID], event[keyContentIndex])
	return TextDone{Output: output, Text: r.str(event[keyText], keyText)}
}

func decodeAudioDelta(r *reader, event map[string]any) any {
	output := r.output(event, event[keyItemID], event[keyContentIndex])
	return AudioDelta{
		Output: output, Audio: r.audio(event[keyDelta]), SampleRate: outputSampleRate, Channels: 1,
	}
}

func decodeAudioDone(r *reader, event map[string]any) any {
	return AudioDone{Output: r.output(event, event[keyItemID], event[keyContentIndex])}
}

func decodeFault(r *reader, event map[string]any) any {
	return FaultEvent{Error: r.fault(event[keyError])}
}
