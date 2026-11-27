package deepgram

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/url"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/gojargo/jargo/audio"
	"github.com/gojargo/jargo/frames"
	"github.com/gojargo/jargo/internal/validate"
	"github.com/gojargo/jargo/language"
	"github.com/gojargo/jargo/service/settings"
	"github.com/gojargo/jargo/service/stt"
)

const (
	// defaultSegmentedBaseURL is the pre-recorded API the segmented service
	// addresses when no base URL is configured.
	defaultSegmentedBaseURL = "https://" + defaultHost
	// defaultSegmentedSTTModel is the model the segmented service transcribes
	// with when the settings name none.
	defaultSegmentedSTTModel = "nova-3-general"
	// defaultDiarizeModel is the diarization model Diarize turns on.
	defaultDiarizeModel = "latest"
)

// SegmentedSTTSettings is the settings of the segmented service (see
// NewSegmentedSTT).
//
// Each setting is sent as the Deepgram query parameter of the same name, except
// Diarize (see DiarizeModel) and the local SpeakerFormat, and a setting with no
// value is left out. Any other Deepgram pre-recorded query parameter can be
// passed through Extra.
type SegmentedSTTSettings struct {
	BaseSTTSettings

	// DetectLanguage detects the spoken language and transcribes in it,
	// overriding Language. The detected language is reported on the transcript.
	DetectLanguage settings.Opt[bool] `settings:"detect_language"`
	// DiarizeModel is the diarization model Diarize turns on: "latest" or a
	// version such as "v2". Diarization is requested with this parameter
	// alone, since Deepgram's diarize parameter selects its oldest model and is
	// refused alongside diarize_model.
	DiarizeModel settings.Opt[string] `settings:"diarize_model"`
	// FillerWords includes filler words such as "uh" and "um".
	FillerWords settings.Opt[bool] `settings:"filler_words"`
}

// SegmentedSTTConfig configures the Deepgram segmented speech-to-text service,
// which transcribes each speech segment with one request to Deepgram's
// pre-recorded API.
type SegmentedSTTConfig struct {
	// APIKey is the Deepgram API key. Required.
	APIKey string `validate:"required"`
	// BaseURL is the base URL of the Deepgram API, scheme included. Override it
	// for a regional endpoint (e.g. "https://api.eu.deepgram.com") or a
	// self-hosted deployment. Empty uses "https://api.deepgram.com".
	BaseURL string
	// SampleRate is the input audio sample rate; 0 uses the pipeline's rate.
	SampleRate int
	// Tag attaches custom billing tags to each request.
	Tag []string
	// MipOptOut opts out of Deepgram's Model Improvement Program; nil leaves it
	// unset.
	MipOptOut *bool
	// Settings overrides the default settings, field by field: a field it does
	// not give keeps its default, and one given no value (settings.Cleared)
	// leaves the parameter out. Nil keeps every default: model
	// "nova-3-general", language "en", punctuation on, and every other boolean
	// setting off.
	Settings *SegmentedSTTSettings
	// HTTPClient makes the requests; nil uses one with the standard library's
	// defaults. Supply one to set a timeout, a proxy, or connection limits of
	// your own.
	HTTPClient *http.Client

	// TTFSP99 overrides the measured transcript latency the turn strategies
	// size their wait by; nil uses stt.DefaultTTFSP99, and zero means no wait.
	TTFSP99 *time.Duration
}

// Validate reports whether the configuration is usable.
func (c SegmentedSTTConfig) Validate() error { return validate.Struct(c) }

// NewSegmentedSTT builds a Deepgram speech-to-text service on the pre-recorded
// API (POST /v1/listen) rather than a streaming WebSocket. It is segmented: the
// VAD upstream delimits each utterance, and each one is transcribed with one
// request carrying the audio as a WAV file. Transcripts are final only; there
// are no interim results.
//
// Settings are read for every request, so an update applies from the next
// segment.
func NewSegmentedSTT(cfg SegmentedSTTConfig) *stt.SegmentService {
	t := newSegmentedTranscriber(cfg)
	// The transcriber builds its frames with the service's help: the user they
	// are attributed to, the processing metrics, the speaker runs.
	t.svc = stt.NewSegmentRunner("DeepgramSegmentedSTT", t, cfg.SampleRate)
	return t.svc
}

// defaultSegmentedSettings is the settings a service starts from before the
// configured ones are applied.
func defaultSegmentedSettings() *SegmentedSTTSettings {
	s := &SegmentedSTTSettings{}
	s.Model = settings.Set(defaultSegmentedSTTModel)
	s.Language = settings.Set(deepgramLanguage(language.English))
	s.DetectEntities = settings.Set(false)
	s.DetectLanguage = settings.Set(false)
	s.Diarize = settings.Set(false)
	s.DiarizeModel = settings.Set(defaultDiarizeModel)
	s.Dictation = settings.Set(false)
	s.FillerWords = settings.Set(false)
	s.Keyterm = settings.Cleared[[]string]()
	s.Keywords = settings.Cleared[[]string]()
	s.LanguageHint = settings.Cleared[[]string]()
	s.Numerals = settings.Set(false)
	s.ProfanityFilter = settings.Cleared[bool]()
	s.Punctuate = settings.Set(true)
	s.Redact = settings.Cleared[[]string]()
	s.Replace = settings.Cleared[[]string]()
	s.Search = settings.Cleared[[]string]()
	s.SmartFormat = settings.Set(false)
	s.Version = settings.Cleared[string]()
	s.SpeakerFormat = settings.Cleared[string]()
	return s
}

// segmentedTranscriber transcribes one segment per request to Deepgram's
// pre-recorded API.
type segmentedTranscriber struct {
	// svc is the service the transcriber runs in.
	svc     *stt.SegmentService
	cfg     SegmentedSTTConfig
	baseURL string
	http    *http.Client

	// store is the settings the service updates. The service writes it, under
	// its own lock, and then calls UpdateSettings, which is the only place it
	// is read once the service is built.
	store *SegmentedSTTSettings

	mu sync.Mutex
	// live is a copy of store taken whenever it changed, which the requests
	// are built from: they run on the service's segment task, while store is
	// written on the frame goroutine.
	live *SegmentedSTTSettings
}

// newSegmentedTranscriber builds the transcriber with its defaults applied.
func newSegmentedTranscriber(cfg SegmentedSTTConfig) *segmentedTranscriber {
	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = defaultSegmentedBaseURL
	}
	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{}
	}

	store := defaultSegmentedSettings()
	if cfg.Settings != nil {
		// The delta is synced first: an unset field of a delta is not given,
		// so a matching key of its Extra can still be promoted to it. On the
		// merged store every field already holds a default and the value
		// would be dropped.
		if delta, err := settings.Copy(cfg.Settings); err == nil {
			d, _ := delta.(*SegmentedSTTSettings)
			syncExtraToFields(d, d.Extra)
			if _, err := settings.Apply(store, d); err != nil {
				slog.Warn("deepgram: the configured settings could not be applied", "err", err)
			}
		}
	}
	syncExtraToFields(store, store.Extra)

	t := &segmentedTranscriber{
		cfg: cfg, baseURL: strings.TrimRight(baseURL, "/"), http: client, store: store,
	}
	t.snapshot()
	return t
}

// snapshot copies the store into live, which the requests are built from.
func (t *segmentedTranscriber) snapshot() {
	c, err := settings.Copy(t.store)
	if err != nil {
		return
	}
	live, _ := c.(*SegmentedSTTSettings)
	t.mu.Lock()
	t.live = live
	t.mu.Unlock()
}

// current is the settings as they stand.
func (t *segmentedTranscriber) current() *SegmentedSTTSettings {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.live
}

// Metadata reports the model in use and the transcript latency the turn
// strategies size their wait by.
func (t *segmentedTranscriber) Metadata() stt.Metadata {
	return stt.Metadata{
		TTFSP99: stt.TTFSP99Or(t.cfg.TTFSP99, stt.DeepgramSegmentedTTFSP99),
		Model:   t.current().Model.Or(""),
	}
}

// Settings is the configuration a caller may change while the pipeline runs,
// implementing stt.SettingsHolder.
func (t *segmentedTranscriber) Settings() any { return t.store }

// PrepareSettings promotes the keys of an update's Extra that name a setting to
// that setting, implementing stt.SettingsPreparer.
func (t *segmentedTranscriber) PrepareSettings(delta any) error {
	if d, ok := delta.(*SegmentedSTTSettings); ok {
		syncExtraToFields(d, d.Extra)
	}
	return nil
}

// UpdateSettings keeps the store unambiguous after an update and takes the
// settings the next request is built from, implementing stt.SettingsUpdater.
// There is no session to reopen: the next segment is transcribed with the
// settings as they now stand.
func (t *segmentedTranscriber) UpdateSettings(context.Context, settings.Changed) (bool, error) {
	syncExtraToFields(t.store, t.store.Extra)
	t.snapshot()
	return false, nil
}

// ServiceLanguage names a language the way Deepgram does, implementing
// stt.LanguageNamer.
func (t *segmentedTranscriber) ServiceLanguage(l language.Language) string {
	return deepgramLanguage(l)
}

// queryParam is one query parameter of a request, in the order it is sent.
type queryParam struct {
	name, value string
}

// queryParams builds the request's query parameters from the settings s.
func (t *segmentedTranscriber) queryParams(s *SegmentedSTTSettings) []queryParam {
	var names []string
	values := map[string]any{}
	set := func(name string, value any) {
		if _, ok := values[name]; !ok {
			names = append(names, name)
		}
		values[name] = value
	}
	pop := func(name string) any {
		value := values[name]
		delete(values, name)
		names = slices.DeleteFunc(names, func(n string) bool { return n == name })
		return value
	}

	forEachSetting(s, func(name string, o option) {
		if slices.Contains(localSTTFields, name) {
			return
		}
		set(name, o.AsAny())
	})
	diarize := pop("diarize")
	diarizeModel := pop("diarize_model")
	if on, _ := diarize.(bool); on {
		set("diarize_model", diarizeModel)
	}
	var tag any
	if t.cfg.Tag != nil {
		tag = t.cfg.Tag
	}
	set("tag", tag)
	var mipOptOut any
	if t.cfg.MipOptOut != nil {
		mipOptOut = *t.cfg.MipOptOut
	}
	set("mip_opt_out", mipOptOut)
	// Extra is a map, so its keys are taken in a stable order.
	for _, key := range slices.Sorted(maps.Keys(s.Extra)) {
		set(key, s.Extra[key])
	}

	var params []queryParam
	for _, name := range names {
		if values[name] == nil {
			continue
		}
		for _, encoded := range queryValues(values[name]) {
			params = append(params, queryParam{name: name, value: encoded})
		}
	}
	return params
}

// encodeQuery encodes params as a query string, keeping their order.
func encodeQuery(params []queryParam) string {
	var b strings.Builder
	for i, p := range params {
		if i > 0 {
			b.WriteByte('&')
		}
		b.WriteString(url.QueryEscape(p.name))
		b.WriteByte('=')
		b.WriteString(url.QueryEscape(p.value))
	}
	return b.String()
}

// queryValues encodes a setting as query parameter values. A list becomes one
// value per item, sent as a repeated parameter, and a bool is spelled the way
// Deepgram expects (true/false).
func queryValues(value any) []string {
	switch v := value.(type) {
	case []string:
		return slices.Clone(v)
	case []any:
		out := make([]string, 0, len(v))
		for _, item := range v {
			if item != nil {
				out = append(out, queryValue(item))
			}
		}
		return out
	default:
		return []string{queryValue(v)}
	}
}

// queryValue encodes one value of a setting.
func queryValue(value any) string {
	switch v := value.(type) {
	case bool:
		return strconv.FormatBool(v)
	case string:
		return v
	default:
		return fmt.Sprint(v)
	}
}

// option is the reflected view of a settings field.
type option interface {
	IsGiven() bool
	AsAny() any
}

// forEachSetting calls fn for every declared setting of the settings struct s
// points at, in declaration order, walking into the embedded structs the
// shared settings come from.
func forEachSetting(s any, fn func(name string, o option)) {
	var walk func(v reflect.Value)
	walk = func(v reflect.Value) {
		for i := range v.NumField() {
			field := v.Type().Field(i)
			value := v.Field(i)
			if o, ok := reflect.TypeAssert[option](value); ok {
				if name := field.Tag.Get("settings"); name != "" && name != "-" {
					fn(name, o)
				}
				continue
			}
			if field.Anonymous && field.Type.Kind() == reflect.Struct {
				walk(value)
			}
		}
	}
	walk(reflect.ValueOf(s).Elem())
}

// syncExtraToFields moves the keys of extra, the Extra of the settings s points
// at, that name one of its declared settings onto that setting, so the
// settings hold the value that will be used.
//
// A key naming a setting that is not given sets it. A key naming a setting is
// removed from extra whether or not it set it, so the two never disagree. Model
// and language are not taken from extra.
func syncExtraToFields(s any, extra map[string]any) {
	if len(extra) == 0 {
		return
	}
	given := map[string]bool{}
	forEachSetting(s, func(name string, o option) {
		if name != "model" && name != "language" {
			given[name] = o.IsGiven()
		}
	})
	for key, value := range extra {
		isGiven, declared := given[key]
		if !declared {
			continue
		}
		if !isGiven {
			if err := setSetting(s, key, value); err != nil {
				slog.Warn("deepgram: an extra setting does not fit the setting it names",
					"setting", key, "err", err)
			}
		}
		delete(extra, key)
	}
}

// setSetting gives the named setting value. A single string given for a list
// setting is taken as a list of one, as Deepgram takes either.
func setSetting(s any, name string, value any) error {
	err := settings.SetNamed(s, name, value)
	if err == nil {
		return nil
	}
	if str, ok := value.(string); ok {
		if listErr := settings.SetNamed(s, name, []string{str}); listErr == nil {
			return nil
		}
	}
	return err
}

// statusError is a request the pre-recorded API answered with a status other
// than 200. Its message is Deepgram's own description of the error, and it
// carries the status so the service can classify the failure: a rejected key
// or setting leaves the service unusable, a rate limit or a server error does
// not.
type statusError struct {
	status  int
	message string
	// requestURL is the URL the request was sent to. It is kept off the
	// message, since its query carries settings such as keyterms.
	requestURL string
}

// Error implements error.
func (e *statusError) Error() string { return fmt.Sprintf("%d %s", e.status, e.message) }

// HTTPStatusCode is the status Deepgram answered with.
func (e *statusError) HTTPStatusCode() int { return e.status }

// requestError is a request that failed before Deepgram answered it. Its
// message leaves out the request URL, whose query carries settings such as
// keyterms; the failure it wraps keeps it.
type requestError struct {
	err *url.Error
}

// Error implements error.
func (e *requestError) Error() string { return e.err.Op + ": " + e.err.Err.Error() }

// Unwrap returns the failure, URL included.
func (e *requestError) Unwrap() error { return e.err }

// withoutRequestURL is err with the request URL kept off its message.
func withoutRequestURL(err error) error {
	if ue, ok := errors.AsType[*url.Error](err); ok {
		return &requestError{err: ue}
	}
	return err
}

// errorDetail builds a readable message from the body of a non-200 response.
// Deepgram reports errors as {"err_code", "err_msg", "request_id"}; any other
// body is returned as it is.
func errorDetail(body []byte) string {
	var e struct {
		ErrCode   string `json:"err_code"`
		ErrMsg    string `json:"err_msg"`
		RequestID string `json:"request_id"`
	}
	if err := json.Unmarshal(body, &e); err == nil && (e.ErrCode != "" || e.ErrMsg != "") {
		var parts []string
		for _, part := range []string{e.ErrCode, e.ErrMsg} {
			if part != "" {
				parts = append(parts, part)
			}
		}
		detail := strings.Join(parts, " - ")
		if e.RequestID != "" {
			detail += " (request_id: " + e.RequestID + ")"
		}
		return detail
	}
	return string(body)
}

// transcribe posts an audio segment to the pre-recorded API and returns the
// response body.
//
// A request that fails on a dropped connection is sent once more. Deepgram
// closes an idle connection after about 5 s, so a request sent on a pooled
// connection as it closes can fail. The HTTP client retries such a request
// itself only when nothing of it was written, or for an idempotent method (RFC
// 9112, section 9.3.1), which excludes POST; repeating this POST only
// transcribes the segment again, so it is retried here the same way. A failure
// to connect at all is not retried.
func (t *segmentedTranscriber) transcribe(ctx context.Context, wav []byte) ([]byte, error) {
	body, err := t.postSegment(ctx, wav)
	if err != nil && droppedConnection(err) {
		slog.Debug("deepgram: connection dropped, retrying the segment", "err", withoutRequestURL(err))
		return t.postSegment(ctx, wav)
	}
	return body, err
}

// postSegment sends one request for an audio segment and returns the response
// body.
func (t *segmentedTranscriber) postSegment(ctx context.Context, wav []byte) ([]byte, error) {
	target := t.baseURL + listenPath + "?" + encodeQuery(t.queryParams(t.current()))
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(wav))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", authToken(t.cfg.APIKey))
	req.Header.Set("Content-Type", "audio/wav")
	resp, err := t.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = resp.Body.Close() }()
	body, err := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, &statusError{status: resp.StatusCode, message: errorDetail(body), requestURL: target}
	}
	if err != nil {
		return nil, err
	}
	return body, nil
}

// droppedConnection reports whether err is a connection dropped under a
// request, as opposed to a failure to connect at all, a timeout or a
// cancellation.
func droppedConnection(err error) bool {
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if opErr, ok := errors.AsType[*net.OpError](err); ok && opErr.Op == "dial" {
		return false
	}
	if netErr, ok := errors.AsType[net.Error](err); ok && netErr.Timeout() {
		return false
	}
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, syscall.EPIPE) ||
		strings.Contains(err.Error(), "server closed idle connection")
}

// listenResponse is the part of a pre-recorded transcription result the
// service reads.
type listenResponse struct {
	Results struct {
		Channels []listenChannel `json:"channels"`
	} `json:"results"`
}

type listenChannel struct {
	DetectedLanguage string              `json:"detected_language"`
	Alternatives     []listenAlternative `json:"alternatives"`
}

type listenAlternative struct {
	Transcript string       `json:"transcript"`
	Languages  []string     `json:"languages"`
	Words      []listenWord `json:"words"`
}

type listenWord struct {
	Word           string `json:"word"`
	PunctuatedWord string `json:"punctuated_word"`
	Speaker        *int   `json:"speaker"`
}

// RunSTT transcribes one segment, 16-bit mono PCM at sampleRate, with one
// request to the pre-recorded API carrying it as a WAV file, implementing
// stt.SegmentRunner. It yields a TranscriptionFrame per transcript (one per
// speaker run when diarization is on), nothing when nothing was heard, and an
// ErrorFrame on failure.
func (t *segmentedTranscriber) RunSTT(ctx context.Context, pcm []byte, sampleRate int) iter.Seq[frames.Frame] {
	return func(yield func(frames.Frame) bool) {
		t.svc.StartProcessingMetrics()
		raw, result, err := t.transcribeResult(ctx, audio.PCMToWAV(pcm, sampleRate, 1))
		if err != nil {
			if ctx.Err() != nil {
				// Canceled with the service: nothing failed.
				return
			}
			yield(t.errorFrame(err))
			return
		}
		t.svc.StopProcessingMetrics(ctx)

		// Audio with no samples comes back with no channels at all.
		channels := result.Results.Channels
		if len(channels) == 0 || len(channels[0].Alternatives) == 0 {
			return
		}
		alternative := channels[0].Alternatives[0]
		text := strings.TrimSpace(alternative.Transcript)
		if text == "" {
			return
		}
		lang := t.resultLanguage(channels[0], alternative)
		slog.Debug("deepgram transcription", "service", t.svc.Name(), "text", text)

		var words []stt.SpeakerWord
		if t.current().Diarize.Or(false) {
			// Words carry punctuation only when punctuate or smart_format is on.
			words = make([]stt.SpeakerWord, 0, len(alternative.Words))
			for _, w := range alternative.Words {
				word := w.PunctuatedWord
				if word == "" {
					word = w.Word
				}
				var speaker any
				if w.Speaker != nil {
					speaker = *w.Speaker
				}
				words = append(words, stt.SpeakerWord{Text: word, Speaker: speaker})
			}
		}
		timestamp := frames.NowTimestamp()
		for _, run := range t.svc.DiarizedRuns(text, words) {
			f := frames.NewTranscriptionFrame(run.Text, t.svc.UserID(), timestamp)
			f.Language = lang
			f.Result = raw
			f.Speaker = run.Speaker
			if !yield(f) {
				return
			}
		}
	}
}

// transcribeResult posts a WAV segment and returns the result both as the raw
// decoded JSON, which the transcripts carry, and as the part the service reads.
func (t *segmentedTranscriber) transcribeResult(
	ctx context.Context, wav []byte,
) (map[string]any, listenResponse, error) {
	var parsed listenResponse
	body, err := t.transcribe(ctx, wav)
	if err != nil {
		return nil, parsed, err
	}
	var raw map[string]any
	if err := json.Unmarshal(body, &raw); err != nil {
		return nil, parsed, fmt.Errorf("decoding the transcription result: %w", err)
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, parsed, fmt.Errorf("decoding the transcription result: %w", err)
	}
	return raw, parsed, nil
}

// errorFrame reports a failure to transcribe a segment. Neither its message nor
// its error quotes the request URL, whose query carries settings such as
// keyterms: both reach the logs, and the message reaches the client. The URL
// stays on the failure the error wraps.
func (t *segmentedTranscriber) errorFrame(err error) *frames.ErrorFrame {
	err = withoutRequestURL(err)
	ef := frames.NewErrorFrame(fmt.Sprintf("%s transcription error: %v", t.svc.Name(), err))
	ef.Err = err
	return ef
}

// resultLanguage is the language of a result: the one Deepgram detected, else
// the language setting.
func (t *segmentedTranscriber) resultLanguage(channel listenChannel, alternative listenAlternative) string {
	if channel.DetectedLanguage != "" {
		return channel.DetectedLanguage
	}
	if len(alternative.Languages) > 0 && alternative.Languages[0] != "" {
		return alternative.Languages[0]
	}
	return t.current().Language.Or("")
}

// Compile-time checks that the transcriber plugs into the segmented service.
var (
	_ stt.SegmentRunner    = (*segmentedTranscriber)(nil)
	_ stt.SettingsUpdater  = (*segmentedTranscriber)(nil)
	_ stt.SettingsPreparer = (*segmentedTranscriber)(nil)
	_ stt.LanguageNamer    = (*segmentedTranscriber)(nil)
	_ stt.Describer        = (*segmentedTranscriber)(nil)
)
