// Package query provides jargo's shared URL-query helpers. Provider Config types
// map their optional fields onto url.Values through these small setters, which
// keep the conditional "omit when unset" logic in one place across providers
// rather than copied into each package.
package query

import (
	"net/url"
	"strconv"
)

// SetBoolTrue sets key to v, defaulting to true when v is nil.
func SetBoolTrue(q url.Values, key string, v *bool) {
	val := true
	if v != nil {
		val = *v
	}
	q.Set(key, strconv.FormatBool(val))
}

// SetBoolOpt sets key only when v is non-nil.
func SetBoolOpt(q url.Values, key string, v *bool) {
	if v != nil {
		q.Set(key, strconv.FormatBool(*v))
	}
}

// SetIntOpt sets key only when v is non-nil.
func SetIntOpt(q url.Values, key string, v *int) {
	if v != nil {
		q.Set(key, strconv.Itoa(*v))
	}
}

// SetFloatOpt sets key only when v is non-nil.
func SetFloatOpt(q url.Values, key string, v *float64) {
	if v != nil {
		q.Set(key, strconv.FormatFloat(*v, 'g', -1, 64))
	}
}

// SetStrOpt sets key only when v is non-empty.
func SetStrOpt(q url.Values, key, v string) {
	if v != "" {
		q.Set(key, v)
	}
}

// AddAll appends each value under key, preserving repeated query parameters.
func AddAll(q url.Values, key string, vals []string) {
	for _, v := range vals {
		q.Add(key, v)
	}
}
