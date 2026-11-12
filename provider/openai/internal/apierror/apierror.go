// Package apierror reads the error an OpenAI API refuses a request with: the
// status, and the error code the body names. The code is what tells a content
// filter's refusal from a malformed request, which share the status 400.
package apierror

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	errs "github.com/gojargo/jargo/utils/errors"
)

// The bounds on how much of a refused response is read: enough of the body to
// find the error code in, and the part of it the error message carries.
const (
	bodyLimit    = 8192
	messageLimit = 512
)

// codeContentFilter is the code a content filter refuses a request with: Azure
// OpenAI's, or an OpenAI-compatible moderation proxy's.
const codeContentFilter = "content_filter"

// Error is a request the API refused: the status it refused with, and the error
// code the body names, if any.
type Error struct {
	*errs.HTTPStatusError
	// Code is the body's error code, empty when it named none.
	Code string
}

// Unwrap returns the status error, so the status and the wrapped failure are
// found through it.
func (e *Error) Unwrap() error { return e.HTTPStatusError }

// FromResponse reads a refused response into an Error. The message is the
// failure wrapped as base, the status, and the start of the body.
func FromResponse(resp *http.Response, base error) *Error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, bodyLimit))
	msg := body
	if len(msg) > messageLimit {
		msg = msg[:messageLimit]
	}
	var parsed struct {
		Error struct {
			Code any `json:"code"`
		} `json:"error"`
	}
	_ = json.Unmarshal(body, &parsed)
	code, _ := parsed.Error.Code.(string)
	return &Error{
		HTTPStatusError: errs.NewHTTPStatusError(resp.StatusCode,
			fmt.Errorf("%w %d: %s", base, resp.StatusCode, msg)),
		Code: code,
	}
}

// Classify returns the category of a refusal whose status alone would mislead:
// a content filter rejects a request with a 400 whose code is content_filter,
// which is content the provider refused, not the permanent invalid request the
// status maps to. Anything else returns the zero category, leaving the status
// to decide.
func Classify(err error) errs.Category {
	if e, ok := errors.AsType[*Error](err); ok &&
		e.Status == http.StatusBadRequest && e.Code == codeContentFilter {
		return errs.ContentFiltered
	}
	return errs.Unset
}
