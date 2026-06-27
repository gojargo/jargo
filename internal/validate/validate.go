// Package validate provides jargo's shared struct validator. Provider Config
// types expose a Validate method backed by it, and the service bases run that
// validation at Setup so an invalid configuration fails when the pipeline starts
// rather than at first use.
package validate

import "github.com/go-playground/validator/v10"

// v is the shared validator. It is safe for concurrent use and caches per-struct
// reflection, so a single instance is reused across all Config types.
//
//nolint:gochecknoglobals // shared validator instance
var v = validator.New(validator.WithRequiredStructEnabled())

// Struct validates s against its `validate` struct tags.
func Struct(s any) error {
	return v.Struct(s)
}
