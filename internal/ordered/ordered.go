// Package ordered writes JSON objects whose keys keep the order they were given
// in. A map would sort them, and the order of the options, the levels and the
// questions a classifier sends is part of what it asks.
package ordered

import (
	"bytes"
	"encoding/json"
)

// Object is a JSON object whose keys are written in the order they were added.
type Object []Field

// Field is one key and its value.
type Field struct {
	Key   string
	Value any
}

// Set adds a key, or replaces its value when it is already there.
func (o *Object) Set(key string, value any) {
	for i := range *o {
		if (*o)[i].Key == key {
			(*o)[i].Value = value
			return
		}
	}
	*o = append(*o, Field{Key: key, Value: value})
}

// Keys is the keys, in order.
func (o Object) Keys() []string {
	keys := make([]string, 0, len(o))
	for _, f := range o {
		keys = append(keys, f.Key)
	}
	return keys
}

// MarshalJSON implements json.Marshaler, writing the keys in order.
func (o Object) MarshalJSON() ([]byte, error) {
	var buf bytes.Buffer
	buf.WriteByte('{')
	for i, f := range o {
		if i > 0 {
			buf.WriteByte(',')
		}
		key, err := json.Marshal(f.Key)
		if err != nil {
			return nil, err
		}
		buf.Write(key)
		buf.WriteByte(':')
		value, err := Marshal(f.Value)
		if err != nil {
			return nil, err
		}
		buf.Write(value)
	}
	buf.WriteByte('}')
	return buf.Bytes(), nil
}

// Marshal encodes v as JSON without escaping HTML, so text reads as it was
// written.
func Marshal(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

// Indent encodes v as JSON indented by two spaces, without escaping HTML: how
// structured data is written into text a model reads.
func Indent(v any) (string, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return "", err
	}
	return string(bytes.TrimRight(buf.Bytes(), "\n")), nil
}
