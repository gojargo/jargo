package bus

import "errors"

// Serialization of bus messages for a bus that leaves the process.
//
// A message crossing a network bus has to survive being written as bytes and
// read back as the same thing. Go loses that on its own: a field typed as an
// interface, a Frame or an arbitrary payload, marshals as whatever it happened
// to hold and comes back as a bare map, with nothing left to say what it was.
//
// The wire form therefore tags such a value with the name of its type, and the
// name is resolved back to a value through a TypeRegistry. That registry is
// this package's own machinery rather than a port: Python reads a type out of
// its own module path at run time, and Go has no equivalent, so the types that
// may cross a bus name themselves in advance.

// MessageSerializer converts bus messages to bytes and back. A bus that leaves
// the process holds one and uses it at each edge.
type MessageSerializer interface {
	// Serialize converts a message to the bytes to send.
	Serialize(m Message) ([]byte, error)
	// Deserialize reconstructs the message those bytes carried.
	Deserialize(data []byte) (Message, error)
}

// SerializeFunc converts one value to its wire form. A TypeAdapter is handed
// one so it can serialize what its own fields hold without knowing how.
type SerializeFunc func(v any) (any, error)

// DeserializeFunc reconstructs one value from its wire form, and is the
// counterpart handed to a TypeAdapter.
type DeserializeFunc func(v any) (any, error)

// TypeAdapter converts values of one type to and from a plain map, for a type
// the serializer cannot take apart itself: one whose state is private, or whose
// wire form is deliberately not its field layout.
//
// Register one with JSONSerializer.RegisterAdapter. Unlike upstream, the
// adapter is not told which type to build: it is registered against exactly one
// type and is the only thing that builds it.
type TypeAdapter interface {
	// Serialize converts obj to a map, using serialize for whatever its own
	// fields hold that it cannot convert itself.
	Serialize(obj any, serialize SerializeFunc) (map[string]any, error)
	// Deserialize rebuilds the value from a map serialize produced.
	Deserialize(data map[string]any, deserialize DeserializeFunc) (any, error)
}

// The failures serialization reports. Each names a shape that arrived where
// another was expected, which on a bus means the two ends disagree about what
// they are carrying.
//
//nolint:gochecknoglobals // sentinel errors
var (
	// ErrWrongAdapter reports an adapter handed a value of another type.
	ErrWrongAdapter = errors.New("bus: the adapter was given the wrong type")
	// ErrUnnamedType reports a type with an adapter but no registered name, so
	// there is nothing to tag it with.
	ErrUnnamedType = errors.New("bus: the type has an adapter but no registered name")
	// ErrWrongShape reports a value that arrived as one JSON shape where
	// another was expected.
	ErrWrongShape = errors.New("bus: the value has the wrong shape")
	// ErrNotJSON reports raw JSON that will not parse.
	ErrNotJSON = errors.New("bus: the raw JSON is not JSON")
)

// The keys a tagged value carries on the wire. They match the shape upstream
// writes, so a bus carrying both is reading the same envelope.
const (
	typeKey = "__type__"
	dataKey = "__data__"
)

// bytesTypeName is the type name raw bytes are tagged with. It is not a
// registered type: every language on a bus has bytes, and they are always
// base64 on the wire.
const bytesTypeName = "bytes"
