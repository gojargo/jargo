package frames

import "time"

// iso8601Millis is ISO 8601 with millisecond precision, which is the form the
// timestamps carried on frames take. Seconds alone are too coarse to tell two
// events of the same turn apart.
const iso8601Millis = "2006-01-02T15:04:05.000Z07:00"

// NowTimestamp is the current UTC time, formatted the way every timestamp
// carried on a frame is: ISO 8601, to the millisecond.
//
// It exists so the transcript timestamps, the assistant turn timestamps and
// anything else stamped onto a frame all read the same way, whichever processor
// produced them.
func NowTimestamp() string {
	return time.Now().UTC().Format(iso8601Millis)
}
