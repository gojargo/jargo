package run

import (
	"bufio"
	"bytes"
	"os"
	"runtime/debug"
	"strconv"
)

// releaseMemory collects garbage and hands the freed memory back to the
// system, so that a resident-memory reading taken next starts from what is
// actually in use. The Python side does the same through gc.collect and
// malloc_trim.
func releaseMemory() {
	debug.FreeOSMemory()
}

// residentBytes reads the process's resident set size from /proc. It reports
// false where there is no /proc to read it from.
func residentBytes() (int64, bool) {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0, false
	}
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		rest, ok := bytes.CutPrefix(sc.Bytes(), []byte("VmRSS:"))
		if !ok {
			continue
		}
		// The value is in kibibytes: "VmRSS:     12345 kB".
		fields := bytes.Fields(rest)
		if len(fields) == 0 {
			return 0, false
		}
		kib, err := strconv.ParseInt(string(fields[0]), 10, 64)
		if err != nil {
			return 0, false
		}
		return kib * 1024, true
	}
	return 0, false
}
