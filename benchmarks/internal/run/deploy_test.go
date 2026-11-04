package run

import (
	"context"
	"os"
	"slices"
	"testing"
	"time"
)

// serveEnv makes the test binary a deployment server, so Deployment can be
// driven end to end against processes of its own.
const serveEnv = "JBENCH_TEST_SERVE"

func TestMain(m *testing.M) {
	if os.Getenv(serveEnv) != "" {
		if err := Serve(context.Background(), os.Stdin, os.Stdout); err != nil {
			os.Exit(1) //nolint:forbidigo // a server process reports its failure through its exit code
		}
		return
	}
	m.Run()
}

func TestShare(t *testing.T) {
	var got []int
	for i := range 3 {
		got = append(got, share(10, 3, i))
	}
	if want := []int{4, 3, 3}; !slices.Equal(got, want) {
		t.Errorf("share(10, 3) = %v, want %v", got, want)
	}
}

// TestDeployment drives two serving processes through the whole protocol and
// checks every metric comes back, with the memory readings positive.
func TestDeployment(t *testing.T) {
	if _, err := os.Stat("/proc/self/smaps_rollup"); err != nil {
		t.Skip("no /proc/<pid>/smaps_rollup to read memory from")
	}
	t.Setenv(serveEnv, "1")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	res, err := Deployment(ctx, []string{os.Args[0]}, 2, 2, 5, 1, 3)
	if err != nil {
		t.Fatalf("Deployment: %v", err)
	}
	if res.Iterations != 3 {
		t.Errorf("iterations = %d, want the 3 rounds", res.Iterations)
	}
	got := map[string]float64{}
	for _, m := range res.Metrics {
		got[m.Unit] = m.Value
	}
	for _, unit := range []string{"ns/round", "ns/frame", "frames/s", "startup-ns/session"} {
		if got[unit] <= 0 {
			t.Errorf("%s = %v, want positive", unit, got[unit])
		}
	}
	if got["baseline-B"] <= 0 || got["B/deployment"] <= 0 {
		t.Errorf("baseline-B = %v, B/deployment = %v, want both positive", got["baseline-B"], got["B/deployment"])
	}
}

// TestDeploymentReportsServerFailure checks a server that cannot start is
// reported rather than waited for.
func TestDeploymentReportsServerFailure(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := Deployment(ctx, []string{"/bin/false"}, 1, 1, 1, 0, 1); err == nil {
		t.Fatal("Deployment = nil error, want the failed server reported")
	}
}
