// Package release holds the release evals: the manifest of example bots and
// the scenarios each one runs. This test checks the manifest stays runnable:
// every bot it names is an example package, and every scenario loads.
package release_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gojargo/jargo/eval"
)

func TestManifestNamesExistingBotsAndLoadableScenarios(t *testing.T) {
	m, err := eval.LoadManifest("manifest.yaml", eval.ManifestOverrides{})
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if len(m.Runs) == 0 {
		t.Fatal("the manifest has no runs")
	}
	for _, run := range m.Runs {
		if run.BotPath == "" {
			t.Errorf("%s: no bot to build", run.Label())
			continue
		}
		if _, err := os.Stat(filepath.Join(run.BotPath, "main.go")); err != nil {
			t.Errorf("%s: bot %s is not an example package: %v", run.Label(), run.BotPath, err)
		}
		if _, err := run.Load(); err != nil {
			t.Errorf("%s: scenario %s: %v", run.Label(), run.Scenario, err)
		}
	}
}
