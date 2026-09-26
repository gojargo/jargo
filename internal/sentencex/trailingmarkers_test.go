package sentencex

import "testing"

func pmMarker(uppercaseBreaks bool) *markerDef {
	return &markerDef{
		matcher: suffixMatcher{suffix: "p.m", ignoreCase: true},
		policy:  markerPolicy{digitBreaks: true, uppercaseBreaks: uppercaseBreaks},
	}
}

const pmPrefix = "the sun sets at 7 "

func TestUppercaseBreaksFalseKeepsSuppression(t *testing.T) {
	m := markerMatch{prefix: pmPrefix, def: pmMarker(false)}
	if markerBypassesSuppression(m, "Tom wakes up", false, &language{}) {
		t.Error("a cont marker must keep suppressing before an uppercase non-starter")
	}
}

func TestStarterOverrideWinsOverUppercaseBreaksFalse(t *testing.T) {
	m := markerMatch{prefix: pmPrefix, def: pmMarker(false)}
	if !markerBypassesSuppression(m, "Tom wakes up", true, &language{}) {
		t.Error("a sentence starter must break even after a cont marker")
	}
}

func TestUppercaseBreaksTrueBreaksOnUppercaseFollower(t *testing.T) {
	m := markerMatch{prefix: pmPrefix, def: pmMarker(true)}
	if !markerBypassesSuppression(m, "Tom wakes up", false, &language{}) {
		t.Error("a break marker must break before an uppercase follower")
	}
}
