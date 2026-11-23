// Package zoomai provides Zoom AI Services text-to-speech: an HTTP service
// that sends each sentence in its own request and streams the audio back.
//
// See https://developers.zoom.us/docs/ai-services/tts/ and https://www.zoom.ai/.
package zoomai

import "github.com/gojargo/jargo/internal/zoomaiutil"

// headerTrackingID carries the ID Zoom support uses to find a request or
// session.
const headerTrackingID = zoomaiutil.HeaderTrackingID
