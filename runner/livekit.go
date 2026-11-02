package runner

import (
	"errors"
	"log/slog"
	"os"
	"time"

	"github.com/livekit/protocol/auth"
)

// ErrLiveKitNotConfigured is returned when the LiveKit server or its
// credentials are not set in the environment.
var ErrLiveKitNotConfigured = errors.New(
	"runner: LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET must be set in environment variables")

// tokenValidity is how long a minted LiveKit token stays valid.
const tokenValidity = 6 * time.Hour

func mintToken(roomName, participantName, apiKey, apiSecret string, agent bool) (string, error) {
	grant := &auth.VideoGrant{RoomJoin: true, Room: roomName, Agent: agent}
	return auth.NewAccessToken(apiKey, apiSecret).
		SetIdentity(participantName).
		SetName(participantName).
		SetVideoGrant(grant).
		SetValidFor(tokenValidity).
		ToJWT()
}

// GenerateToken mints a LiveKit access token for a participant to join
// roomName.
func GenerateToken(roomName, participantName, apiKey, apiSecret string) (string, error) {
	return mintToken(roomName, participantName, apiKey, apiSecret, false)
}

// GenerateTokenWithAgent mints a LiveKit access token for an agent
// participant, so LiveKit clients know an agent has joined.
func GenerateTokenWithAgent(roomName, participantName, apiKey, apiSecret string) (string, error) {
	return mintToken(roomName, participantName, apiKey, apiSecret, true)
}

// LiveKitCredentials returns the LiveKit server URL, API key and API secret
// from LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET.
func LiveKitCredentials() (url, apiKey, apiSecret string, err error) {
	url = os.Getenv("LIVEKIT_URL")
	apiKey = os.Getenv("LIVEKIT_API_KEY")
	apiSecret = os.Getenv("LIVEKIT_API_SECRET")
	if url == "" || apiKey == "" || apiSecret == "" {
		return "", "", "", ErrLiveKitNotConfigured
	}
	return url, apiKey, apiSecret, nil
}

// GenerateSessionTokens mints distinct tokens for the bot and the caller of
// one session. The identities carry the start of the session id, so sessions
// sharing a room do not collide: LiveKit evicts the earlier connection when
// another joins with the same identity.
func GenerateSessionTokens(roomName, sessionID, apiKey, apiSecret string) (agentToken, userToken string, err error) {
	suffix := sessionID
	if len(suffix) > 8 {
		suffix = suffix[:8]
	}
	agentToken, err = GenerateTokenWithAgent(roomName, "jargo Agent-"+suffix, apiKey, apiSecret)
	if err != nil {
		return "", "", err
	}
	userToken, err = GenerateToken(roomName, "User-"+suffix, apiKey, apiSecret)
	if err != nil {
		return "", "", err
	}
	return agentToken, userToken, nil
}

// ErrLiveKitRoomNotSet is returned by ConfigureLiveKit when no room is named.
var ErrLiveKitRoomNotSet = errors.New(
	"runner: no LiveKit room specified: set LIVEKIT_ROOM_NAME in your environment")

// ConfigureLiveKit reads the LiveKit server, credentials and room from the
// environment (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
// LIVEKIT_ROOM_NAME) and mints the bot's token for the room. It also mints a
// token for a user and logs it, for testing.
func ConfigureLiveKit() (url, token, roomName string, err error) {
	roomName = os.Getenv("LIVEKIT_ROOM_NAME")
	if roomName == "" {
		return "", "", "", ErrLiveKitRoomNotSet
	}
	url, apiKey, apiSecret, err := LiveKitCredentials()
	if err != nil {
		return "", "", "", err
	}
	token, err = GenerateTokenWithAgent(roomName, "jargo Agent", apiKey, apiSecret)
	if err != nil {
		return "", "", "", err
	}
	userToken, err := GenerateToken(roomName, "User", apiKey, apiSecret)
	if err != nil {
		return "", "", "", err
	}
	slog.Info("runner: LiveKit user token", "token", userToken)
	return url, token, roomName, nil
}
