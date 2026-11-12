package security_test

import (
	"strings"
	"testing"

	"github.com/gojargo/jargo/utils/security"
)

// redactCase is a message and what RedactSecrets should make of it.
type redactCase struct {
	in, want string
}

func assertRedacts(t *testing.T, cases []redactCase) {
	t.Helper()
	for _, c := range cases {
		if got := security.RedactSecrets(c.in); got != c.want {
			t.Errorf("RedactSecrets(%q)\n got  %q\n want %q", c.in, got, c.want)
		}
	}
}

func assertKeeps(t *testing.T, messages ...string) {
	t.Helper()
	for _, m := range messages {
		if got := security.RedactSecrets(m); got != m {
			t.Errorf("RedactSecrets(%q) = %q, want it as written", m, got)
		}
	}
}

func TestRedactSecretsMasksACredentialInAPrintedHeaderMap(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"Sarvam API error: headers: {'api-subscription-key': 'sk_live_abc123', " + // gitleaks:allow (a test fixture)
			"'User-Agent': 'sarvamai/0.1.28'}, status_code: 429",
		"Sarvam API error: headers: {'api-subscription-key': '[REDACTED]', " +
			"'User-Agent': 'sarvamai/0.1.28'}, status_code: 429",
	}})
}

func TestRedactSecretsMasksAPresignedURLsCredentialsAndKeepsTheRest(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"Timed out fetching file from URL: 'https://bucket.s3.amazonaws.com/doc.pdf" +
			"?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAEXAMPLE%2F20261014" +
			"&X-Amz-Signature=deadbeef&X-Amz-Security-Token=FQoGZXIvYXdz'",
		"Timed out fetching file from URL: 'https://bucket.s3.amazonaws.com/doc.pdf" +
			"?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=[REDACTED]" +
			"&X-Amz-Signature=[REDACTED]&X-Amz-Security-Token=[REDACTED]'",
	}})
}

func TestRedactSecretsMasksAKeyInAQueryString(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"wss://api.example.com/ws?api_key=SECRET&version=v1 isn't a valid URI",
		"wss://api.example.com/ws?api_key=[REDACTED]&version=v1 isn't a valid URI",
	}})
}

func TestRedactSecretsKeepsAnAuthSchemeAndMasksItsCredential(t *testing.T) {
	assertRedacts(t, []redactCase{{"Authorization: Bearer abc123", "Authorization: Bearer [REDACTED]"}})
}

func TestRedactSecretsMasksACamelCaseCredentialName(t *testing.T) {
	assertRedacts(t, []redactCase{{"apiKey=foo, accessToken: bar", "apiKey=[REDACTED], accessToken: [REDACTED]"}})
}

func TestRedactSecretsMasksAURLsPassword(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"Could not connect to redis://default:hunter2@redis:6379/0",
		"Could not connect to redis://default:[REDACTED]@redis:6379/0",
	}})
}

func TestRedactSecretsMasksTokensRecognizableByFormat(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"rejected sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAA for AKIAIOSFODNN7EXAMPLE " +
			"joining https://acme.daily.co/room?t=eyJhbGciOiJIUzI1NiJ9.eyJyIjoicm9vbSJ9." +
			"c2lnbmF0dXJl",
		"rejected [REDACTED] for [REDACTED] joining https://acme.daily.co/room?t=[REDACTED]",
	}})
}

func TestRedactSecretsMasksEachProviderTokenFormat(t *testing.T) {
	for _, token := range []string{
		"sk-proj-" + strings.Repeat("a", 74) + "T3BlbkFJ" + strings.Repeat("b", 74), // OpenAI
		"ABIA" + strings.Repeat("A", 16),                                            // AWS
		"ABSK" + strings.Repeat("a", 120) + "==",                                    // Amazon Bedrock, long-lived
		"bedrock-api-key-YmVkcm9jay5hbWF6b25hd3MuY29t" + strings.Repeat("a", 40),    // short-lived (gitleaks:allow, a test fixture)
		"AIza" + strings.Repeat("a", 35),                                            // Google
		"github_pat_" + strings.Repeat("a", 82),                                     // GitHub
		"xapp-1-A0123456789-0123456789012-" + strings.Repeat("a", 64),               // Slack
		"hf_" + strings.Repeat("a", 34),                                             // Hugging Face
		"pplx-" + strings.Repeat("a", 48),                                           // Perplexity
		"SK" + strings.Repeat("0", 32),                                              // Twilio
		"eyJhbGciOiJub25lIn0.eyJzdWIiOiIxMjM0NSJ9.",                                 // unsigned JWT (gitleaks:allow, a test fixture)
	} {
		in := "failed with " + token + " here"
		if got := security.RedactSecrets(in); got != "failed with [REDACTED] here" {
			t.Errorf("RedactSecrets(%q) = %q, want the token masked", token[:20], got)
		}
	}
}

func TestRedactSecretsMasksASlackWebhooksSecretPath(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"posting to https://hooks.slack.com/services/" + strings.Repeat("a", 44),
		"posting to https://hooks.slack.com/services/[REDACTED]",
	}})
}

func TestRedactSecretsMasksAMultilinePrivateKey(t *testing.T) {
	pem := "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0B\n-----END PRIVATE KEY-----"
	assertRedacts(t, []redactCase{{
		`{"private_key": "` + pem + `", "client_email": "svc@x"}`,
		`{"private_key": "[REDACTED]", "client_email": "svc@x"}`,
	}})
}

func TestRedactSecretsMasksAValueAfterAnyAssignmentSeparator(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"apiKey := abc123, 'secret' => 'def456', creds=ghi789",
		"apiKey := [REDACTED], 'secret' => '[REDACTED]', creds=[REDACTED]",
	}})
}

func TestRedactSecretsMasksAnAWSSignatureAuthorizationHeader(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"Authorization: AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/20261014/" +
			"us-east-1/s3/aws4_request, SignedHeaders=host, Signature=fe5f80f77d5fa3be",
		"Authorization: AWS4-HMAC-SHA256 [REDACTED], SignedHeaders=host, Signature=[REDACTED]",
	}})
}

func TestRedactSecretsMasksAQuotedValueCutShortByTruncation(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"PGMQueue(host='db', password='hunter2...",
		"PGMQueue(host='db', password='[REDACTED]",
	}})
}

func TestRedactSecretsMasksAResumableUploadSessionID(t *testing.T) {
	assertRedacts(t, []redactCase{{
		"https://example.googleapis.com/upload/files?upload_id=ABC123",
		"https://example.googleapis.com/upload/files?upload_id=[REDACTED]",
	}})
}

func TestRedactSecretsLeavesAMessageWithoutCredentialsAsWritten(t *testing.T) {
	assertKeeps(t,
		"wss://api.elevenlabs.io/v1/speech-to-text?model_id=scribe&keyterms=pipecat",
		"Error code: 400 - {'error': {'message': 'max_tokens: 100 is too low'}}",
		`Cartesia API error (status 400): {"error": "bad voice", "voice_id": "abc"}`,
		"Error: timeout at 12:30",
		"",
	)
}

func TestRedactSecretsTwiceReadsTheSameAsOnce(t *testing.T) {
	once := security.RedactSecrets("Failed to fetch file from URL 'https://x/a?sig=abc&se=2026': 403, " +
		"url='https://x/a?sig=abc&se=2026' Authorization: Bearer abc")
	if twice := security.RedactSecrets(once); twice != once {
		t.Errorf("redacted twice = %q, want %q", twice, once)
	}
}

func TestRedactSecretsALongChainOfPairs(t *testing.T) {
	security.RedactSecrets(strings.Repeat("a:", 5000) + "a")
}

func TestRedactSecretsACredentialAtTheEndOfALongChainIsStillMasked(t *testing.T) {
	if got := security.RedactSecrets(strings.Repeat("a:", 5000) + "token=hunter2"); strings.Contains(got, "hunter2") {
		t.Error("the credential at the end of the chain was left in")
	}
}

func TestRedactSecretsMasksACredentialNestedAFewPairsDeep(t *testing.T) {
	assertRedacts(t, []redactCase{{"a:b:c:https://x/p?token=abc&n=1", "a:b:c:https://x/p?token=[REDACTED]&n=1"}})
}

func TestRedactSecretsKeepsALongChainOfPairsWithoutCredentialsAsWritten(t *testing.T) {
	assertKeeps(t, strings.Repeat("a:", 5000)+"z")
}

func TestRedactSecretsKeepsTheDetailAfterAnExceptionNameHoldingACredentialWord(t *testing.T) {
	assertKeeps(t,
		"KeyError: 'audio_content'",
		"InvalidTokenError: signature has expired",
		"ExpiredSignatureError: Signature has expired",
		"TokenExpiredError: jwt expired",
		"NoCredentialsError: Unable to locate credentials",
		"jwt.exceptions.ExpiredSignatureError: Signature has expired",
		"jwt.exceptions.InvalidTokenError: bad",
		"SomeWarning: a warning",
	)
}

func TestRedactSecretsStillMasksACredentialAfterAnExceptionName(t *testing.T) {
	assertRedacts(t, []redactCase{{"KeyError: 'k' api_key=abc123", "KeyError: 'k' api_key=[REDACTED]"}})
}

func TestRedactSecretsStillMasksWhatOnlyLooksLikeAnExceptionName(t *testing.T) {
	assertRedacts(t, []redactCase{
		{"AuthError: Bearer abc123", "AuthError: Bearer [REDACTED]"},
		{"TokenError: Basic dXNlcjpwYXNz", "TokenError: Basic [REDACTED]"},
		{"credentials_error: hunter2", "credentials_error: [REDACTED]"},
		{"secret_warning: hunter2", "secret_warning: [REDACTED]"},
		{"X-Auth-Error: xyz789", "X-Auth-Error: [REDACTED]"},
		{"{'api_key_error': 'abc'}", "{'api_key_error': '[REDACTED]'}"},
		{"'TokenError': 'abc'", "'TokenError': '[REDACTED]'"},
		{`"TokenError": "abc"`, `"TokenError": "[REDACTED]"`},
		{"TokenError=abc", "TokenError=[REDACTED]"},
	})
}

func TestRedactSecretsMasksACredentialPairThatFollowsAnotherPair(t *testing.T) {
	assertRedacts(t, []redactCase{
		{"ValueError: Authorization: Bearer abc123", "ValueError: Authorization: Bearer [REDACTED]"},
		{"Unknown error occurred: api_key: abc123", "Unknown error occurred: api_key: [REDACTED]"},
		{"error: api_key = abc123", "error: api_key = [REDACTED]"},
		{"X: 'Authorization': 'Bearer abc123'", "X: 'Authorization': '[REDACTED]'"},
		{"KeyError: api_key: abc123", "KeyError: api_key: [REDACTED]"},
		{"api_key: token: abc123", "api_key: token: [REDACTED]"},
		{"error: headers=Authorization: Bearer abc123", "error: headers=Authorization: Bearer [REDACTED]"},
		{"error: c=token: abc123", "error: c=token: [REDACTED]"},
		{"a:b:token: abc123", "a:b:token: [REDACTED]"},
	})
}

func TestRedactSecretsMasksACredentialsValueThatReadsLikeAName(t *testing.T) {
	assertRedacts(t, []redactCase{
		{"token: abc: def", "token: [REDACTED] def"},
		{"password: token:", "password: [REDACTED]"},
	})
}

func TestRedactSecretsMasksACredentialAtTheEndOfALongSpacedChain(t *testing.T) {
	got := security.RedactSecrets(strings.Repeat("a: ", 5001) + "password: hunter2")
	if !strings.HasSuffix(got, "password: [REDACTED]") {
		t.Errorf("RedactSecrets ends %q, want the password masked", got[len(got)-30:])
	}
}

func TestRedactSecretsAPrefixDoesNotChangeWhatAMessageMasks(t *testing.T) {
	messages := []string{
		"api_key: abc123",
		"Authorization: Bearer abc123",
		"'token': 'abc123'",
		"headers=Authorization: Bearer abc123",
		"url=https://x/p?token=abc123&n=1",
		"KeyError: 'audio_content'",
		"InvalidSignature: Signature verification failed",
	}
	for _, prefix := range []string{"Unknown error occurred: ", "Error: ", "Deepgram error: ", "a:b:"} {
		for _, m := range messages {
			if got, want := security.RedactSecrets(prefix+m), prefix+security.RedactSecrets(m); got != want {
				t.Errorf("RedactSecrets(%q) = %q, want %q", prefix+m, got, want)
			}
		}
	}
}

// A webhook's path is told by the URL ahead of it as written, even when that
// URL ends a token masked just before it.
func TestRedactSecretsMasksAWebhookPathAfterATokenRunningIntoItsURL(t *testing.T) {
	path := strings.Repeat("a", 22)
	assertRedacts(t, []redactCase{{
		"hooks.slack.com/services/" + path + "hooks.slack.com/services/" + path,
		"hooks.slack.com/services/[REDACTED].slack.com/services/[REDACTED]",
	}})
}
