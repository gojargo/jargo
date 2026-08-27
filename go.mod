module github.com/gojargo/jargo

go 1.26

toolchain go1.26.6

require (
	github.com/anthropics/anthropic-sdk-go v1.66.0
	github.com/aws/aws-sdk-go-v2 v1.43.7
	github.com/aws/aws-sdk-go-v2/config v1.32.38
	github.com/aws/aws-sdk-go-v2/credentials v1.19.37
	github.com/aws/aws-sdk-go-v2/service/bedrockruntime v1.57.4
	github.com/aws/aws-sdk-go-v2/service/polly v1.60.7
	github.com/aws/aws-sdk-go-v2/service/transcribestreaming v1.38.4
	github.com/coder/websocket v1.8.15
	github.com/ebitengine/purego v0.10.2
	github.com/go-playground/validator/v10 v10.30.3
	github.com/gojargo/go-resample v1.0.1
	github.com/google/uuid v1.6.0
	github.com/jfreymuth/pulse v0.1.3
	github.com/livekit/protocol v1.50.4
	github.com/livekit/server-sdk-go/v2 v2.18.1
	github.com/modelcontextprotocol/go-sdk v1.7.0
	github.com/neurosnap/sentences v1.1.2
	github.com/pion/opus v0.1.1-0.20260706164138-4d863e517a7a
	github.com/pion/webrtc/v4 v4.2.18
	github.com/spf13/cobra v1.10.2
	github.com/spf13/pflag v1.0.10
	github.com/spf13/viper v1.21.0
	github.com/vmihailenco/msgpack/v5 v5.4.1
	github.com/yalue/onnxruntime_go v1.35.0
	github.com/yuin/goldmark v1.8.5
	go.opentelemetry.io/otel v1.45.0
	go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp v1.45.0
	go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp v1.45.0
	go.opentelemetry.io/otel/metric v1.45.0
	go.opentelemetry.io/otel/sdk v1.45.0
	go.opentelemetry.io/otel/sdk/metric v1.45.0
	go.opentelemetry.io/otel/trace v1.45.0
	golang.org/x/oauth2 v0.36.0
	golang.org/x/text v0.41.0
	gonum.org/v1/gonum v0.17.0
	google.golang.org/grpc v1.83.1
	google.golang.org/protobuf v1.36.12
	gopkg.in/yaml.v3 v3.0.1
)

require (
	buf.build/gen/go/bufbuild/protovalidate/protocolbuffers/go v1.36.11-20260415201107-50325440f8f2.1 // indirect
	buf.build/go/protovalidate v1.2.0 // indirect
	buf.build/go/protoyaml v0.7.0 // indirect
	cel.dev/expr v0.25.2 // indirect
	github.com/antlr4-go/antlr/v4 v4.13.1 // indirect
	github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream v1.7.18 // indirect
	github.com/aws/aws-sdk-go-v2/feature/ec2/imds v1.18.38 // indirect
	github.com/aws/aws-sdk-go-v2/internal/configsources v1.4.38 // indirect
	github.com/aws/aws-sdk-go-v2/internal/endpoints/v2 v2.7.38 // indirect
	github.com/aws/aws-sdk-go-v2/internal/v4a v1.4.39 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding v1.13.17 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/presigned-url v1.13.38 // indirect
	github.com/aws/aws-sdk-go-v2/service/signin v1.5.7 // indirect
	github.com/aws/aws-sdk-go-v2/service/sso v1.33.7 // indirect
	github.com/aws/aws-sdk-go-v2/service/ssooidc v1.38.7 // indirect
	github.com/aws/aws-sdk-go-v2/service/sts v1.45.7 // indirect
	github.com/aws/smithy-go v1.27.8 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/benbjohnson/clock v1.3.5 // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/bep/debounce v1.2.1 // indirect
	github.com/buger/jsonparser v1.1.2 // indirect
	github.com/cenkalti/backoff/v5 v5.0.3 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/dennwc/iters v1.2.2 // indirect
	github.com/frostbyte73/core v0.1.1 // indirect
	github.com/fsnotify/fsnotify v1.10.1 // indirect
	github.com/gabriel-vasile/mimetype v1.4.13 // indirect
	github.com/gammazero/deque v1.2.1 // indirect
	github.com/go-logr/logr v1.4.4 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/go-playground/locales v0.14.1 // indirect
	github.com/go-playground/universal-translator v0.18.1 // indirect
	github.com/go-viper/mapstructure/v2 v2.4.0 // indirect
	github.com/golang-jwt/jwt/v5 v5.3.1 // indirect
	github.com/google/cel-go v0.29.0 // indirect
	github.com/google/jsonschema-go v0.4.3 // indirect
	github.com/gorilla/websocket v1.5.4-0.20250319132907-e064f32e3674 // indirect
	github.com/grpc-ecosystem/grpc-gateway/v2 v2.29.0 // indirect
	github.com/inconshreveable/mousetrap v1.1.0 // indirect
	github.com/invopop/jsonschema v0.14.0 // indirect
	github.com/jxskiss/base62 v1.1.0 // indirect
	github.com/klauspost/compress v1.18.7 // indirect
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	github.com/leodido/go-urn v1.4.0 // indirect
	github.com/lithammer/shortuuid/v4 v4.2.0 // indirect
	github.com/livekit/mageutil v0.0.0-20250511045019-0f1ff63f7731 // indirect
	github.com/livekit/mediatransportutil v0.0.0-20260605212259-862d4a7bcb1e // indirect
	github.com/livekit/psrpc v0.7.2 // indirect
	github.com/magefile/mage v1.17.2 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/nats-io/nats.go v1.52.0 // indirect
	github.com/nats-io/nkeys v0.4.16 // indirect
	github.com/nats-io/nuid v1.0.1 // indirect
	github.com/pb33f/ordered-map/v2 v2.3.1 // indirect
	github.com/pelletier/go-toml/v2 v2.2.4 // indirect
	github.com/pion/datachannel v1.6.2 // indirect
	github.com/pion/dtls/v3 v3.1.5 // indirect
	github.com/pion/ice/v4 v4.4.0 // indirect
	github.com/pion/interceptor v0.1.47 // indirect
	github.com/pion/logging v0.2.4 // indirect
	github.com/pion/mdns/v2 v2.1.0 // indirect
	github.com/pion/randutil v0.1.0 // indirect
	github.com/pion/rtcp v1.2.17 // indirect
	github.com/pion/rtp v1.10.5 // indirect
	github.com/pion/sctp v1.11.1 // indirect
	github.com/pion/sdp/v3 v3.0.19 // indirect
	github.com/pion/srtp/v3 v3.0.12 // indirect
	github.com/pion/stun/v3 v3.1.6 // indirect
	github.com/pion/transport/v4 v4.0.2 // indirect
	github.com/pion/turn/v5 v5.0.12 // indirect
	github.com/prometheus/client_golang v1.23.2 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.68.1 // indirect
	github.com/prometheus/procfs v0.20.1 // indirect
	github.com/puzpuzpuz/xsync/v4 v4.5.0 // indirect
	github.com/redis/go-redis/v9 v9.20.0 // indirect
	github.com/sagikazarmark/locafero v0.11.0 // indirect
	github.com/segmentio/asm v1.1.3 // indirect
	github.com/segmentio/encoding v0.5.4 // indirect
	github.com/sourcegraph/conc v0.3.1-0.20240121214520-5f936abd7ae8 // indirect
	github.com/spf13/afero v1.15.0 // indirect
	github.com/spf13/cast v1.10.0 // indirect
	github.com/standard-webhooks/standard-webhooks/libraries v0.0.1 // indirect
	github.com/subosito/gotenv v1.6.0 // indirect
	github.com/tidwall/gjson v1.18.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	github.com/twitchtv/twirp v8.1.3+incompatible // indirect
	github.com/vmihailenco/tagparser/v2 v2.0.0 // indirect
	github.com/wlynxg/anet v0.0.5 // indirect
	github.com/yosida95/uritemplate/v3 v3.0.2 // indirect
	github.com/zeebo/xxh3 v1.1.0 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/otel/exporters/otlp/otlptrace v1.45.0 // indirect
	go.opentelemetry.io/proto/otlp v1.11.0 // indirect
	go.uber.org/atomic v1.11.0 // indirect
	go.uber.org/multierr v1.11.0 // indirect
	go.uber.org/zap v1.28.0 // indirect
	go.uber.org/zap/exp v0.3.0 // indirect
	go.yaml.in/yaml/v3 v3.0.4 // indirect
	go.yaml.in/yaml/v4 v4.0.0-rc.2 // indirect
	golang.org/x/crypto v0.54.0 // indirect
	golang.org/x/exp v0.0.0-20260603202125-055de637280b // indirect
	golang.org/x/net v0.57.0 // indirect
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/time v0.15.0 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20260803160001-6ac0973c030d // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260803160001-6ac0973c030d // indirect
)

replace github.com/pion/opus => github.com/gojargo/opus v0.1.1-0.20260710133549-77e753c0dd49
