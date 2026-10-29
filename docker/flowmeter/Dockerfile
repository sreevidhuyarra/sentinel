# DistriNet's corrected CICFlowMeter at the version behind the CIC-IDS2017 "improved" dataset
# the models were trained on: commit 55f60ad (29 July 2022). Its feature set matches the
# dataset exactly (Total TCP Flow Time, ICMP code/type, RST flag counts; none of the 2023
# additions). Later commits rename Total TCP Flow Time and change how subflows and the
# backward initial window are computed, which would silently shift the models' inputs.
#
# The repository has no Dockerfile at that commit; this follows the build steps of the
# Dockerfile it gained in August 2023 (b0c716f), which did not change the build itself.
#
#   docker compose --profile pcap build flowmeter
#   docker run --rm --network none -v <pcaps>:/tmp/pcap/input:ro -v <out>:/tmp/pcap/output \
#       sentinel-flowmeter:55f60ad /tmp/pcap/input /tmp/pcap/output
FROM alpine:latest AS builder
ARG COMMIT=55f60addd21ada7635a79ee3c596a389756552a6
RUN apk add --update --no-cache git unzip openjdk8 libpcap-dev
RUN git clone https://github.com/GintsEngelen/CICFlowMeter.git /CICFlowMeter \
    && git -C /CICFlowMeter checkout --detach "$COMMIT"
WORKDIR /CICFlowMeter
RUN ./gradlew distZip
RUN UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o build/distributions/CICFlowMeter-*.zip -d build/distributions

FROM alpine:latest
RUN apk add --update --no-cache openjdk8 libpcap-dev
WORKDIR /CICFlowMeter
COPY --from=builder /CICFlowMeter/build/distributions/CICFlowMeter-* .
# The CLI loads native libraries from a path relative to bin/.
WORKDIR /CICFlowMeter/bin
ENTRYPOINT ["./cfm"]
