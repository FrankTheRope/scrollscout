#!/usr/bin/env bash
# Download the published canon_2um ink predictions of the PHerc0139 segments
# that also have ink_9um labels. ~500 MB. Requires awscli (no credentials).
set -e
B=s3://vesuvius-challenge-open-data/PHerc0139/segments
mkdir -p data/0139_pred/raw
for SEG in $(aws s3 ls --no-sign-request $B/ | grep -E "w0(28|29|35|39|40|41|43|44)_" | awk '{print $2}'); do
  W=$(echo "$SEG" | sed -E 's/.*-(w0[0-9]+)_.*/\1/')
  F=$(aws s3 ls --no-sign-request "$B/$SEG"ink-detection/ | awk '{print $4}' | grep "2.399um" | head -1)
  if [ -z "$F" ]; then echo "$W: no 2.399um prediction"; continue; fi
  echo "$W <- $F"
  aws s3 cp --no-sign-request --quiet "$B/$SEG"ink-detection/"$F" "data/0139_pred/raw/$W.tif"
done
