#!/bin/bash

MODEL_FILE=$1
SIG_FILE=$2
KEY_FILE=$3

if [ ! -f "$MODEL_FILE" ] || [ ! -f "$SIG_FILE" ] || [ ! -f "$KEY_FILE" ]; then
  echo "Usage: $0 <model_file> <signature_file> <public_key_file>"
  exit 1
fi

echo "Verifying model integrity..."
if cosign verify-blob --key "$KEY_FILE" --signature "$SIG_FILE" "$MODEL_FILE"; then
  echo "✅ Model verified successfully!"
  echo "Safe to use this model for inference"
  exit 0
else
  echo "❌ Model verification failed!"
  echo "DO NOT use this model - it may have been tampered with"
  exit 1
fi
