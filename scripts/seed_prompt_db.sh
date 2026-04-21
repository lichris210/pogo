#!/usr/bin/env bash
#
# Seed the v2 prompt DB on S3 from seed_prompts.json.
#
# Run from the project root:
#   bash scripts/seed_prompt_db.sh
#
set -euo pipefail

BUCKET="${POGO_KNOWLEDGE_BUCKET:-pogo-knowledge-base}"
PROMPTS_KEY="prompt_db/prompts.json"
EMBEDDINGS_KEY="prompt_db/embeddings.npy"
SEED_FILE="seed_prompts.json"

# Ensure the project root is on sys.path so `python -m prompt_db.ingest` works
# even if the user has shadowed PYTHONPATH.
export PYTHONPATH="${PYTHONPATH:-.}"
case ":${PYTHONPATH}:" in
    *:.:*) ;;
    *) PYTHONPATH=".:${PYTHONPATH}" ;;
esac
export PYTHONPATH

echo "==> Checking AWS credentials..."
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    echo "ERROR: AWS credentials are not configured." >&2
    echo "       Run 'aws configure' or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY." >&2
    exit 1
fi
aws sts get-caller-identity --output text --query 'Arn' | sed 's/^/    identity: /'

echo "==> Checking that s3://${BUCKET} exists..."
if ! aws s3api head-bucket --bucket "${BUCKET}" >/dev/null 2>&1; then
    echo "ERROR: Bucket s3://${BUCKET} is missing or inaccessible." >&2
    exit 1
fi

if [ ! -f "${SEED_FILE}" ]; then
    echo "ERROR: ${SEED_FILE} not found. Run this script from the project root." >&2
    exit 1
fi

echo "==> Running ingest (python -m prompt_db.ingest --seed-file ${SEED_FILE})..."
python -m prompt_db.ingest --seed-file "${SEED_FILE}"

echo "==> Verifying s3://${BUCKET}/${PROMPTS_KEY}..."
PROMPTS_SIZE=$(aws s3api head-object \
    --bucket "${BUCKET}" --key "${PROMPTS_KEY}" \
    --query 'ContentLength' --output text)
if [ -z "${PROMPTS_SIZE}" ] || [ "${PROMPTS_SIZE}" = "None" ] || [ "${PROMPTS_SIZE}" -le 0 ]; then
    echo "ERROR: ${PROMPTS_KEY} is missing or empty." >&2
    exit 1
fi
echo "    ${PROMPTS_KEY}: ${PROMPTS_SIZE} bytes"

echo "==> Verifying s3://${BUCKET}/${EMBEDDINGS_KEY}..."
EMBEDDINGS_SIZE=$(aws s3api head-object \
    --bucket "${BUCKET}" --key "${EMBEDDINGS_KEY}" \
    --query 'ContentLength' --output text)
if [ -z "${EMBEDDINGS_SIZE}" ] || [ "${EMBEDDINGS_SIZE}" = "None" ] || [ "${EMBEDDINGS_SIZE}" -le 0 ]; then
    echo "ERROR: ${EMBEDDINGS_KEY} is missing or empty." >&2
    exit 1
fi
echo "    ${EMBEDDINGS_KEY}: ${EMBEDDINGS_SIZE} bytes"

echo "==> Counting ingested prompts..."
COUNT=$(aws s3 cp "s3://${BUCKET}/${PROMPTS_KEY}" - \
    | python -c "import json, sys; print(len(json.load(sys.stdin)))")
echo "    Ingested ${COUNT} prompts into s3://${BUCKET}/prompt_db/"
