# POGO Architecture

> **POGO** (Prompt Optimization with Grounded Output) is a serverless RAG application that generates optimized LLM prompts grounded in research documentation. A user describes a task and selects a target model; POGO retrieves the most relevant prompt-engineering research and uses Claude 3.5 Haiku to craft a model-specific, research-backed prompt.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Request Flow](#2-request-flow)
3. [Lambda Function](#3-lambda-function)
4. [API Gateway](#4-api-gateway)
5. [Frontend](#5-frontend)
6. [Vector Store (S3 + NumPy)](#6-vector-store-s3--numpy)
7. [Bedrock Calls](#7-bedrock-calls)
8. [Knowledge Base Sources](#8-knowledge-base-sources)
9. [Other Infrastructure](#9-other-infrastructure)
10. [IAM & Permissions](#10-iam--permissions)
11. [Configuration Reference](#11-configuration-reference)
12. [Deployment](#12-deployment)
13. [CI/CD & Version Control](#13-cicd--version-control)

---

## 1. System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Browser / API Client                     │
│                    HTTP POST {task, model}                        │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              API Gateway HTTP API  (us-east-1)                   │
│              Route: POST /generate                               │
│              CORS: *, Content-Type                               │
└─────────────────────────────┬────────────────────────────────────┘
                              │ Lambda invoke
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Lambda: pogo-prompt-generator                       │
│              Python 3.12 | 512 MB | 60 s timeout                │
│                                                                  │
│  ① Parse & validate {task, model}                               │
│  ② embed_query(task)  ──────────────────────────────────┐       │
│  ③ Load embeddings.npy + chunks.pkl from S3 (cold start)│       │
│  ④ Cosine similarity search → top-5 chunks              │       │
│  ⑤ generate_prompt(task, model, chunks) ────────────────┤       │
│  ⑥ Return JSON {optimized_prompt, sources_used}         │       │
└────────────────┬────────────────────────────────────────┼───────┘
                 │ S3 GetObject                            │ Bedrock InvokeModel
                 ▼                                         ▼
┌──────────────────────────┐          ┌───────────────────────────────────┐
│  S3: pogo-knowledge-base │          │  Amazon Bedrock  (us-east-1)      │
│  index/embeddings.npy    │          │                                   │
│  index/chunks.pkl        │          │  • Titan Embed v2  (256-dim)      │
└──────────────────────────┘          │    embed_query + index builds     │
                                      │                                   │
                                      │  • Claude 3.5 Haiku               │
                                      │    generate_prompt                │
                                      └───────────────────────────────────┘

┌──────────────────────────┐
│  S3: pogo-web-ui         │  ← static website hosting
│  pogo.html               │    served directly to browser
└──────────────────────────┘
```

---

## 2. Request Flow

```
Browser
  │
  │  POST /generate
  │  Content-Type: application/json
  │  { "task": "...", "model": "claude|gpt|gemini" }
  │
  ▼
API Gateway  ──→  Lambda (pogo-prompt-generator)
                     │
                     ├─ [cold start] S3 GetObject: embeddings.npy → numpy array
                     ├─ [cold start] S3 GetObject: chunks.pkl     → list[dict]
                     │
                     ├─ Bedrock InvokeModel (Titan Embed v2)
                     │    input:  task text (≤8000 chars)
                     │    output: 256-dim float32 vector
                     │
                     ├─ cosine_sim = query_vec @ embeddings.T
                     │    → argsort → top 5 chunk indices
                     │
                     ├─ Bedrock InvokeModel (Claude 3.5 Haiku)
                     │    system: POGO role + instructions
                     │    user:   task + model profile + 5 research chunks
                     │    output: optimized prompt + explanation (max 1500 tokens)
                     │
                     └─ 200 { task, model, optimized_prompt, sources_used }
```

On **warm** Lambda containers, the S3 reads are skipped — `_embeddings` and `_chunks` are cached as module-level globals.

---

## 3. Lambda Function

**Name:** `pogo-prompt-generator`  
**File:** `lambda/handler.py`  
**Entry point:** `handler.lambda_handler`  
**Runtime:** Python 3.12  
**Memory:** 512 MB  
**Timeout:** 60 s  
**Trigger:** API Gateway HTTP API (POST /generate)

### Key functions

| Function | Purpose |
|----------|---------|
| `lambda_handler(event, context)` | Entry point — parses HTTP body, orchestrates search + generation, returns HTTP response |
| `embed_query(text)` | Calls Bedrock Titan Embed v2; returns 256-dim normalized numpy vector |
| `search(query)` | Loads S3 index on first call (cached globally); computes cosine similarity; returns top-5 chunks |
| `generate_prompt(task, model, chunks)` | Builds system + user messages; calls Bedrock Claude 3.5 Haiku; returns raw text |

### Global cold-start cache

```python
_embeddings: np.ndarray | None   # shape (N, 256), loaded from S3 once
_chunks:     list[dict]  | None  # [{"text": str, "source": str}, ...]
_bedrock:    boto3.client         # reused across invocations
```

### Request / response contract

**Request body (JSON)**
```json
{ "task": "string (required)", "model": "claude|gpt|gemini (optional, default: claude)" }
```

**Success response (HTTP 200)**
```json
{
  "task": "string",
  "model": "string",
  "optimized_prompt": "string  (markdown code block with the prompt)",
  "sources_used": ["anthropic_tutorial.txt", "openai_guide.txt", "..."]
}
```

**Error responses**
| Code | Body |
|------|------|
| 400 | `{"error": "task is required"}` |
| 400 | `{"error": "model must be one of: claude, gpt, gemini"}` |
| 500 | `{"error": "<exception message>"}` |

### Model profiles (hardcoded in handler)

```python
MODEL_PROFILES = {
    "claude": {
        "strengths": "nuanced reasoning, long context, responds well to role/persona framing and XML tags",
        "techniques": "chain-of-thought, XML tags for structure, role assignment, thinking step-by-step",
        "avoid": "overly rigid formatting constraints, unnecessary repetition",
    },
    "gpt": {
        "strengths": "literal instruction following, structured output, function calling, JSON mode",
        "techniques": "explicit JSON schema, few-shot examples, clear step-by-step instructions",
        "avoid": "overly conversational framing, vague instructions",
    },
    "gemini": {
        "strengths": "very long context, multimodal input, strong synthesis across documents",
        "techniques": "explicit context placement, structured reasoning, document grounding",
        "avoid": "assuming short context behavior, ignoring placement of key information",
    },
}
```

### Packaging

NumPy must be bundled in the deployment ZIP (it is not available in the default Lambda runtime). `deploy.sh` creates `/tmp/pogo-lambda.zip` containing `handler.py` + a `numpy/` package directory.

---

## 4. API Gateway

**Type:** HTTP API (API Gateway v2)  
**Name:** `pogo-api`  
**Region:** `us-east-1`  
**Stage:** `$default` (auto-deploy enabled)  
**Protocol:** HTTP/1.1

### Routes

| Method | Route | Backend | Payload format version |
|--------|-------|---------|------------------------|
| POST | `/generate` | Lambda (pogo-prompt-generator) | `2.0` |

### CORS configuration

```
AllowOrigins: *
AllowMethods: POST, OPTIONS
AllowHeaders: Content-Type
```

### Integration

- Type: AWS_PROXY (Lambda proxy integration)
- Lambda invoke permission granted to API Gateway principal via `aws lambda add-permission`

### Endpoint format

```
https://<api-id>.execute-api.us-east-1.amazonaws.com/generate
```

---

## 5. Frontend

**File:** `pogo.html` (single file, ~400 lines)  
**Framework:** None — pure HTML + CSS + vanilla JavaScript  
**Hosting:** S3 static website (`pogo-web-ui` bucket)  
**Dependencies:** Zero npm packages

### UI component map

```
pogo.html
├── Header
│   ├── Brand label "POGO"
│   ├── Headline "Craft the perfect prompt."
│   └── Subheadline "Grounded in research. Tailored to your model."
│
├── Model Selector Tabs
│   ├── [Claude]   → selectedModel = "claude"
│   ├── [GPT]      → selectedModel = "gpt"
│   └── [Gemini]   → selectedModel = "gemini"
│
├── Input Area
│   ├── <textarea> task description
│   │     Cmd/Ctrl+Enter → generate()
│   └── Hint text
│
├── Generate Button
│   └── onClick → generate()
│
├── Loading Shimmer bar (hidden during idle)
│
├── Error Box (hidden when no error)
│
├── Results Section (hidden until first generation)
│   ├── Sources row (pill tags from sources_used[])
│   ├── Prompt Card
│   │   ├── Copy button (navigator.clipboard API)
│   │   └── <pre><code> block (fullPromptText)
│   ├── Rationale section (technique explanations)
│   └── Meta line (model name + source count)
│
└── Examples Section (4 pre-filled task examples)
    └── onClick → populate textarea + focus
```

### JavaScript state

| Variable | Type | Purpose |
|----------|------|---------|
| `selectedModel` | string | Active model tab ("claude" / "gpt" / "gemini") |
| `fullPromptText` | string | Latest generated prompt, used by Copy button |
| `isLoading` | boolean | Disables button, shows shimmer during fetch |

### API call (inside `generate()`)

```javascript
fetch(API_URL, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ task, model: selectedModel }),
})
```

`API_URL` is a constant on line 397 of `pogo.html` — must be updated with the real API Gateway URL after deployment.

---

## 6. Vector Store (S3 + NumPy)

POGO uses **no dedicated vector database**. Instead, embeddings and chunk metadata are stored as binary files in S3 and loaded into Lambda memory for in-process similarity search.

### S3 objects

| Key | Format | Contents |
|-----|--------|---------|
| `index/embeddings.npy` | NumPy binary | float32 array, shape `(N, 256)` — one row per chunk |
| `index/chunks.pkl` | Python pickle | `list[dict]` — `[{"text": str, "source": str}, ...]` |

### Embedding model

| Property | Value |
|----------|-------|
| Model | Amazon Bedrock Titan Text Embeddings v2 |
| Model ID | `amazon.titan-embed-text-v2:0` |
| Dimension | 256 |
| Normalization | Yes (L2-normalized, cosine-sim compatible) |
| Max input | 8000 characters (truncated before sending) |

### Chunking (build_index_titan.py)

| Parameter | Value |
|-----------|-------|
| Chunk size | 1200 characters |
| Overlap | 200 characters |
| Strategy | Sentence-aware split (preserves sentence boundaries) |

### Search (lambda/handler.py)

```python
# 1. embed incoming query
query_vec = embed_query(task)                   # shape (256,)

# 2. cosine similarity against all stored vectors
sims = query_vec @ embeddings.T                 # shape (N,)

# 3. top-K indices
top_indices = np.argsort(sims)[::-1][:TOP_K]   # TOP_K = 5

# 4. return chunk dicts
return [chunks[i] for i in top_indices]
```

### Index building scripts

| Script | Purpose | Notes |
|--------|---------|-------|
| `pogo/scripts/build_index_titan.py` | **Active.** Builds S3 index using Bedrock Titan | Current production approach |
| `pogo/scripts/build_index.py` | Legacy. Uses FAISS + sentence-transformers | Kept for reference; not used in Lambda |
| `pogo/scripts/ingest.py` | Deprecated. Local sentence-transformers only | Earliest prototype |

---

## 7. Bedrock Calls

All Bedrock calls go through a single `boto3.client("bedrock-runtime", region_name="us-east-1")` instance cached at module scope in the Lambda.

### Call 1 — Titan Text Embeddings v2

| Property | Value |
|----------|-------|
| Model ID | `amazon.titan-embed-text-v2:0` |
| Invoked in | `lambda/handler.py:embed_query()` and `build_index_titan.py` |
| Purpose | Encode user task (and knowledge base chunks) into 256-dim vectors |
| Input | `{"inputText": text}` |
| Output | `{"embedding": [float, ...]}` (256 values) |
| Throttle guard (index build) | 50 ms sleep between batch calls |

### Call 2 — Claude 3.5 Haiku

| Property | Value |
|----------|-------|
| Model ID | `us.anthropic.claude-3-5-haiku-20241022-v1:0` |
| Invoked in | `lambda/handler.py:generate_prompt()` |
| Purpose | Generate optimized, research-grounded prompt for the target model |
| Max tokens | 1500 |
| System message | POGO role definition + output format instructions |
| User message | Task description + model profile (strengths/techniques/avoid) + top-5 research chunks |
| Output format | Markdown code block containing the prompt + technique explanations + source citations |

### Bedrock invocation pattern

```python
response = _bedrock.invoke_model(
    modelId=MODEL_ID,
    body=json.dumps(payload),
    contentType="application/json",
    accept="application/json",
)
result = json.loads(response["body"].read())
```

---

## 8. Knowledge Base Sources

All source documents live in `pogo/documents/` (text files committed to git; PDFs excluded via `.gitignore`).

| File | Lines | Topic |
|------|-------|-------|
| `anthropic_guide.txt` | ~50 | Anthropic prompt engineering overview |
| `anthropic_tutorial.txt` | ~1,881 | Comprehensive Anthropic prompting tutorial with examples |
| `dair_guide.txt` | ~40 | DAIR.ai prompt engineering guide |
| `gemini_multimodal.txt` | ~1,372 | Google Gemini multimodal + prompting guide |
| `google_guide.txt` | ~564 | Google general LLM prompting guidance |
| `openai_guide.txt` | ~661 | OpenAI prompt engineering best practices |

**Total:** ~4,568 lines / ~396 KB

**Topics covered:** Chain-of-Thought, Tree of Thoughts, ReAct, Self-Consistency, Zero-Shot CoT, APE, DSPy, RAG, few-shot learning, prompt injection, model-specific behavioral guidance.

---

## 9. Other Infrastructure

### S3 Buckets

| Bucket | Purpose | Key contents |
|--------|---------|-------------|
| `pogo-knowledge-base` | Vector index storage | `index/embeddings.npy`, `index/chunks.pkl` |
| `pogo-web-ui` | Static website hosting | `pogo.html` (index document) |

Both buckets are in `us-east-1`. `pogo-web-ui` has static website hosting enabled.

### No persistent database

The application is **fully stateless**:
- No DynamoDB tables
- No RDS
- No ElastiCache
- Session state lives in the browser; Lambda state is ephemeral (warm container cache only)

### No message queues

- No SQS queues
- No SNS topics
- All calls are synchronous request-response

### CloudWatch Logs

Auto-created by `AWSLambdaBasicExecutionRole`:
- Log group: `/aws/lambda/pogo-prompt-generator`

---

## 10. IAM & Permissions

### Lambda execution role

**Name:** `pogo-lambda-role`  
**Trust policy:** `lambda.amazonaws.com`

**Attached policies:**

| Policy | ARN / Type | Grants |
|--------|-----------|--------|
| AWSLambdaBasicExecutionRole | AWS Managed | CloudWatch Logs write |
| AmazonS3ReadOnlyAccess | AWS Managed | S3 GetObject on all buckets |
| AmazonBedrockFullAccess | AWS Managed | Bedrock InvokeModel |

### Resource-based policy (Lambda)

API Gateway is granted `lambda:InvokeFunction` on `pogo-prompt-generator` via:

```bash
aws lambda add-permission \
  --function-name pogo-prompt-generator \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:us-east-1:*:$API_ID/*/*/generate"
```

---

## 11. Configuration Reference

### Lambda constants (lambda/handler.py)

| Constant | Value | Description |
|----------|-------|-------------|
| `S3_BUCKET` | `pogo-knowledge-base` | Bucket containing vector index |
| `EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `GEN_MODEL_ID` | `us.anthropic.claude-3-5-haiku-20241022-v1:0` | Bedrock generation model |
| `TOP_K` | `5` | Chunks returned by similarity search |

### Index build constants (build_index_titan.py)

| Constant | Value | Description |
|----------|-------|-------------|
| `DOCS_DIR` | `pogo/documents` | Source document directory |
| `S3_BUCKET` | `pogo-knowledge-base` | Destination bucket |
| `CHUNK_SIZE` | `1200` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `EMBED_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Embedding model |
| `EMBED_BATCH_DELAY` | `0.05` s | Throttle guard between embedding calls |

### Deployment constants (deploy.sh)

| Variable | Value |
|----------|-------|
| `FUNCTION_NAME` | `pogo-prompt-generator` |
| `ROLE_NAME` | `pogo-lambda-role` |
| `API_NAME` | `pogo-api` |
| `REGION` | `us-east-1` |
| `RUNTIME` | `python3.12` |
| `HANDLER` | `handler.lambda_handler` |
| `TIMEOUT` | `60` |
| `MEMORY` | `512` |

### Frontend constant (pogo.html line 397)

```javascript
const API_URL = "https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/generate";
```

Must be updated with the actual API Gateway URL after each deployment.

---

## 12. Deployment

POGO has no automated CI/CD. Everything is deployed manually in order:

### Step 1 — Prerequisites

```bash
pip install boto3 numpy pymupdf   # local build tools
aws configure                     # set up credentials + default region (us-east-1)
```

### Step 2 — Build vector index

```bash
python3 pogo/scripts/build_index_titan.py
```

- Reads all `.txt` files from `pogo/documents/`
- Chunks text (1200 chars, 200 overlap)
- Embeds each chunk via Bedrock Titan (256-dim)
- Saves `embeddings.npy` + `chunks.pkl` to `s3://pogo-knowledge-base/index/`

### Step 3 — Deploy Lambda + API Gateway

```bash
chmod +x deploy.sh && ./deploy.sh
```

Script actions:
1. Create (or retrieve) IAM role `pogo-lambda-role` with required policies
2. Package `lambda/handler.py` + `numpy/` into `/tmp/pogo-lambda.zip`
3. Create or update Lambda function `pogo-prompt-generator`
4. Create or retrieve HTTP API `pogo-api`
5. Wire `POST /generate` → Lambda integration
6. Grant API Gateway invoke permission on Lambda
7. Print the API endpoint URL

### Step 4 — Update frontend

Edit `pogo.html` line 397 — replace `YOUR_API_ID` with the real API Gateway ID printed by `deploy.sh`.

### Step 5 — Host frontend

```bash
aws s3 mb s3://pogo-web-ui --region us-east-1
aws s3 website s3://pogo-web-ui --index-document pogo.html
aws s3 cp pogo.html s3://pogo-web-ui/pogo.html --content-type "text/html"
```

### Step 6 — Enable CORS (if not set by deploy.sh)

```bash
aws apigatewayv2 update-api \
  --api-id YOUR_API_ID \
  --cors-configuration 'AllowOrigins=*,AllowMethods=POST,OPTIONS,AllowHeaders=Content-Type' \
  --region us-east-1
```

---

## 13. CI/CD & Version Control

**VCS:** Git (GitHub — `lichris210/pogo`)  
**Default branch:** `main`  
**Current work branch:** `claude/document-architecture-dzivE`

**No automated CI/CD pipeline.** There are no:
- GitHub Actions workflows
- AWS CodePipeline definitions
- CloudFormation / SAM / CDK / Terraform templates

All infrastructure is provisioned imperatively by `deploy.sh` and manual AWS CLI commands.

### .gitignore exclusions

| Pattern | Reason |
|---------|--------|
| `venv/`, `__pycache__/` | Python environment artifacts |
| `.env` | Credentials / secrets |
| `pogo/documents/*.pdf` | Large binary files — only `.txt` committed |
| `pogo/faiss_output/`, `faiss_index/` | Large binary FAISS index files |
| `*.zip` | Lambda deployment packages |
| `.vscode/`, `.idea/` | IDE configs |

### Runtime dependency summary

| Context | Dependencies |
|---------|-------------|
| Lambda runtime | `boto3` (built-in), `numpy` (bundled in ZIP) |
| Index build | `boto3`, `numpy`, `pymupdf` (optional, for PDF parsing) |
| Frontend | None (pure HTML/CSS/JS) |
