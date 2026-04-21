import os
import json
import pickle
import tempfile
import boto3
import numpy as np

S3_BUCKET = "pogo-knowledge-base"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
GEN_MODEL_ID = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
TOP_K = 5

MODEL_PROFILES = {
    "claude": {
        "strengths": "nuanced reasoning, long context, responds well to role/persona framing and XML tags",
        "techniques": "chain-of-thought, XML tags for structure, role assignment, thinking step-by-step",
        "avoid": "overly rigid formatting constraints, unnecessary repetition"
    },
    "gpt": {
        "strengths": "literal instruction following, structured output, function calling, JSON mode",
        "techniques": "explicit JSON schema, few-shot examples, clear step-by-step instructions",
        "avoid": "overly conversational framing, vague instructions"
    },
    "gemini": {
        "strengths": "very long context, multimodal input, strong synthesis across documents",
        "techniques": "explicit context placement, structured reasoning, document grounding",
        "avoid": "assuming short context behavior, ignoring placement of key information"
    }
}

_embeddings = None
_chunks = None
_bedrock = None


def get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    return _bedrock


def load_resources():
    global _embeddings, _chunks
    if _embeddings is not None:
        return

    s3 = boto3.client("s3")
    tmp = tempfile.gettempdir()

    # Download embeddings
    emb_path = os.path.join(tmp, "embeddings.npy")
    s3.download_file(S3_BUCKET, "index/embeddings.npy", emb_path)
    _embeddings = np.load(emb_path)

    # Download chunks
    chunks_path = os.path.join(tmp, "chunks.pkl")
    s3.download_file(S3_BUCKET, "index/chunks.pkl", chunks_path)
    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)

    print("Resources loaded successfully")


def embed_query(text):
    """Embed a single query using Bedrock Titan."""
    bedrock = get_bedrock()
    response = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({
            "inputText": text[:8000],
            "dimensions": 256,
            "normalize": True
        })
    )
    result = json.loads(response["body"].read())
    return np.array(result["embedding"], dtype="float32")


def search(query, top_k=TOP_K):
    """Search using cosine similarity with numpy."""
    load_resources()
    query_vec = embed_query(query)

    # Cosine similarity (vectors are already normalized)
    similarities = _embeddings @ query_vec

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(_chunks[idx])
    return results


def generate_prompt(task, model, chunks):
    profile = MODEL_PROFILES.get(model, MODEL_PROFILES["claude"])
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    ])

    system_prompt = """You are POGO - a Prompt Optimization & Generation Oracle.
Your job is to generate a highly optimized, ready-to-use prompt based on:
1. The user's task description
2. The target model's behavioral profile
3. Relevant prompt engineering research

Always return:
- The optimized prompt (clearly labeled, ready to copy-paste)
- A brief explanation of which techniques you used and why
- Which research/sources informed your choices"""

    user_message = f"""Task: {task}

Target Model: {model.upper()}
Model Profile:
- Strengths: {profile['strengths']}
- Best techniques: {profile['techniques']}
- Avoid: {profile['avoid']}

Relevant Research:
{context}

Generate an optimized prompt for this task."""

    bedrock = get_bedrock()
    response = bedrock.invoke_model(
        modelId=GEN_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1500,
            "messages": [{"role": "user", "content": user_message}],
            "system": system_prompt
        })
    )

    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def lambda_handler(event, context):
    # CORS preflight — respond before any routing / heavy imports
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or ""
    ).upper()
    if method == "OPTIONS":
        return {
            "statusCode": 204,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            },
            "body": "",
        }

    # Route /optimize requests to the v2 orchestrator
    raw_path = event.get("rawPath", event.get("path", ""))
    if raw_path.endswith("/optimize"):
        from orchestrator.orchestrator import handle_message
        return handle_message(event)

    # --- v1 /generate handler (unchanged) ---
    try:
        body = json.loads(event.get("body", "{}"))
        task = body.get("task", "").strip()
        model = body.get("model", "claude").strip().lower()

        if not task:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Please provide a task description"})
            }

        if model not in MODEL_PROFILES:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": f"Model must be one of: {list(MODEL_PROFILES.keys())}"})
            }

        chunks = search(task)
        optimized_prompt = generate_prompt(task, model, chunks)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({
                "task": task,
                "model": model,
                "optimized_prompt": optimized_prompt,
                "sources_used": list(set(c["source"] for c in chunks))
            })
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
