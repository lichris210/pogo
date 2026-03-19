# POGO — YouTube Video Outline
# Target: 5-8 minutes
# Covers: What you built, why, which services, technical explanation

---

## INTRO (30-45 sec)
**What to show:** POGO web UI on screen

"Every company using AI right now is leaving performance on the table because
their employees don't know how to prompt well. Most people just type something
into ChatGPT and hope for the best.

I built POGO — Prompt Optimization and Generation Oracle — a tool that takes
your task, picks the right model, and generates a research-backed, optimized
prompt grounded in actual academic papers.

Let me show you how it works, and then I'll walk through the AWS architecture
behind it."

---

## DEMO (1-2 min)
**What to show:** Live demo in browser at the S3 URL

1. Open POGO in browser
2. Select "Claude" as the target model
3. Type: "Analyze customer churn data and suggest retention strategies"
4. Click Generate
5. Show the prompt typing out
6. Point out:
   - The sources it pulled (chain_of_thought.pdf, dspy.pdf, etc.)
   - The XML structure (tailored to Claude)
   - The rationale explaining WHY each technique was chosen
7. Switch to GPT, same task — show how the output changes to JSON schema
8. Try a different task: "Build a multi-step reasoning agent"
   - Show it pulls ReAct and zero-shot CoT papers instead

"Notice it's not just reformatting the same prompt — it's pulling different
research papers depending on the task and adapting the structure to each model."

---

## THE PROBLEM / BUSINESS CASE (30-45 sec)
**What to show:** Slide or talking head

"Prompt engineering is still mostly trial and error. There are hundreds of
research papers on techniques like chain-of-thought, ReAct, tree-of-thoughts,
and most people have never heard of them.

POGO bridges that gap. It takes peer-reviewed research and model-specific
behavioral profiles and uses them to generate prompts that are actually
grounded in what works — not just vibes.

The use case is any company deploying LLMs at scale. Instead of every employee
guessing at prompts, they describe their task and get an optimized one backed
by citations."

---

## ARCHITECTURE WALKTHROUGH (2-3 min)
**What to show:** Architecture diagram (use the one from the README or draw on screen)

"Here's how it works end to end. Six AWS services."

### S3 — Knowledge Base
"First, I have an S3 bucket storing my knowledge base — 17 research papers
and model guides, chunked into about 1,500 segments with overlapping context
so no insight gets lost at a chunk boundary. Each chunk is embedded as a
256-dimensional vector using Bedrock Titan."

### API Gateway
"When a user submits a task from the web UI, it hits an API Gateway endpoint
— a simple HTTP POST to /generate."

### Lambda
"API Gateway triggers a Lambda function — this is where all the logic lives.
The function does three things:"

### Step 1: Bedrock Titan Embeddings
"First, it takes the user's task description and embeds it into the same
256-dimensional vector space using Bedrock Titan Embeddings. This converts
the text into a numerical representation that captures its meaning."

### Step 2: Vector Search (NumPy)
"Then it loads the pre-computed embeddings from S3 and does a cosine
similarity search using NumPy — no database needed. It finds the top 5
most semantically similar chunks from the research papers."

### Step 3: Bedrock Claude 3.5 Haiku
"Finally, it bundles those research chunks with a model-specific behavioral
profile — Claude likes XML tags, GPT likes JSON schemas, Gemini handles
long context well — and sends everything to Bedrock Claude 3.5 Haiku, which
generates the final optimized prompt with citations."

### S3 — Static Website
"The web UI is a single HTML file hosted on a second S3 bucket configured
for static website hosting. It calls the API Gateway endpoint directly."

---

## TECHNICAL DECISIONS (45-60 sec)
**What to show:** Code snippets or talking head

"A few technical decisions worth calling out:

**Why not OpenSearch for vector search?**
OpenSearch Serverless charges about 24 cents an hour even when idle. For a
project with a $200 budget, that would eat through funds fast. NumPy cosine
similarity on 1,500 vectors is instant and costs nothing.

**Why Titan Embeddings instead of a local model?**
Lambda has a 250MB deployment limit. Sentence-transformers with PyTorch is
over 2GB. By using Titan for embeddings, my entire Lambda deployment is a
single 3KB Python file with zero custom dependencies.

**Chunk quality matters more than quantity.**
My first index used 500-character chunks with hard cutoffs mid-sentence —
results were terrible. After switching to 1,200-character sentence-aware
chunks with 200-character overlap, retrieval accuracy improved dramatically."

---

## RECAP (30 sec)
**What to show:** Architecture diagram or web UI

"To recap — POGO uses six AWS services: two S3 buckets, Bedrock Titan
Embeddings, Bedrock Claude, Lambda, and API Gateway. It solves a real
problem — making prompt engineering accessible and research-backed instead
of guesswork.

The code is on GitHub — link in the description. Thanks for watching."

---

## PRODUCTION NOTES

- Record the demo FIRST while the API is live and working
- If Lambda cold-starts during recording, just note "there's a brief cold
  start on the first request" — shows you understand the tradeoff
- Screen record with QuickTime (Cmd+Shift+5 on Mac)
- Keep it conversational, not scripted word-for-word
- Architecture diagram can be drawn in Excalidraw (free) or just use the
  ASCII one from the README as a slide
