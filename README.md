# POGO — Prompt Optimization & Generation Oracle

A serverless, research-backed prompt engineering tool built on AWS. POGO takes a task description and target model, searches a knowledge base of prompt engineering research papers, and generates an optimized, citation-grounded prompt tailored to how that specific model behaves.

**Live Demo:** http://pogo-web-ui.s3-website-us-east-1.amazonaws.com/pogo.html

**Video Walkthrough:** [YouTube Link]

---

## How It Works

```
User submits:                        POGO returns:
┌──────────────────────┐             ┌──────────────────────────┐
│ Task description     │             │ Optimized prompt         │
│ Target model         │────────────▶│ Techniques used + why    │
│ (Claude/GPT/Gemini)  │             │ Research sources cited   │
└──────────────────────┘             └──────────────────────────┘
```

**Under the hood:**

1. User's task is embedded using **Amazon Bedrock Titan Embeddings**
2. Cosine similarity search finds the most relevant research from a vector knowledge base stored in **S3**
3. Retrieved research + model-specific behavioral profile are bundled together
4. **Bedrock Claude 3.5 Haiku** generates an optimized prompt grounded in the research
5. Result is returned via **API Gateway → Lambda**

---

## Architecture

```
┌─────────────┐     ┌───────────────┐     ┌──────────────────────┐
│   Browser   │────▶│ API Gateway   │────▶│   Lambda Function    │
│  (S3 Site)  │     │ (HTTP API)    │     │                      │
└─────────────┘     └───────────────┘     │  1. Embed query      │
                                          │     (Bedrock Titan)  │
                                          │                      │
                                          │  2. Vector search    │
                                          │     (S3 + NumPy)     │
                                          │                      │
                                          │  3. Generate prompt  │
                                          │     (Bedrock Claude) │
                                          └──────────────────────┘
```

### AWS Services Used (6)

| Service | Purpose |
|---------|---------|
| **S3** (knowledge base) | Stores vector embeddings and chunked research papers |
| **S3** (static website) | Hosts the web UI |
| **Bedrock** (Titan Embeddings) | Converts text into vector representations for semantic search |
| **Bedrock** (Claude 3.5 Haiku) | Generates optimized prompts from research context |
| **Lambda** | Serverless compute — runs the entire pipeline per request |
| **API Gateway** | HTTP endpoint that accepts POST requests and routes to Lambda |

---

## Knowledge Base

POGO's recommendations are grounded in 17 sources spanning academic research and official model documentation:

**Reasoning & Prompting Techniques**
- Chain-of-Thought Prompting (Wei et al., 2022)
- Tree of Thoughts (Yao et al., 2023)
- ReAct: Reasoning + Acting (Yao et al., 2022)
- Self-Consistency (Wang et al., 2022)
- Zero-Shot Chain of Thought (Kojima et al., 2022)

**Prompt Optimization**
- Automatic Prompt Engineer / APE (Zhou et al., 2023)
- DSPy: Compiling Declarative LM Calls (Khattab et al., 2023)

**Context & Retrieval**
- RAG: Retrieval-Augmented Generation (Lewis et al., 2020)
- Lost in the Middle (Liu et al., 2023)
- Few-Shot Learners / GPT-3 (Brown et al., 2020)

**Security**
- Prompt Injection Attacks (Perez & Ribeiro, 2022)

**Model-Specific Guides**
- Anthropic Prompt Engineering Guide & Tutorial
- OpenAI Prompt Engineering Guide
- Google Gemini Prompting Guide
- DAIR.ai Prompt Engineering Guide

---

## Model Profiles

POGO tailors output based on each model's documented strengths:

```python
MODEL_PROFILES = {
    "claude": {
        "strengths": "nuanced reasoning, long context, role/persona framing, XML tags",
        "techniques": "chain-of-thought, XML structure, role assignment",
        "avoid": "overly rigid formatting, unnecessary repetition"
    },
    "gpt": {
        "strengths": "literal instruction following, structured output, JSON mode",
        "techniques": "explicit JSON schema, few-shot examples, step-by-step",
        "avoid": "overly conversational framing, vague instructions"
    },
    "gemini": {
        "strengths": "very long context, multimodal, strong synthesis",
        "techniques": "document grounding, explicit context placement",
        "avoid": "assuming short context behavior"
    }
}
```

---

## Project Structure

```
pogo/
├── lambda/
│   └── handler.py              # Lambda function — search + generation
├── pogo/
│   ├── documents/              # Source PDFs and text files (not committed)
│   └── scripts/
│       ├── build_index.py      # Local indexer (sentence-transformers + FAISS)
│       └── build_index_titan.py # Cloud indexer (Bedrock Titan + NumPy)
├── pogo.html                   # Web UI (hosted on S3)
├── deploy.sh                   # Lambda + API Gateway deployment script
└── README.md
```

---

## API Usage

**Endpoint:** `POST /generate`

**Request:**
```json
{
  "task": "Analyze customer churn data and suggest retention strategies",
  "model": "claude"
}
```

**Response:**
```json
{
  "task": "Analyze customer churn data...",
  "model": "claude",
  "optimized_prompt": "...",
  "sources_used": ["chain_of_thought.pdf", "anthropic_tutorial.txt", "dspy.pdf"]
}
```

**Supported models:** `claude`, `gpt`, `gemini`

---

## Setup & Deployment

### Prerequisites
- AWS account with Bedrock access
- Python 3.12+
- AWS CLI configured

### 1. Clone and set up
```bash
git clone https://github.com/YOUR_USERNAME/pogo.git
cd pogo
python3 -m venv venv
source venv/activate
pip install boto3 numpy PyMuPDF
```

### 2. Build the knowledge base
Place research PDFs and text guides in `pogo/documents/`, then:
```bash
python3 pogo/scripts/build_index_titan.py
```

### 3. Deploy Lambda + API Gateway
```bash
chmod +x deploy.sh
./deploy.sh
```

### 4. Enable CORS
```bash
aws apigatewayv2 update-api \
  --api-id YOUR_API_ID \
  --cors-configuration AllowOrigins='*',AllowMethods='POST,OPTIONS',AllowHeaders='Content-Type' \
  --region us-east-1
```

### 5. Host the UI
```bash
aws s3 mb s3://pogo-web-ui --region us-east-1
aws s3 website s3://pogo-web-ui --index-document pogo.html
aws s3 cp pogo.html s3://pogo-web-ui/pogo.html --content-type "text/html"
```

---

## Cost

Total project cost: **under $5** (Bedrock inference + S3 storage). All services operate within or near free-tier limits for this scale of usage.

---

## Built By

Christopher Li — MSBA Candidate, Cal Poly San Luis Obispo (Orfalea College of Business)

Cloud Computing Capstone, Spring 2026 — Prof. Osterbur
