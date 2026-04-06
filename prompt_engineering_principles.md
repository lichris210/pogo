# Prompt Engineering Principles

This document consolidates **directly actionable** findings from the uploaded papers and guides into principles you can apply when writing **LLM system prompts** and surrounding agent prompts.

## How to use this document

- Treat these as **defaults**, not dogma.
- Some principles are **single-prompt techniques**. Others are **pipeline-level techniques** that usually require multiple model calls or external tools.
- When a principle is highly model-dependent, that is stated explicitly.

---

## Structural best practices

- **Put durable rules in the highest-priority instruction channel.**
  - **Why it works:** System/developer instructions are the right place for identity, goals, constraints, tone, and house rules. User messages should supply task inputs, not override the application’s core behavior.
  - **Sources:** OpenAI Guide; Anthropic Tutorial.

- **State the task, constraints, and output format explicitly.**
  - **Why it works:** Models follow concrete instructions more reliably than implied intent. If you want no preamble, a single label, exact JSON, or a specific tone, say so directly.
  - **Sources:** Anthropic Tutorial; Google Guide; OpenAI Guide.

- **Structure the prompt into named sections.**
  - **Why it works:** Separating `role`, `context`, `task`, `constraints`, `examples`, and `output_format` reduces ambiguity and makes the prompt easier for the model to parse.
  - **Sources:** Google Guide; OpenAI Guide; Anthropic Tutorial.

- **Use explicit delimiters to separate instructions from data. Prefer XML-style tags for templated inputs.**
  - **Why it works:** The model is less likely to confuse raw user content with instructions when the boundaries are obvious.
  - **Sources:** Anthropic Tutorial; Google Guide; OpenAI Guide.

- **Wrap every untrusted variable in its own tag or clearly marked field.**
  - **Why it works:** This makes the start and end of user-provided content unambiguous and reduces accidental reinterpretation of user text as prompt logic.
  - **Sources:** Anthropic Tutorial; Prompt Injection.

- **Use specific placeholder names in prompt templates.**
  - **Why it works:** Clear variable names make templates easier to maintain and reduce accidental misuse when prompts get long or are reused across workflows.
  - **Sources:** Anthropic Tutorial.

- **Put the immediate task or user question near the end of a long prompt, after context.**
  - **Why it works:** This acts like a final reminder of what the model should do now, which often helps long prompts stay on task.
  - **Sources:** Anthropic Tutorial.

- **When format matters, ask for a schema and show the opening of the desired output if needed.**
  - **Why it works:** Prefilling with `<tag>` or `{` can strongly steer the model into the right output shape.
  - **Sources:** Anthropic Tutorial; OpenAI Guide; Google Guide.

- **Use structured outputs when exact machine-readable format matters.**
  - **Why it works:** Schema-constrained JSON is more reliable than hoping the model will freestyle valid structure.
  - **Sources:** OpenAI Guide.

- **Use role prompting when domain lens, tone, or reasoning style matters.**
  - **Why it works:** Giving the model an identity like “logic tutor,” “strict classifier,” or “senior architect” can improve both style and task performance.
  - **Sources:** Anthropic Tutorial; Google Guide.

- **Include the intended audience when style matters.**
  - **Why it works:** “Explain like a lawyer” and “explain to a beginner founder” are different tasks. Audience information sharpens tone, depth, and vocabulary.
  - **Sources:** Anthropic Tutorial.

- **Give the model a valid fallback path.**
  - **Why it works:** Telling the model when it may abstain, say it is unsure, or request clarification reduces hallucinated confidence.
  - **Sources:** Anthropic Tutorial.

- **Ask for evidence extraction before the final answer when accuracy is fragile.**
  - **Why it works:** Forcing the model to surface relevant evidence first can reduce distractor-driven mistakes and make failures easier to debug.
  - **Sources:** Anthropic Tutorial; ReAct; RAG.

- **Assume tiny wording changes can matter.**
  - **Why it works:** Prompt behavior can shift on seemingly small edits. Good prompts should be scrubbed for ambiguity, stray tokens, and accidental cues.
  - **Sources:** Anthropic Tutorial; Chain-of-Thought; Prompt Injection.

---

## Chain-of-thought and reasoning findings

- **For multi-step tasks, explicitly ask for planning or step-by-step reasoning.**
  - **Why it works:** Multi-step reasoning prompts improve performance on arithmetic, symbolic, and commonsense tasks that standard answer-only prompting often misses.
  - **Sources:** Chain-of-Thought; Zero-shot CoT; Anthropic Tutorial; Google Guide.

- **Use a zero-shot reasoning trigger as a baseline before building elaborate few-shot reasoning prompts.**
  - **Why it works:** A simple instruction like “think step by step” is a strong baseline and can deliver large gains without example engineering.
  - **Sources:** Zero-shot CoT; Automatic Prompt Engineer.

- **For strict answer formatting, separate reasoning from answer extraction.** *(Pipeline-level)*
  - **Why it works:** One call can produce reasoning, and a second call can convert that reasoning into a tightly formatted final answer. This reduces format drift.
  - **Sources:** Zero-shot CoT.

- **Do not force chain-of-thought on every task.**
  - **Why it works:** Chain-of-thought is not uniformly beneficial. It can hurt smaller models and is unnecessary overhead on tasks that do not require explicit decomposition.
  - **Sources:** Chain-of-Thought.

- **Treat visible reasoning prompts as most useful for tasks that genuinely require multi-step decomposition.**
  - **Why it works:** The strongest gains in the literature are on reasoning-heavy tasks, not on every prompt category.
  - **Sources:** Chain-of-Thought; Zero-shot CoT.

- **Use self-critique or explicit validation before returning the final answer.**
  - **Why it works:** Asking the model to review its output against the original constraints catches missed requirements and shallow misreads.
  - **Sources:** Google Guide; Anthropic Tutorial.

- **Model family matters: GPT-style models usually benefit from more explicit instructions, while reasoning models often do better with high-level goals.**
  - **Why it works:** Different model classes respond differently to micromanagement versus delegation.
  - **Sources:** OpenAI Guide.

---

## Few-shot selection criteria

- **Use few-shot examples when behavior or formatting is hard to specify cleanly in prose.**
  - **Why it works:** Examples often teach the model faster than long verbal instructions, especially for tone, labeling style, and output structure.
  - **Sources:** Anthropic Tutorial; OpenAI Guide; GPT-3 / Few-Shot Learners.

- **Choose examples that match the target output format as closely as possible.**
  - **Why it works:** Models often infer the repeated output pattern from examples more strongly than the underlying task description.
  - **Sources:** Zero-shot CoT; OpenAI Guide; Anthropic Tutorial.

- **Prefer diverse examples over near-duplicates.**
  - **Why it works:** Diversity helps the model generalize the intended rule instead of overfitting to one narrow pattern.
  - **Sources:** OpenAI Guide; Automatic Prompt Engineer.

- **Include common edge cases in the examples.**
  - **Why it works:** If a failure mode matters in production, showing it in-example is often the fastest way to suppress it.
  - **Sources:** Anthropic Tutorial; OpenAI Guide.

- **Use examples that demonstrate both behavior and formatting.**
  - **Why it works:** Few-shot prompts are especially good at teaching exact response shape, not just semantic intent.
  - **Sources:** Anthropic Tutorial; OpenAI Guide.

- **Test example order, especially for classification-style tasks.**
  - **Why it works:** Example permutation can change results materially on some tasks. Do not assume one ordering is representative.
  - **Sources:** Chain-of-Thought; GPT-3 / Few-Shot Learners.

- **Do not assume you need examples from the exact same data distribution.**
  - **Why it works:** Reasoning examples can still help even when they come from different datasets, as long as they teach the right pattern.
  - **Sources:** Chain-of-Thought.

- **Do not assume one annotator style is uniquely correct.**
  - **Why it works:** Chain-of-thought prompts were robust across annotators and writing styles, so optimize for clarity and usefulness, not one canonical voice.
  - **Sources:** Chain-of-Thought.

- **Use chain-of-thought exemplars instead of answer-only exemplars on hard reasoning tasks.**
  - **Why it works:** Reasoning demonstrations outperform standard answer-only few-shot prompts on multi-step tasks.
  - **Sources:** Chain-of-Thought.

- **More examples help until context or diminishing returns kick in, but better examples beat more mediocre ones.**
  - **Why it works:** Increasing standard examples alone did not close the gap to chain-of-thought prompts, and example count eventually hits context and marginal-value limits.
  - **Sources:** Chain-of-Thought; GPT-3 / Few-Shot Learners.

- **Larger models tend to use in-context examples more effectively than smaller ones.**
  - **Why it works:** Few-shot gains scale with model size; bigger models are generally better at pattern pickup from context.
  - **Sources:** GPT-3 / Few-Shot Learners.

- **Search over multiple candidate instructions instead of trusting your first decent wording.** *(Optimization-level)*
  - **Why it works:** Prompt quality varies widely, and automatic search reliably finds better instructions than greedy manual drafting.
  - **Sources:** Automatic Prompt Engineer.

- **When automatically searching prompts, sample many candidates, but expect diminishing returns.** *(Optimization-level)*
  - **Why it works:** More candidate instructions increase the chance of finding a strong prompt, but gains flatten after enough samples.
  - **Sources:** Automatic Prompt Engineer.

- **Prepending a better instruction to an existing few-shot prompt can improve the whole prompt without changing the examples.**
  - **Why it works:** Instruction quality compounds with example quality.
  - **Sources:** Automatic Prompt Engineer.

---

## Retrieval, grounding, and long-context management

- **Use retrieval when factual freshness, provenance, or updateability matter.**
  - **Why it works:** Retrieved evidence makes it easier to inspect sources and update knowledge without changing model weights.
  - **Sources:** RAG; DAIR Guide; OpenAI Guide.

- **Tell the model to answer from provided evidence, not from memory, when grounding matters.**
  - **Why it works:** This reduces unsupported fabrication and makes failures easier to audit.
  - **Sources:** RAG; Anthropic Tutorial; ReAct.

- **Do not dump more retrieved documents than the model can effectively use.**
  - **Why it works:** Long-context performance saturates early; more documents often add cost and distraction with little gain.
  - **Sources:** Lost in the Middle; RAG.

- **Place the most relevant evidence at the beginning or end of the context.**
  - **Why it works:** Models show primacy and recency effects and perform worst when the key evidence sits in the middle.
  - **Sources:** Lost in the Middle.

- **Do not bury critical instructions or facts in the middle of long prompts.**
  - **Why it works:** Even long-context models degrade sharply on mid-context retrieval.
  - **Sources:** Lost in the Middle.

- **Prefer short, relevant chunks over large raw document dumps.**
  - **Why it works:** Cleaner chunks reduce distraction and make the relevant evidence easier to recover.
  - **Sources:** RAG; Lost in the Middle.

- **Use retrieval ordering as a tunable prompt variable, not a fixed afterthought.**
  - **Why it works:** Context position itself changes performance, so ranking and placement are part of prompt design.
  - **Sources:** Lost in the Middle.

- **For factual QA, retrieved evidence often improves specificity, diversity, and factuality versus parametric-only prompting.**
  - **Why it works:** The model can ground generation in retrieved passages instead of relying only on frozen internal knowledge.
  - **Sources:** RAG.

---

## Agentic and multi-call techniques that emerged from the literature

- **When pure reasoning hallucinates, interleave reasoning with actions.** *(Pipeline-level)*
  - **Why it works:** ReAct-style prompting lets the model reason, retrieve new information, update its plan, and correct itself instead of free-associating through an unsupported chain-of-thought.
  - **Sources:** ReAct.

- **Use reasoning traces to drive actions, not as an end in themselves.** *(Pipeline-level)*
  - **Why it works:** In ReAct, reasoning is useful because it helps decide what to do next and how to interpret results from tools or environments.
  - **Sources:** ReAct.

- **For planning-heavy tasks, generate multiple candidate thoughts and evaluate them instead of committing to a single chain early.** *(Pipeline-level)*
  - **Why it works:** Tree-of-Thoughts shows that branching search, self-evaluation, and backtracking outperform single-path greedy reasoning on search-like problems.
  - **Sources:** Tree of Thoughts.

- **Allow backtracking when early decisions are high-leverage.** *(Pipeline-level)*
  - **Why it works:** Greedy one-pass decoding performs badly when the task requires exploration or correction of earlier choices.
  - **Sources:** Tree of Thoughts.

- **Use self-consistency on hard reasoning tasks.** *(Pipeline-level)*
  - **Why it works:** Sampling multiple reasoning paths and selecting the most consistent final answer beats greedy single-path chain-of-thought.
  - **Sources:** Self-Consistency.

- **Modularize complex agent prompts into smaller subprompts with explicit signatures or step contracts.** *(Pipeline-level)*
  - **Why it works:** Multi-stage systems are brittle when implemented as one giant hand-written prompt. Modular steps are easier to optimize, test, and swap.
  - **Sources:** DSPy; ReAct; Tree of Thoughts.

- **Optimize agent prompts against metrics, not aesthetics.** *(Pipeline-level)*
  - **Why it works:** DSPy-style compilation shows that prompt quality improves when demonstrations and instructions are selected against a validation metric rather than manual taste.
  - **Sources:** DSPy; Automatic Prompt Engineer.

- **Prefer declarative task specs plus optimized demonstrations over giant handcrafted prompt chains as pipelines grow.** *(Pipeline-level)*
  - **Why it works:** Hard-coded prompt templates are brittle across inputs, domains, and model changes.
  - **Sources:** DSPy.

---

## Evaluation criteria

- **Define success criteria before prompt iteration.**
  - **Why it works:** Prompt engineering is much more efficient when “good” is measurable beforehand.
  - **Sources:** Anthropic Overview; Anthropic Tutorial.

- **Build empirical evals instead of judging prompts by vibes.**
  - **Why it works:** Prompt behavior is noisy and wording-sensitive. Evals let you compare variants and detect regressions.
  - **Sources:** OpenAI Guide; Anthropic Overview; Automatic Prompt Engineer; DSPy.

- **Use held-out validation data to choose prompt variants.**
  - **Why it works:** Without a held-out set, you risk optimizing to anecdotes or overfitting to your own manual tests.
  - **Sources:** Automatic Prompt Engineer; DSPy.

- **Measure format adherence separately from task correctness.**
  - **Why it works:** A prompt can be semantically right but operationally unusable if it breaks the required schema or wrapper tags.
  - **Sources:** Anthropic Tutorial; OpenAI Guide.

- **Measure grounding separately from answer quality.**
  - **Why it works:** A fluent answer can still be unsupported. Use metrics that reward both correctness and evidence alignment when grounding matters.
  - **Sources:** DSPy; RAG; ReAct.

- **For long-context prompts, evaluate sensitivity to evidence position.**
  - **Why it works:** Average accuracy hides a major failure mode: some models look strong until the relevant document moves into the middle.
  - **Sources:** Lost in the Middle.

- **Evaluate across paraphrases, example orders, and prompt variants.**
  - **Why it works:** Robust prompts should not collapse when the wording or example order shifts.
  - **Sources:** Chain-of-Thought; Automatic Prompt Engineer.

- **Pin model versions in production.**
  - **Why it works:** Prompt behavior can change across snapshots, so reproducibility requires version pinning.
  - **Sources:** OpenAI Guide.

- **Track cost and latency separately from prompt quality.**
  - **Why it works:** Some failures are better solved by a different model, smaller context, or external tool rather than more prompt edits.
  - **Sources:** Anthropic Overview; OpenAI Guide.

---

## Robustness and security

- **Assume any user-controlled text can contain adversarial instructions.**
  - **Why it works:** Simple string substitution is enough for goal hijacking and prompt leaking.
  - **Sources:** Prompt Injection.

- **Do not treat delimiters alone as a security control.**
  - **Why it works:** Delimiters change attack behavior, but they are not a sufficient defense. In the cited attack paper, delimiters actually improved attack success in some settings.
  - **Sources:** Prompt Injection.

- **Keep sensitive instructions and secrets out of the same free-form generation channel as user content when possible.**
  - **Why it works:** Prompt leaking attacks target exactly this setup.
  - **Sources:** Prompt Injection.

- **If you must template user input into a prompt, keep important instructions outside that user-controlled region and consider reinforcing the task after the inserted input.**
  - **Why it works:** Prompts with text after `{user_input}` were harder to attack than prompts that ended with user input.
  - **Sources:** Prompt Injection.

- **Use stop sequences, output caps, and post-processing when the task allows.**
  - **Why it works:** Constraining how much the model can continue reduces opportunities for attack spillover and prompt leakage.
  - **Sources:** Prompt Injection; Anthropic Tutorial.

- **Moderate or validate outputs when misuse cost is high.**
  - **Why it works:** Prompting alone is not a guaranteed safety boundary.
  - **Sources:** Prompt Injection; DAIR Guide.

---

## Common anti-patterns

- **Vague prompts that expect the model to infer omitted constraints.**
  - **Why it hurts:** The model fills the gaps with its own defaults, which often do not match application requirements.
  - **Sources:** Anthropic Tutorial; Google Guide.

- **Mixing instructions and user data in one undelimited blob.**
  - **Why it hurts:** The model may rewrite, classify, or obey the wrong text span.
  - **Sources:** Anthropic Tutorial; Prompt Injection.

- **Relying on one successful run as proof the prompt is good.**
  - **Why it hurts:** Prompt performance is stochastic and sensitive to wording, ordering, and model version.
  - **Sources:** OpenAI Guide; Automatic Prompt Engineer; Chain-of-Thought.

- **Burying the key fact in the middle of long context.**
  - **Why it hurts:** This is one of the clearest and most reproducible long-context failure modes.
  - **Sources:** Lost in the Middle.

- **Stuffing more retrieved documents into the prompt instead of improving retrieval quality and ordering.**
  - **Why it hurts:** Extra context quickly stops helping and can make the model less effective.
  - **Sources:** Lost in the Middle; RAG.

- **Assuming chain-of-thought is a universal fix.**
  - **Why it hurts:** It can add latency, format drift, and even reduce accuracy on some models or tasks.
  - **Sources:** Chain-of-Thought.

- **Using examples with the wrong answer format.**
  - **Why it hurts:** The model often copies the format pattern more strongly than the intended task.
  - **Sources:** Zero-shot CoT; Anthropic Tutorial.

- **Hard-coding giant prompt templates for multi-stage systems.**
  - **Why it hurts:** These prompts are brittle, hard to debug, and hard to retune across models or domains.
  - **Sources:** DSPy.

- **Overfitting the prompt to one model family or snapshot.**
  - **Why it hurts:** Prompt behavior does not transfer perfectly across architectures or versions.
  - **Sources:** OpenAI Guide; Chain-of-Thought.

- **Assuming the model will “think” helpfully without any explicit planning or validation scaffold.**
  - **Why it hurts:** If the task needs decomposition, hidden assumptions often stay hidden until the final answer is already wrong.
  - **Sources:** Anthropic Tutorial; Google Guide; Zero-shot CoT.

---

## Multimodal-specific principles

- **For image/video/file tasks, be unusually explicit about what to extract and how to format it.**
  - **Why it works:** Multimodal prompts are especially prone to generic answers unless the task and output fields are clearly specified.
  - **Sources:** Gemini Multimodal Guide; Google Guide.

- **If multimodal outputs are too generic, first ask the model to describe the input before doing the higher-level task.**
  - **Why it works:** This surfaces whether the model actually noticed the salient details before you trust its reasoning.
  - **Sources:** Gemini Multimodal Guide.

- **Use few-shot multimodal examples when style or extraction format matters.**
  - **Why it works:** Cross-modal examples teach the mapping from input type to desired response structure.
  - **Sources:** Gemini Multimodal Guide.

- **If multimodal prompts hallucinate, reduce temperature and shorten the requested output.**
  - **Why it works:** Lower-variance generation is less likely to invent extra unsupported detail.
  - **Sources:** Gemini Multimodal Guide.

---

## What emerged most strongly across the literature

If you only keep ten defaults when writing agent system prompts, keep these:

1. Separate **role / constraints / context / task / output format** into explicit sections.
2. Keep durable application rules in the **system/developer** layer.
3. Wrap every user-controlled variable in explicit tags or fields.
4. Be painfully explicit about **what to output** and **what not to output**.
5. Use **few-shot examples** when format or behavior is hard to verbalize.
6. For multi-step tasks, add **planning / step-by-step reasoning / validation**.
7. For factual tasks, use **retrieval + grounding**, not memory-only prompting.
8. Put the most relevant evidence at the **start or end** of long context.
9. Evaluate prompt changes with **held-out tests and fixed model versions**.
10. Treat prompting as one lever in a larger system: for hard tasks, use **multi-call orchestration**, tools, retrieval, and safety checks.

---

## Source key

- **Anthropic Overview** — `anthropic_guide.txt`
- **Anthropic Tutorial** — `anthropic_tutorial.txt`
- **Automatic Prompt Engineer** — *Large Language Models are Human-Level Prompt Engineers*
- **Chain-of-Thought** — *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*
- **DAIR Guide** — `dair_guide.txt`
- **DSPy** — *DSPY: Compiling Declarative Language Model Calls into Self-Improving Pipelines*
- **Gemini Multimodal Guide** — `gemini_multimodal.txt`
- **Google Guide** — `google_guide.txt`
- **GPT-3 / Few-Shot Learners** — *Language Models are Few-Shot Learners*
- **Lost in the Middle** — *Lost in the Middle: How Language Models Use Long Contexts*
- **OpenAI Guide** — `openai_guide.txt`
- **Prompt Injection** — *Ignore Previous Prompt: Attack Techniques For Language Models*
- **RAG** — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- **ReAct** — *ReAct: Synergizing Reasoning and Acting in Language Models*
- **Self-Consistency** — *Self-Consistency Improves Chain of Thought Reasoning in Language Models*
- **Tree of Thoughts** — *Tree of Thoughts: Deliberate Problem Solving with Large Language Models*
- **Zero-shot CoT** — *Large Language Models are Zero-Shot Reasoners*
