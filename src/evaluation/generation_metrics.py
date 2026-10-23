"""
LLM-as-judge metrics, implemented from the RAGAS definitions.
(The ragas package itself clashes with this project's LangChain version.)

faithfulness      : share of the answer's claims that the retrieved context supports.
                    1 - faithfulness is the hallucination rate.
answer_relevancy  : does the answer address the question? The judge writes questions
                    the answer would fit; we compare them to the real question with embeddings.
context_recall    : share of the reference answer's facts that appear in the retrieved context.
                    Low recall = retrieval missed something the answer needed.
"""
import json
import re

import numpy as np

FAITHFULNESS_PROMPT = """Break the ANSWER into short standalone factual claims.
For each claim, decide if the CONTEXT directly supports it.

Return ONLY JSON: {{"claims": [{{"claim": "...", "supported": true}}]}}
If the answer only says it could not find the information, return {{"claims": []}}.

CONTEXT:
{context}

ANSWER:
{answer}"""

RELEVANCY_PROMPT = """Write 3 different questions that the following answer would be a good response to.
Return ONLY JSON: {{"questions": ["...", "...", "..."]}}

ANSWER:
{answer}"""

RECALL_PROMPT = """Split the REFERENCE answer into its individual facts.
For each fact, decide if it can be found in the CONTEXT.

Return ONLY JSON: {{"facts": [{{"fact": "...", "in_context": true}}]}}

CONTEXT:
{context}

REFERENCE:
{reference}"""


def _json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(match.group(0)) if match else {}


def faithfulness(judge, answer, contexts):
    claims = _json(judge.invoke(FAITHFULNESS_PROMPT.format(
        context="\n\n".join(contexts), answer=answer)).content).get("claims", [])

    if not claims:
        # "I could not find it" makes no claims, so it cannot hallucinate
        return 1.0

    return sum(bool(c.get("supported")) for c in claims) / len(claims)


def answer_relevancy(judge, embeddings, question, answer):
    generated = _json(judge.invoke(RELEVANCY_PROMPT.format(answer=answer)).content).get("questions", [])

    if not generated:
        return 0.0

    q = np.array(embeddings.embed_query(question))
    g = np.array(embeddings.embed_documents(generated))

    cosine = g @ q / (np.linalg.norm(g, axis=1) * np.linalg.norm(q))
    return float(np.mean(cosine))


def context_recall(judge, reference, contexts):
    facts = _json(judge.invoke(RECALL_PROMPT.format(
        context="\n\n".join(contexts), reference=reference)).content).get("facts", [])

    if not facts:
        return 0.0

    return sum(bool(f.get("in_context")) for f in facts) / len(facts)
