from src.config import LLM_PROVIDER, OLLAMA_MODEL, GROQ_MODEL, JUDGE_MODEL


def _groq(model, temperature):
    from langchain_groq import ChatGroq
    # gpt-oss models "think" before answering; low effort keeps them fast
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    # max_retries: the free tier rate-limits bursts, so back off and retry instead of failing
    return ChatGroq(model=model, temperature=temperature, max_retries=8, **extra)


def get_llm(temperature=0):

    if LLM_PROVIDER == "groq":
        return _groq(GROQ_MODEL, temperature)

    from langchain_ollama import ChatOllama
    # reasoning=False turns off qwen's "thinking" mode, which is much slower
    return ChatOllama(model=OLLAMA_MODEL, temperature=temperature, reasoning=False)


def get_judge():
    """LLM used to grade answers in the evaluation."""
    if LLM_PROVIDER == "groq":
        return _groq(JUDGE_MODEL, 0)
    return get_llm()
