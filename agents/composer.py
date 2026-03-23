import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import LLMRouter

logger = logging.getLogger("LLMRouter.Composer")

# Keywords that signal a multi-domain query
COMPOSITION_SIGNALS = [
    " and ", " also ", " then ", " plus ",
    " as well as ", " along with ", " additionally ",
    " furthermore ", " both ", " with "
]

# Presentation order — explanation before implementation before analysis
PRESENTATION_ORDER = ["general", "medical", "legal", "math", "code"]

# Domain-specific task extraction patterns
DOMAIN_EXTRACTION = {
    "code": {
        "prefix": "Write clean, well-commented code only. Task: ",
        "keywords": ["implement", "write", "code", "program", 
                     "function", "class", "algorithm", "python", 
                     "java", "javascript", "c++", "script"]
    },
    "math": {
        "prefix": "Provide mathematical analysis only (no code). Topic: ",
        "keywords": ["solve", "integral", "derivative", "calculate",
                     "proof", "equation", "complexity", "big o",
                     "differentiate", "integrate", "eigenvalue"]
    },
    "medical": {
        "prefix": "Provide accurate medical information only. Question: ",
        "keywords": ["symptoms", "disease", "diabetes", "diagnosis",
                     "treatment", "medicine", "health", "clinical",
                     "warning signs", "cancer", "infection"]
    },
    "legal": {
        "prefix": "Provide legal information only. Question: ",
        "keywords": ["lawsuit", "criminal", "civil", "law", "legal",
                     "rights", "prosecution", "contract", "arrest",
                     "habeas", "statute"]
    },
    "general": {
        "prefix": "Provide a clear conceptual explanation only. Topic: ",
        "keywords": ["explain", "what is", "describe", "history",
                     "philosophy", "concept", "meaning", "difference between"]
    }
}

DOMAIN_FOCUSED_PROMPTS = {
    "general": (
        "Provide a clear conceptual explanation only (no code). "
        "Explain the concept, how it works, and when to use it. Topic: "
    ),
    "code": (
        "Write clean, well-commented code only (no explanation of concept, "
        "no complexity analysis). Just the implementation. Task: "
    ),
    "math": (
        "Provide mathematical complexity analysis only (no code, no general explanation). "
        "Cover time complexity (best/average/worst case) and space complexity "
        "with Big-O notation. Topic: "
    ),
    "medical": (
        "Provide accurate medical information only. Be specific and clinical. Topic: "
    ),
    "legal": (
        "Provide relevant legal information only. Be precise. Topic: "
    )
}


def detect_domains(user_prompt: str) -> list:
    """
    Detects which domains are present in a multi-part query.
    Returns list of detected domains in order of appearance.
    """
    # Reuse signal lists for detection
    prompt_lower = user_prompt.lower()
    detected = []

    # Map signal lists for consistency
    domain_signals = {
        "code": ["code", "implement", "python", "java", "script", "c++"],
        "math": ["complexity", "calculate", "solve", "proof", "big o"],
        "medical": ["symptoms", "disease", "diagnosis", "treatment"],
        "legal": ["legal", "law", "rights", "contract"],
        "general": ["explain", "what is", "describe", "concept"]
    }

    for domain, keywords in domain_signals.items():
        if any(kw in prompt_lower for kw in keywords):
            if domain not in detected:
                detected.append(domain)

    return detected


def is_multi_domain(user_prompt: str) -> bool:
    prompt_lower = user_prompt.lower()

    has_signal = any(signal in prompt_lower for signal in COMPOSITION_SIGNALS)
    if not has_signal:
        return False

    detected = detect_domains(user_prompt)
    specific_domains = [d for d in detected if d != "general"]

    # Also check if prompt looks like multiple separate questions
    # (comma-separated long clauses are a strong signal)
    comma_clauses = [c.strip() for c in user_prompt.split(",") if len(c.strip()) > 20]
    has_multiple_clauses = len(comma_clauses) >= 2

    if len(specific_domains) >= 2 and has_multiple_clauses:
        return True

    if len(specific_domains) >= 2:
        return True

    return False


def decompose_query(user_prompt: str) -> list:
    """
    Breaks a multi-domain query into sub-tasks.
    Returns list of {"domain": str, "sub_prompt": str}
    """
    detected_domains = detect_domains(user_prompt)
    
    # Sort by presentation order — explanation before implementation before analysis
    detected_domains.sort(
        key=lambda d: PRESENTATION_ORDER.index(d) 
        if d in PRESENTATION_ORDER else 99
    )
    
    sub_tasks = []
    for domain in detected_domains:
        sub_prompt = _build_sub_prompt(domain, user_prompt)
        sub_tasks.append({"domain": domain, "sub_prompt": sub_prompt})
    
    return sub_tasks


def _extract_relevant_subquery(domain: str, original_prompt: str) -> str:
    """
    Extracts the sentence or clause most relevant to the given domain
    from a multi-part prompt. Falls back to the full prompt if extraction fails.
    """
    # Split on common multi-query separators
    separators = [", and ", " and ", ", ", ". ", "? ", "! "]
    sentences = [original_prompt]
    for sep in separators:
        new_sentences = []
        for s in sentences:
            new_sentences.extend(s.split(sep))
        sentences = new_sentences

    sentences = [s.strip().rstrip("?,. ") for s in sentences if len(s.strip()) > 10]

    if not sentences:
        return original_prompt

    keywords = DOMAIN_EXTRACTION.get(domain, {}).get("keywords", [])
    
    # Score each sentence by how many domain keywords it contains
    best_sentence = None
    best_score = 0

    for sentence in sentences:
        sentence_lower = sentence.lower()
        score = sum(1 for kw in keywords if kw in sentence_lower)
        if score > best_score:
            best_score = score
            best_sentence = sentence

    # If no sentence matched, fall back to full prompt
    chosen = best_sentence if best_sentence and best_score > 0 else original_prompt
    prefix = DOMAIN_EXTRACTION.get(domain, {}).get("prefix", "Answer: ")
    return f"{prefix}{chosen}"


def _build_sub_prompt(domain: str, original_prompt: str) -> str:
    return _extract_relevant_subquery(domain, original_prompt)


def merge_responses(sub_results: list) -> str:
    """
    Merges multiple specialist responses into a single coherent output.
    Each section is clearly labeled with its domain.
    """
    if not sub_results:
        return ""

    if len(sub_results) == 1:
        return sub_results[0]["response"]

    domain_labels = {
        "general": "📖 Concept & Explanation",
        "code": "💻 Implementation",
        "math": "📐 Complexity Analysis",
        "medical": "🏥 Medical Information",
        "legal": "⚖️ Legal Information"
    }

    sections = []
    for result in sub_results:
        domain = result["domain"]
        label = domain_labels.get(domain, f"🔹 {domain.title()}")
        sections.append(f"{label}\n{'─' * 40}\n{result['response']}")

    return "\n\n".join(sections)
