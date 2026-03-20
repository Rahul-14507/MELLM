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

# Map domain to the specific task keywords that belong to it
DOMAIN_TASK_PATTERNS = {
    "general": ["explain", "what is", "describe", "overview", "concept"],
    "code": ["give", "write", "implement", "code", "java", "python", "c++"],
    "math": ["analyse", "analyze", "complexity", "time complexity", 
             "space complexity", "big o", "proof"],
    "medical": ["symptoms", "diagnosis", "treatment", "causes"],
    "legal": ["legal", "law", "rights", "liability"]
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
    """
    Returns True if the query appears to span multiple domains.
    Requires both a composition signal word AND multiple domain keywords.
    """
    prompt_lower = user_prompt.lower()
    has_signal = any(signal in prompt_lower for signal in COMPOSITION_SIGNALS)
    if not has_signal:
        return False

    detected = detect_domains(user_prompt)
    return len(detected) >= 2


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


def _build_sub_prompt(domain: str, original_prompt: str) -> str:
    """Builds a tightly focused sub-prompt for a specific domain."""
    # Extract the core topic by removing domain-signal words
    topic = original_prompt
    for signals in DOMAIN_TASK_PATTERNS.values():
        for signal in signals:
            topic = topic.lower().replace(signal, "").strip()
    
    # Clean up conjunctions left over
    for word in [" and ", " also ", " plus ", " then ", "  "]:
        topic = topic.replace(word, " ").strip()
    
    prefix = DOMAIN_FOCUSED_PROMPTS.get(
        domain, 
        "Answer the following concisely: "
    )
    return f"{prefix}{original_prompt}"


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
