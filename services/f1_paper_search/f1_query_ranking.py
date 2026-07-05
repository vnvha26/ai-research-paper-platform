import math
import re
from collections import defaultdict

from services.f1_paper_search.f1_search_config import ALL_STOP_WORDS


def extract_keywords(user_text: str) -> list[str]:
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", user_text.lower())
    words = clean.split()
    return [word for word in words if word not in ALL_STOP_WORDS and len(word) > 1]


def build_tsquery(keywords: list[str], operator: str = "AND") -> str:
    if not keywords:
        return ""
    separator = " & " if operator == "AND" else " | "
    return separator.join(keywords)


class SimpleBM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> list[str]:
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", (text or "").lower())
        return [word for word in clean.split() if word not in ALL_STOP_WORDS and len(word) > 1]

    def score_all(self, query_keywords: list[str], corpus: list[dict]) -> dict[str, float]:
        if not query_keywords or not corpus:
            return {}

        documents = {}
        for paper in corpus:
            text = " ".join(filter(None, [
                paper.get("abstract", ""),
                paper.get("intro", ""),
                paper.get("method", ""),
                paper.get("conclusion", ""),
            ]))
            documents[paper["paper_id"]] = self._tokenize(text)

        document_count = len(documents)
        average_length = sum(len(tokens) for tokens in documents.values()) / document_count
        if average_length == 0:
            return {paper_id: 0.0 for paper_id in documents}
        inverse_document_frequency = {}
        for keyword in query_keywords:
            frequency = sum(1 for tokens in documents.values() if keyword in tokens)
            inverse_document_frequency[keyword] = math.log(
                (document_count - frequency + 0.5) / (frequency + 0.5) + 1
            )

        scores = {}
        for paper_id, tokens in documents.items():
            term_frequencies = defaultdict(int)
            for token in tokens:
                term_frequencies[token] += 1

            score = 0.0
            for keyword in query_keywords:
                term_frequency = term_frequencies.get(keyword, 0)
                numerator = term_frequency * (self.k1 + 1)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * len(tokens) / average_length
                )
                if denominator:
                    score += inverse_document_frequency.get(keyword, 0) * numerator / denominator
            scores[paper_id] = score

        maximum = max(scores.values()) if scores else 0.0
        if maximum > 0:
            return {paper_id: score / maximum for paper_id, score in scores.items()}
        return scores
