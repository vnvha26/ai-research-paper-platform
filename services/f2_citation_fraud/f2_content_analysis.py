import json
import math
from collections import OrderedDict

from services.f2_citation_fraud.f2_fraud_config import MAX_BM25_QUERY_KEYWORDS, MAX_SECTION_CHARS
from services.f1_paper_search.f1_search_config import SECTION_WEIGHTS
from services.f1_paper_search.f1_query_ranking import SimpleBM25, extract_keywords


def normalize_author_ids(authors_raw):
    if isinstance(authors_raw, str):
        try:
            authors_raw = json.loads(authors_raw)
        except json.JSONDecodeError:
            authors_raw = [authors_raw]

    author_ids = set()
    author_names = {}
    for author in authors_raw or []:
        if isinstance(author, dict):
            author_id = (
                author.get("author_id")
                or author.get("authorId")
                or author.get("id")
                or author.get("name")
            )
            name = author.get("name") or str(author_id or "")
        else:
            author_id = author
            name = str(author)

        if author_id:
            author_id = str(author_id).strip()
            author_ids.add(author_id)
            author_names[author_id] = str(name).strip()
    return author_ids, author_names


def section_texts(data):
    return {
        "core": data.get("title_abs", ""),
        "intro": data.get("intro", ""),
        "method": data.get("method", ""),
        "conclusion": data.get("conclusion", ""),
    }


def encode_sections(nlp_model, data):
    vectors = {}
    for section, text in section_texts(data).items():
        text = (text or "").strip()
        if text:
            vectors[section] = nlp_model.encode([text[:MAX_SECTION_CHARS]])[0]
    return vectors


def weighted_section_cosine(source_vectors, cited_vectors):
    scores = {}
    for section in SECTION_WEIGHTS:
        if section in source_vectors and section in cited_vectors:
            source = source_vectors[section]
            cited = cited_vectors[section]
            if hasattr(source, "tolist"):
                source = source.tolist()
            if hasattr(cited, "tolist"):
                cited = cited.tolist()
            dot_product = sum(float(left) * float(right) for left, right in zip(source, cited))
            source_norm = math.sqrt(sum(float(value) ** 2 for value in source))
            cited_norm = math.sqrt(sum(float(value) ** 2 for value in cited))
            denominator = source_norm * cited_norm
            score = dot_product / denominator if denominator else 0.0
            scores[section] = max(0.0, min(1.0, score))

    total_weight = sum(SECTION_WEIGHTS[section] for section in scores)
    if total_weight <= 0:
        return 0.0, {}
    weighted_score = sum(
        SECTION_WEIGHTS[section] * scores[section]
        for section in scores
    ) / total_weight
    return weighted_score, scores


def build_bm25_scores(source_data, cited_rows):
    if len(cited_rows) < 2:
        return {}

    source_text = " ".join(section_texts(source_data).values())
    keywords = list(OrderedDict.fromkeys(extract_keywords(source_text)))
    corpus = [
        {
            "paper_id": paper_id,
            "abstract": data.get("abstract", ""),
            "intro": data.get("intro", ""),
            "method": data.get("method", ""),
            "conclusion": data.get("conclusion", ""),
        }
        for paper_id, data in cited_rows.items()
    ]
    return SimpleBM25().score_all(keywords[:MAX_BM25_QUERY_KEYWORDS], corpus)
