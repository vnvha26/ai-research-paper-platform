try:
    import nltk
    from nltk.corpus import stopwords
except ModuleNotFoundError:
    nltk = None
    stopwords = None


FALLBACK_ENGLISH_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "with",
}

if nltk is not None and stopwords is not None:
    try:
        nltk.data.find("corpora/stopwords")
        ENGLISH_STOP_WORDS = set(stopwords.words("english"))
    except LookupError:
        ENGLISH_STOP_WORDS = FALLBACK_ENGLISH_STOP_WORDS
else:
    ENGLISH_STOP_WORDS = FALLBACK_ENGLISH_STOP_WORDS

ACADEMIC_STOP_WORDS = {
    "using", "based", "proposed", "novel", "new", "study", "analysis",
    "paper", "method", "approach", "toward", "towards", "via", "within",
    "across", "without", "framework",
}
ALL_STOP_WORDS = ENGLISH_STOP_WORDS.union(ACADEMIC_STOP_WORDS)

SECTION_WEIGHTS = {
    "core": 0.55,
    "intro": 0.15,
    "method": 0.20,
    "conclusion": 0.10,
}

ALPHA_COSINE = 0.60
ALPHA_BM25 = 0.40
ACTIVE_CONFIG_NAME = "core_heavy"
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K_LEXICAL = 500
TOP_K_SEMANTIC = 300
MULTIVEC_COLLECTION = "papers_multivec"
