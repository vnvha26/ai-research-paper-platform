"""F3: graph-based research paper recommendation."""

from services.f3_recommender.inference import recommend_related_papers


def train():
    """Load training dependencies only when an F3 training run is requested."""
    from services.f3_recommender.trainer import train as train_f3

    return train_f3()


__all__ = ["recommend_related_papers", "train"]
