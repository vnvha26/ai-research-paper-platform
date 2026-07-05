def analyze_paper_fraud(connections, paper_id):
    from services.f2_citation_fraud.f2_fraud_service import analyze_paper_fraud as analyze

    return analyze(connections, paper_id)


__all__ = ["analyze_paper_fraud"]
