"""Serializable vote summaries and independently configurable decisions."""


def decide(votes, *, min_reads=3, min_score=0, min_mean=.7, min_margin=0, min_share=.7):
    best = max(votes, key=lambda key: votes[key]["score"], default="")
    winner = votes.get(best, {"score": 0, "count": 0, "confidence_sum": 0})
    score, count = winner["score"], winner["count"]
    runner = max((v["score"] for key, v in votes.items() if key != best), default=0)
    total = sum(v["score"] for v in votes.values())
    mean = winner["confidence_sum"] / count if count else 0
    share = score / total if total else 0
    checks = {"reads": count >= min_reads, "score": score >= min_score, "mean": mean >= min_mean,
              "margin": score - runner >= min_margin, "share": share >= min_share}
    return {
        "best": best, "reads": count, "score": round(score, 4), "mean": round(mean, 4),
        "margin": round(score - runner, 4), "share": round(share, 4),
        "confirmed": bool(best) and all(checks.values()),
        "unmet": [key for key, passed in checks.items() if not passed],
    }


def merge_votes(left, right):
    combined = {name: dict(values) for name, values in left.items()}
    for name, values in right.items():
        total = combined.setdefault(name, {"score": 0, "count": 0, "confidence_sum": 0})
        for field in ("score", "count", "confidence_sum"):
            total[field] += values[field]
    return combined


def classification_decisions(summaries, config):
    return {
        name: decide(summaries.get(name, {}),
                     min_reads=getattr(config, f"{prefix}_confirm_reads"),
                     min_mean=getattr(config, f"{prefix}_confirm_mean"),
                     min_share=getattr(config, f"{prefix}_confirm_share"))
        for name, prefix in (("vehicle_type", "type"), ("brand", "brand"))
    }
