"""
routes/report.py

GET /api/v1/report/weekly/{athlete_id}
    — Aggregates last 7 days of history and generates an
      AI-written personalised weekly report using Claude.
"""

from fastapi  import APIRouter, HTTPException, Depends
from typing   import Dict, Any
from datetime import datetime, timedelta
import urllib.request, urllib.error, json, os

from models.auth_deps       import get_current_user
from services.athlete_store import AthleteStore

router = APIRouter(prefix="/report", tags=["Report"])

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def _call_claude(prompt: str) -> str:
    """Call Claude claude-sonnet-4-20250514 to generate a report narrative."""
    if not ANTHROPIC_API_KEY:
        return (
            "AI narrative unavailable — set the ANTHROPIC_API_KEY environment "
            "variable to enable personalised report generation."
        )
    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 600,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type":      "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"].strip()
    except Exception as e:
        return f"AI narrative unavailable: {str(e)}"


def _aggregate_week(history: list) -> Dict[str, Any]:
    """Compute stats from the last 7 log entries."""
    week = history[:7]
    if not week:
        return {}

    scores    = [e["risk_score"]    for e in week]
    acwrs     = [e["acwr"]          for e in week]
    loads     = [e["daily_load"]    for e in week]
    sleeps    = [e.get("sleep_hours", 7)  for e in week]
    soreness  = [e.get("soreness",   2)   for e in week]
    cats      = [e["risk_category"] for e in week]

    peak_day  = week[scores.index(max(scores))]
    best_day  = week[scores.index(min(scores))]

    cat_counts = {"low": 0, "elevated": 0, "moderate": 0, "high": 0}
    for c in cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1

    # Sleep trend: positive = improving, negative = worsening
    mid = len(sleeps) // 2
    sleep_trend = round(
        (sum(sleeps[:mid]) / max(mid, 1)) - (sum(sleeps[mid:]) / max(len(sleeps) - mid, 1)),
        2,
    )

    return {
        "days_logged":       len(week),
        "avg_risk_score":    round(sum(scores)   / len(scores),   1),
        "max_risk_score":    round(max(scores),  1),
        "min_risk_score":    round(min(scores),  1),
        "avg_acwr":          round(sum(acwrs)    / len(acwrs),    3),
        "max_acwr":          round(max(acwrs),   3),
        "avg_load":          round(sum(loads)    / len(loads),    1),
        "peak_load":         round(max(loads),   1),
        "avg_sleep":         round(sum(sleeps)   / len(sleeps),   2),
        "min_sleep":         round(min(sleeps),  2),
        "avg_soreness":      round(sum(soreness) / len(soreness), 2),
        "max_soreness":      round(max(soreness),2),
        "risk_distribution": cat_counts,
        "high_risk_days":    cat_counts.get("high", 0),
        "sleep_trend":       sleep_trend,   # positive = recent nights better
        "peak_risk_day":     peak_day,
        "best_risk_day":     best_day,
        "dates":             [e["date"] for e in week],
    }


def _build_prompt(athlete: dict, stats: dict) -> str:
    sport = athlete.get("sport", "sport")
    name  = athlete.get("name",  "the athlete")
    age   = athlete.get("age",   "unknown")

    return f"""You are a sports science injury prevention coach. Write a concise, 
personalised weekly training report for {name}, a {age}-year-old {sport} athlete.

LAST 7 DAYS DATA:
- Days logged: {stats['days_logged']}
- Average risk score: {stats['avg_risk_score']}/100
- Peak risk score: {stats['max_risk_score']}/100 (on {stats['peak_risk_day']['date']})
- Best risk score: {stats['min_risk_score']}/100 (on {stats['best_risk_day']['date']})
- High-risk days: {stats['high_risk_days']} out of {stats['days_logged']}
- Risk distribution: {stats['risk_distribution']}
- Average ACWR: {stats['avg_acwr']} (danger if > 1.5)
- Peak ACWR: {stats['max_acwr']}
- Average training load: {stats['avg_load']} AU
- Average sleep: {stats['avg_sleep']}h (minimum: {stats['min_sleep']}h)
- Sleep trend: {'improving' if stats['sleep_trend'] > 0 else 'worsening'} ({stats['sleep_trend']:+.1f}h shift)
- Average soreness: {stats['avg_soreness']}/5 (peak: {stats['max_soreness']}/5)

Write in 3 short paragraphs:
1. Overall week summary (2-3 sentences, honest assessment)
2. Key risk factors identified this week (2-3 sentences)
3. Specific, actionable recommendations for NEXT week (3-4 bullet points)

Keep the tone professional but encouraging. Be specific with numbers.
Do not use markdown headers or bold. Use plain text only."""


@router.get("/weekly/{athlete_id}", summary="Generate AI weekly report for an athlete")
async def weekly_report(
    athlete_id: str,
    current_user: dict = Depends(get_current_user),
) -> Dict[str, Any]:

    # Ownership check
    athlete = AthleteStore.get(athlete_id)
    if not athlete:
        raise HTTPException(status_code=404, detail="Athlete not found.")
    if athlete.get("owner_id") != current_user["user_id"] and \
            current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Access denied.")

    history = AthleteStore.get_history(athlete_id, limit=7)
    if len(history) < 3:
        raise HTTPException(
            status_code=422,
            detail="Need at least 3 logged days to generate a weekly report.",
        )

    stats     = _aggregate_week(history)
    narrative = _call_claude(_build_prompt(athlete, stats))

    # Determine overall week grade
    avg = stats["avg_risk_score"]
    if avg < 25:
        grade, grade_label = "A", "Excellent week"
    elif avg < 40:
        grade, grade_label = "B", "Good week"
    elif avg < 55:
        grade, grade_label = "C", "Caution advised"
    elif avg < 70:
        grade, grade_label = "D", "High-risk week"
    else:
        grade, grade_label = "F", "Critical — rest needed"

    return {
        "athlete_id":   athlete_id,
        "athlete_name": athlete.get("name"),
        "sport":        athlete.get("sport"),
        "week_ending":  history[0]["date"],
        "week_starting": history[-1]["date"] if len(history) > 1 else history[0]["date"],
        "grade":        grade,
        "grade_label":  grade_label,
        "stats":        stats,
        "narrative":    narrative,
        "generated_at": datetime.utcnow().isoformat(),
    }