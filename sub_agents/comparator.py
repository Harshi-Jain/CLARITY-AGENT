"""Comparator agent — scores each option against the user's criteria.

Delegates the actual arithmetic to compute_weighted_score() so the final
number is deterministic and auditable rather than an LLM-eyeballed average.
"""

import os
from google.adk.agents import Agent

from schemas import OptionComparison
from tools.scoring import compute_weighted_score

MODEL = os.getenv("CLARITY_MODEL", "gemini-2.5-flash")

comparator_agent = Agent(
    name="comparator",
    model=MODEL,
    description="Scores each option against the user's stated criteria.",
    instruction="""You have the DecisionCriteria (state key: 'decision_criteria')
and OptionFindings for every option (state key: 'option_findings').

For each option:
1. Score it 0-10 on each of the user's priorities, with a one-line rationale
   per score, grounded in the research findings — not vibes.
2. Flag violates_must_have = true if it clearly fails any must_have.
3. Call compute_weighted_score with your criterion scores and reasonable
   weights (weight earlier-listed priorities higher, since priorities is
   already roughly ordered) to get the deterministic weighted_total.

Output one OptionComparison per option. Be willing to score an option low —
your job is accurate differentiation, not diplomacy.""",
    tools=[compute_weighted_score],
    output_schema=OptionComparison,
    output_key="option_comparisons",
)
