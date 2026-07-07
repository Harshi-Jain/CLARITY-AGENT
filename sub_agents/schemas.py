"""Structured data contracts passed between sub-agents.

Using explicit schemas (instead of free-text handoffs) is what lets the
Comparator and Recommender reason over consistent fields, and is what lets
you show judges a clean, inspectable data flow instead of a black box.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DecisionCriteria(BaseModel):
    """What the Clarifier agent produces once it understands the user's goal."""

    decision_summary: str = Field(
        description="One-sentence restatement of the decision being made."
    )
    options: List[str] = Field(
        description="The candidate options the user is choosing between."
    )
    must_haves: List[str] = Field(
        default_factory=list,
        description="Hard constraints. Any option violating one should be down-ranked.",
    )
    priorities: List[str] = Field(
        description="Soft criteria the user cares about, roughly in order of importance."
    )
    budget_or_limits: Optional[str] = Field(
        default=None, description="Any explicit budget, timeline, or resource limit."
    )


class OptionFindings(BaseModel):
    """What the Research agent produces for a single option."""

    option_name: str
    summary: str = Field(description="2-4 sentence synthesis of what was found.")
    pros: List[str]
    cons: List[str]
    sources: List[str] = Field(default_factory=list)


class CriterionScore(BaseModel):
    criterion: str
    score: float = Field(ge=0, le=10)
    rationale: str


class OptionComparison(BaseModel):
    """What the Comparator agent produces for a single option."""

    option_name: str
    criterion_scores: List[CriterionScore]
    weighted_total: float
    violates_must_have: bool = False


class Recommendation(BaseModel):
    """Final output of the Recommender agent."""

    recommended_option: str
    confidence: float = Field(
        ge=0, le=1, description="Recommender's self-assessed confidence in this verdict."
    )
    reasoning: str
    key_tradeoffs: List[str]
    needs_more_info: bool = Field(
        default=False,
        description="True if confidence is below threshold and the loop should continue.",
    )
    follow_up_question: Optional[str] = Field(
        default=None,
        description="Set only when needs_more_info is True — routed back to the Clarifier.",
    )
