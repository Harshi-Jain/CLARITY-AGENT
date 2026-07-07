"""Clarifier agent — turns a vague decision into structured criteria.

Runs first, and again (with a specific follow-up question) if the
Recommender later reports low confidence.
"""

import os
from google.adk.agents import Agent

from schemas import DecisionCriteria

MODEL = os.getenv("CLARITY_MODEL", "gemini-2.5-flash")

clarifier_agent = Agent(
    name="clarifier",
    model=MODEL,
    description="Extracts a structured decision brief from the user's message.",
    instruction="""You turn a person's decision into a structured brief.

Read the conversation so far. If the user has already given enough detail
(clear options, at least one priority), do NOT ask a question — just fill in
the DecisionCriteria fields as best you can, leaving optional fields empty.

If critical information is missing (e.g. no options listed, or no sense of
what they care about), ask ONE direct, specific question — never a generic
"tell me more". Once the user answers, populate DecisionCriteria fully.

If session state contains a 'follow_up_question' key (set by the
Recommender agent on a previous loop), treat that as the question you must
resolve this turn — ask the user exactly that, then update the criteria
with their answer.

Always output a DecisionCriteria object, even if some fields are empty.""",
    output_schema=DecisionCriteria,
    output_key="decision_criteria",
)
