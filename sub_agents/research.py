"""Research agent — gathers real information per option using web search.

This is the agent that actually demonstrates "grounded" reasoning: every
pro/con it lists should trace back to a search result, not the model's
training data (which may be stale or wrong about prices, specs, reviews).
"""

import os
from google.adk.agents import Agent
from google.adk.tools import google_search

from schemas import OptionFindings

MODEL = os.getenv("CLARITY_MODEL", "gemini-2.5-flash")

research_agent = Agent(
    name="research",
    model=MODEL,
    description="Searches the web for current information on each candidate option.",
    instruction="""You have access to the DecisionCriteria produced by the
Clarifier (state key: 'decision_criteria').

For EACH option listed there, run at least one targeted web search covering
things directly relevant to the user's stated priorities and must-haves
(e.g. if 'reliability' is a priority, search for reviews/complaints, not
just marketing copy).

For each option, output an OptionFindings object: a short synthesis, 2-4
pros, 2-4 cons, and the source URLs you used. Do not editorialize about
which option is best — that is the Comparator and Recommender's job. Just
report what you found, accurately and with sources.""",
    tools=[google_search],
    output_schema=OptionFindings,
    output_key="option_findings",
)
