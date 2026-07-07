"""Entry point for the Clarity decision agent.

Architecture:
    root_agent (LoopAgent)
        clarifier    -> writes decision_criteria
        research     -> writes option_findings (uses google_search)
        comparator   -> writes option_comparisons (uses compute_weighted_score)
        recommender  -> writes recommendation, decides loop continues or stops

Run locally with:
    adk web        # browser chat UI
    adk run        # terminal chat

Requires GOOGLE_API_KEY set in .env (see .env.example).
"""

import os
from dotenv import load_dotenv
from google.adk.agents import LoopAgent

from sub_agents.clarifier import clarifier_agent
from sub_agents.research import research_agent
from sub_agents.comparator import comparator_agent
from sub_agents.recommender import recommender_agent

load_dotenv()

MAX_LOOP_ITERATIONS = int(os.getenv("MAX_LOOP_ITERATIONS", "3"))

# A LoopAgent runs its sub_agents in order, repeating the whole sequence
# until either a sub-agent sets tool_context.actions.escalate = True
# (see recommender.py's submit_recommendation) or max_iterations is hit —
# whichever comes first. This is what turns four
