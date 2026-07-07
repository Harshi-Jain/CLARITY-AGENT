"""Recommender agent — synthesizes the final verdict and decides whether the
Clarifier -> Research -> Comparator -> Recommender loop should run again.

The confidence threshold is what makes this a genuine agentic loop instead of
a fixed pipeline: the agent itself decides when it knows enough to stop.
"""

import os
from google.adk.agents import Agent
from google.adk.tools import ToolContext

from schemas import Recommendation

MODEL = os.getenv("CLARITY_MODEL", "gemini-2.5-flash")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


def submit_recommendation(
    recommended_option: str,
    confidence: float,
    reasoning: str,
    key_tradeoffs: list[str],
    follow_up_question: str,
    tool_context: ToolContext,
) -> dict:
    """Finalize the recommendation, or signal that another loop iteration is needed.

    Pass follow_up_question="" when confidence is already high enough — there
    is nothing to ask. When confidence is below the threshold, follow_up_question
    must contain the single most useful thing to ask the user next.
    """
    high_confidence = confidence >= CONFIDENCE_THRESHOLD

    if high_confidence:
        # Signals the LoopAgent to stop iterating — we have a final answer.
        tool_context.actions.escalate = True
        tool_context.state["final_recommendation"] = {
            "recommended_option": recommended_option,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_tradeoffs": key_tradeoffs,
            "needs_more_info": False,
            "follow_up_question": None,
        }
    else:
        # Leave escalate unset so the LoopAgent runs another round; stash the
        # question where the Clarifier agent knows to look for it.
        tool_context.state["follow_up_question"] = follow_up_question

    return {"status": "recorded", "high_confidence": high_confidence}


recommender_agent = Agent(
    name="recommender",
    model=MODEL,
    description="Synthesizes a final recommendation and judges its own confidence.",
    instruction=f"""You have DecisionCriteria (state key: 'decision_criteria')
and every OptionComparison (state key: 'option_comparisons').

Pick the option with the best combination of highest weighted_total and no
must-have violations. Write 2-4 sentences of reasoning and list the key
tradeoffs a reasonable person might weigh differently.

Honestly assess your own confidence (0-1). Confidence should be LOW if:
- research findings were thin or conflicting for the top options
- the user's priorities were vague or the criteria felt guessed rather than stated
- two options scored within ~1 point of each other on weighted_total

The threshold for a final answer is {CONFIDENCE_THRESHOLD}. If your
confidence is below that, do not guess — identify the single question whose
answer would most change the outcome, and call submit_recommendation with
that question. Otherwise call submit_recommendation with an empty
follow_up_question string.""",
    tools=[submit_recommendation],
    output_schema=Recommendation,
    output_key="recommendation",
)
