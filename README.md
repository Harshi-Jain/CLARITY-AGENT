# CLARITY AGENT — ALL FILES
# Copy each section below into GitHub using "Add file -> Create new file".
# The filename to use is shown in the header comment above each section.
# ==============================================================================


# ==============================================================================
# FILE: README.md
# ==============================================================================

# Clarity — a personal decision agent

Clarity helps you make a real decision (which apartment, which laptop, which
job offer) instead of just chatting about it. It clarifies what you actually
care about, researches your real options on the web, scores them against your
priorities, and gives you a recommendation with an honest confidence level —
asking a follow-up question and looping again if it isn't sure yet.

## Why this problem

Most "AI decision help" is a single prompt that produces a confident-sounding
but ungrounded opinion. Clarity is built to be honest about uncertainty and
grounded in current information, which is exactly where a single LLM call
falls short and a multi-agent, tool-using system earns its keep.

## Architecture

```
root_agent (LoopAgent, max 3 iterations)
  1. clarifier    -> DecisionCriteria      (asks a question only if needed)
  2. research     -> OptionFindings[]      (google_search per option)
  3. comparator   -> OptionComparison[]    (compute_weighted_score tool)
  4. recommender  -> Recommendation        (submit_recommendation tool)
                     |
                     +-- confidence >= threshold -> escalate=True, loop stops
                     +-- confidence <  threshold -> follow_up_question saved
                                                      to state, loop repeats
```

Each agent reads and writes to shared session state via ADK's `output_key`,
so the handoffs are structured data (Pydantic schemas in `schemas.py`), not
free text — this is what lets the Comparator and Recommender reason reliably
over what came before.

## Concepts from the course this demonstrates

- **Multi-agent orchestration** — a coordinator (`LoopAgent`) delegating to
  specialized sub-agents rather than one agent doing everything.
- **Tool use** — `google_search` for grounded, current information;
  `compute_weighted_score` as a deterministic tool so the final ranking
  doesn't depend on LLM arithmetic.
- **Structured output** — every sub-agent produces a typed Pydantic object,
  not prose, so downstream agents can rely on it.
- **Agentic loop with self-assessed confidence** — the Recommender decides
  *for itself* whether it knows enough to stop, which is a genuinely
  agentic behavior rather than a fixed N-step pipeline.
- **Session state as memory** — criteria and prior findings persist across
  loop iterations so a follow-up question doesn't restart from scratch.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/clarity-agent.git
cd clarity-agent
pip install -r requirements.txt
cp .env.example .env   # then add your GOOGLE_API_KEY
adk web                # opens a local chat UI
# or
adk run                # terminal chat
```

## Project layout

```
clarity-agent/
  agent.py                 # root_agent — required entry point for adk web/run
  schemas.py                # shared Pydantic contracts between agents
  sub_agents/
    clarifier.py
    research.py
    comparator.py
    recommender.py
  tools/
    scoring.py               # compute_weighted_score
  requirements.txt
  .env.example
```

## Known gaps to close before submission

- **No test conversations yet.** Run 3-5 real decisions through it (a cheap
  one, a high-stakes one, a vague one) and paste transcripts into your
  writeup — judges want to see it actually behave, not just read the code.
- **No evaluation harness.** Consider ADK's eval tooling, or even a simple
  script that checks: did it call search for every option? did confidence
  correlate with cases where you'd expect it to be uncertain?
- **ADK's exact API surface moves fast.** The imports and Agent/LoopAgent
  parameters here reflect the patterns taught in the course, but check
  `google-adk`'s current docs/changelog if anything fails to import —
  parameter names like `output_schema` vs `response_schema` have shifted
  across versions.
- **Cost/latency**: the loop can call the model 4x per iteration, up to 3
  iterations. Fine for a demo; worth mentioning as a tradeoff in your writeup.
- **No UI beyond `adk web`.** A simple Streamlit or Gradio front end showing
  the comparison table visually would make your demo video much stronger.


# ==============================================================================
# FILE: requirements.txt
# ==============================================================================

google-adk
pydantic>=2.0
python-dotenv


# ==============================================================================
# FILE: .env.example
# ==============================================================================

# Copy this file to .env and fill in your key
GOOGLE_API_KEY=your_gemini_api_key_here

# Model used by every sub-agent (override per-agent in code if you want
# a cheaper/faster model for simple sub-agents like Clarifier)
CLARITY_MODEL=gemini-2.5-flash

# Confidence (0-1) the Recommender must reach before the loop stops
CONFIDENCE_THRESHOLD=0.75

# Safety cap so the loop can't run forever if confidence never clears
MAX_LOOP_ITERATIONS=3


# ==============================================================================
# FILE: schemas.py
# ==============================================================================

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


# ==============================================================================
# FILE: agent.py
# ==============================================================================

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
# whichever comes first. This is what turns four separate agents into one
# self-correcting research-and-decide loop instead of a fixed pipeline.
root_agent = LoopAgent(
    name="clarity_orchestrator",
    description=(
        "Helps a person make a decision by clarifying their criteria, "
        "researching real options on the web, scoring them, and "
        "recommending one — asking follow-up questions if confidence is low."
    ),
    sub_agents=[
        clarifier_agent,
        research_agent,
        comparator_agent,
        recommender_agent,
    ],
    max_iterations=MAX_LOOP_ITERATIONS,
)


# ==============================================================================
# FILE: sub_agents/__init__.py
# (leave this file completely empty — just create it and commit)
# ==============================================================================


# ==============================================================================
# FILE: sub_agents/clarifier.py
# ==============================================================================

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


# ==============================================================================
# FILE: sub_agents/research.py
# ==============================================================================

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


# ==============================================================================
# FILE: sub_agents/comparator.py
# ==============================================================================

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


# ==============================================================================
# FILE: sub_agents/recommender.py
# ==============================================================================

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


# ==============================================================================
# FILE: tools/__init__.py
# (leave this file completely empty — just create it and commit)
# ==============================================================================


# ==============================================================================
# FILE: tools/scoring.py
# ==============================================================================

"""A deterministic scoring tool.

Letting the LLM eyeball a weighted average is unreliable and hard to audit.
Instead the Comparator agent calls this tool with the scores it assigned per
criterion, and the tool does the arithmetic. This is a good example of
"use a tool for anything that should not depend on model arithmetic."
"""

from typing import Dict, List


def compute_weighted_score(
    criterion_scores: Dict[str, float],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Compute a normalized weighted score.

    Args:
        criterion_scores: e.g. {"price": 7, "reliability": 9}
        weights: e.g. {"price": 0.4, "reliability": 0.6} — need not sum to 1,
            they are normalized here.

    Returns:
        A dict with the weighted total and the normalized weights actually used,
        so the agent can cite exactly how the number was derived.
    """
    total_weight = sum(weights.get(k, 0) for k in criterion_scores) or 1.0
    normalized = {k: weights.get(k, 0) / total_weight for k in criterion_scores}
    weighted_total = sum(
        criterion_scores[k] * normalized[k] for k in criterion_scores
    )
    return {
        "weighted_total": round(weighted_total, 2),
        "normalized_weights": {k: round(v, 3) for k, v in normalized.items()},
    }


# ==============================================================================
# END OF FILE
# ==============================================================================
