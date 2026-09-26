"""Model-based planner for the latent world model.

Importing this package loads PyTorch. The environment, schema, and corpus
commands do not import it. Install the optional ``ml`` extra first.

The search is a small cross-entropy method over a short action sequence.
Candidates are scored by rolling the world model in latent space and
summing Lotka-Volterra regulation rewards of the decoded states.
``planning_regret`` compares that planner, closed loop, with a zero dose
and a seeded random dose.
"""

from worldforge.plan.cem import (
    ActionPlan,
    CEMConfig,
    plan_actions,
    score_action_sequence,
)
from worldforge.plan.objective import OBJECTIVE_NAME, default_target, regulation_reward
from worldforge.plan.report import (
    ACTION_SEQUENCE_FORMAT,
    PLAN_REPORT_FORMAT,
    REGRET_DEFINITION,
    ActionSequenceReport,
    PlanReport,
    action_sequence_report,
    format_action_sequence,
    format_plan_report,
    planning_regret,
    read_action_sequence,
    read_plan_report,
    write_action_sequence,
    write_plan_report,
)

__all__ = [
    "ACTION_SEQUENCE_FORMAT",
    "ActionPlan",
    "ActionSequenceReport",
    "CEMConfig",
    "OBJECTIVE_NAME",
    "PLAN_REPORT_FORMAT",
    "PlanReport",
    "REGRET_DEFINITION",
    "action_sequence_report",
    "default_target",
    "format_action_sequence",
    "format_plan_report",
    "plan_actions",
    "planning_regret",
    "read_action_sequence",
    "read_plan_report",
    "regulation_reward",
    "score_action_sequence",
    "write_action_sequence",
    "write_plan_report",
]
