"""engine -- the discovery tier + campaign orchestration."""
from .agents import Agent, Bundle, Maturation, MockAgent
from .campaign import CampaignResult, run_campaign
from .claude_agent import ClaudeAgentError, ClaudeCodeAgent
from .discovery_roles import (
    ClaudeReviewerAdversary,
    ClaudeSignificanceAdversary,
    ClaudeSynthesizer,
    MockReviewerAdversary,
    MockSignificanceAdversary,
    MockSynthesizer,
    ReviewerVerdict,
    SignificanceVerdict,
    Synthesis,
    decide_arc,
    is_mature,
)
from .bandit import Bandit, BanditError
from .deadman import (
    DeadMansSwitch,
    EscalationChannel,
    MockTransport,
    TransportError,
    clock_trustworthy,
)
from .fingerprint import (
    MATCH,
    PAGE,
    SCORED_NO_RESCORE,
    Fingerprint,
    MockEnvProbe,
    record_fingerprint,
    resume_verify,
)
from .health import DEGRADE, HALT, QUARANTINE, RETRY, HaltError, HealthGate, Probe
from .steering import (
    BREADTH,
    DEPTH,
    SteeringPolicy,
    choose_vein,
    is_dead_end,
    is_high_patience,
    passes_cheap_gate,
    should_prune,
)
from .ledger import Ledger, SQLiteLedger, canary_probe
from .pool import (
    DELIVERABLE,
    NO_ARC,
    CampaignClose,
    FamilyReport,
    PoolReport,
    close_campaign,
    depth_completion,
    finalize_campaign,
    group_and_report,
    replication_conjunction,
)
from .supervisor import (
    BACKSTOP,
    BASE_CASE,
    HALTED,
    HEALTH_HALT,
    Budget,
    HaltFlag,
    SupervisorState,
    base_case_reached,
    run_supervisor,
)

__all__ = [
    "Agent", "Bundle", "Maturation", "MockAgent",
    "ClaudeCodeAgent", "ClaudeAgentError",
    "MockReviewerAdversary", "MockSignificanceAdversary", "MockSynthesizer",
    "ClaudeReviewerAdversary", "ClaudeSignificanceAdversary", "ClaudeSynthesizer",
    "ReviewerVerdict", "SignificanceVerdict", "Synthesis", "is_mature", "decide_arc",
    "run_campaign", "CampaignResult",
    "close_campaign", "PoolReport",
    "depth_completion", "replication_conjunction", "group_and_report",
    "finalize_campaign", "CampaignClose", "FamilyReport", "DELIVERABLE", "NO_ARC",
    "Bandit", "BanditError",
    "SteeringPolicy", "choose_vein", "should_prune", "is_high_patience",
    "is_dead_end", "passes_cheap_gate", "DEPTH", "BREADTH",
    "HealthGate", "Probe", "HaltError", "HALT", "QUARANTINE", "RETRY", "DEGRADE",
    "Ledger", "SQLiteLedger", "canary_probe",
    "Fingerprint", "MockEnvProbe", "record_fingerprint", "resume_verify",
    "MATCH", "PAGE", "SCORED_NO_RESCORE",
    "Budget", "HaltFlag", "SupervisorState", "base_case_reached", "run_supervisor",
    "BASE_CASE", "HALTED", "HEALTH_HALT", "BACKSTOP",
    "DeadMansSwitch", "EscalationChannel", "MockTransport", "TransportError", "clock_trustworthy",
]
