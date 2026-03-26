from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AuditJudgeDecision(BaseModel):
    verdict: Literal["pass", "fail", "needs_human_review"]
    risk: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    findings: List[str] = Field(default_factory=list)
    should_promote_to_expert: bool = False
    suggested_gate: Literal["blocking", "regression", "audit"] = "audit"


class HumanReviewRecord(BaseModel):
    status: Literal[
        "pending",
        "false_alarm",
        "confirmed_issue",
        "promote_to_expert",
    ] = "pending"
    reviewer: Optional[str] = None
    notes: str = ""
    corrected_input: Dict[str, Any] = Field(default_factory=dict)
    corrected_expected: Dict[str, Any] = Field(default_factory=dict)
    target_gate: Optional[Literal["blocking", "regression"]] = None
    resolved_at: Optional[str] = None
    promotion_exported_at: Optional[str] = None
    promotion_file: Optional[str] = None


class AuditReviewRecord(BaseModel):
    id: str
    task: Literal["planner", "extractor", "variety_match"]
    gate: str = "audit"
    input: Dict[str, Any] = Field(default_factory=dict)
    expected: Dict[str, Any] = Field(default_factory=dict)
    observed_output: Dict[str, Any] = Field(default_factory=dict)
    normalized_observed_output: Dict[str, Any] = Field(default_factory=dict)
    source: Dict[str, Any] = Field(default_factory=dict)
    rule_grade: Dict[str, Any] = Field(default_factory=dict)
    ai_judge: Optional[AuditJudgeDecision] = None
    human_review: HumanReviewRecord = Field(default_factory=HumanReviewRecord)


class ProductionAuditBatch(BaseModel):
    task: Literal["planner", "extractor", "variety_match"]
    line: Literal["production_audit"] = "production_audit"
    sampling_scope: Literal["standalone", "context_dependent"] = "standalone"
    replay_mode: Literal["standalone_replay", "judge_only"] = "standalone_replay"
    review: Dict[str, str] = Field(
        default_factory=lambda: {
            "primary": "ai_judge",
            "secondary": "human_spotcheck",
        }
    )
    generated_at: Optional[str] = None
    source_store: Optional[str] = None
    cases: List[Dict[str, Any]] = Field(default_factory=list)
