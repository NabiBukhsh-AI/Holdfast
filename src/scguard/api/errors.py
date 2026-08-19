"""RFC 9457 problem details and the error taxonomy. TASK-027, spec 18.9.

Every error the API can return is enumerated here with its HTTP status and whether a client
should retry. An error that is not in this table cannot be returned, which is what makes the
contract testable.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

PROBLEM_CONTENT_TYPE = "application/problem+json"
PROBLEM_BASE_URI = "https://holdfast.invalid/problems"


class ErrorCode(StrEnum):
    """Spec 18.9. Stable identifiers: clients branch on these, not on prose."""

    INVALID_ROLE = "INVALID_ROLE"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    TURN_CONFLICT = "TURN_CONFLICT"
    COMPACTION_CONFLICT = "COMPACTION_CONFLICT"
    CONTENT_TOO_LARGE = "CONTENT_TOO_LARGE"
    RATE_LIMITED = "RATE_LIMITED"
    REGISTRY_UNAVAILABLE = "REGISTRY_UNAVAILABLE"
    EXTRACTOR_UNAVAILABLE = "EXTRACTOR_UNAVAILABLE"
    BUDGET_INVALID = "BUDGET_INVALID"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    CONSTRAINT_NOT_FOUND = "CONSTRAINT_NOT_FOUND"


ERROR_TAXONOMY: dict[ErrorCode, tuple[int, bool, str]] = {
    ErrorCode.INVALID_ROLE: (400, False, "Non-user turn submitted to the turn endpoint"),
    ErrorCode.VALIDATION_ERROR: (400, False, "Malformed request body"),
    ErrorCode.UNAUTHORIZED: (401, False, "Missing or invalid bearer token"),
    ErrorCode.FORBIDDEN: (403, False, "The token lacks the required scope"),
    ErrorCode.SESSION_NOT_FOUND: (404, False, "Unknown session"),
    ErrorCode.CONSTRAINT_NOT_FOUND: (404, False, "Unknown constraint"),
    ErrorCode.TURN_CONFLICT: (409, False, "Same turn index, different content hash"),
    ErrorCode.COMPACTION_CONFLICT: (409, False, "Compaction index already assembled with different input"),
    ErrorCode.CONTENT_TOO_LARGE: (413, False, "Turn content exceeds the configured cap"),
    ErrorCode.BUDGET_INVALID: (422, False, "Registry budget unset or zero"),
    ErrorCode.RATE_LIMITED: (429, True, "Rate limit exceeded"),
    ErrorCode.REGISTRY_UNAVAILABLE: (
        503,
        True,
        "Registry store unavailable. The service does NOT degrade to an empty registry.",
    ),
    ErrorCode.EXTRACTOR_UNAVAILABLE: (503, True, "Extractor unreachable"),
}


class ProblemDetail(BaseModel):
    """RFC 9457 problem details object."""

    model_config = ConfigDict(frozen=True)

    type: str
    title: str
    status: int
    detail: str
    code: ErrorCode
    instance: str | None = None
    retryable: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


class APIError(Exception):
    """Raised anywhere in the request path; rendered as a problem document by the handler."""

    def __init__(
        self,
        code: ErrorCode,
        detail: str,
        *,
        instance: str | None = None,
        **extra: Any,
    ) -> None:
        status, retryable, title = ERROR_TAXONOMY[code]
        super().__init__(detail)
        self.code = code
        self.status = status
        self.retryable = retryable
        self.title = title
        self.detail = detail
        self.instance = instance
        self.extra = extra

    def to_problem(self) -> ProblemDetail:
        return ProblemDetail(
            type=f"{PROBLEM_BASE_URI}/{self.code.value.lower()}",
            title=self.title,
            status=self.status,
            detail=self.detail,
            code=self.code,
            instance=self.instance,
            retryable=self.retryable,
            extra=self.extra,
        )
