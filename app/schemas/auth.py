"""Auth request/response schemas.

Inherits :class:`ApiModel`, so every field is camelCase on the wire and
snake_case in Python -- the same rule as every other schema in this package.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, field_validator

from app.core.auth import ROLE_RANK, normalise_email
from app.schemas.common import ApiModel

#: Minimum password length. Deliberately above the "8 characters" advice that
#: every breach list has made worthless.
MIN_PASSWORD_LENGTH = 10

#: Addresses that ship with a known-password list. Checking them costs nothing
#: and turns "the user picked something already in a wordlist" into a form
#: error instead of a live credential.
BREACHED_PASSWORDS = frozenset(
    {
        "password",
        "password1",
        "password123",
        "1234567890",
        "12345678901",
        "qwertyuiop",
        "iloveyou123",
        "administrator",
        "letmein1234",
        "welcome1234",
    }
)


class RegisterRequest(ApiModel):
    """``POST /api/v1/auth/register`` body.

    ``workspace_name`` is the business created alongside the account, because a
    new user must land on a usable workspace rather than an empty shell.
    """

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=256)
    full_name: str | None = Field(default=None, max_length=200)
    workspace_name: str = Field(default="My Business", min_length=1, max_length=200)

    @field_validator("email")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return normalise_email(value)

    @field_validator("password")
    @classmethod
    def _policy(cls, value: str) -> str:
        if value.lower() in BREACHED_PASSWORDS:
            raise ValueError("this password appears in a list of breached passwords")
        if value.strip() != value:
            raise ValueError("the password must not start or end with whitespace")
        return value


class LoginRequest(ApiModel):
    """``POST /api/v1/auth/login`` body."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return normalise_email(value)


class RefreshRequest(ApiModel):
    """``POST /api/v1/auth/refresh`` body. The token being rotated."""

    token: str = Field(min_length=1, max_length=512)


class WorkspaceMembership(ApiModel):
    """One workspace the caller belongs to, with the role they hold there.

    Shaped for the frontend's portfolio switcher: enough to render a list of
    businesses with the right affordances, and nothing that identifies another
    user.
    """

    workspace_id: str
    business_id: str
    business_name: str
    role: str = Field(description="owner | member | viewer")
    #: True for the workspace the API resolves when no header is sent.
    is_primary: bool = False


class UserOut(ApiModel):
    """The authenticated user. Never carries ``password_hash``."""

    id: str
    email: str
    full_name: str | None = None
    is_active: bool = True
    last_login_at: datetime | None = None
    created_at: datetime | None = None


class AuthTokenResponse(ApiModel):
    """What register / login / refresh return.

    ``tokenType`` is always ``"Bearer"`` so the client can build the header
    without hard-coding it.
    """

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = Field(description="Seconds until expires_at.")
    user: UserOut
    workspaces: list[WorkspaceMembership] = Field(default_factory=list)
    #: The workspace a request resolves to when no ``X-Workspace-Id`` is sent.
    active_workspace_id: str | None = None


class LogoutResponse(ApiModel):
    """``POST /api/v1/auth/logout``. Idempotent by design."""

    revoked: bool = Field(
        description="False when the token was already revoked or unknown."
    )


class MeResponse(ApiModel):
    """``GET /api/v1/auth/me`` -- the frontend's boot check."""

    user: UserOut
    workspaces: list[WorkspaceMembership] = Field(default_factory=list)
    active_workspace_id: str | None = None
    #: The highest role the caller holds on ``active_workspace_id``, so the UI
    #: can hide owner-only affordances without a second round trip.
    active_role: str | None = None


def role_rank(role: str | None) -> int:
    """Numeric rank of a role, 0 when unknown.

    Re-exported from ``app.core.auth`` so a schema module never has to guess
    the ordering itself.
    """
    return ROLE_RANK.get(role or "", 0)


class CompleteProfileRequest(ApiModel):
    """``POST /api/v1/auth/complete-profile`` body for new OAuth users."""

    full_name: str = Field(min_length=1, max_length=200)
    workspace_name: str = Field(min_length=1, max_length=200)


class CompleteProfileResponse(ApiModel):
    """Response for ``POST /api/v1/auth/complete-profile``."""

    workspaces: list[WorkspaceMembership] = Field(default_factory=list)
    active_workspace_id: str | None = None


__all__ = [
    "BREACHED_PASSWORDS",
    "MIN_PASSWORD_LENGTH",
    "AuthTokenResponse",
    "LoginRequest",
    "LogoutResponse",
    "MeResponse",
    "RefreshRequest",
    "RegisterRequest",
    "CompleteProfileRequest",
    "CompleteProfileResponse",
    "UserOut",
    "WorkspaceMembership",
    "role_rank",
]