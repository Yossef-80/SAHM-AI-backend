# Supabase Auth Migration Plan

## Current State
- Custom auth: email/password + PBKDF2 + opaque server-side sessions (sha256 token stored)
- JWT issued by backend, signed with `JWT_SECRET`
- Tables: `users`, `memberships`, `sessions`, `businesses`
- Frontend: `useAuthStore` with `sessionStorage` token, `sahm_session` marker cookie

## Target State
- Supabase Auth handles: signup, login, password reset, email confirmation, OAuth (Google, etc.)
- Supabase issues JWTs (JWKS at `https://<ref>.supabase.co/auth/v1/.well-known/jwks.json`)
- Backend verifies Supabase JWT via JWKS, maps `sub` → local `business_id` via `profiles` table
- Local tables kept: `businesses`, `memberships` (renamed to `profiles`), `profiles` links `supabase_user_id` → `business_id` + `role`
- `users` + `sessions` tables **dropped** (or archived)

## Migration Steps

### 1. Run Alembic on Supabase (BLOCKER - must run from WSL/Docker)
```bash
# From WSL/Docker/other host:
python -m alembic -c app/db/migrations/alembic.ini upgrade head
python scripts/seed_demo_data.py --reset
```

### 2. Add `profiles` table (new migration)
```sql
-- Links Supabase auth.users (sub) to our business/role model
CREATE TABLE profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    supabase_user_id UUID NOT NULL UNIQUE,  -- maps to auth.users.id
    business_id TEXT NOT NULL REFERENCES businesses(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('owner','member','viewer')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_profiles_supabase_user ON profiles(supabase_user_id);
CREATE INDEX idx_profiles_business ON profiles(business_id);
```

### 3. Backend changes
- Remove: `app/api/auth.py` (register, login, refresh, logout, me)
- Remove: `app/core/auth.py` (password hashing, session management, Principal)
- Add: `app/core/supabase_auth.py` (JWKS verification, token parsing, profile loading)
- Update: `app/api/deps.py` - `get_principal` now validates Supabase JWT
- Update: `app/db/models.py` - drop `User`, `Session`, `Membership`; add `Profile`
- Update: `app/schemas/auth.py` - new response shapes

### 4. Frontend changes
- Install `@supabase/supabase-js`
- Replace `useAuthStore` with Supabase client + auth state listener
- Login/Register pages use `supabase.auth.signInWithPassword` / `signUp`
- Google OAuth: `supabase.auth.signInWithOAuth({ provider: 'google' })`
- Token auto-refresh handled by Supabase client
- `X-Workspace-Id` header still sent (business_id from profile)

### 5. Supabase Dashboard
- Enable Email provider (default)
- Enable Google provider: add `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET`
- Set Site URL: `http://localhost:3000` (dev) / production URL
- Add redirect URLs: `http://localhost:3000/auth/callback`

### 6. Environment variables
```env
# Supabase (already in .env)
DATABASE_URL=postgresql+asyncpg://...

# NEW - for Supabase Auth
SUPABASE_URL=https://fmvwhrduiwloqravbeqi.supabase.co
SUPABASE_ANON_KEY=<from dashboard>
SUPABASE_SERVICE_ROLE_KEY=<from dashboard, server-only>
```

## Rollback Plan
- Keep old `users`/`sessions`/`memberships` tables renamed (`_legacy_*`)
- Feature flag `USE_SUPABASE_AUTH` in config
- Can flip back in < 5 min if issues

## Testing Checklist
- [ ] Register with email/password → email confirmation → login
- [ ] Login with Google OAuth
- [ ] Password reset flow
- [ ] Token refresh (auto, silent)
- [ ] `GET /auth/me` returns profile + workspaces
- [ ] Role enforcement (owner/member/viewer) works
- [ ] Workspace switching (`X-Workspace-Id`) works
- [ ] Admin endpoints still gated
- [ ] All existing API endpoints work with new auth