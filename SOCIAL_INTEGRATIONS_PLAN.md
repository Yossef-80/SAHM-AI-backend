# Social Integrations, Limits, and Delivery Plan

## Decision

Sahm will provide personalized marketing analysis from a business's own social
accounts. A user connects an account through the provider's OAuth consent
screen, selects the Page, professional account, channel, or ad account they
manage, and Sahm synchronizes only the data that the provider grants.

Sahm must not collect a user's social password or scrape personal profiles,
private accounts, DMs, or competitor analytics. The product analyzes authorized
business content, not every item a person has ever posted on social media.

## Free-For-User Product Policy

The product should not charge users a separate fee to connect an eligible
social account. Select official APIs with a no-fee developer entry point and
build on their normal quotas.

This does not mean the feature has zero cost:

- Meta, TikTok, Google, and LinkedIn can change their API terms, quotas, or app
  review rules. We cannot promise that any third-party API will remain free
  forever.
- Sahm still pays for hosting, encrypted database storage, object storage,
  background jobs, and LLM/embedding usage. These costs must be covered by the
  product's general plan or capped with a per-workspace allowance.
- Advertising spend is always paid by the business to the ad platform and is
  not an integration fee.
- A user must hold the required role on the account they connect. A personal
  account does not become accessible simply because its owner uses Sahm.

The first release should use only free official API access where available,
apply strict sync limits, and never rely on paid scraping vendors.

## What We Can Analyze

| Provider | Eligible account | Data to request | First-release decision |
| --- | --- | --- | --- |
| Meta Ads | Ad account the user can manage | Campaign, ad set, and ad delivery metrics; spend, impressions, reach, clicks, and configured conversion actions | Build first |
| Facebook | Page the user manages | Page posts, captions, media references, engagement counts, approved post comments, and available post/Page insights | Build first |
| Instagram | Professional/creator/business account owned by the user | Media, captions, hashtags, media references, approved comments, and available media insights | Build first |
| TikTok | User-authorized creator account | Public video metadata and captions through approved OAuth scopes | Build second; do not promise reach or comments initially |
| YouTube | Channel the user authorizes | Videos, titles/descriptions, public comments where permitted, and channel/video analytics | Build second |
| LinkedIn | Company Page or marketing account with approved access | Marketing campaign and company Page data | Later; access is more restricted |
| Competitors | No connected account | Public website research and permitted public ad-library data only | Research only; no organic performance claims |

## Important Limitations

### Personal accounts are out of scope

We will not promise to import all content from a personal Facebook profile or
personal Instagram account. Meta's supported business integrations are designed
around Pages, professional Instagram accounts, and ad accounts. The connected
user must be an admin or have an appropriate role.

### "All data" is not a safe promise

Each provider decides which fields, historical window, metrics, and comments
are available to an approved app. API pagination, rate limits, deleted posts,
privacy settings, retention rules, and provider review can limit the result.
The UI must say what was synced, when it was synced, and which fields were not
available.

### Comments need explicit consent

Comments can contain personal information. The connection screen must include a
separate `Include post comments in analysis` option. We should store only the
comment text and necessary post relationship, redact obvious personal data
before model analysis, and never import DMs or private messages.

### Images are optional analysis inputs

We may analyze images belonging to the connected business to identify visual
style, product visibility, text overlays, and creative patterns. Store a media
reference and a small derived analysis by default; avoid copying full-resolution
media unless it is necessary and the user has opted in.

### Metrics need truthful labels

Reach, impressions, views, clicks, engagement, conversion actions, and ROAS are
not interchangeable. Store the original provider metric name, retrieval date,
reporting period, and source account. The analyst must never invent a missing
metric or call a generic conversion number "purchases".

## OAuth and Sync Design

```text
User clicks Connect
  -> Sahm creates a short-lived OAuth state and PKCE verifier
  -> Provider login and consent screen
  -> Provider redirects to Sahm callback with authorization code
  -> Backend exchanges code for tokens and encrypts them at rest
  -> User selects eligible account(s)
  -> Connector performs an initial, rate-limited sync
  -> Incremental job syncs new/changed content and metric snapshots
  -> Analysis service produces evidence-backed recommendations
```

Rules:

1. Tokens, refresh tokens, client secrets, and OAuth codes stay on the server.
2. Encrypt tokens at rest; never log or send them to the browser after storage.
3. Request the smallest scopes needed for the enabled feature.
4. A provider connection has a status: `connecting`, `connected`,
   `needs_reauth`, `error`, or `disconnected`.
5. Disconnect revokes access when supported, deletes tokens immediately, and
   queues deletion of imported content and derived analysis.
6. Use incremental sync cursors and scheduled jobs, not repeated full imports.

## Data Model to Add

Create these provider-neutral records:

- `social_connection`: provider, business, provider user ID, encrypted tokens,
  granted scopes, expiration, consent flags, status, and last sync time.
- `social_account`: connection, account type, provider account ID, display name,
  selected/not selected state, and account role.
- `social_content`: provider, account, external post ID, permalink, posted time,
  caption/text, hashtags, media type, media reference, and content hash.
- `social_metric_snapshot`: content or account, provider metric name, value,
  reporting period, retrieved time, and raw-source reference.
- `social_comment`: only when opted in; external comment ID, content ID, text,
  created time, and deletion state. Do not retain profile data unless required.
- `analysis_run`: input snapshot IDs, model/version, result, evidence, status,
  and deletion linkage.

Keep source records separate from `BrandMemory`. The analysis service can create
short, approved learnings such as "Reels with a product-first hook performed
best", but raw posts and metrics must remain auditable source data.

## Personalized Analysis Outcomes

The social data should improve the existing agents in concrete ways:

| Agent | Outcome from connected data |
| --- | --- |
| Nour, Researcher | Content themes, audience questions, recurring objections, and permitted competitor/ad evidence |
| Omar, Strategist | Channel mix, content pillars, post cadence, and campaign hypotheses grounded in actual performance |
| Layla, Creative | Brand voice examples, visual-style cues, high-performing hooks, prohibited claims, and content variations |
| Karim, Analyst | Post/ad trend reports, reach and engagement changes, creative fatigue signals, and missing-data warnings |
| Salma, Optimizer | Human-approved recommendations such as refresh a fatigued creative or reuse a successful hook |

Every insight must include an evidence link to a content item or metric snapshot,
the provider, the reporting period, and a confidence level.

## Delivery Plan

### Phase 0: Import-first analysis (2-3 days)

- Add CSV/JSON export upload and paste-text input for posts and metrics.
- Build `social_content`, `social_metric_snapshot`, and `analysis_run`.
- Produce a personalized report with voice, themes, cadence, top posts, and
  recommendations.
- This validates the product without waiting for external app approval.

### Phase 1: Meta read-only integration (4-6 days of engineering)

- Replace the current pasted-token route with OAuth start/callback routes.
- Add account selection for an ad account, Facebook Page, and eligible Instagram
  professional account.
- Implement initial and incremental read-only syncs.
- Show sync scope, last-sync timestamp, unavailable fields, and reauthorization.
- Keep campaign creation, publishing, budget updates, and pausing disabled.

### Phase 2: Meta analysis and approval workflow (2-4 days)

- Feed normalized content and metrics into Karim and Layla.
- Add comment analysis only for explicitly opted-in connections.
- Generate recommendations as visible change diffs.
- Require confirmation before any advertising mutation.

### Phase 3: TikTok and YouTube (one provider at a time)

- Implement the same OAuth, account-selection, normalized-data, sync, and
  deletion contract for each provider.
- Ship TikTok public-video analysis first.
- Ship YouTube video/channel analytics next.
- Do not add a provider until its scopes and approval requirements are verified
  with a test account.

### Phase 4: LinkedIn and publishing actions

- Add LinkedIn only after Meta read-only analysis is stable.
- Add provider write actions only after read paths, audit logs, human approval,
  and error recovery are proven.

## Technical Changes Needed in This Repository

1. Replace the manual `POST /integrations/meta/connect` token payload with
   OAuth start and callback endpoints.
2. Align backend integration endpoints and response schemas with the frontend
   `MetaConnection` contract.
3. Replace the placeholder `get_social_profile` tool with provider-specific
   connectors; do not describe unavailable social data as supported.
4. Add the provider-neutral source and metric tables above, migrations, sync
   cursor jobs, retention/deletion jobs, and test fixtures.
5. Map Meta conversion actions and values explicitly instead of relying on a
   generic `conversions` field.
6. Add a consent screen, privacy policy, data deletion workflow, and audit log
   before public launch.

## Sources to Validate During Implementation

- Meta Instagram Platform: https://developers.facebook.com/docs/instagram-platform/
- Meta Marketing APIs: https://developers.facebook.com/docs/marketing-apis/
- TikTok Display API: https://developers.tiktok.com/docs/en/display-api-get-started
- YouTube Data API OAuth: https://developers.google.com/youtube/v3/guides/authentication
- YouTube Analytics API: https://developers.google.com/youtube/analytics/reference/
- LinkedIn Marketing Platform: https://learn.microsoft.com/en-us/linkedin/marketing/integrations/marketing-integrations-overview

