-- One-time, data-preserving reconciliation for the Supabase database that
-- reported Alembic revision a7d8c3e219b4 and already has content_items.
-- It applies the schema changes from b60e91f42c73 through c3e81b7a4d26,
-- then advances alembic_version only after every DDL statement succeeds.
--
-- BEFORE RUNNING:
--   1) Stop the backend so its dev create_all() cannot race this script.
--   2) Take a Supabase backup.
--   3) Run this only if `alembic current` still returns a7d8c3e219b4.
--
-- The transaction aborts if the starting revision or expected table shape is
-- different. It does not drop tables or delete/update application rows.

BEGIN;

DO $preflight$
DECLARE
    current_revision text;
    revision_rows integer;
BEGIN
    IF to_regclass('public.alembic_version') IS NULL THEN
        RAISE EXCEPTION 'public.alembic_version is missing; stopping without changes';
    END IF;

    SELECT count(*), min(version_num)
      INTO revision_rows, current_revision
      FROM public.alembic_version;

    IF revision_rows <> 1 OR current_revision IS DISTINCT FROM 'a7d8c3e219b4' THEN
        RAISE EXCEPTION
            'Expected exactly one Alembic revision a7d8c3e219b4; found count=%, revision=%; stopping',
            revision_rows, current_revision;
    END IF;

    IF to_regclass('public.businesses') IS NULL
       OR to_regclass('public.campaigns') IS NULL
       OR to_regclass('public.creatives') IS NULL THEN
        RAISE EXCEPTION 'A required base table (businesses, campaigns, or creatives) is missing';
    END IF;
END
$preflight$;

-- b60e91f42c73: create content calendar table if the app did not already do so.
CREATE TABLE IF NOT EXISTS public.content_items (
    id varchar(64) NOT NULL,
    business_id varchar(64) NOT NULL,
    campaign_id varchar(64),
    date date NOT NULL,
    time varchar(5),
    channel varchar(20) NOT NULL,
    format varchar(20) NOT NULL,
    title varchar(500) NOT NULL,
    hook text,
    caption text,
    hashtags json NOT NULL,
    status varchar(20) NOT NULL,
    pillar varchar(200),
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT content_items_pkey PRIMARY KEY (id),
    CONSTRAINT ck_content_items_channel
        CHECK (channel IN ('instagram','facebook','tiktok','email','blog')),
    CONSTRAINT ck_content_items_format
        CHECK (format IN ('post','reel','story','ad','email','article')),
    CONSTRAINT ck_content_items_status
        CHECK (status IN ('idea','draft','scheduled','published')),
    CONSTRAINT fk_content_items_business_id_businesses
        FOREIGN KEY (business_id) REFERENCES public.businesses (id) ON DELETE CASCADE,
    CONSTRAINT fk_content_items_campaign_id_campaigns
        FOREIGN KEY (campaign_id) REFERENCES public.campaigns (id) ON DELETE SET NULL
);

DO $content_items_check$
DECLARE
    missing_columns text;
BEGIN
    SELECT string_agg(required.name, ', ' ORDER BY required.name)
      INTO missing_columns
      FROM unnest(ARRAY[
          'id','business_id','campaign_id','date','time','channel','format',
          'title','hook','caption','hashtags','status','pillar','created_at','updated_at'
      ]) AS required(name)
     WHERE NOT EXISTS (
         SELECT 1
           FROM information_schema.columns c
          WHERE c.table_schema = 'public'
            AND c.table_name = 'content_items'
            AND c.column_name = required.name
     );

    IF missing_columns IS NOT NULL THEN
        RAISE EXCEPTION
            'public.content_items exists but is missing expected columns: %. No data was changed',
            missing_columns;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.content_items'::regclass
           AND contype = 'p'
    ) THEN
        RAISE EXCEPTION 'public.content_items has no primary key; stopping rather than guessing';
    END IF;
END
$content_items_check$;

DO $content_items_constraints$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.content_items'::regclass
           AND conname = 'ck_content_items_channel'
    ) THEN
        ALTER TABLE public.content_items
            ADD CONSTRAINT ck_content_items_channel
            CHECK (channel IN ('instagram','facebook','tiktok','email','blog'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.content_items'::regclass
           AND conname = 'ck_content_items_format'
    ) THEN
        ALTER TABLE public.content_items
            ADD CONSTRAINT ck_content_items_format
            CHECK (format IN ('post','reel','story','ad','email','article'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.content_items'::regclass
           AND conname = 'ck_content_items_status'
    ) THEN
        ALTER TABLE public.content_items
            ADD CONSTRAINT ck_content_items_status
            CHECK (status IN ('idea','draft','scheduled','published'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM information_schema.table_constraints tc
          JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_catalog = tc.constraint_catalog
           AND kcu.constraint_schema = tc.constraint_schema
           AND kcu.constraint_name = tc.constraint_name
           AND kcu.table_schema = tc.table_schema
           AND kcu.table_name = tc.table_name
          JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_catalog = tc.constraint_catalog
           AND ccu.constraint_schema = tc.constraint_schema
           AND ccu.constraint_name = tc.constraint_name
         WHERE tc.constraint_type = 'FOREIGN KEY'
           AND tc.table_schema = 'public'
           AND tc.table_name = 'content_items'
           AND kcu.column_name = 'business_id'
           AND ccu.table_schema = 'public'
           AND ccu.table_name = 'businesses'
           AND ccu.column_name = 'id'
    ) THEN
        ALTER TABLE public.content_items
            ADD CONSTRAINT fk_content_items_business_id_businesses
            FOREIGN KEY (business_id) REFERENCES public.businesses (id) ON DELETE CASCADE;
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM information_schema.table_constraints tc
          JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_catalog = tc.constraint_catalog
           AND kcu.constraint_schema = tc.constraint_schema
           AND kcu.constraint_name = tc.constraint_name
           AND kcu.table_schema = tc.table_schema
           AND kcu.table_name = tc.table_name
          JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_catalog = tc.constraint_catalog
           AND ccu.constraint_schema = tc.constraint_schema
           AND ccu.constraint_name = tc.constraint_name
         WHERE tc.constraint_type = 'FOREIGN KEY'
           AND tc.table_schema = 'public'
           AND tc.table_name = 'content_items'
           AND kcu.column_name = 'campaign_id'
           AND ccu.table_schema = 'public'
           AND ccu.table_name = 'campaigns'
           AND ccu.column_name = 'id'
    ) THEN
        ALTER TABLE public.content_items
            ADD CONSTRAINT fk_content_items_campaign_id_campaigns
            FOREIGN KEY (campaign_id) REFERENCES public.campaigns (id) ON DELETE SET NULL;
    END IF;
END
$content_items_constraints$;

CREATE INDEX IF NOT EXISTS ix_content_items_business_date
    ON public.content_items (business_id, date);
CREATE INDEX IF NOT EXISTS ix_content_items_business_id
    ON public.content_items (business_id);
CREATE INDEX IF NOT EXISTS ix_content_items_campaign_id
    ON public.content_items (campaign_id);
CREATE INDEX IF NOT EXISTS ix_content_items_date
    ON public.content_items (date);
CREATE INDEX IF NOT EXISTS ix_content_items_status
    ON public.content_items (status);

-- d2f1a7c04b18: widen only; this preserves every existing URL value.
ALTER TABLE public.creatives ALTER COLUMN url TYPE text;

-- b7c8d9e0f1a2: composite query indexes.
CREATE INDEX IF NOT EXISTS ix_brand_memory_items_biz_kind
    ON public.brand_memory_items (business_id, kind);
CREATE INDEX IF NOT EXISTS ix_conversations_biz_updated
    ON public.conversations (business_id, updated_at);
CREATE INDEX IF NOT EXISTS ix_messages_conv_created
    ON public.messages (conversation_id, created_at);

-- c3e81b7a4d26: create saved audiences if dev create_all() did not already.
CREATE TABLE IF NOT EXISTS public.saved_audiences (
    id varchar(64) NOT NULL,
    business_id varchar(64) NOT NULL,
    name varchar(200) NOT NULL,
    description text,
    targeting jsonb NOT NULL,
    source varchar(20) NOT NULL DEFAULT 'manual',
    is_archived boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT saved_audiences_pkey PRIMARY KEY (id),
    CONSTRAINT fk_saved_audiences_business_id_businesses
        FOREIGN KEY (business_id) REFERENCES public.businesses (id) ON DELETE CASCADE
);

DO $saved_audiences_check$
DECLARE
    missing_columns text;
BEGIN
    SELECT string_agg(required.name, ', ' ORDER BY required.name)
      INTO missing_columns
      FROM unnest(ARRAY[
          'id','business_id','name','description','targeting','source',
          'is_archived','created_at','updated_at'
      ]) AS required(name)
     WHERE NOT EXISTS (
         SELECT 1
           FROM information_schema.columns c
          WHERE c.table_schema = 'public'
            AND c.table_name = 'saved_audiences'
            AND c.column_name = required.name
     );

    IF missing_columns IS NOT NULL THEN
        RAISE EXCEPTION
            'public.saved_audiences exists but is missing expected columns: %. No data was changed',
            missing_columns;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.saved_audiences'::regclass
           AND contype = 'p'
    ) THEN
        RAISE EXCEPTION 'public.saved_audiences has no primary key; stopping rather than guessing';
    END IF;
END
$saved_audiences_check$;

-- create_all() supplies ORM-side defaults; make the equivalent DB defaults explicit.
ALTER TABLE public.saved_audiences ALTER COLUMN source SET DEFAULT 'manual';
ALTER TABLE public.saved_audiences ALTER COLUMN is_archived SET DEFAULT false;
ALTER TABLE public.saved_audiences ALTER COLUMN created_at SET DEFAULT CURRENT_TIMESTAMP;
ALTER TABLE public.saved_audiences ALTER COLUMN updated_at SET DEFAULT CURRENT_TIMESTAMP;

DO $saved_audiences_fk$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM information_schema.table_constraints tc
          JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_catalog = tc.constraint_catalog
           AND kcu.constraint_schema = tc.constraint_schema
           AND kcu.constraint_name = tc.constraint_name
           AND kcu.table_schema = tc.table_schema
           AND kcu.table_name = tc.table_name
          JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_catalog = tc.constraint_catalog
           AND ccu.constraint_schema = tc.constraint_schema
           AND ccu.constraint_name = tc.constraint_name
         WHERE tc.constraint_type = 'FOREIGN KEY'
           AND tc.table_schema = 'public'
           AND tc.table_name = 'saved_audiences'
           AND kcu.column_name = 'business_id'
           AND ccu.table_schema = 'public'
           AND ccu.table_name = 'businesses'
           AND ccu.column_name = 'id'
    ) THEN
        ALTER TABLE public.saved_audiences
            ADD CONSTRAINT fk_saved_audiences_business_id_businesses
            FOREIGN KEY (business_id) REFERENCES public.businesses (id) ON DELETE CASCADE;
    END IF;
END
$saved_audiences_fk$;

CREATE INDEX IF NOT EXISTS ix_saved_audiences_business_id
    ON public.saved_audiences (business_id);
CREATE INDEX IF NOT EXISTS ix_saved_audiences_is_archived
    ON public.saved_audiences (is_archived);
CREATE INDEX IF NOT EXISTS ix_saved_audiences_business_updated
    ON public.saved_audiences (business_id, updated_at);

-- Add both columns selected by the app. Both are nullable, so existing campaign rows remain valid.
ALTER TABLE public.campaigns
    ADD COLUMN IF NOT EXISTS audience_id varchar(64);
ALTER TABLE public.campaigns
    ADD COLUMN IF NOT EXISTS audience_snapshot jsonb;

DO $campaign_audience_check$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'campaigns'
           AND column_name = 'audience_id'
           AND data_type = 'character varying'
           AND character_maximum_length = 64
    ) THEN
        RAISE EXCEPTION 'campaigns.audience_id exists with an unexpected type; stopping without advancing Alembic';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'campaigns'
           AND column_name = 'audience_snapshot'
           AND data_type = 'jsonb'
    ) THEN
        RAISE EXCEPTION 'campaigns.audience_snapshot exists with an unexpected type; stopping without advancing Alembic';
    END IF;

    IF NOT EXISTS (
        SELECT 1
          FROM information_schema.table_constraints tc
          JOIN information_schema.key_column_usage kcu
            ON kcu.constraint_catalog = tc.constraint_catalog
           AND kcu.constraint_schema = tc.constraint_schema
           AND kcu.constraint_name = tc.constraint_name
           AND kcu.table_schema = tc.table_schema
           AND kcu.table_name = tc.table_name
          JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_catalog = tc.constraint_catalog
           AND ccu.constraint_schema = tc.constraint_schema
           AND ccu.constraint_name = tc.constraint_name
         WHERE tc.constraint_type = 'FOREIGN KEY'
           AND tc.table_schema = 'public'
           AND tc.table_name = 'campaigns'
           AND kcu.column_name = 'audience_id'
           AND ccu.table_schema = 'public'
           AND ccu.table_name = 'saved_audiences'
           AND ccu.column_name = 'id'
    ) THEN
        ALTER TABLE public.campaigns
            ADD CONSTRAINT fk_campaigns_audience_id_saved_audiences
            FOREIGN KEY (audience_id) REFERENCES public.saved_audiences (id) ON DELETE SET NULL;
    END IF;
END
$campaign_audience_check$;

CREATE INDEX IF NOT EXISTS ix_campaigns_audience_id
    ON public.campaigns (audience_id);

-- Record head only after all schema changes above succeeded in this transaction.
DO $advance_revision$
DECLARE
    revision_rows integer;
    current_revision text;
BEGIN
    SELECT count(*), min(version_num)
      INTO revision_rows, current_revision
      FROM public.alembic_version;

    IF revision_rows <> 1 OR current_revision IS DISTINCT FROM 'a7d8c3e219b4' THEN
        RAISE EXCEPTION
            'Alembic revision changed during repair (count=%, revision=%); rolling back',
            revision_rows, current_revision;
    END IF;

    UPDATE public.alembic_version
       SET version_num = 'c3e81b7a4d26'
     WHERE version_num = 'a7d8c3e219b4';
END
$advance_revision$;

COMMIT;
