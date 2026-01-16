CREATE TABLE IF NOT EXISTS "User" (
    id uuid PRIMARY KEY,
    identifier text UNIQUE NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    "createdAt" timestamptz DEFAULT NOW(),
    "updatedAt" timestamptz DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "Thread" (
    id uuid PRIMARY KEY,
    name text,
    "userId" uuid REFERENCES "User"(id),
    tags jsonb,
    metadata jsonb DEFAULT '{}'::jsonb,
    "createdAt" timestamptz DEFAULT NOW(),
    "updatedAt" timestamptz DEFAULT NOW(),
    "deletedAt" timestamptz
);

CREATE TABLE IF NOT EXISTS "Step" (
    id uuid PRIMARY KEY,
    "threadId" uuid REFERENCES "Thread"(id) ON DELETE CASCADE,
    "parentId" uuid,
    name text,
    type text NOT NULL,
    input text,
    output text,
    metadata jsonb DEFAULT '{}'::jsonb,
    "startTime" timestamptz,
    "endTime" timestamptz,
    "createdAt" timestamptz DEFAULT NOW(),
    "showInput" text,
    "isError" boolean DEFAULT false
);

CREATE TABLE IF NOT EXISTS "Element" (
    id uuid PRIMARY KEY,
    "threadId" uuid REFERENCES "Thread"(id) ON DELETE CASCADE,
    "stepId" uuid REFERENCES "Step"(id) ON DELETE SET NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    mime text,
    name text,
    "objectKey" text,
    url text,
    "chainlitKey" text,
    display text,
    size integer,
    language text,
    page integer,
    props jsonb DEFAULT '{}'::jsonb,
    "autoPlay" boolean,
    "playerConfig" jsonb
);

CREATE TABLE IF NOT EXISTS "Feedback" (
    id uuid PRIMARY KEY,
    "stepId" uuid REFERENCES "Step"(id) ON DELETE CASCADE,
    name text,
    value double precision,
    comment text
);
