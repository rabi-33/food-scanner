create table if not exists public.inspections (
  id uuid primary key,
  created_at timestamptz not null,
  status text not null,
  expiry_date date,
  filename text,
  ocr_text jsonb not null default '[]'::jsonb,
  barcodes jsonb not null default '[]'::jsonb,
  date_candidates jsonb not null default '[]'::jsonb
);

alter table public.inspections enable row level security;