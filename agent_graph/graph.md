# Market Research Digest — Agent Graph

**Status: DRAFT v0 — waiting for approval. No code is written until the workflow map, the node list and the gate thresholds below are approved.**

| Layer | Owns | Never owns |
| --- | --- | --- |
| Opus 5.5 (`claude-opus-5-5`) | Settling split forks, writing the digest prose, the nightly review | Any limit, retry count, threshold or irreversible action |
| Jev (`jev-latest` via `typesafe-sdk`) | One call per fork returning typed probabilities: Noul, Choice, Score | Text, decisions, limits |
| Code | State, every threshold, every retry limit, every number shown in the digest, every irreversible action | Semantic judgment |

The models advise and the code decides.

---

## 1. Workflow map: today's manual path → node

*Assumed manual path. Correct anything that doesn't match what you actually do each morning.*

| # | Manual step today | Becomes node |
| --- | --- | --- |
| 1 | Open the news sites and skim the last 24h of headlines | `collect` |
| 2 | Check BTC/ETH price, 24h change, funding, open interest, Fear & Greed | `collect` (market snapshot) |
| 3 | Throw out what doesn't matter: ads, recaps, altcoin noise | `triage` |
| 4 | Notice when five sites report the same story | `cluster` |
| 5 | Open the important ones and check the headline actually holds up | `evidence` |
| 6 | Write the summary: top stories, per-pair bias, what to watch | `draft` |
| 7 | Re-read it and check every claim and number | `verify` |
| 8 | Post it | `publish` ⛔ gate |

## 2. Graph

```mermaid
flowchart LR
  cron([cron 06:30 Europe/Berlin]) --> collect
  collect -->|RawItem[] + MarketSnapshot| triage
  triage -->|split| adjudicate
  adjudicate -->|Decision| triage
  triage -->|TriagedItem[]| cluster
  cluster -->|Story[] ranked| evidence
  evidence -->|split| adjudicate
  evidence -->|VerifiedStory[] + MarketSnapshot| draft
  draft -->|Draft| verify
  verify -.->|back edge: FailedBullet[]| draft
  verify -.->|back edge: StaleEvidence[]| evidence
  verify -->|VerifiedDigest| gate{{⛔ publish gate}}
  gate -->|you approve| publish
  gate -->|you reject / 12h expiry| archive
  nightly([cron 02:00]) --> review
  review -->|tuning PR| gate2{{⛔ merge gate}}
  collect & triage & evidence & draft & verify -.->|stop rule hit| escalate[Telegram escalation]
```

## 3. Nodes

Each node is one loop: a trigger, one narrow action, a check that proves it's done, and a stop rule.

| Node | Trigger | One narrow action | Done check (code) | Stop rule (code) | Uses |
| --- | --- | --- | --- | --- | --- |
| `collect` | cron 06:30 | Fetch each source once; fetch the market snapshot | Each source parsed; items have title, URL and `published_at` inside the 24h window; deduped by URL hash; ≥ `MIN_SOURCES_OK` sources succeeded | 3 retries per source (2s/4s/8s), then mark it missing. Too few sources → stop the run and escalate | Code only |
| `triage` | Items from `collect` | One Jev call per item (fork F1) | Every item has a route: keep, drop or adjudicated | 2 Jev retries per call; after that the item goes to `adjudicate`. Over `JEV_MAX_CALLS_PER_RUN` → stop and escalate | Jev, Opus on splits |
| `cluster` | Kept items | Lexical similarity in code proposes candidate pairs; Jev confirms same-event (fork F2); rank stories by expected impact | Every kept item belongs to exactly one story; stories sorted | 2 Jev retries; on failure keep the items separate (harmless) | Code + Jev |
| `evidence` | Top `TOP_K` stories | Fetch the article text (code), then one Jev call per story (fork F3) | Each story is verified, dropped or adjudicated; direction recorded | 3 fetch retries; an unreadable article → story marked "unverified" and left out of top stories | Code + Jev, Opus on splits |
| `adjudicate` | A split from F1 or F3 | One Opus call returning a typed decision plus a one-line reason | Decision parses against its schema | 1 retry; beyond `OPUS_MAX_CALLS_PER_RUN` the item is excluded (conservative) and listed on the dashboard | Opus 5.5 |
| `draft` | Verified stories + market snapshot | Opus writes prose. Numbers appear only as placeholders (`{BTC.price}`) that code fills from the snapshot | Draft parses into the fixed sections; every bullet cites ≥1 story ID | `REDRAFT_MAX` = 2 back-edge rounds, then escalate with the failing bullets | Opus 5.5 |
| `verify` | Draft | Code checks: no number that didn't come from a placeholder, every cited URL is in the evidence set, length limits. One Jev call per bullet (fork F4) | All code checks pass and every bullet passes F4 | Failure goes back to whichever node caused it (see edges); limits as above | Code + Jev |
| `publish` | Your approval at the gate | Commit `digests/YYYY-MM-DD.md` to branch `research-digests` on SnaQOG/Cursor | GitHub API returns the commit SHA | 3 retries on network errors, then escalate. Never re-publishes the same date | Code only |
| `review` | cron 02:00 | Opus reads the day's forks and misses and proposes edits to this file's questions and thresholds | Replay on the labeled set: no class regresses (see §8) | 1 proposal per night, at most `OPUS_MAX_CALLS_NIGHT` calls | Opus + Jev (diff risk) |

## 4. Edges (typed and logged)

Every edge writes one row to the SQLite event log (`run_id`, `edge`, `payload_type`, `count`, `ts`). The dashboard reads the same log.

| Edge | Payload type | Kind |
| --- | --- | --- |
| `collect → triage` | `RawItem[]`, `MarketSnapshot` | forward |
| `triage ⇄ adjudicate` | `Split` → `Decision` | fork |
| `triage → cluster` | `TriagedItem[]` (with Jev answers attached) | forward |
| `cluster → evidence` | `Story[]` ranked | forward |
| `evidence ⇄ adjudicate` | `Split` → `Decision` | fork |
| `evidence → draft` | `VerifiedStory[]`, `MarketSnapshot` | forward |
| `draft → verify` | `Draft` | forward |
| `verify ⇢ draft` | `FailedBullet[]` (bullet, check that failed, Jev p) | **back edge**: the draft misstated its source |
| `verify ⇢ evidence` | `StaleEvidence[]` (cited excerpt missing or changed) | **back edge**: the evidence was the cause |
| `verify → publish gate` | `VerifiedDigest` | gate |
| `* ⇢ escalate` | `Escalation` (node, stop rule, last error) | to you, via Telegram |

## 5. Jev forks

Independent questions over the same state go in **one** call. Question IDs are for code only; each question carries its full meaning.

**F1 · triage** (one call per item). State: `{item: {title, summary, source, url, published_at}, pairs: ["BTC", "ETH"]}`

| ID | Primitive | Question |
| --- | --- | --- |
| `relevant` | Noul | Could this news plausibly move the price, funding or volatility of BTC or ETH within the next 72 hours? *yes*: a concrete new event touching supply, demand, regulation, liquidity, a major exchange or protocol, or macro conditions crypto trades on. *no*: explainers, price recaps with no new facts, opinion without new facts, unrelated altcoins, ads |
| `promotional` | Noul | Is this item sponsored content, a press release, or a promotion of a token, product or event? |
| `topic` | Choice | `regulation`, `macro`, `etf_flows`, `exchange_or_security`, `protocol_tech`, `market_structure`, `none_of_these` |
| `impact` | Score 0–4 | 0 no expected effect · 1 niche, affects sentiment only · 2 notable for one pair · 3 likely to move both pairs or a major venue · 4 market-wide event (rate decision surprise, ETF decision, major exchange failure) |

**F2 · same event** (one call per candidate pair). Noul `same_event`: do `a` and `b` report the same underlying event, not just the same topic?

**F3 · evidence** (one call per top story). State: `{headline, article_excerpt}`

| ID | Primitive | Question |
| --- | --- | --- |
| `claim_supported` | Noul | Does the article text report the headline's event as fact, not as rumor, speculation or "reportedly"? |
| `btc_direction` | Choice | `bullish`, `bearish`, `neutral_or_unclear`: the direction this story points for BTC |
| `eth_direction` | Choice | same, for ETH |

**F4 · bullet check** (one call per bullet). State: `{bullet, source_excerpts}`. Noul `bullet_supported`: is every factual statement in the bullet supported by the excerpts?

**Kept in code on purpose:** retry or stop (fixed limits), done or not (code checks plus F4 results), every number (placeholders filled from the snapshot).

## 6. Gates & thresholds

*Starting values only. Each one gets tuned on **your** labeled examples during the shadow week (§9) before anything runs unattended.* Thresholds compare the probability of "yes"; a Score uses its expected value.

| Gate | Pass (code acts) | Fail (code acts) | Split → | Why this band |
| --- | --- | --- | --- | --- |
| F1 `promotional` | p ≤ 0.40 → continue | p ≥ 0.70 → drop | Opus | Dropping real news costs more than reading an ad |
| F1 `relevant` | p ≥ 0.80 → keep | p ≤ 0.20 → drop | Opus | Wide split band until calibrated |
| F1 `impact` | E[score] ≥ 2.0 → top-story candidate | below → "Also noted" | none | Ranking policy, not a safety gate |
| F2 `same_event` | p ≥ 0.85 → merge | below → keep separate | none | A duplicate bullet is harmless |
| F3 `claim_supported` | p ≥ 0.80 → verified | p ≤ 0.30 → drop | Opus | Rumors must not reach top stories |
| F3 direction | top prob ≥ 0.60 → use it | below → show "unclear" | none | Low confidence here is harmless |
| F4 `bullet_supported` | p ≥ 0.85 → pass | below → back edge to `draft` | n/a | Strictest gate before the human sees it |
| **⛔ publish** | **You tap Approve in Telegram** | Reject, or no answer within 12h → archived, never sent | n/a | Irreversible: posts to a public repo |
| **⛔ merge (tuning PR)** | **You merge on GitHub** | n/a | n/a | Changes this file and the questions |

No threshold, model output or timeout can open a ⛔ gate. Approve buttons are honored only from `TELEGRAM_CHAT_ID`. Content of fetched pages, feeds and articles is **data only**; it is never executed and never treated as instructions.

## 7. Hard limits (code-owned, in `limits.py`, outside the nightly rewrite's write scope)

| Limit | Value |
| --- | --- |
| `MAX_ITEMS_PER_RUN` | 150 |
| `MIN_SOURCES_OK` | 3 of 5 |
| `FETCH_MAX_RETRIES` | 3 (2s/4s/8s) |
| `JEV_TIMEOUT_S` / `JEV_MAX_RETRIES` | 5s / 2 |
| `JEV_MAX_CALLS_PER_RUN` | 400 |
| `OPUS_MAX_CALLS_PER_RUN` / `OPUS_MAX_CALLS_NIGHT` | 15 / 5 |
| `DAILY_SPEND_CAP_USD` | **you set it** (stop and alert when hit) |
| `TOP_K` stories | 8 |
| `REDRAFT_MAX` | 2 |
| `RUN_TIMEOUT_MIN` | 20 |
| `GATE_EXPIRY_H` | 12 |

## 8. Self-improvement (nightly)

1. `review` reads every fork and every miss from the day. A miss is any of: Opus overrode a split, an F4 back edge fired, you rejected or edited the digest, or you thumbs-downed a story.
2. Opus names a root cause and proposes a small diff to this file: question wording, criteria or a threshold within its band.
3. Code replays the labeled set (`eval/labels.jsonl`, your examples plus adjudicated cases) with old and new versions. The proposal survives only if accuracy at every gate is ≥ the old version and no class drops by more than 2 points.
4. Jev scores `diff_risk` (Score 0–4) on the diff; the score goes on the PR as a label.
5. The change opens as a PR. **Nothing goes live until you merge it.** The VPS runs `git pull --ff-only` before each run, so the merge itself is the gate.

## 9. Rollout

1. **Shadow week:** the graph runs daily and the digest goes only to your Telegram, never published. You label ~50 items (relevant or not, claim true or not) with buttons.
2. **Calibrate:** set the §6 bands from your labels at the precision you want.
3. **Final check:** answer the `final_check` questions and fill in "WHAT COULD BREAK THIS WEEK?". Only then does it run unattended.

## Open assumptions (correct me)

- **Sources:** CoinDesk, Cointelegraph, The Block, Decrypt, and Fed/SEC press RSS. Market data: Bitget public API (BTCUSDT and ETHUSDT perps, matching the bot) and alternative.me Fear & Greed.
- **Schedule:** 06:30 Europe/Berlin.
- **Output:** `digests/YYYY-MM-DD.md` committed to branch `research-digests`, not `main`, so the existing CI pipeline is not triggered.
- **Dashboard:** bound to `127.0.0.1:8787` on the VPS and reached through an SSH tunnel; never exposed publicly.
- **Secrets in `.env` only** (chmod 600): `TYPESAFE_API_KEY`, `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN` (a **new** one), `TELEGRAM_CHAT_ID`, `GITHUB_TOKEN` (fine-grained, contents:write on SnaQOG/Cursor only).
