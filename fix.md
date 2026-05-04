# Fix Plan: Hari's Comments

**Deadlines:** Abstract May 4 · Full paper May 6

Hari's comments split into three buckets: **clarity rewrites** (most, mostly easy), **one substantive metric concern** (Δμ definition includes A₁), and **a missing-baselines concern in §5**.

---

## P0 — Must fix before May 4 (Abstract + Intro)

### 1. Abstract: foreground contributions, not just numbers
Hari's main complaint: the abstract reads as a list of results without a clear "what we contribute and why it matters." Restructure as:
- (a) State the problem (memory compression hides toxicity)
- (b) Pivot quickly to the three contributions — empirical finding (laundering), new metric (SPG), state-aware mitigation
- (c) For each, say **what the impact is**, not just what it is

Concretely: "we propose state-aware mitigation" should become "state-aware mitigation reduces Δμ from X to Y and SPG from X to Y."

### 2. Intro ¶2: the τ confusion (highest-impact fix)
Hari is genuinely confused, and reviewers will be too. Current text reports turn-level scores 0.030, 0.051, 0.010 — all far below τ=0.5 — and asks reader to care. Add one sentence early in intro:

> The standard threshold τ=0.5 is for filtering individual messages as overtly toxic; we are not claiming the laundered messages cross τ. We are claiming the **gap** between toxic-condition and neutral-condition outputs is statistically significant and behaviorally meaningful, even when both sides are below τ.

Then re-anchor numerical claims around Δμ and SPG (the gap), not around absolute scores.

### 3. Intro: "detoxify toxicity → reference"
Add citation for Detoxify (`unitary/detoxify`) on first mention.

### 4. Intro & abstract: clarify Δμ = 0.13 vs 0.38 units
Add parenthetical: "Δμ is the paired difference in mean Detoxify scores, in [0,1] units." Apply both in abstract and intro.

### 5. Intro: clarify the 0.110 turn-2 spike
Same units issue. State explicitly these are Detoxify scores.

### 6. Contributions list: rewrite as "novelty (vs prior work) + impact"
Hari is explicit: "in one sentence, explain the novelty of the contribution, and in one sentence, explain the impact." Each bullet needs a contrast sentence.

Example for SPG: *"Prior toxicity evaluation operates per-message above τ; SPG is the first metric to detect behavioral influence in classifier-clean states. Impact: it surfaces a 0.139 propagation gap that standard monitors miss entirely."*

---

## P0 — §3 (Problem) substantive issue

### 7. Δμ definition includes the focal node A₁ ⚠️ requires recomputation
**Hari is right.** Currently μ(G) averages over all nodes, including A₁ — whose toxicity is by construction high in the toxic condition. A positive Δμ could be entirely driven by A₁ itself.

**Fix:** Redefine
$$\mu(G) = \frac{1}{|V \setminus \mathcal{V}_{A_1}|} \sum_{v \in V \setminus \mathcal{V}_{A_1}} \mathrm{tox}(v),$$
i.e., average over **descendants of A₁ only**, excluding A₁-authored nodes.

**Action:** Need to recompute Table 1 with corrected definition. Propagation should still be highly significant (A₁ is one node of many; downstream effect is real), but magnitudes will shift slightly.

---

## P1 — §3 and §4 clarity fixes (target May 5)

### 8. §3.3: replace "hostile tone" / "combative responses" with measurable terms
Either say "toxicity (as measured by Detoxify)" or define operationally.

### 9. §3.3 ¶3 (parametric bias): "this channel interacts with 1 and 2"
Replace with named references: "parametric bias interacts with transcript backflow (channel i) and memory laundering (channel ii)." Apply this naming convention everywhere — including §3.4.

### 10. §3.4: define S and W explicitly
Currently the objective uses θ', S, W without unpacking. Add: "where S is the read-side context sanitizer and W is the write-side state gate (defined in §4)."

### 11. §3: state explicitly that one agent can author multiple nodes
Hari noticed this only became clear in §4.1. Add to §3.1 right after defining φ: V → A:
> "Note that φ is many-to-one: a single agent can author multiple nodes in the graph. In particular, A₁ may be assigned to multiple nodes under the multi-injection condition."

### 12. §3: add 3-sentence summary paragraph at end
Section 3 is long; Hari wants a bridge to §4. Suggested:
> "In summary, we have formalized agent-mediated discussion as a state machine with three persistence channels — transcript, memory, and parameters — and shown that any robust unlearning method must address all three. Section 4 introduces the corresponding three-pathway intervention framework."

### 13. §4.7 (DPO): "same toxic context C_T" is confusing
Hari is right — under the neutral condition, A₁'s post is neutral, so descending context isn't toxic. Fix: clarify that for DPO data construction, **both** responses (m_t⁺ and m_t⁻) are evaluated against the toxic-condition context (we replay both responses on the toxic context tree). Current paper says this but Hari didn't parse it; rewrite more carefully, ideally with a small diagram.

### 14. §4: add summary paragraph at end
Same reason as §3.

### 15. §4: explain novelty of read/write sanitization
Hari asks: what's novel beyond just redacting input? Answer: the **symmetric architecture** (read AND write, before AND after generation) plus the **state-as-target framing** — prior sanitization work treats it as input filtering only.

---

## P1 — §5 baselines (Hari's biggest §5 concern)

### 16. §5: discuss baselines (or explicitly state their absence)
Currently §5 has no baseline section. Hari says one paragraph is fine. Cover:
- **Output-only Detoxify filter** (already in §6.6 as a baseline; mention in §5 too)
- **DPO-only** — already planned
- **No-intervention** — already there
- Explicit statement: "We are not aware of prior published methods designed for multi-agent state-channel toxicity propagation. The closest comparators are output filtering and DPO, which we evaluate."

### 17. §5: cite or claim originality for AUTC, propagation radius, cascade fraction, TTFT
Hari asks where these metrics come from.
- AUTC, TTFT — likely adapted from epidemiology / cascade literature
- Propagation radius — graph-theoretic but as toxicity metric is yours
- Cascade fraction — yours
- SPG — yours (already flagged)

Add a citation for AUTC-style metrics (info-cascade literature, e.g., Goel/Watts) or explicitly state we adapt these to toxicity-propagation.

### 18. §5: convert bullets to prose
Hari called this out specifically: "we evaluate 3 memory settings" + 3 bullet lines, robustness axes, phenomenon block, mitigation block. Two-column NeurIPS format eats space with bullets. Replace with chunky paragraphs using inline "(a), (b), (c)" structure. Frees space for figures.

---

## P1 — §6 fixes

### 19. §6.1: "geometry depends on conversation structure" → "effect depends on conversation structure"
Trivial wording fix.

### 20. §6.1: report Max toxicity (in addition to mean)
Add column or sentence to laundering table.

### 21. SPG metric definition concerns
Hari has two issues:
- **(a)** Is the expectation over time slices or at a particular t? Formula is ambiguous. Fix: clarify in §3/§5 that SPG aggregates over all (M_t, v_{t+1}) pairs across rollouts where tox(M_t) < τ, then takes the difference of conditional means across toxic/neutral.
- **(b)** "Why is SPG needed when mean toxicity exists?" Answered by the result: mean memory toxicity is ~0.085 (well below any threshold), but SPG = 0.139 reveals the hidden behavioral gap. Make this rhetorical move up front: "Mean toxicity averages over states without conditioning on classifier-clean status; SPG isolates the gap that survives after thresholding, which is what a deployed safety monitor would actually see."

---

## Summary by deadline

| Deadline | Tasks |
|---|---|
| **May 4** (abstract) | #1 abstract restructure, #4 unit clarifications |
| **May 5** (intro + §3 substance) | #2 τ confusion, #3 Detoxify cite, #5 unit clarifications, #6 contributions rewrite, #7 Δμ recomputation, #8–#12 §3 clarity fixes |
| **May 6** (rest) | #13–#15 §4 fixes, #16–#18 §5 baselines + bullets→prose, #19–#21 §6 fixes |

**Code change required:** only #7 (Δμ-without-A₁ recomputation). Everything else is writing.