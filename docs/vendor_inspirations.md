# Vendor Inspirations

MosaicAlpha was built alongside four open-source repositories. Each one influenced
a specific aspect of the design. This document describes what was borrowed conceptually
and what was not.

**No code was copied from any of these repositories.** They are used as design
references only, and are not listed as package dependencies. They are expected to
be cloned into a sibling `../vendor/` directory for local reference, but MosaicAlpha
runs correctly whether or not they are present.

---

## agent-reach

**Influence area:** Data connector philosophy

agent-reach demonstrated a clean pattern for giving AI agents stable, modular access
to external data sources. Each connector is a thin adapter that handles authentication,
caching, and rate limiting, exposing a uniform interface to the rest of the pipeline.

MosaicAlpha's data layer follows the same philosophy: `yfinance`, FRED, and GDELT
connectors each live in a dedicated module, handle their own caching (Parquet files,
`--offline-sample` mode), and expose a uniform `DataFrame` interface to the feature
engineering layer. Adding a new data source (e.g., NOAA, SEC EDGAR) means writing
one new adapter module without touching the research pipeline.

---

## G0DM0D3

**Influence area:** Multi-model evaluation and research-review idea

G0DM0D3 demonstrated the value of side-by-side model comparison in a research workflow,
where multiple models are evaluated simultaneously and their outputs are compared
head-to-head.

MosaicAlpha's four-way ablation study (price-only vs price+macro vs price+news vs
price+macro+news) borrows this comparative mindset. Rather than testing one configuration
and reporting its IC, the pipeline always runs all four configurations in the same walk-forward
framework so the marginal contribution of each data source is immediately visible.

The future LLM research-reviewer layer on the roadmap is directly inspired by G0DM0D3's
multi-model evaluation approach: given an `ExperimentRecord`, an optional LLM reviewer
would generate a structured critique highlighting potential look-ahead risks, statistical
concerns, and limitation caveats.

---

## openhuman

**Influence area:** Local memory and experiment registry

openhuman emphasised local memory, managed services, and a simple-but-powerful design
philosophy. The project demonstrated that complex agent behaviour can be grounded in
human-readable, file-based state rather than opaque databases.

MosaicAlpha's experiment registry (`memory/experiments/`) is a direct application of
this idea. Every research run is persisted as a timestamped directory containing
`metadata.json`, `metrics.json`, and `report.md`. The entire registry is:

- **Human-readable**: any text editor can inspect it
- **Diffable**: `git diff memory/experiments/` shows exactly what changed between runs
- **Queryable**: `mosaic list-experiments` and `mosaic show-experiment` provide CLI access
- **Testable**: the registry format is pure JSON with no database dependency

The researcher stays in the loop on every decision because the decision is recorded
in a file they can read.

---

## Understand-Anything

**Influence area:** Research knowledge graph and explainability layer

Understand-Anything demonstrated that unstructured content — code, documentation,
research notes — can be turned into a searchable, queryable knowledge graph. The project
showed that making relationships between entities explicit (rather than implicit in
prose) dramatically improves navigability.

MosaicAlpha's graph layer (`mosaic_alpha/graph/`) applies the same idea to experiment
lineage. Datasets, features, models, experiments, metrics, findings, and limitations
are linked in a directed graph with typed nodes and edges:

- `dataset → produces → experiment` (which datasets fed which experiments)
- `experiment → uses → feature` (which features were active)
- `experiment → evaluated_by → metric` (which metrics were measured)
- `experiment → has_limitation → limitation` (what caveats apply)

Limitation nodes are deduplicated by MD5 hash so the same limitation text across
multiple experiments maps to a single node. This surfaces systemic limitations
(e.g., "look-ahead risk from FRED vintage") across all affected experiments in a
single graph query.

---

## Attribution and licensing

Each repository listed above is an independent open-source project with its own
license. MosaicAlpha does not redistribute or incorporate any code from these
projects. All MosaicAlpha source code is original.

To clone the vendor repos into the expected sibling directory:

```bash
mkdir -p ../vendor
cd ../vendor
git clone https://github.com/HunterMRocha/agent-reach
git clone https://github.com/HunterMRocha/G0DM0D3
git clone https://github.com/HunterMRocha/openhuman
git clone https://github.com/HunterMRocha/Understand-Anything
```

The dashboard's **Vendor Inspirations** page (`mosaic dashboard` → Vendor Inspirations)
reads README previews from these directories if they are present locally.
