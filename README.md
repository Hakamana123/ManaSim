# ManaSim

<div align="center">

**Domain-adaptive swarm simulation for organisational and educational 
decision-making.**

Built on [MiroFish](https://github.com/666ghj/MiroFish) and 
[OASIS](https://github.com/camel-ai/oasis).  
A [ManaEd](http://ManaEd.app) open source project.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![GitHub Stars](https://img.shields.io/github/stars/Hakamana123/ManaSim?style=flat-square)](https://github.com/Hakamana123/ManaSim/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/Hakamana123/ManaSim?style=flat-square)](https://github.com/Hakamana123/ManaSim/network)

</div>

---

## What is ManaSim?

MiroFish simulates public opinion on social media. ManaSim simulates how 
real human segments respond to policies, decisions, and artefacts inside 
organisations and learning environments.

You upload a policy, curriculum design, org change proposal, or any 
artefact. ManaSim researches how people like your stakeholders actually 
behave — drawing from academic literature, government data, and Reddit 
discussions — then simulates their response before you deploy anything 
in the real world.

---

## How it works

### 1. Select your domain and upload your artefact
Choose a simulation domain (education, organisational change, policy, 
healthcare, or define your own). Upload the document you want to test.

### 2. Research agent runs
ManaSim's research agent automatically scrapes three source types before 
every simulation:

- **Academic and government sources** — peer-reviewed literature and 
  empirical studies relevant to your domain and artefact type, via 
  Semantic Scholar and open government data
- **Reddit** — relevant subreddit discussions surfacing real sentiment, 
  resistance patterns, and lived experience (via PRAW)
- **Case studies** — documented examples of similar artefacts implemented 
  in comparable contexts

This runs fresh on every simulation. Agents are grounded in current 
evidence, not static templates.

### 3. Agent generation
The research agent produces a structured set of human segments — 
documented groups known to respond differently to this type of artefact 
in this domain. Each segment becomes a class of agents with behavioural 
profiles derived from evidence, not generic personality randomisation.

### 4. Simulation
Agents interact with the artefact and with each other across a 
domain-appropriate environment. ManaSim environments model the structures 
relevant to your context — classrooms, team meetings, policy consultations, 
change rollouts — not social media feeds.

### 5. Outcome report and validation
ManaSim produces a structured prediction report. Where historical data 
is available, outcomes are scored against known real-world results using 
the validation framework. Benchmark scenarios with documented outcomes 
are included in the repo.

---

## Key differences from MiroFish

| | MiroFish | ManaSim |
|---|---|---|
| Simulation environment | Twitter / Reddit social media | Classroom, organisation, policy |
| Agent generation | Generic personality profiles | Research-grounded human segments |
| Source data | User-uploaded documents only | Live academic, government, Reddit |
| Memory layer | Zep Cloud (rate-limited) | Pluggable backend (Supabase stub included) |
| Validation | None | Benchmark scenarios with scored outcomes |
| Domain focus | General purpose | Education and organisational change |

---

## Architecture
```
ManaSim/
├── backend/
│   └── app/
│       └── services/
│           ├── memory/
│           │   ├── base.py          ← abstract memory interface
│           │   └── supabase.py      ← Supabase implementation stub
│           ├── entity_reader.py
│           ├── graph_builder.py
│           ├── graph_tools.py
│           └── memory_updater.py
├── docs/
│   ├── memory-layer.md              ← how to implement a memory backend
│   └── supabase-schema.sql          ← reference schema (pgvector)
└── frontend/                        ← Vue.js SPA
```

---

## Quick Start

### Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 18+ | `node -v` |
| Python | 3.11–3.12 | `python --version` |
| uv | Latest | `uv --version` |

### 1. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env`:
```env
# LLM — any OpenAI-compatible endpoint
LLM_API_KEY=your_api_key
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL_NAME=qwen-plus

# Memory backend
MEMORY_BACKEND=supabase

# Reddit API (required for research agent)
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=manasim:v0.1 (by u/yourusername)
```

### 2. Install dependencies
```bash
npm run setup:all
```

### 3. Start
```bash
npm run dev
```

Frontend: http://localhost:3000  
API: http://localhost:5001

### Docker
```bash
cp .env.example .env
docker compose up -d
```

---

## Memory backend

ManaSim uses a pluggable memory layer. The default backend is Supabase 
but any backend can be implemented by creating a new module under 
`services/memory/`.

See [docs/memory-layer.md](docs/memory-layer.md) for the full interface 
contract and implementation guide.

To implement the Supabase backend:
1. Apply `docs/supabase-schema.sql` to your Supabase project
2. Add `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` to your `.env`
3. Implement the methods in `services/memory/supabase.py`

---

## Contributing

ManaSim is an open research contribution. Contributions are welcome in 
three areas:

- **Memory backends** — implement `MemoryBackend` for Neo4j, Postgres, 
  or other stores
- **Domain configs** — add new domain definitions with source lists and 
  segment templates
- **Validation scenarios** — contribute benchmark scenarios with 
  documented real-world outcomes

---

## Licence

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

Any modified version distributed or run as a network service must also 
be open source under the same licence.

---

## Credits

ManaSim is built on [MiroFish](https://github.com/666ghj/MiroFish) by 
Guo Hangjiang, with the simulation engine powered by 
[OASIS](https://github.com/camel-ai/oasis) from CAMEL-AI. We thank both 
teams for their open source contributions.
