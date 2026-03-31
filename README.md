# ManaSim

A domain-adaptive swarm simulation engine for organisational and 
educational decision-making. Built on MiroFish and OASIS, extended 
with a research agent, domain focusing layer, and validation framework.

## What is ManaSim?

MiroFish simulates public opinion on social media. ManaSim simulates 
how real human segments respond to policies, decisions, and artefacts 
inside organisations and learning environments.

You upload a policy, curriculum design, org change proposal, or any 
artefact. ManaSim researches how people like your stakeholders actually 
behave — drawing from academic literature, government data, and Reddit 
discussions — then simulates their response before you deploy anything 
in the real world.

## How it works

### 1. Select your domain and upload your artefact
Choose a simulation domain (education, healthcare, organisational 
change, policy, or define your own). Upload the document you want to 
test — a policy draft, curriculum plan, change proposal, or similar.

### 2. Research agent runs
ManaSim's research agent automatically scrapes three source types 
before every simulation:

- **Academic and government sources** — peer-reviewed literature, 
  government reports, empirical studies relevant to your domain and 
  artefact type
- **Reddit** — relevant subreddit discussions surfacing real sentiment, 
  resistance patterns, and lived experience
- **Case studies** — documented examples of similar artefacts being 
  implemented in comparable contexts

This happens fresh on every run. Agents are grounded in current 
evidence, not static templates.

### 3. Agent generation
The research agent produces a structured set of human segments — 
documented groups known to respond differently to this type of 
artefact in this domain. Each segment becomes a class of agents with 
behavioural profiles derived from the evidence, not from generic 
personality randomisation.

Examples in an educational domain might include: disengaged learners, 
high-achieving students, reluctant facilitators, early adopter 
instructors. In an organisational domain: resistant middle managers, 
executive champions, frontline workers, external stakeholders.

### 4. Simulation
Agents interact with the artefact and with each other across a 
domain-appropriate environment. Unlike MiroFish's social media 
environment, ManaSim environments model the structures relevant to 
your context — classrooms, team meetings, policy consultations, 
change rollouts.

### 5. Outcome report and validation
ManaSim produces a structured prediction report. Where historical 
data is available, outcomes are scored against known real-world 
results using the validation framework. Benchmark scenarios with 
documented outcomes are included in the repo so you can test 
simulation fidelity before running novel scenarios.

## Key differences from MiroFish

| | MiroFish | ManaSim |
|---|---|---|
| Simulation environment | Twitter / Reddit social media | Classroom, organisation, policy |
| Agent generation | Generic personality profiles | Research-grounded human segments |
| Source data | User-uploaded documents only | Live academic, government, Reddit sources |
| Memory layer | Zep Cloud (rate-limited) | Supabase (self-hosted, no rate limits) |
| Validation | None | Benchmark scenarios with scored outcomes |
| Domain focus | General purpose | Education and organisational change |

## Architecture
```
ManaSim
├── research-agent/         # Scrapes and structures source data per run
│   ├── academic/           # Semantic Scholar, government APIs
│   ├── social/             # Reddit via PRAW
│   └── cases/              # Case study retrieval and parsing
├── focusing-layer/         # Domain configuration and agent generation
│   ├── domains/            # Domain definitions (education, org, custom)
│   └── personas/           # Segment templates generated from research
├── environments/           # Simulation environments
│   ├── classroom/
│   └── organisation/
├── memory/
│   └── supabase/           # Replaces Zep Cloud
├── validation/
│   ├── scenarios/          # Benchmark scenarios with known outcomes
│   ├── scoring/            # Pluggable scoring rubrics
│   └── datasets/           # Anonymised reference datasets
└── simulation/             # OASIS engine (CAMEL-AI, Apache 2.0)
```

## Credits

ManaSim is built on [MiroFish](https://github.com/666ghj/MiroFish) 
by Guo Hangjiang and the [OASIS](https://github.com/camel-ai/oasis) 
framework by CAMEL-AI. The validation framework draws on the AITTTS 
framework for educational simulation scoring.

## Contributing

The validation layer is designed to be framework-agnostic. If you 
have a domain-specific rubric or benchmark dataset, contributions 
are welcome.
