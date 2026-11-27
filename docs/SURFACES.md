# Public surfaces

Every place where ringzero is described to people. All of them must agree with the code. Any change that a user
can see (a feature, a command, a number, a requirement, a model option, a default) updates every surface in the
same piece of work, then `tools/surfaces_check.sh` must pass.

| Surface | Source in this repo | Published by | Audience |
|---|---|---|---|
| README | `README.md` | GitHub (on push) | users: features, try it, editions, credits |
| Technical notes | `INFO.md` | GitHub | developers: every command, numbers, layout, licenses |
| Beta guide | `docs/BETA.md` | GitHub | testers: requirements, run commands, known limits, checklist |
| Demos | `docs/DEMOS.md`, `tools/demo/` | GitHub, the data disk | users: copy-paste examples |
| Architecture | `docs/ARCHITECTURE.md` + `docs/ARCHITECTURE_DIAGRAMS.pdf` | GitHub, `tools/arch_pdf.sh` | developers |
| Roadmap | `docs/ROADMAP.md` | GitHub, `tools/roadmap.sh` | everyone |
| Built-in manual | `docs/MANUAL.md` | compiled into the kernel (T22, vectors T38) | the agent and `/man` |
| In-OS help | `/help` text in `kernel/src/shell.rs` | the kernel | users at the prompt |
| Website | `site/` | `tools/publish_site.sh` (tcsenpai.github.io/ringzero-site) | everyone |
| Download page (ISOs) | `release/hf/README.md`, `release/hf/LICENSE.md` | `hf upload` to Pleistocene/ringzero | testers |
| GitHub release notes | `release/NOTES.md` (per release) | `gh release create` | testers |
| GitHub About | description, homepage, topics | `gh repo edit` | everyone |
| Behaviour specs | `openspec/specs/*/spec.md` | GitHub | developers, agents: what the system does, with code sources |
| Result docs | `docs/*_RESULT.md` | GitHub | records of measurements: not rewritten, new results get new sections |

Rules (also in AGENTS.md):
- Public text is English, plain and factual, measured numbers only, no em dashes, no marketing words.
- No hostnames, IP addresses, machine paths or personal names on any surface.
- Italian appears only where it documents bilingual support (keyword lists, the manual's "Asked as" lines) or
  inside quoted test inputs in result docs.
- The behaviour specs in `openspec/specs/` are the reference for every other surface. A behaviour change is an
  OpenSpec change (propose, apply, archive) next to its mycelium task, and the archive updates the specs.
- `tools/spec_check.py`, run by `tools/surfaces_check.sh`, checks that the commands and `!set` keys in the specs,
  INFO.md and the /help table in kernel/src/shell.rs are the same set; `openspec validate --specs --strict`
  checks the spec format.
