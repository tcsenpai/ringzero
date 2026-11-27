# Contributing to ringzero

ringzero is an experiment that wants to become an operating system people can actually use. Ideas, bug reports,
specs and code are all welcome, from a one-line fix to a new driver.

## Ways to help

| You have | Do this |
|---|---|
| a bug | open an issue: edition (first line of the boot banner), host CPU (`!cpu`), guest RAM, what you typed, the serial output around it (`!log 50`) |
| an idea for a feature or a change in behaviour | propose it as an **OpenSpec change** (below), as a pull request or an issue |
| a fix or a feature ready to go | open a pull request (below) |
| a question | open a discussion or an issue |

## Propose a change with OpenSpec

The behaviour of ringzero is written down in [openspec/specs/](openspec/specs/): every command, tool, setting and
default, with the code it comes from. A change in behaviour starts as a proposal next to those specs, so it can be
discussed before anyone writes code.

```sh
npm install -g @fission-ai/openspec      # the OpenSpec CLI
openspec list --specs                    # what the system does today
openspec show settings                   # one capability in detail
```

Create `openspec/changes/<short-name>/` with:
- `proposal.md`: why, and what changes for the user;
- `specs/<capability>/spec.md`: the requirements and scenarios that are added, changed or removed;
- `tasks.md`: the steps to implement it (optional for a pure proposal).

`openspec validate <short-name> --strict` checks the format. Open a pull request with only the change folder to
propose it; the same folder carries the implementation later, and archiving the change merges it into the specs.

Design rule to keep in mind: **ringzero presents itself as a Unix.** The models already know Linux, so state is
exposed as files under familiar paths (`/proc`, `/sys`, `/dev`, `/etc`) before adding special tools. Exact Linux
formats are not required.

## Pull requests

- Build and test on Linux with KVM (`make model`, `make iso`, `make disk`, `make run-local`; tools in
  [INFO.md](INFO.md#build-and-run)). Never rely on a test that did not run: say what you ran and what you saw.
- Commit messages start with one of `feat`, `fix`, `refactor`, `docs`, `style`, `test`, `build`.
- Keep every public surface in step with the code: README, INFO, the docs, the built-in manual, `/help`, the
  specs and the website. The list is in [docs/SURFACES.md](docs/SURFACES.md); run `tools/surfaces_check.sh`
  before opening the pull request.
- Numbers in docs are measured, not estimated. Plain English, no marketing words.
- Code is GPL-2.0-or-later. New dependencies must be MIT, Apache, BSD, zlib, MPL or GPL-2.0(-or-later), no
  GPL-3.0-only code (see the license notes in INFO.md).
- AI-assisted contributions are fine; say so in one line in the pull request.

## Where to start

- [docs/ROADMAP.md](docs/ROADMAP.md) lists what is planned. Good first areas: more files under `/proc` and `/sys`,
  pipes and redirection between `/bin` programs, more WASI programs in `make wasm-apps`, demos in
  [docs/DEMOS.md](docs/DEMOS.md), tests in `tools/`.
- Bigger pieces: drivers for real hardware (network, NVMe, USB keyboards), a POSIX shell, other model families in
  the engine.

Thanks for building it with us.
