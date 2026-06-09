---
name: new-project
description: >
  Read the docs BEFORE writing code whenever we start using a new external project, library,
  framework, tool, or file format — or try a new part of one we already use. Vendor the
  authoritative docs into docs/refs/<project>/ as a persistent, searchable reference, then do
  an in-depth search of those docs + the project's own examples for OUR specific use case to
  see whether guidance already exists, and only then implement. Use when: adding/integrating a
  new dependency or format; doing something new with an existing one; or when files/APIs "don't
  add up" and we're trial-and-erroring against an external tool. The project's docs are the
  source of truth — not assumptions, not training-data memory, not "it looks like a one-liner."
---

# new-project — read the docs before you build

**Version: v1**

We adopt a lot of external machinery (XRoboToolkit, placo, MuJoCo, i2rt, LeRobot, robot model
formats…). The expensive failures here come from **implementing against assumptions instead of
reading the project's own docs and examples first.** When "the files don't add up," it is almost
always guidance that *existed and wasn't read*. This skill is the discipline that prevents that.

## The one rule

**The external project's docs + examples are the source of truth.** Not your memory of the API,
not what the file "looks like," not a close-enough analogy. Before you write code against a new
project (or a new corner of a known one): read its docs, **keep them as a local reference**, and
**search them in depth for our exact use case**. Write code last, citing what you followed.

## When this triggers

- Starting to use any external project/library/framework/tool/format we haven't used before.
- A **new** part of one we already use (an API, config path, format field, or feature we
  haven't exercised).
- We're trial-and-erroring against an external tool, or "the files don't add up." That is the
  signal we skipped this skill — stop and run it now.

## Procedure

### 1. Scope & pin the version
- Name the exact project, the **version we actually have** (`pip show <pkg>` / lockfile / git
  commit / release tag), and where its authoritative docs live (repo README, docs site,
  `examples/`, source). Note the license.
- Docs must match the installed version. Reading the latest docs for an older pinned release is
  a classic source of "the API doesn't match."

### 2. Vendor the docs as a persistent reference
Save authoritative material into **`docs/refs/<project>/`** in this repo so it is searchable and
survives across sessions — do not re-fetch the same pages every time.
- Capture: the overview/README, the specific doc pages for our use case, the **canonical example
  scripts**, and the actual **format/schema spec** if a file format is involved.
- Web access: use the global **`/browse`** skill (never the chrome MCP tools). GitHub blob pages
  are JS-heavy — fetch `https://raw.githubusercontent.com/<org>/<repo>/<ref>/<path>` instead.
- At the top of each saved file, record: **source URL + version/commit + date fetched.**
- Maintain `docs/refs/<project>/INDEX.md` — a **use-case → where-the-guidance-lives** map, so the
  next search is a lookup, not a re-read.

### 3. Deep search for OUR use case — before any code
This is the in-depth pass the work depends on. Search the vendored docs **and the project's own
examples** for the specific thing we are about to do:
- Has someone already done **exactly this**? (Examples directories are gold — model new code on
  the closest example.)
- The canonical config / API call for it.
- Required fields, naming conventions, and **format constraints** (e.g. which frame/link/site
  name, which path scheme, which units/angle convention).
- Documented **gotchas and known issues** — including the project's GitHub issues and CHANGELOG.

Cover the whole use case, not the first hit. If the search is large, fan it out (multiple
Explore agents / a workflow) rather than skimming one page.

### 4. Verify, don't assume
- Confirm capabilities/constraints against the **installed** version: `dir(obj)`, a 5-line probe
  script, or read the installed source. (On the sim port, two "X can't do Y" assumptions each
  cost hours — and both were wrong.)
- Confirm that names/paths actually resolve: link/body/site names, mesh paths, schemes like
  `package://`, env vars. A name that's off by one underscore fails silently or in a confusing way.

### 5. Implement, citing the source
- Model the code on the canonical example. In a comment, **cite the doc/example/issue** you
  followed (repo path or URL + version).
- Use the **real source asset/config**, never a "close enough" substitute — a substitution
  silently changes behavior you depend on.

### 6. Capture what you learned
- Write the **gotchas** and a **working minimal example** back into `docs/refs/<project>/`, and
  update `INDEX.md`, so the next use skips the search.
- If it's a project-wide simulator gotcha, also feed the `simulation` skill; if it's a durable
  project fact, save a memory.

## The cautionary tale this skill exists to prevent

Adding the YAM arm was painful because we built against assumptions instead of reading the
formats and the framework's own behavior:
- **Two disagreeing sources of truth.** placo (IK) wanted a URDF; MuJoCo (sim) used an MJCF;
  the two were exported separately and were inconsistent (~6 cm + ~180° in the EE frame). Every
  orientation attempt broke until we read that **placo can load an MJCF directly**
  (`placo.Flags.mjcf`) and pointed IK at the *same* model the sim uses.
- **Names that don't add up.** URDF links are `link_6` (underscore); MJCF bodies are `link6`
  (none). Three drifted copies of `yam.xml` existed; only one had the body renamed to `link_6` to
  match the IK config — so the same `link_name="link_6"` worked against one copy and failed
  against the others.
- **A handoff doc that hadn't been reconciled with the code** told us `link_name="link6"`, which
  matches no working setup.

Reading placo's MJCF support, the URDF/MJCF format conventions, and verifying the actual frame
names up front would have avoided nearly all of it. That is exactly steps 2–4 above.

## Reference layout (convention)

```
docs/refs/
  <project>/
    INDEX.md        use-case → where the guidance lives (start here every time)
    README.md       vendored overview (source URL + version + date at top)
    examples/       the canonical example scripts we model code on
    gotchas.md      hard-won surprises + the working minimal example
    spec.md         format/schema spec, if a file format is involved
```

A project may be a library (`xrobotoolkit_teleop`, `placo`) or a format/domain
(`robot-models` for URDF/MJCF). Keep one folder per source of truth.
