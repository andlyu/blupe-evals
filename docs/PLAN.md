# blupe-evals — plan

## Goal (north star)

**Make it easy for a user to run teleop evals on any robot** — switching between **teleop and
policy** to **measure policy success**. The operator should be **far from the arm**. (Sim-first,
then real.)

The work decomposes into four design parts: **Safety · Setup/Integration · Success ·
Generalizability** (below). We **execute** them in four sim-first steps:

## Execution roadmap (4 steps, each ends in a check)

**Step 1 — YAM working on the Jetson** (local; no remote yet; sim → real)
1. Unblock YAM sim — fix `yam.urdf` mesh paths (resolvable relative) + the missing `link_6` mesh.
   *Check:* placo loads the URDF; `--check` passes (MuJoCo loads, `home` applies, IK reaches).
2. YAM under safe control in sim — `SimRobot(yam)` behind `SafeRobot` + `Console`.
   *Check:* headless safety tests (clamp, disarm, return-home) + sim arm moves under a stub driver.
3. YAM teleop in sim — Quest drives YAM in MuJoCo on the Jetson. *Check:* grip clutches, arm moves.
4. Real YAM — `Robot` adapter over i2rt; sim-first verify, then the same loop on the arm.
   *Check:* real YAM teleops with the speed clamp + graceful return-to-home.

**Step 2 — Basic policy + switching + scoring**
- Integrate a basic policy via the `run(robot, stop)` seam (stub first), switch teleop ↔ policy
  through the gate, and score pass/fail tracked run-over-run.
  *Check:* switch to a basic policy mid-session, kill cuts it; run twice → success rate shown.

**Step 3 — Works from far away, relying on cameras**
- Operator node split from the robot node; robot cameras streamed back; watch + kill remotely.
  *Check:* drive/watch from another machine with live camera; pull the link → arm holds (fail-safe).

**Step 4 — Easily (re)set up the repo for an arm** (cross-cutting; enables 1–3)
- Guided setup + preflight **doctor** (actionable ✗s) + the **add-an-arm checklist** + the
  **`--check`** validator, so a fresh Jetson or a new arm is fast and reproducible.
  *Check:* clean Jetson → doctor passes → `--check --arm <x>` green with no manual fiddling.

**The eval task is the user's, not ours** (scope cut). We don't author tasks, scenes, objects, or
success criteria — the user tells us what they want to evaluate. We provide the **machinery**:
- **Run a trial** + **switch teleop ↔ policy** through the gate.
- **Reset to start** — via teleop, or the user's own reset hook.
- **Record operator-marked pass/fail** run-over-run. Automatic success-checking stays the user's
  call, later.

Real hardware is not a separate step — each step is **sim-first, then graduate the same loop to real.**

## Foundation & repo decision

blupe-evals = **eval-console rebuilt on top of the XRoboToolkit teleop repo.** The XRoboToolkit
sim/teleop stack is the base input; we port eval-console's proven core onto it — the `Robot`/
`run(robot, stop)` boundary (`interface.py`), the `Console` gate + `SafeRobot` (`console.py`), and
the `ArmProfile`/`ARMS` registry (`arms.py`). Old `blupe-eval-console` is the reference, not a
dependency. Pure-Python core is headless-testable.

---

## Part 1 — Safety (the layer we never cede)

Two relationships, not one — **we own teleop; the user owns the policy.** One driver at a time.

1. **Teleop — we own it.** Our driver writes to the arm, so we **smooth + speed-limit our own
   teleop** (a hard per-joint velocity cap, rate-independent) → no joints jumping, no strange
   motions. (The XRoboToolkit framework doesn't do this — placo IK + `dt` only — so we add it.)
2. **Policy — the user owns it; we watch + halt.** The policy's `run` loop commands the arm
   directly; we **do not rewrite its commands** — silently clamping would corrupt what the eval
   measures. Instead we **monitor and halt**: a kill switch that's manual *and* auto-triggered when
   the policy breaches speed / workspace / joint bounds. `stop()` disarms instantly and holds, even
   if the loop ignores it (no-jump handoff).
3. **Sim-first verification** — the *same* `run(robot, stop)` runs against a `SimRobot` (from the
   arm's MJCF) **before** the real `Robot`. Catch jumpy/bad behavior where nothing can break.
4. **Safety authority is robot-side** — because the operator/VR is remote, the clamp + kill switch
   live on the **robot node** (near the arm). A network cut is then **fail-safe**: lose the link →
   arm stops/holds, never runs blind. Operator can *request*; robot node *enforces*.
5. **Graceful shutdown — return-to-home before disconnect.** When the robot is turned off, the arm is
   **eased to a home pose under control first, then torque-off / disconnect** — so it never drops or
   ends in a bad configuration. (Uses `Robot.home()` then the safe torque-off sequence.) The
   watchdog-limp is only the *uncontrolled* fallback for a crash / network drop, not the normal path.

Boundary: **we own teleop + oversight; the user owns the policy's commands** (we halt it, we don't
rewrite it); a physical E-stop stays theirs. Sim holds on stop; on an uncontrolled drop the real YAM
limps (~400 ms Damiao watchdog) — the graceful path (#5) returns it home first.

**Open:** sim-first = enforced gate vs. recommended? And on real hardware, is the policy **halt-only**,
or does it also keep a hard backstop clamp as a last resort (accepting that a clamp alters the run)?

---

## Part 2 — Setup / Integration (two things to integrate: the robot AND the policy)

Sequencing: **sim-readiness first, then merge to hardware.** The headline artifacts are a concrete
**checklist** + a **`--check` validator** that tells you exactly what's missing.

### Add an arm — checklist (`docs/add-an-arm.md`)
**Sim-ready (phase A)**
1. Gather the arm package: **URDF + meshes** (+ MJCF, or generate from URDF).
2. Fix mesh paths → **resolvable relative paths**; every link incl. the EE has its mesh.
   *(This was the YAM blocker: `package://` paths + a missing `link_6` mesh.)*
3. Build the MJCF scene: **actuators + `home` keyframe + teleop target mocap**.
4. Fill the **`ArmProfile`**: EE frame/link name (IK target), joint limits, home pose, speed cap.
5. Register it in the arm registry.
6. Run **`--check`** → placo loads URDF, MuJoCo loads scene, home applies, IK reaches. ✅ sim-ready.
7. **Live sim teleop**: Quest clutches + moves the arm in MuJoCo. ✅

**Hardware-ready (phase B)**
8. Write the thin **`Robot` adapter** (`read/command/info/home`) over their stack (e.g. i2rt/CAN).
9. Set **real EE site, gripper, CAN/joint config, watchdog + torque-off**.
10. Set **`SafeRobot` params**: velocity cap + workspace box.
11. **Sim-first verify** the run loop, then swap in the real `Robot`. ✅ hardware-ready.

### Integrate a policy — checklist (parallel)
- Seam: implement **`run(robot, stop)`** (calls their model, their cadence) — or a simpler
  `predict_action(obs) → action` we wrap in a default loop.
- We ship a **template + a stub policy** so it runs before their real model is wired.
- **Sim-first**, then the same loop on real. Selectable in the console (swap teleop ↔ policy).

### Guided install
- A **preflight "doctor"**: checks prereqs and prints the exact fix on each ✗ (not a stack trace).
- Platform-aware, **Orin/Linux first**; the setup *experience* is the bar, packaging deferred.

---

## Part 3 — Success (measure the policy, trust the number)

What we want the customer to achieve: a feedback loop on "is my policy good / improving?"

- **Run a task as a repeatable trial** (sim or real), mark **pass/fail**.
- **Track success run-over-run** — persisted history; this run vs. prior. *(eval-console v1 had no
  persistence — this is new.)*
- **Generate a report for users (core)** — runs + success rate + run-vs-run + failure notes, as a
  shareable artifact.
- **Operate from far away / a different room** — run teleop or watch a policy remotely.
- **Switch teleop ↔ policy** — the console gate (teleop to set up/reset, hand off to policy, cut).
- **Camera-in-the-loop on policy runs** — the **cameras the robot sees** stream live to the operator
  while the policy drives, so you can **watch from another room** and **kill** it if it misbehaves.
- **Save video / episode data (ideally)** — the same live stream tapped to disk: per-run video +
  episode (state/action/obs per step) for **replays** and the report's failure insights.

**Open:** minimum report contents (rate + history only, or + per-run notes/video links)?
Video/episode capture in v1 or later?

---

## Part 4 — Generalizability (a design discipline + clean seams)

Mostly a principle we thread through the code: **anything device-, platform-, or stack-specific
lives behind an interface; the core (Safety, gate, eval) references none.** The seams:

- **Input seam (teleop)** — don't hardcode VR. Quest now; **keyboard later** (and others beyond).
  The teleop driver is one impl of an input interface.
- **Control-platform seam** — generalize across the platforms that actually drive the robot
  (i2rt/CAN, ROS, DROID/polymetis, …) behind the `Robot` adapter; nothing above it knows the stack.
- **Machine seam (Mac/Linux)** — keep it clean so Mac *can* slot in, but **low priority** today
  (XRoboToolkit teleop binaries are Linux/Windows-only; Mac does sim/eval now, live Quest teleop on
  Mac is upstream-blocked).
- **Operator ⇄ robot node split** — what's connected to the **VR (operator node)** is separate from
  what **controls the arm (robot node)**, and the **VR is expected to be far from the arm**. Never
  assume co-location. Network seam between them: input + commands one way, **camera + state** back;
  designed for distance, latency, and drops. (Safety authority pinned robot-side — see Part 1.)

**Open:** how many input devices to design the seam for now (just Quest, or Quest + keyboard)?
Remote bar — same-LAN or true over-the-internet operation?

---

## Architecture (cross-cutting)

```
 OPERATOR NODE (thin, may be far)              ROBOT NODE (near the arm)
  teleop in (VR now / keyboard later) ──cmd──► Console gate ─► we smooth+clamp ─► Robot adapter ─► arm
  watch + KILL ─────────────────────halt─────► monitor + halt ◄── run(robot, stop)  ← user's policy (drives arm directly)
  view: robot cameras + state ◄──────feed───── camera/state publisher          [sim-first: SimRobot]
                         network seam (distance / latency / drop = fail-safe)
```
(Teleop = ours, smoothed + speed-clamped. Policy = the user's, commands the arm directly; we only
**monitor + halt** — never rewrite its commands.)

- **Core (port from eval-console):** `interface.py` (Robot/Observation/run), `console.py`
  (Console + SafeRobot), `arms.py` (ArmProfile/ARMS).
- **From XRoboToolkit base:** sim teleop via `MujocoTeleopController` + placo IK; the Quest input.
- **New:** input-seam abstraction, run-log + report, camera publish/record, the doctor/setup,
  the network seam.

## Decisions still to pin
1. Sim-first — enforced gate vs. recommended.
2. Minimum report contents; video/episode capture v1 vs. later.
3. Input seam scope now (Quest only vs. Quest + keyboard).
4. Remote bar — LAN vs. internet.
