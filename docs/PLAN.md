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
6. **Move-to-home is speed-limited.** Every move to the home pose — the `GO_HOME` state *and* the
   shutdown path — must obey a **rate-independent max-speed cap** (rad/s): the arm eases home
   **slowly and predictably**, never a fast snap. (A per-frame step clamp varies with loop rate; the
   cap must be in rad/s, same as the teleop velocity cap in #1.)

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
12. **Verify the robot is being served** — before relying on the connection, run
    **`scripts/orin/check_serve.py`** (exit 0 = served). The serve replies with its `start_joints`
    handshake *only* once the real arm is actually connected, so a down serve / powered-off arm /
    dead CAN is caught **at setup** — not discovered mid-eval when you suddenly can't connect and it
    becomes a whole mess. (`--hold` also exercises the command path.) Live equivalent: the headset
    HUD shows **`ROBOT OFF`** until the `CONNECT` mirror receives that handshake. ✅ served.

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

## Remote topology (decided 2026-06-10)

Resolves "remote bar" (was open question 4): **internet, via tailnet, with operator-side
processing.** Diagrams: `docs/remote-topology.png` (where things run, ports, build status) and
`docs/what-happens-where.png` (what each piece does). Protocol: `docs/XR-INPUT-BRIDGE.md`.

- **Operator site (anywhere):** Quest (stock app, zero changes) + a **Mac laptop** on the same
  local Wi-Fi. The Quest's NAT-hostile link (video is *inbound* to Quest:12345) stays local;
  the Quest never needs to be reachable across the internet.
- **Mac = processing node.** The closed Linux-only PC Service + `xrobotoolkit_sdk` run in an
  **arm64 Docker container** (`docker/xr-bridge/`) with a small bridge republishing XR state on
  a socket; the eval (IK, sim, policy, HUD/encode) runs **natively on macOS** behind an input
  seam (`XR_INPUT=bridge|sdk|stub` — this is also Part 4's input seam; `stub` enables headless
  tests and keyboard later). Fallback if the service misbehaves in Docker: full Linux VM with
  bridged networking.
- **Robot site:** the Orin keeps only `YAM_control/yam_real_serve.py` (robot-side safety:
  clamp, hold-on-drop, torque-off) + CAN. Safety authority stays robot-side per Part 1 #4.
- **Cross-internet links (tailscale on Mac + Orin only):** the joints stream
  (`--serve-host`, :5599 newline-JSON) and, at M2, camera frames Orin → Mac (HUD composited on
  the Mac, re-encoded to the Quest locally). Heavy/latency-sensitive traffic (video, input)
  never leaves the operator's Wi-Fi. WAN drop = serve holds = fail-safe.

## Customer transport (decided 2026-06-11)

For customers, **no VPN — the robot node dials OUT to a relay we host** (outbound-only TLS
on 443: WSS v1, WebRTC/LiveKit endgame with camera frames as video tracks). The relay gives
us auth (per-robot tokens), tenancy, audit/metering, and a server-side kill switch. No VPN
for anyone — WE use the same relay (dogfood from day one). Our seams are already
direction-agnostic byte streams (joints JSON, camera frames), so this swaps transports in
`RobotLink`/the relays without touching robot-side safety (clamp, hold-on-drop, watchdog).
Needed from us: hosted relay + token issuance + dial-out wrappers on both ends + kill endpoint.

## Multi-arm + open release (decided 2026-06-12)

Decisions: next arm = **SO-101** (sim first, hardware after); distribution = **open GitHub
repo** (no closed components for now); audience = **pilot customers we onboard** (docs may
assume the YAM/Orin/Quest recipe, SO-101 broadens it).

This makes PLAN Part 2 concrete. Staged, each stage independently shippable:

**Stage 1 — ArmProfile + registry (the de-hardcoding).**
`arms/<name>/arm.json` + assets per arm; YAM becomes the first bundle. Profile fields:
`n_joints, ee_link, mjcf, urdf, home_keyframe, max_vel, serve {host, port}, gripper
{open, close}, cameras (default)`. `eval_yam_vr.py --arm yam` (default) loads it;
`eval_yam_states` constants (N_ARM=6, EE, MAX_VEL) become profile-driven via `E.load_arm()`;
policies take paths from the profile. Exit check: YAM runs exactly as today via its bundle.
SO-101 is 5-DOF + gripper — the registry must survive n_joints ≠ 6 everywhere ([:6] sweeps).

**Stage 2 — SO-101 sim bundle (phase A of add-an-arm).**
Vendor the MJCF from mujoco_menagerie (`trs_so_arm100`; verify license + that it matches
SO-101 rev), GENERATE the URDF from the MJCF (one source of truth — the gen_yam_urdf.py
pattern; teleop-integration.md's headline rule: identical link/joint names), scene.xml with
actuators + `home` keyframe + vis-target mocap + offwidth 1920, fill the profile, register.
Exit check: `--check` passes + live sim teleop from the headset + a pick_place trial in sim.

**Stage 3 — `arm check` validator + doctor (the guided-install experience).**
`scripts/arm_check.py --arm so101`: placo loads URDF, MuJoCo loads scene, home applies,
IK reaches a probe target, link names consistent across xml/urdf — each ✗ prints the fix.
Plus repo doctor: deps, docker bridge, ports, announcer, headset reachability.

**Stage 4 — hardware seam doc (phase B). DONE 2026-06-12, split by audience:**
`docs/integrate-your-hardware.md` (the common case: connect YOUR unit of a supported
embodiment — sim sanity check, serve, fleet one-liner, first contact) and
`docs/add-an-embodiment.md` (the advanced case: new arm MODEL — ArmSpec + consistent
MJCF/URDF + the serve wire protocol & safety contract). Both linked from the README
("I want to…" table). SO-101 hardware driver (feetech/lerobot) = its own later step.

**Stage 5 — GitHub-ready.**
Commit the working tree (FIRST — two threads of uncommitted work), LICENSE (decide; deps
are MIT/Apache), `.gitignore` runs/ + tokens, README = quickstart for pilots (today's
runbook + headset ritual + report workflow), pyproject + uv pins, config file
(`blupe.toml`: arm, hosts, task) to replace the CLI sprawl, one-command `up` script.
CI smoke (fake-quest headless e2e) once offscreen GL on runners is sorted.

Order: 1 → 2 → 3 can ship before 4/5; 5's commit step should happen IMMEDIATELY regardless.

## Decisions still to pin
1. Sim-first — enforced gate vs. recommended.
2. Minimum report contents; video/episode capture v1 vs. later.
3. Input seam scope now (Quest only vs. Quest + keyboard).
4. LICENSE for the open release (Apache-2.0 recommended; deps MIT/Apache-compatible).
5. Does `runs/` (trial videos) ever belong in git? (recommend: no — .gitignore + zip export).
