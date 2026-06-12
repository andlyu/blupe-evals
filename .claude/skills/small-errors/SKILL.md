---
name: small-errors
description: >
  Quick symptom -> fix lookup for known teleop papercuts (no re-debugging). Use when: a
  familiar small failure appears (dead joystick/menu, stale saved IP, NO SIGNAL flash,
  B-button surprises, headset doze fallout) — check here FIRST before investigating; or
  when a small error gets a confirmed quick fix — add it here. Deep root-caused outages
  belong in the infra skill's Issue log instead; link entries there when one exists.
---

# small-errors — known papercuts and their 30-second fixes

Check this list BEFORE debugging. Each entry: symptom → fix → how to confirm. If the fix
doesn't work, it's not this papercut — escalate to the `infra` skill playbooks.

Rule of thumb for what lives here: if the remedy is a restart, a reconnect, or a retype,
it's a small error. If the remedy required reading source code, it goes in `infra`
ISSUE-NNN and gets only a pointer here.

## Joystick won't move the menu highlight (video still fine)

- **The headset now self-diagnoses this**: a red **"NO CONTROLLER INPUT"** banner appears in
  the HUD within 2 s of controller data stopping (liveness = head-pose jitter, bridge mode).
- **Symptom:** stick left/right does nothing in the eval menu; video/cameras keep working
  (input and video are SEPARATE paths — video proves nothing about input). Eval log shows
  `[dbg] ... jx=+0.00` forever while the stick moves, and `[eval] controller input LOST`.
- **Fix, in order:** (1) Network panel → tap the popup IP → **Controller + Send ON** —
  covers the common cause (headset doze / panel session died; the Quest never auto-reconnects).
  (2) Still dead → `docker restart xr-bridge`, then redo step 1 (vendor PC service inside the
  container wedges after abrupt disconnects: `docker logs xr-bridge` shows
  `JSON parsing error ... empty input` + `device missing`).
- **Confirm:** banner disappears; `grep "controller input" /tmp/eval_live.log` shows `back`;
  stick moves the yellow cursor.

## Headset doze killed everything (input + video at once)

- **Symptom:** took the headset off for a minute; now neither input nor video.
- **Fix:** the Quest kills ALL its sockets on sleep. Network panel → popup tap → Controller+
  Send ON; Camera panel → Listen → Confirm. No Mac-side restarts needed.
- **Prevent:** disable Auto-Sleep (Settings → Power) or tape the proximity sensor.

## Camera panel connects to nothing after changing networks (green screen)

- **Symptom:** green screen on Listen at a new location.
- **Fix:** the camera panel's saved IP is the OLD network's. Retype the Mac's current IP
  (`ipconfig getifaddr en0`), Listen → Confirm. One retype per network change — it re-saves.
- **Deep cause when retyping doesn't fix it:** infra ISSUE-003 (zombie control connection;
  fixed by preemption, but older eval builds deadlock).

## No popup in the Network panel

- **Symptom:** app open, but no IP-select dialog appears.
- **Fix:** the announcer isn't running or announces a stale IP (it auto-detects at START
  only): `pkill -f "xrtk_announce[.]py"` then
  `nohup .venv/bin/python scripts/xrtk_announce.py --unicast <quest-ip> &`. Re-open the app.
- **Confirm:** `/tmp/xrtk_announce.log` shows the CURRENT Mac IP and the right subnet.

## NO SIGNAL flashes in the camera view

- **Symptom:** a camera panel shows "NO SIGNAL camN (Ns)" briefly.
- **Fix:** none needed if it heals within ~4 s — the grabber auto-reopens dead streams
  (counts up forever = older eval build; restart the eval). Frequent reopens logged as
  `[camera] ... reopening` = chase the channel drop (infra playbook).

## B button sends the arm home unexpectedly

- **Symptom:** pressed B to toggle flat↔3D in the stereo view; the arm started moving.
- **Fix:** that's the known collision — B is BOTH the app's flat/3D toggle and our
  B=GO_HOME shortcut. Avoid B in stereo view; remap one of them if it keeps biting.
