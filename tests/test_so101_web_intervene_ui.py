from pathlib import Path


SOURCE = Path("scripts/so101_web_intervene.py").read_text()


def test_so101_control_ui_labels_semantic_camera_cards() -> None:
    assert 'data-camera="front"' in SOURCE
    assert 'data-camera="side"' in SOURCE
    assert 'data-camera="wrist"' in SOURCE
    assert '<div class="cam-badge">FRONT</div>' in SOURCE
    assert '<div class="cam-badge">SIDE</div>' in SOURCE
    assert '<div class="cam-badge">WRIST</div>' in SOURCE


def test_so101_control_ui_uses_named_camera_snapshot_routes() -> None:
    assert "startMjpegImage('camera-front', '/camera/front.mjpg');" in SOURCE
    assert "startMjpegImage('camera-side', '/camera/side.mjpg');" in SOURCE
    assert "startMjpegImage('camera-wrist', '/camera/wrist.mjpg');" in SOURCE
    assert "class MjpegCameraSource:" in SOURCE
    assert "self.mjpeg_cameras: dict[str, MjpegCameraSource]" in SOURCE
    assert "renderCameraLabels(data);" in SOURCE


def test_sam3_preview_route_has_policy_running_guard_method() -> None:
    assert "def is_policy_running(self) -> bool:" in SOURCE
    assert "if controller.is_policy_running() or controller._eval_running():" in SOURCE


def test_sam3_prompt_defaults_to_black_cylinder() -> None:
    assert "black cylinder along with the insides" in SOURCE


def test_eval_recording_starts_after_delay_and_only_during_policy_execute() -> None:
    assert 'DEFAULT_EVAL_RECORD_START_DELAY_S = float(os.environ.get("SO101_EVAL_RECORD_START_DELAY_S", "10"))' in SOURCE
    assert '"record_start_delay_s": DEFAULT_EVAL_RECORD_START_DELAY_S' in SOURCE
    assert 'start_attempt_recording(trigger="policy_delay", episode_kind="policy")' in SOURCE
    assert 'and self.mode == "policy"' in SOURCE
    assert 'and self.stage == "execute"' in SOURCE
    assert '"record_only_policy_execute": True' in SOURCE
    assert '"trajectory_time_source": trajectory_time_source' in SOURCE


def test_intervention_recordings_start_on_intervention_and_are_marked() -> None:
    assert 'trigger="intervention_start"' in SOURCE
    assert 'episode_kind="intervention"' in SOURCE
    assert '"had_intervention": had_intervention' in SOURCE
    assert '"control_sources": control_sources' in SOURCE
    assert 'episode_kind = "mixed_intervention"' in SOURCE
    assert '"intervention_id": intervention_request.get("id")' in SOURCE
    assert "stopped_by_request = self.intervention_stop_event.is_set()" in SOURCE
    assert 'if state == "complete" and not stopped_by_request:' in SOURCE
    assert "self._resume_paused_policy()" in SOURCE
    assert "def _restart_recording_for_intervention(" in SOURCE
    assert '"record_start_trigger": "intervention_restart"' in SOURCE
    assert '"discarded_policy_leadin": True' in SOURCE
    assert '"outcome": "policy_leadin_restarted_for_intervention"' in SOURCE
    assert 'name_prefix="so101_intervention_recording"' in SOURCE
    assert "restarted_recording_dir" in SOURCE


def test_live_view_start_recording_uses_visible_prompt_and_dataset_name() -> None:
    assert 'id="liveInstruction"' in SOURCE
    assert 'id="evalDatasetName"' in SOURCE
    assert 'id="liveRecordStatus"' in SOURCE
    assert 'id="liveSuccessIndicator"' in SOURCE
    assert 'onclick="startEval()">Start Recording</button>' in SOURCE
    assert 'id="stopRecordingButton"' in SOURCE
    assert 'id="headerStopRecordingButton"' in SOURCE
    assert 'onclick="stopRecording()" disabled>Stop Recording</button>' in SOURCE
    assert "document.getElementById('liveInstruction')?.addEventListener('input', () => syncModelPrompt('liveInstruction'))" in SOURCE
    assert "document.getElementById('instruction')?.addEventListener('input', () => syncModelPrompt('instruction'))" in SOURCE
    assert "function modelPromptValue()" in SOURCE
    assert "function liveRecordStatusText(data)" in SOURCE
    assert "function renderLiveSuccess(data)" in SOURCE
    assert "renderLiveSuccess(data);" in SOURCE
    assert "SUCCESS ${count}" in SOURCE
    assert "async function startLivePolicyRecording()" in SOURCE
    assert "if (lastStatus?.recording?.running)" in SOURCE
    assert "await stopRecording();" in SOURCE
    assert "instruction: modelPromptValue()" in SOURCE
    assert "dataset_name: val('evalDatasetName')" in SOURCE
    assert "const policyAlreadyRunning = !!lastStatus && lastStatus.mode === 'policy' && !lastStatus.eval?.running" in SOURCE
    assert "await api('/api/record/start'" in SOURCE
    assert "if (!String(e.message || '').includes('motion is already running')) throw e;" in SOURCE
    assert "await startLivePolicyRecording();" in SOURCE
    assert "record_start_trigger: 'live_button_existing_policy'" in SOURCE
    assert "e.config?.dataset_name || e.dataset_name || val('evalDatasetName')" in SOURCE
    assert "setLiveRecordStatus('starting recording...', 'warn')" in SOURCE
    assert "setLiveRecordStatus('stopping recording...', 'warn')" in SOURCE
    assert "setLiveRecordStatus(`error: ${startError}`, 'bad')" in SOURCE
    assert "recording armed; starts in ${wait}s" in SOURCE
    assert "startButton.textContent = rec.running ? 'Stop Recording' : e.running ? 'Recording...' : 'Start Recording';" in SOURCE
    assert "startButton.className = rec.running ? 'danger' : 'primary';" in SOURCE
    assert "|| data.mode === 'intervention'" in SOURCE
    assert "|| data.mode === 'stopping'" in SOURCE
    assert "|| !!rec.running" in SOURCE
    assert "|| !!intervention.active" in SOURCE


def test_record_only_success_finalizes_episode_and_restarts_after_delay() -> None:
    assert "def _handle_record_only_success(" in SOURCE
    assert "def _record_only_success_worker(" in SOURCE
    assert "record_rollover_active" in SOURCE
    assert '"rollover_active": rollover_active' in SOURCE
    assert '"rollover_remaining_s": rollover_remaining_s' in SOURCE
    assert '"running": record_running or rollover_active' in SOURCE
    assert 'outcome="success"' in SOURCE
    assert 'reason="mask_success"' in SOURCE
    assert "cancel_rollover=False" in SOURCE
    assert '"record_start_trigger": "post_success_delay"' in SOURCE
    assert '"previous_success_recording_dir": final_dir or str(recording_dir)' in SOURCE
    assert 'name_prefix="so101_policy_recording"' in SOURCE
    assert "record-only episode finalized success" in SOURCE
    assert "record-only next episode armed" in SOURCE


def test_eval_recordings_are_grouped_under_dataset_name() -> None:
    assert 'dataset_name: str = ""' in SOURCE
    assert '"dataset_name": dataset_name' in SOURCE
    assert '"dataset_slug": dataset_slug' in SOURCE
    assert 'summary_dir = RECORD_ROOT / (dataset_slug or f"so101_eval_{timestamp}")' in SOURCE
    assert 'root_dir=summary_dir' in SOURCE
    assert '"episodes_root": str(summary_dir)' in SOURCE
    assert 'dataset_name=str(data.get("dataset_name", ""))' in SOURCE
    assert 'dataset_name = str(data.get("dataset_name", "")).strip()' in SOURCE
    assert 'root_dir=RECORD_ROOT / dataset_slug if dataset_slug else None' in SOURCE
    assert 'root.rglob("*")' in SOURCE
