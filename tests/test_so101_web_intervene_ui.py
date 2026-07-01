from pathlib import Path


SOURCE = Path("scripts/so101_web_intervene.py").read_text()


def test_so101_control_ui_labels_semantic_camera_cards() -> None:
    assert 'data-camera="front"' in SOURCE
    assert 'data-camera="side"' in SOURCE
    assert 'data-camera="wrist"' in SOURCE
    assert 'data-camera="masks"' in SOURCE
    assert '<div class="cam-badge">FRONT</div>' in SOURCE
    assert '<div class="cam-badge">SIDE</div>' in SOURCE
    assert '<div class="cam-badge">WRIST</div>' in SOURCE
    assert '<div class="cam-badge">MASKS</div>' in SOURCE


def test_so101_control_ui_uses_named_camera_snapshot_routes() -> None:
    assert "function startPreviewImage(id, path)" in SOURCE
    assert "CAMERA_PREVIEW_REFRESH_MS" in SOURCE
    assert "CAMERA_PREVIEW_STALL_MS" in SOURCE
    assert "startPreviewImage('camera-front', '/camera/front.jpg');" in SOURCE
    assert "startPreviewImage('camera-side', '/camera/side.jpg');" in SOURCE
    assert "startPreviewImage('camera-wrist', '/camera/wrist.jpg');" in SOURCE
    assert "class MjpegCameraSource:" in SOURCE
    assert "self.mjpeg_cameras: dict[str, MjpegCameraSource]" in SOURCE
    assert "renderCameraLabels(data);" in SOURCE


def test_sam3_preview_route_has_policy_running_guard_method() -> None:
    assert "def is_policy_running(self) -> bool:" in SOURCE
    assert "if controller.is_policy_running() or controller._eval_running():" in SOURCE


def test_sam3_prompt_defaults_to_black_cylinder_with_insides() -> None:
    assert "black cylinder along with insides" in SOURCE


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


def test_live_view_can_record_only_intervention_episodes() -> None:
    assert 'id="evalInterventionsOnly" type="checkbox"' in SOURCE
    assert "Intervention episodes only" in SOURCE
    assert "body.page-setup:not(.live-view) .monitor-only" in SOURCE
    assert "label.intervention-only-option" in SOURCE
    assert "record_interventions_only: onlyInterventions" in SOURCE
    assert 'record_interventions_only: bool = False' in SOURCE
    assert '"record_interventions_only": record_interventions_only' in SOURCE
    assert '"record_interventions_only": True' in SOURCE
    assert "def _record_interventions_only_enabled(" in SOURCE
    assert "def _start_intervention_only_recording(" in SOURCE
    assert '"record_start_trigger": "intervention_only"' in SOURCE
    assert '"control_sources_expected": ["leader_delta"]' in SOURCE
    assert 'name_prefix=f"so101_intervention_run{attempt:04d}"' in SOURCE
    assert 'outcome="intervention" if state == "complete" else "failure"' in SOURCE
    assert "record_interventions_only and run_num == 1 and self._motion_running()" in SOURCE
    assert "eval policy run {run_num} attached to existing policy" in SOURCE
    assert "if record_interventions_only or not record_episodes or attempt_record_dir is not None:" in SOURCE
    assert "intervention_requested = bool(self.intervention_status.get(\"requested\"))" in SOURCE
    assert "if intervention_requested and self._motion_running():" in SOURCE
    assert "if not record_interventions_only and tracker_success_count > last_tracker_success_count:" in SOURCE
    assert "if not record_interventions_only and now >= watchdog_deadline:" in SOURCE
    assert "from_eval=bool(config.get(\"from_eval\", False))" in SOURCE
    assert "controller.stop_eval()" in SOURCE
    assert "intervention episodes armed" in SOURCE


def test_live_view_start_recording_uses_visible_prompt_and_dataset_name() -> None:
    assert 'id="liveInstruction"' in SOURCE
    assert 'id="evalDatasetName"' in SOURCE
    assert 'id="liveRecordStatus"' in SOURCE
    assert 'id="liveSuccessIndicator"' in SOURCE
    assert 'id="liveSuccessOverlay"' in SOURCE
    assert 'id="liveSam3RerunButton"' in SOURCE
    assert 'onclick="rerunSam3Masks(\'front\', false)">Rerun SAM3</button>' in SOURCE
    assert "async function rerunSam3Masks(cameraOverride = '', useEditorConfig = true)" in SOURCE
    assert "const camera = cameraOverride || cfg.camera;" in SOURCE
    assert "const payload = {camera, ball_seed_slot: selectedBallSeedSlot};" in SOURCE
    assert "if (useEditorConfig) {" in SOURCE
    assert 'prompt=str(data["prompt"]) if "prompt" in data else None' in SOURCE
    assert 'min_score=float(data["min_score"]) if "min_score" in data else None' in SOURCE
    assert "setLiveRecordStatus(`rerunning SAM3 masks (${camera})`, 'warn');" in SOURCE
    assert "sam3RerunButton.disabled = data.mode === 'stopping';" in SOURCE
    assert 'Rerun SAM3 is disabled while stopping' in SOURCE
    assert 'Rerun SAM3 only works while idle' not in SOURCE
    assert 'id="sam3MinScore" type="number" value="0.25"' in SOURCE
    assert "body.live-view .cams { grid-template-columns: repeat(4" in SOURCE
    assert "body.page-monitor .cams { grid-template-columns: repeat(4" in SOURCE
    assert "getElementById('successOverlay')" not in SOURCE


def test_live_view_can_save_sam3_frames_and_predictions() -> None:
    assert "SAM_LABEL_ROOT = RAW_RECORD_ROOT / \"sam-labels\"" in SOURCE
    assert "DEFAULT_SAM_LABEL_SAVE_FPS" in SOURCE
    assert 'id="liveSam3SaveButton"' in SOURCE
    assert 'onclick="saveSam3Predictions(\'front\')">Save Frames + Predictions</button>' in SOURCE
    assert "async function saveSam3Predictions(camera = 'front')" in SOURCE
    assert "button.textContent = saving ? 'Stopping Save...' : 'Starting Save...';" in SOURCE
    assert "function renderSamLabelSaving(data)" in SOURCE
    assert "`Stop Saving (${labels.count || 0})`" in SOURCE
    assert "const path = saving ? '/api/success/save_predictions/stop' : '/api/success/save_predictions/start';" in SOURCE
    assert "def save_success_predictions(" in SOURCE
    assert "camera: Any = \"front\"" in SOURCE
    assert "def start_sam_label_saving(self, camera: Any = \"front\", fps: float = DEFAULT_SAM_LABEL_SAVE_FPS)" in SOURCE
    assert "def stop_sam_label_saving(self, join: bool = False)" in SOURCE
    assert "def _sam_label_saving_loop(self, camera_name: str, fps: float, session_dir: Path)" in SOURCE
    assert '"sam_labels": self._sam_label_snapshot_locked()' in SOURCE
    assert '\"kind\": \"so101_sam_predictions\"' in SOURCE
    assert '\"prediction_frame\": frame_paths.get(cam.name)' in SOURCE
    assert 'annotations_path = session_dir / "annotations.jsonl"' in SOURCE
    assert '\"container\": {' in SOURCE
    assert '\"ball\": {' in SOURCE
    assert 'elif parsed.path == "/api/success/save_predictions":' in SOURCE
    assert "controller.save_success_predictions(camera=data.get(\"camera\", \"front\"))" in SOURCE
    assert 'elif parsed.path == "/api/success/save_predictions/start":' in SOURCE
    assert 'elif parsed.path == "/api/success/save_predictions/stop":' in SOURCE


def test_success_mask_stream_renders_fresh_camera_frames_with_cached_masks() -> None:
    assert "SUCCESS_MASK_STREAM_FPS" in SOURCE
    assert "def render_success_overlay_jpeg" in SOURCE
    assert "controller.render_success_overlay_jpeg(\"front\")" in SOURCE
    assert "controller.success_condition.wait(timeout=1.0)" not in SOURCE


def test_stop_state_does_not_force_live_view() -> None:
    auto_active_start = SOURCE.index("const autoActive = (")
    auto_active_end = SOURCE.index("  );", auto_active_start)
    auto_active_block = SOURCE[auto_active_start:auto_active_end]

    assert "data.mode === 'policy'" in auto_active_block
    assert "data.mode === 'intervention'" in auto_active_block
    assert "data.mode === 'stopping'" not in auto_active_block
    assert 'class="monitor-actions"' not in SOURCE
    assert '<input id="evalWithTeleop" type="checkbox" checked>' in SOURCE
    assert '<input id="evalRecordEpisodes" type="checkbox" checked>' in SOURCE
    assert 'id="liveInterventionToggleButton"' not in SOURCE
    assert 'id="liveStopEvalButton"' not in SOURCE
    assert 'id="liveClearEvalButton"' not in SOURCE
    assert "Teleop Intervention" not in SOURCE
    assert 'onclick="startEval()">Start Eval</button>' in SOURCE
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
    assert "cameras: ['front', 'wrist', 'side']" in SOURCE
    assert "const policyAlreadyRunning = !!lastStatus && lastStatus.mode === 'policy' && !lastStatus.eval?.running" in SOURCE
    assert "await api('/api/record/start'" in SOURCE
    assert "setLiveRecordStatus('stopping current MolmoAct before eval...', 'warn');" in SOURCE
    assert "await api('/api/stop', {});" in SOURCE
    assert "await waitForMotionIdle();" in SOURCE
    assert "record_start_trigger: 'live_button_existing_policy'" in SOURCE
    assert "e.config?.dataset_name || e.dataset_name || val('evalDatasetName')" in SOURCE
    assert "setLiveRecordStatus('starting recording...', 'warn')" in SOURCE
    assert "const evalStatusEl = document.getElementById('evalStatus');" in SOURCE
    assert "if (evalStatusEl) evalStatusEl.textContent" in SOURCE


def test_live_view_has_record_teleop_button_for_continuous_capture() -> None:
    assert 'id="startTeleopRecordingButton"' in SOURCE
    assert 'onclick="startLiveTeleopRecording()">Record Teleop</button>' in SOURCE
    assert "async function startLiveTeleopRecording()" in SOURCE
    assert "name_prefix: 'so101_teleop_recording'" in SOURCE
    assert "capture_mode: 'continuous'" in SOURCE
    assert "record_start_trigger: 'live_teleop_button'" in SOURCE
    assert "episode_kind: 'teleop'" in SOURCE
    assert "control_sources_expected: ['leader_delta', 'manual']" in SOURCE
    assert "setLiveRecordStatus('stopping recording...', 'warn')" in SOURCE
    assert "controller.export_recorded_dataset" in SOURCE
    assert "resolve_record_export_cameras" in SOURCE
    assert "append=True" in SOURCE
    assert "auto_cameras: true" in SOURCE
    assert "skip_unusable=True" in SOURCE
    assert "setLiveRecordStatus(`error: ${startError}`, 'bad')" in SOURCE
    assert "recording armed; starts in ${wait}s" in SOURCE
    assert "controller.stop_recording(stop_policy=True)" in SOURCE
    assert "startButton.textContent = e.running ? 'Eval Running' : rec.running ? 'Recording...' : 'Start Eval';" in SOURCE
    assert "startButton.className = 'primary';" in SOURCE
    assert "|| data.mode === 'intervention'" in SOURCE
    assert "|| data.mode === 'stopping'" in SOURCE
    assert "|| !!rec.running" in SOURCE
    assert "|| !!intervention.active" in SOURCE


def test_policy_advanced_controls_configure_realtime_chunking() -> None:
    assert "DEFAULT_REALTIME_CHUNKING" in SOURCE
    assert 'SO101_POLICY_REALTIME_CHUNKING' in SOURCE
    assert 'REALTIME_QUERY_FRACTION = float(os.environ.get("SO101_POLICY_REALTIME_QUERY_FRACTION", "0.5"))' in SOURCE
    assert 'DEFAULT_HZ = float(os.environ.get("SO101_POLICY_HZ", "30"))' in SOURCE
    assert 'class="advanced-policy setup-only"' in SOURCE
    assert '<summary>Advanced</summary>' in SOURCE
    assert 'id="hz" type="number" value="30"' in SOURCE
    assert 'id="realtimeChunking" type="checkbox"' in SOURCE
    assert 'id="realtimeQueryFraction" type="number" value="0.5"' in SOURCE
    assert 'function checked(id, fallback = false)' in SOURCE
    assert "realtime_chunking: checked('realtimeChunking', false)" in SOURCE
    assert "realtime_query_fraction: Number(val('realtimeQueryFraction'))" in SOURCE
    assert "realtime_query_fraction_value" in SOURCE
    assert '"captured_at": captured_at' in SOURCE
    assert "stale_steps = int(round(stale_s * hz))" in SOURCE
    assert "skip_actions={skipped_steps}" in SOURCE
    assert "chunk = chunk[skipped_steps:]" in SOURCE
    assert "cur_cmd = self.read_state()" in SOURCE
    assert "max(1, min(n, int(round(n * query_fraction))))" in SOURCE
    assert "dispatch_after_steps = 1 if skipped_steps else" in SOURCE
    assert "chunk {pending.chunk_index} realtime underrun; waiting for query" in SOURCE
    assert '"policy_config": dict(self.active_policy_config or {})' in SOURCE


def test_episode_editor_marks_compaction_status() -> None:
    assert "def _compaction_status(" in SOURCE
    assert '"compacted": bool(compressed)' in SOURCE
    assert "'Dataset not compacted'" in SOURCE
    assert "raw frame playback may be choppy" in SOURCE
    assert "compacted_root_exists" in SOURCE


def test_episode_editor_uses_joint_graph_without_position_table() -> None:
    assert "Joint Positions" not in SOURCE
    assert "function renderJointPositions(sample)" not in SOURCE
    assert "joint-table" not in SOURCE
    assert 'canvas id="jointGraph"' in SOURCE
    assert "shoulder_pan" in SOURCE


def test_episode_editor_shows_joint_motion_graph() -> None:
    assert "Joint Motion" in SOURCE or "Joint graph" in SOURCE
    assert 'canvas id="jointGraph"' in SOURCE
    assert "function renderJointGraph()" in SOURCE
    assert "function graphSeries(jointIdx)" in SOURCE
    assert "function velocityMagnitudeSeries()" in SOURCE
    assert "Math.sqrt(sumSq)" in SOURCE
    assert "series = joints.map" in SOURCE
    assert "const yForPosition" in SOURCE
    assert "const yForVelocity" in SOURCE
    assert "drawGraphLine(ctx, item.rows, 'state'" in SOURCE
    assert "drawGraphLine(ctx, item.rows, 'action'" in SOURCE
    assert "drawGraphLine(ctx, velocityRows, 'velocityMagnitude'" in SOURCE
    assert "avg_velocity_magnitude" in SOURCE
    assert "velocity magnitude" in SOURCE
    assert "window.addEventListener('resize', () => renderJointGraph())" in SOURCE


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
    assert '"dataset_name": meta.get("dataset_name")' in SOURCE
    assert '"dataset_slug": meta.get("dataset_slug")' in SOURCE
    assert "ep?.dataset_slug || ep?.dataset_name" in SOURCE
    assert "def _count_jsonl_rows(" in SOURCE
    assert "for path in root.iterdir()" in SOURCE
    assert "for child in path.iterdir()" in SOURCE
