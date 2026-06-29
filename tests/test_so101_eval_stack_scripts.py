from pathlib import Path


START_SCRIPT = Path("scripts/start_so101_eval_stack.sh").read_text()
STOP_SCRIPT = Path("scripts/stop_so101_eval_stack.sh").read_text()
CHECK_SCRIPT = Path("scripts/check_so101_eval_stack.sh").read_text()
OVERVIEW = Path("docs/SO101-EVAL-STACK-OVERVIEW.md").read_text()
SAM_DOC = Path("docs/SO101-4090-SAM-EVAL.md").read_text()
ENV_EXAMPLE = Path("config/so101_eval_stack.env.example").read_text()
README = Path("README.md").read_text()


def test_so101_eval_stack_has_two_command_entrypoint() -> None:
    assert "scripts/start_so101_eval_stack.sh" in README
    assert "scripts/check_so101_eval_stack.sh" in README
    assert "scripts/stop_so101_eval_stack.sh" in README
    assert "scripts/start_so101_eval_stack.sh" in OVERVIEW
    assert "scripts/check_so101_eval_stack.sh" in OVERVIEW
    assert "scripts/stop_so101_eval_stack.sh" in OVERVIEW
    assert "scripts/start_so101_eval_stack.sh" in SAM_DOC
    assert "scripts/check_so101_eval_stack.sh" in SAM_DOC
    assert "scripts/stop_so101_eval_stack.sh" in SAM_DOC


def test_start_script_owns_gpu_services_tunnels_and_local_ui() -> None:
    assert "remote MolmoAct2 policy server on :8202" in START_SCRIPT
    assert "remote SAM3 prompt server on :8213" in START_SCRIPT
    assert "remote SAM2 tracker on :8214" in START_SCRIPT
    assert "local SSH tunnels for :8202/:8213/:8214" in START_SCRIPT
    assert 'POLICY_PORT="${SO101_POLICY_PORT:-8202}"' in START_SCRIPT
    assert 'SAM3_PORT="${SO101_SAM3_PORT:-8213}"' in START_SCRIPT
    assert 'SAM2_PORT="${SO101_SAM2_PORT:-8214}"' in START_SCRIPT
    assert 'CAMERA_RELAY_PORT="${SO101_CAMERA_RELAY_PORT:-8089}"' in START_SCRIPT
    assert 'UI_PORT="${SO101_WEB_PORT:-8092}"' in START_SCRIPT
    assert 'scripts/molmoact2_policy_runner.py' in START_SCRIPT
    assert 'scripts/sam3_prompt_ui.py' in START_SCRIPT
    assert 'SAM2_TRACKER="${SO101_SAM2_TRACKER:-image}"' in START_SCRIPT
    assert 'SAM2_SCRIPT="scripts/sam2_track_ui.py"' in START_SCRIPT
    assert 'SAM2_EXPECTED_MODE="sam2_image"' in START_SCRIPT
    assert 'SAM2_SCRIPT="scripts/sam2_video_track_ui.py"' in START_SCRIPT
    assert 'SAM2_EXPECTED_MODE="sam2_video"' in START_SCRIPT
    assert 'REMOTE_POLICY_PATH="${MOLMOACT2_POLICY_PATH:-__none__}"' in START_SCRIPT
    assert 'if [ "$POLICY_PATH" = "__none__" ]; then' in START_SCRIPT
    assert 'MOLMOACT2_IMAGE_KEYS_B64="$(printf \'%s\' "$MOLMOACT2_IMAGE_KEYS" | base64 | tr -d \'\\n\')"' in START_SCRIPT
    assert '"$MOLMOACT2_IMAGE_KEYS_B64" \\' in START_SCRIPT
    assert 'IMAGE_KEYS="$(printf \'%s\' "$IMAGE_KEYS_B64" | base64 -d)"' in START_SCRIPT
    assert 'SAM3_READY_PATH="${SO101_SAM3_READY_PATH:-/}"' in START_SCRIPT
    assert '"http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}"' in START_SCRIPT
    assert "ssh \"${SSH_OPTS[@]}\"" in START_SCRIPT
    assert '-L "${POLICY_PORT}:127.0.0.1:${POLICY_PORT}"' in START_SCRIPT
    assert '-L "${SAM3_PORT}:127.0.0.1:${SAM3_PORT}"' in START_SCRIPT
    assert '-L "${SAM2_PORT}:127.0.0.1:${SAM2_PORT}"' in START_SCRIPT
    assert "stop_existing_local_motion" in START_SCRIPT
    assert 'post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/record/stop"' in START_SCRIPT
    assert 'post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/eval/stop"' in START_SCRIPT
    assert 'post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/stop"' in START_SCRIPT
    assert '"$REPO_ROOT/scripts/launch_so101_eval_ui.sh"' in START_SCRIPT
    assert 'SO101_SUCCESS_SAM3_MIN_SCORE="${SO101_SUCCESS_SAM3_MIN_SCORE:-0.25}"' in START_SCRIPT
    assert 'SO101_SUCCESS_BALL_SAM3_PROMPT="${SO101_SUCCESS_BALL_SAM3_PROMPT:-blue rubber ball}"' in START_SCRIPT
    assert 'SO101_SUCCESS_BALL_SAM3_EVERY_N_FRAMES="${SO101_SUCCESS_BALL_SAM3_EVERY_N_FRAMES:-100}"' in START_SCRIPT
    assert 'SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES="${SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES:-2}"' in START_SCRIPT


def test_stop_script_stops_motion_local_processes_tunnels_and_remote_services() -> None:
    assert 'post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/record/stop"' in STOP_SCRIPT
    assert 'post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/eval/stop"' in STOP_SCRIPT
    assert 'post_json_best_effort "http://127.0.0.1:${UI_PORT}/api/stop"' in STOP_SCRIPT
    assert 'stop_screen "$UI_SCREEN"' in STOP_SCRIPT
    assert 'stop_screen "$RELAY_SCREEN"' in STOP_SCRIPT
    assert 'stop_screen "$TUNNEL_SCREEN"' in STOP_SCRIPT
    assert 'pkill -f \'scripts/molmoact2_policy_runner.py\'' in STOP_SCRIPT
    assert 'pkill -f \'scripts/sam3_prompt_ui.py\'' in STOP_SCRIPT
    assert 'pkill -f \'scripts/sam2_video_track_ui.py\'' in STOP_SCRIPT
    assert 'pkill -f \'scripts/sam2_track_ui.py\'' in STOP_SCRIPT
    assert 'SO101_STOP_VAST_INSTANCE' in STOP_SCRIPT
    assert 'vastai stop instance "$VAST_INSTANCE_ID"' in STOP_SCRIPT


def test_check_script_verifies_local_stack_and_optional_remote_gpu() -> None:
    assert 'SO101_CHECK_UI' in CHECK_SCRIPT
    assert 'SO101_CHECK_REMOTE' in CHECK_SCRIPT
    assert '"http://127.0.0.1:${CAMERA_RELAY_PORT}/health"' in CHECK_SCRIPT
    assert '"http://127.0.0.1:${POLICY_PORT}/health"' in CHECK_SCRIPT
    assert '"http://127.0.0.1:${SAM3_PORT}${SAM3_READY_PATH}"' in CHECK_SCRIPT
    assert '"http://127.0.0.1:${SAM2_PORT}/health"' in CHECK_SCRIPT
    assert '"http://127.0.0.1:${UI_PORT}/api/status?log_limit=1"' in CHECK_SCRIPT
    assert 'ssh "${ssh_opts[@]}" "${GPU_USER}@${GPU_HOST}"' in CHECK_SCRIPT
    assert "SO101 eval stack preflight passed." in CHECK_SCRIPT


def test_stack_env_example_documents_mutable_vast_config() -> None:
    assert "SO101_GPU_HOST=ssh2.vast.ai" in ENV_EXAMPLE
    assert "SO101_GPU_PORT=12394" in ENV_EXAMPLE
    assert "MOLMOACT2_CHECKPOINT_PATH=allenai/MolmoAct2-SO100_101" in ENV_EXAMPLE
    assert "MOLMOACT2_NORM_TAG=so100_so101_molmoact2" in ENV_EXAMPLE
    assert "SO101_SAM3_READY_PATH=/" in ENV_EXAMPLE
    assert "SO101_SAM2_TRACKER=image" in ENV_EXAMPLE
    assert "SO101_SUCCESS_BALL_SAM3_EVERY_N_FRAMES=100" in ENV_EXAMPLE
    assert "SO101_SUCCESS_BALL_SAM2_EVERY_N_FRAMES=2" in ENV_EXAMPLE
    assert "SO101_STOP_REMOTE_SERVICES=1" in ENV_EXAMPLE


def test_overview_documents_current_runtime_topology() -> None:
    assert "Browser -> SO101 eval UI :8092" in OVERVIEW
    assert "Camera relay :8089 -> local USB cameras" in OVERVIEW
    assert ":8202 -> GPU MolmoAct2 policy server" in OVERVIEW
    assert ":8213 -> GPU SAM3 prompt server" in OVERVIEW
    assert ":8214 -> GPU SAM2 tracker" in OVERVIEW
    assert "Converted v2.1 datasets are used for MolmoAct2 LoRA training." in OVERVIEW
