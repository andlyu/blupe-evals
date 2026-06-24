import unittest

from scripts.relay_handoff import (can_switch_state, policy_start_accepted,
                                   release_teleop_link_for_policy,
                                   should_attach_teleop_link,
                                   should_release_command_link_for_state,
                                   should_retry_policy_start)


class FakeLink:
    def __init__(self):
        self.closed = []

    def close(self, shutdown=False):
        self.closed.append(shutdown)


class RelayHandoffTests(unittest.TestCase):
    def test_policy_handoff_releases_socket_without_disconnect(self):
        link = FakeLink()

        result = release_teleop_link_for_policy(connect_arm=True, link=link)

        self.assertTrue(result.connect_arm)
        self.assertIsNone(result.link)
        self.assertTrue(result.released)
        self.assertEqual(link.closed, [False])

    def test_policy_handoff_without_link_preserves_connect_state(self):
        result = release_teleop_link_for_policy(connect_arm=True, link=None)

        self.assertTrue(result.connect_arm)
        self.assertIsNone(result.link)
        self.assertFalse(result.released)

    def test_does_not_attach_teleop_while_policy_is_running(self):
        self.assertFalse(
            should_attach_teleop_link(
                connect_arm=True,
                relay_arm_status="serve_ready",
                state="POLICY",
                link=None,
            )
        )

    def test_does_not_attach_teleop_while_idle(self):
        self.assertFalse(
            should_attach_teleop_link(
                connect_arm=True,
                relay_arm_status="serve_ready",
                state="IDLE",
                link=None,
            )
        )

    def test_attaches_teleop_only_when_teleop_is_active(self):
        self.assertTrue(
            should_attach_teleop_link(
                connect_arm=True,
                relay_arm_status="serve_ready",
                state="TELEOP",
                link=None,
            )
        )

    def test_attaches_for_go_home(self):
        self.assertTrue(
            should_attach_teleop_link(
                connect_arm=True,
                relay_arm_status="serve_ready",
                state="GO_HOME",
                link=None,
            )
        )

    def test_no_reconnect_when_connect_is_off(self):
        self.assertFalse(
            should_attach_teleop_link(
                connect_arm=False,
                relay_arm_status="serve_ready",
                state="TELEOP",
                link=None,
            )
        )

    def test_no_reconnect_when_link_already_exists(self):
        self.assertFalse(
            should_attach_teleop_link(
                connect_arm=True,
                relay_arm_status="serve_ready",
                state="TELEOP",
                link=FakeLink(),
            )
        )

    def test_idle_releases_existing_command_link(self):
        self.assertTrue(should_release_command_link_for_state("IDLE", FakeLink()))

    def test_policy_releases_existing_command_link(self):
        self.assertTrue(should_release_command_link_for_state("POLICY", FakeLink()))

    def test_teleop_and_go_home_keep_existing_command_link(self):
        self.assertFalse(should_release_command_link_for_state("TELEOP", FakeLink()))
        self.assertFalse(should_release_command_link_for_state("GO_HOME", FakeLink()))

    def test_policy_active_blocks_mode_switches(self):
        self.assertFalse(
            can_switch_state(current_state="POLICY", target_state="TELEOP", policy_active=True)
        )
        self.assertFalse(
            can_switch_state(current_state="POLICY", target_state="GO_HOME", policy_active=True)
        )

    def test_policy_done_can_return_to_idle(self):
        self.assertTrue(
            can_switch_state(current_state="POLICY", target_state="IDLE", policy_active=True)
        )

    def test_policy_start_accepts_started_and_already_running(self):
        self.assertTrue(policy_start_accepted({"ok": True, "data": {"result": "POLICY started"}}))
        self.assertTrue(
            policy_start_accepted({"ok": True, "data": {"result": "policy already running"}})
        )

    def test_policy_start_retries_transient_serve_off_after_connect(self):
        data = {"ok": True, "data": {"result": "REFUSED: serve is off; Turn ON first"}}

        self.assertTrue(
            should_retry_policy_start(
                data=data,
                attempt=0,
                max_attempts=3,
                relay_arm_status="serve_ready",
            )
        )

    def test_policy_start_does_not_retry_serve_off_when_not_connected(self):
        data = {"ok": True, "data": {"result": "REFUSED: serve is off; Turn ON first"}}

        self.assertFalse(
            should_retry_policy_start(
                data=data,
                attempt=0,
                max_attempts=3,
                relay_arm_status="idle",
            )
        )

    def test_policy_start_stops_retrying_at_limit(self):
        data = {"ok": True, "data": {"result": "REFUSED: serve is off; Turn ON first"}}

        self.assertFalse(
            should_retry_policy_start(
                data=data,
                attempt=2,
                max_attempts=3,
                relay_arm_status="serve_ready",
            )
        )


if __name__ == "__main__":
    unittest.main()
