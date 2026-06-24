import unittest

from relay.relay import (CONTROL_PORT, Relay, Robot, _policy_running,
                         _policy_started, _policy_stopped)


class FakeWriter:
    def __init__(self, on_close=None):
        self.closed = False
        self.on_close = on_close

    def close(self):
        self.closed = True
        if self.on_close is not None:
            self.on_close()


class RelayControlTests(unittest.TestCase):
    def test_policy_running_reads_robot_heartbeat(self):
        self.assertTrue(_policy_running({"policy": {"running": "pick_place"}}))
        self.assertFalse(_policy_running({"policy": {"running": None}}))
        self.assertFalse(_policy_running({}))

    def test_policy_started_results(self):
        self.assertTrue(_policy_started({"result": "POLICY started"}))
        self.assertTrue(_policy_started({"result": "policy already running"}))
        self.assertFalse(_policy_started({"result": "REFUSED: serve is off"}))

    def test_policy_stopped_results(self):
        self.assertTrue(_policy_stopped({"result": "POLICY stopped"}))
        self.assertTrue(_policy_stopped({"result": "POLICY killed"}))
        self.assertTrue(_policy_stopped({"result": "no policy running"}))
        self.assertFalse(_policy_stopped({"result": "POLICY started"}))


class RelayCloseChannelTests(unittest.IsolatedAsyncioTestCase):
    async def test_close_control_channel_closes_writers_and_waits_for_pop(self):
        relay = Relay(state_path=None, env_tokens={}, admin_token="")
        robot = Robot(writer=FakeWriter())

        def pop_channel():
            robot.channels.pop(1, None)

        writer = FakeWriter(on_close=pop_channel)
        robot.channels[1] = {
            "port": CONTROL_PORT,
            "peer": "test",
            "since": 0.0,
            "_writers": (writer,),
        }

        result = await relay._close_channels(robot, CONTROL_PORT, "run_policy")

        self.assertTrue(writer.closed)
        self.assertEqual(result["closed"], 1)
        self.assertEqual(result["remaining"], 0)
        self.assertEqual(result["reason"], "run_policy")


if __name__ == "__main__":
    unittest.main()
