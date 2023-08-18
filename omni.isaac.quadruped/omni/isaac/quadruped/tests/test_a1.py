# Copyright (c) 2018-2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# NOTE:
#   omni.kit.test - std python's unittest module with additional wrapping to add suport for async/await tests
#   For most things refer to unittest docs: https://docs.python.org/3/library/unittest.html
import omni.kit.test
import omni.kit.commands
import carb.tokens
import asyncio
import numpy as np
from omni.isaac.core import World
from omni.isaac.quadruped.robots.unitree import Unitree
from omni.isaac.core.utils.physics import simulate_async
from omni.isaac.quadruped.utils.rot_utils import get_xyz_euler_from_quaternion
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import create_new_stage_async


class TestA1(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        World.clear_instance()
        await create_new_stage_async()
        # This needs to be set so that kit updates match physics updates
        self._physics_rate = 400
        carb.settings.get_settings().set_bool("/app/runLoops/main/rateLimitEnabled", True)
        carb.settings.get_settings().set_int("/app/runLoops/main/rateLimitFrequency", int(self._physics_rate))
        carb.settings.get_settings().set_int("/persistent/simulation/minFrameRate", int(self._physics_rate))

        self._physics_dt = 1 / self._physics_rate
        self._world = World(stage_units_in_meters=1.0, physics_dt=self._physics_dt, rendering_dt=32 * self._physics_dt)
        await self._world.initialize_simulation_context_async()

        self._world.scene.add_default_ground_plane(
            z_position=0,
            name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
            static_friction=0.2,
            dynamic_friction=0.2,
            restitution=0.01,
        )

        self._base_command = [1.0, 0, 0, 0]
        self._stage = omni.usd.get_context().get_stage()
        self._timeline = omni.timeline.get_timeline_interface()

        self._path_follow = False
        self._auto_start = True

        await omni.kit.app.get_app().next_update_async()

        pass

    async def tearDown(self):
        await omni.kit.app.get_app().next_update_async()
        self._timeline.stop()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            print("tearDown, assets still loading, waiting to finish...")
            await asyncio.sleep(1.0)
        await omni.kit.app.get_app().next_update_async()
        pass

    async def test_a1_add(self):
        self._path_follow = False
        self._auto_start = True

        await self.spawn_a1()
        await omni.kit.app.get_app().next_update_async()

        self._a1 = self._a1 = self._world.scene.get_object("A1")
        await omni.kit.app.get_app().next_update_async()
        self.assertEqual(self._a1.num_dof, 12)
        self.assertTrue(get_prim_at_path("/World/A1").IsValid(), True)

        print("robot articulation passed")
        await omni.kit.app.get_app().next_update_async()

        # if dc interface is valid, that means the prim is likely imported correctly

    async def test_robot_move_command(self):
        self._path_follow = False
        self._auto_start = True

        await self.spawn_a1()
        await omni.kit.app.get_app().next_update_async()
        self._a1 = self._a1 = self._world.scene.get_object("A1")

        self.start_pos = np.array(self._a1.get_world_pose()[0])

        await simulate_async(seconds=2.0)

        self.current_pos = np.array(self._a1.get_world_pose()[0])

        print(str(self.current_pos))
        delta = np.linalg.norm(self.current_pos[0] - self.start_pos[0])

        self.assertTrue(delta > 0.5)

        pass

    async def test_robot_move_forward_waypoint(self):
        self._path_follow = True
        self._auto_start = True

        await self.spawn_a1(waypoints=[np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])])
        await omni.kit.app.get_app().next_update_async()
        self._a1 = self._world.scene.get_object("A1")
        await omni.kit.app.get_app().next_update_async()

        self.start_pos = np.array(self._a1.get_world_pose()[0])

        await simulate_async(seconds=1.5)

        self.current_pos = np.array(self._a1.get_world_pose()[0])

        delta = self.current_pos - self.start_pos
        print(str(delta))
        # x should be around 1, y, z should be around 0
        self.assertAlmostEquals(0.5, delta[0], 0)
        self.assertTrue(abs(delta[1]) < 0.1)
        self.assertTrue(abs(delta[2]) < 0.1)

    async def test_robot_turn_waypoint(self):
        self._path_follow = False
        self._auto_start = True
        # turn 90 degrees
        await self.spawn_a1()  # waypoints=[np.array([0.0, 0.0, -1.57])])
        await omni.kit.app.get_app().next_update_async()
        self._a1 = self._world.scene.get_object("A1")
        await omni.kit.app.get_app().next_update_async()
        self._base_command = [0.0, 0.0, 1.0, 0.0]

        self.start_quat = np.array(self._a1.get_world_pose()[1][[1, 2, 3, 0]])

        await simulate_async(seconds=1.5)

        self.current_quat = np.array(self._a1.get_world_pose()[1][[1, 2, 3, 0]])

        self.start_pos = get_xyz_euler_from_quaternion(self.start_quat)
        self.current_pos = get_xyz_euler_from_quaternion(self.current_quat)

        delta = np.array(abs(self.current_pos) - abs(self.start_pos))
        print(str(delta))
        self.assertTrue(abs(delta[2]) < 0.1)
        self.assertTrue(abs(delta[1]) < 0.1)
        self.assertTrue(abs(delta[0]) > 3.14 / 4)

    # Add this test when the controller has better side movement performance

    # async def test_robot_shift(self):
    #     await self.spawn_a1()

    #     # move side ways at 1.8 m/s (due to tuning, it is likely slower than that)
    #     self._base_command = [0.0, 1.8, 0, 0]
    #     await omni.kit.app.get_app().next_update_async()
    #     self._a1 = self._world.scene.get_object("A1")
    #     await omni.kit.app.get_app().next_update_async()

    #     self.start_pos = np.array(self.dc.get_rigid_body_pose(self._a1._root_handle).p)

    #     await simulate_async(seconds=10.0)

    #     self.current_pos = np.array(self.dc.get_rigid_body_pose(self._a1._root_handle).p)

    #     delta = self.current_pos - self.start_pos

    #     print("delta: " + str(delta))
    #     print("start: " + str(self.start_pos))
    #     print("current: " + str(self.current_pos))
    #     # y should be around 0.5, x, z should be around 0
    #     self.assertTrue(abs(delta[1]) > 0.5)
    #     self.assertTrue(abs(delta[0]) < 0.1)
    #     self.assertTrue(abs(delta[2]) < 0.1)

    async def spawn_a1(self, waypoints=None, model="A1"):
        self._prim_path = "/World/" + model

        self._a1 = self._world.scene.get_object("A1")

        if self._a1 is None:
            self._a1 = self._world.scene.add(
                Unitree(
                    prim_path=self._prim_path,
                    name=model,
                    position=np.array([0, 0, 0.40]),
                    physics_dt=self._physics_dt,
                    model=model,
                    way_points=waypoints,
                )
            )

        self._a1._qp_controller.ctrl_state_reset()

        self._world.add_physics_callback("a1_advance", callback_fn=self.on_physics_step)
        await self._world.reset_async()
        return

    def on_physics_step(self, step_size):
        if self._a1 and self._a1._handle:
            self._a1.advance(
                dt=step_size, goal=self._base_command, path_follow=self._path_follow, auto_start=self._auto_start
            )
