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
from omni.isaac.core.utils.stage import create_new_stage_async
from omni.isaac.core.utils.prims import get_prim_at_path


class TestGo1(omni.kit.test.AsyncTestCase):
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

        self._base_command = [0.0, 0, 0, 0]
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

    async def test_go1_add(self):
        self._path_follow = False
        self._auto_start = True

        await self.spawn_go1(model="Go1")
        await omni.kit.app.get_app().next_update_async()

        self._go1 = self._world.scene.get_object("Go1")

        self.assertEqual(self._go1.num_dof, 12)  # actually verify this number
        self.assertTrue(get_prim_at_path("/World/Go1").IsValid(), True)
        print("articulation check passed")
        await omni.kit.app.get_app().next_update_async()

        # if dc interface is valid, that means the prim is likely imported correctly

    async def spawn_go1(self, waypoints=None, model="Go1"):
        self._prim_path = "/World/" + model

        self._go1 = self._world.scene.get_object("Go1")

        if self._go1 is None:
            self._go1 = self._world.scene.add(
                Unitree(
                    prim_path=self._prim_path,
                    name=model,
                    position=np.array([1, 1, 0.45]),
                    physics_dt=self._physics_dt,
                    model=model,
                    way_points=waypoints,
                )
            )

        self._go1._qp_controller.ctrl_state_reset()

        self._world.add_physics_callback("go1_advance", callback_fn=self.on_physics_step)
        await self._world.reset_async()
        return

    def on_physics_step(self, step_size):
        if self._go1 and self._go1._handle:
            # print(self._base_command)
            self._go1.advance(
                dt=step_size, goal=self._base_command, path_follow=self._path_follow, auto_start=self._auto_start
            )
