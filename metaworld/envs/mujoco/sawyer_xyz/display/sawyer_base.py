from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv
import random


class SawyerXYZEnvDisplay(SawyerXYZEnv):

    max_path_length = 1000

    def _random_table_and_floor(self):
        self.sim.model.geom_matid[
            self.sim.model.geom_name2id('floor')
        ] = random.randint(0, 3)

        table_id = random.randint(4, 7)
        self.sim.model.geom_matid[
            self.sim.model.geom_name2id('table_top')
        ] = table_id
        self.sim.model.geom_matid[
            self.sim.model.geom_name2id('table_body')
        ] = table_id

