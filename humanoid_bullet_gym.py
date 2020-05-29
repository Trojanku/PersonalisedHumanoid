import time
import json

from pybullet_envs.deep_mimic.learning.rl_world import RLWorld
from pybullet_envs.deep_mimic.learning.ppo_agent import PPOAgent
from pybullet_envs.deep_mimic.env.pybullet_deep_mimic_env import PyBulletDeepMimicEnv
from pybullet_utils.arg_parser import ArgParser

TIME_STEP = 1. / 240.
ENABLE_DRAW = True
ANIMATING = True
STEP = False
MAX_STEPS = 10000
STEPS_COUNTER = 0

args_file = "args/run_humanoid3d_backflip_args.txt"
agent_file = "agents/humanoid_agent_ppo.txt"
motion_file = "data/motions/humanoid3d_backflip.txt"


def load_agent_data(path) -> json:
    with open(path) as file:
        json_data = json.load(file)

    return json_data


def update_world(world, time_step):
    world.update(time_step)

    global STEPS_COUNTER
    STEPS_COUNTER += 1

    reward = world.env.calc_reward(agent_id=id)
    end_episode = world.env.is_episode_end()

    if end_episode or STEPS_COUNTER >= MAX_STEPS:
        STEPS_COUNTER = 0
        world.end_episode()
        world.reset()


def build_world(enable_draw, arg_file):
    arg_parser = ArgParser()

    arg_parser.load_file(arg_file)
    arg_parser.parse_string("motion_file")

    env = PyBulletDeepMimicEnv(arg_parser=arg_parser,
                               enable_draw=enable_draw)

    world = RLWorld(env, arg_parser)

    agent_data = load_agent_data(agent_file)

    PPOAgent(world=world,
             id=id,
             json_data=agent_data)

    return world


if __name__ == '__main__':

    world = build_world(True, args_file)

    while world.env._pybullet_client.isConnected():
        time_step = TIME_STEP
        time.sleep(time_step)
        keys = world.env.getKeyboardEvents()

        if world.env.isKeyTriggered(keys, ' '):
            ANIMATING = not ANIMATING
        if world.env.isKeyTriggered(keys, 'i'):
            STEP = True
        if ANIMATING or STEP:
            update_world(world, time_step)
            step = False
