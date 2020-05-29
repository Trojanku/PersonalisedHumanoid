from humanoid_bullet_gym import update_world, build_world

TIME_STEP = 1. / 240.

arg_file = "args/train_test_args.txt"
world = None

if __name__ == '__main__':
    enable_draw = False
    world = build_world(enable_draw, arg_file)

    done = False
    while not done:
        update_world(world, TIME_STEP)

    world.shutdown()
