import model_defs

if __name__ == '__main__':
    # a script to test PA. Works now but I have no idea what's calling _render()
    net = model_defs.tcn
    run = net()
    a, b = run.play_episode()
    c = run.calc_future_reward(a, b, 0.5)
    print(c)
