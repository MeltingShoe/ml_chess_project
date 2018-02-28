import model_defs

if __name__ == '__main__':
    # a script to test PA. Works now but I have no idea what's calling _render()
    net = model_defs.tcn
    run = net()
    a = run.play_episode()
    print(a)
