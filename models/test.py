import model_defs

if __name__ == '__main__':
    # a script to test PA. Works now but I have no idea what's calling _render()
    net = model_defs.fc_test
    run = net()
    a, b = run.play_episode()
    c = run.calc_future_reward(a, b, 0.5)
    print(c)
    run.training_session(c['dataloader'], 10)

