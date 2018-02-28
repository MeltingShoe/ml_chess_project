import model_defs

# a script to test PA. Works now but I have no idea what's calling _render()
net = model_defs.tcn
run = net()
run.perform_action()
print('success?')
