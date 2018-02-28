import model_defs

# just a script to test PA. Currently not 100% functional
net = model_defs.tcn
run = net()
run.evaluate()
print('success?')
