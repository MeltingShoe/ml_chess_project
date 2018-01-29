'''
This will be the master server for distributed training
It will use the play_api to generate training data by playing models against each other
It will handle client requests and ensure all active models are trained equally
It will also run validation tests where all active models play a round robin style tournament and generate evaluation metrics
These metrics will be served on a webpage
'''
