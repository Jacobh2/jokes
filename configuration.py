class Config(object):
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 32
    size = 256
    num_layers = 3
    vocab_size = 25004
    steps_per_checkpoint = 2000
    epochs = 10