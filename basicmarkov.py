import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions
initial=tfd.Categorical(probs=[0.2,0.8])
transition=tfd.Categorical(probs=[[0.5,0.5],[0.9,0.1]])
observation=tfd.Normal(loc=[0.,15.],scale=[5.,10.])
model=tfd.HiddenMarkovModel(
    initial_distribution=initial,
    transition_distribution=transition,
    observation_distribution=observation,
    num_steps=2)
mean=model.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy)
