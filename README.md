## Critical time-dependent phenomena in diffusion generative models

### Abstract

In recent years, diffusion generative models have become state-of-the-art for
tasks such as image, video, and audio generation, among others. More recently,
there has been growing interest in studying the statistical mechanics of these
models, driven by the observation of apparent phase transitions during the
sampling process. More specifically, a symmetry breaking that resembles the one
encountered in the Ising model. In this proposal, a theoretical description of the
diffusion models is presented, explaining what diffusion models are, how they can
be studied from perspective of equilibrium statistical mechanics, and how critical
phenomena emerge in a simple case. Additionally, a simple simulation using a
feed forward network and two delta functions as the initial distributions provides
some insight into the model's behavior near the critical point. The main objectives
include a deeper investigation of this critical phenomenon, with particular focus
on questions concerning the relationship between data dimensionality and the
number of spins in the Ising model, as well as the emergence of scaling-free
properties. Furthermore, the development of new models is proposed to allow for
a more detailed observation and analysis of these critical behaviors.

### code info

*two_deltas.ipynb* is a jupyter notebook that contains a feed forward neural network trained to reconstruct a two delta initial distribution from pure noise.
