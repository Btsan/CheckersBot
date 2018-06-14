# CheckersBot
[Play against it](https://havorax.github.io/CheckersBot/)

American Checkers/English Draughts AI

---

A checkers playing agent that heuristically evaluates a given game state with a deep neural network.

The neural network model used is a multi-layer perceptron. The [demo AI](https://havorax.github.io/CheckersBot/) was initially trained with weakly supervised learning and further trained with reinforcement learning, by playing against itself.

## Room for Improvements

##### Look-Ahead
The checkers agent, after learning from 10,000 games against itself, performs badly by itself. This may be because it only learned to evaluate the immediate value of a game state and not the long-term value that also considers the opponent's actions.

This issue can be resolved by a look-ahead decision algorithm (such as the hardcoded MinMax algorithm in the [web demo](https://havorax.github.io/CheckersBot/)) but is time-consuming to calculate. A better solution would be to have the checkers agent train against a MinMax agent, so that it can learn to estimate the long-term MinMax value of game states. This erases the need to have the costly look-ahead algorithm, providng better run-time performance and AI gameplay level.

##### Convolutions
Another *possible* improvement is the usage of a convolutional neural network, instead of the multi-layer perceptron model. I had originally used an MLP, because the model considers the entire board state, without any loss of information. However, an MLP would have to learn the correct response to all possible game states, which are too many to consider. The number of game states also suggests a large number of weights and layers, than actually used in my application.

While I think an MLP model actually works quite well for checkers, and could possibly encode the good responses to all of game states it'd take very long to reach that level. A convolutional neural network would generalize the board state (as well as possible loss of information) and train an agent to achieve a high gameplay level sooner than an MLP agent. 

## Tools Used

The model was implemented in [Keras](https://keras.io/), a deep learning library for Python.

The [web demo](https://havorax.github.io/CheckersBot/) uses [TensorFlow.js](https://js.tensorflow.org/) to run the Keras model.
