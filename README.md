# CheckersBot
[Play against it](https://havorax.github.io/CheckersBot/)

American Checkers/English Draughts AI

---

A checkers playing agent that heuristically evaluates a given game state with a deep neural network.

The neural network model used is a multi-layer perceptron. The [demo AI](https://havorax.github.io/CheckersBot/) was initially trained with weakly supervised learning and further trained with reinforcement learning, by playing against itself.

## Tools Used

The model was implemented in [Keras](https://keras.io/), a deep learning library for Python.

The [web demo](https://havorax.github.io/CheckersBot/) uses [TensorFlow.js](https://js.tensorflow.org/) to run the Keras model.
