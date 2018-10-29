import numpy as np
from keras.models import model_from_json
import checkers


json_file = open('disc.json', 'r') # disc = rl
disc_json = json_file.read()
json_file.close()

meta_model = model_from_json(disc_json)
meta_model.load_weights('disc.h5')
meta_model.compile(optimizer='adadelta', loss='mean_squared_error')

base_model = model_from_json(disc_json)
base_model.load_weights('disc.h5')
base_model.compile(optimizer='adadelta', loss='mean_squared_error')

data = np.zeros((1, 32))
labels = np.zeros(1)
win = lose = draw = 0
for gen in range(0, 150):
	for game in range(0, 200):
		temp_data = np.zeros((1, 32))
		board = checkers.expand(checkers.np_board())
		player = np.sign(np.random.random() - 0.5)
		turn = 0
		while (True):
			moved = False
			boards = np.zeros((0, 32))
			if (player == 1):
				boards = checkers.generate_next(board)
			else:
				boards = checkers.generate_next(checkers.reverse(board))

			score = meta_model.predict_on_batch(boards) if (player == 1)  else base_model.predict_on_batch(boards)

			max = [0, score[0, 0]]
			for i, x in enumerate(score[:, 0]):
				if (x >= max[1] and np.random.random() > 0.2) or not moved:
					moved = True
					max = [i, x]

			if (player == 1):
				board = checkers.expand(boards[max[0]])
				temp_data = np.vstack((temp_data, checkers.compress(board)))
			else:
				board = checkers.reverse(checkers.expand(boards[max[0]]))

			if (game == 199 and gen > 145):
				print('\nTurn player: %d - Confidence = %f' %(player, max[1]))
				print(board)
			player = -player

			# punish losers more for short games, reward winners less for longer games
			winner = checkers.game_winner(board)
			if (winner == 1):
				win = win + 1
				temp_labels = meta_model.predict_on_batch(temp_data[1:]) + (np.reshape(np.linspace(0.01, 0.1, temp_data.shape[0] - 1), (temp_data.shape[0]-1, 1)) * (1./turn))
				data = np.vstack((data, temp_data[1:]))
				labels = np.vstack((labels, temp_labels))
				break
			elif (turn >= 200 or winner == -1):
				if (winner == - 1):
					lose = lose + 1
				else:
					draw = draw + 1
				temp_labels = meta_model.predict_on_batch(temp_data[1:]) - (np.reshape(np.linspace(0.0001, 0.001, temp_data.shape[0] - 1), (temp_data.shape[0]-1, 1)) * (201 - turn))
				data = np.vstack((data, temp_data[1:]))
				labels = np.vstack((labels, temp_labels))
				break
			turn = turn + 1
		print(np.array([gen*200 + game, win, lose, draw]))
		if ((game+1) % 100 == 0):
			meta_model.fit(data[1:], labels[1:], epochs=16, batch_size=256, verbose=0)
			if ((game+1) % 10 == 0 and np.random.random() > 0.5):
				data = np.zeros((1, 32))
				labels = np.zeros(1)

	meta_model.save_weights('disc.h5')
	base_model.load_weights('disc.h5')