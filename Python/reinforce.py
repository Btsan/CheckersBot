import numpy as np
from keras.models import model_from_json

def num_captured(board):
	return 12 - np.sum(board < 0)

def num_branches(board, x, y):
	count = 0
	if (board[x, y] >= 1 and x < 6):
		if (y < 6):
			if (board[x+1, y+1] < 0 and board[x+2, y+2] == 0):
				board[x+2, y+2] = board[x, y]
				board[x, y] = 0
				temp = board[x+1, y+1]
				board[x+1, y+1] = 0
				count += num_branches(board, x+2, y+2) + 1
				board[x+1, y+1] = temp
				board[x, y] = board[x+2, y+2]
				board[x+2, y+2] = 0
		if (y > 1):
			if (board[x+1, y-1] < 0 and board[x+2, y-2] == 0):
				board[x+2, y-2] = board[x, y]
				board[x, y] = 0
				temp = board[x+1, y-1]
				board[x+1, y-1] = 0
				count += num_branches(board, x+2, y-2) + 1
				board[x+1, y-1] = temp
				board[x, y] = board[x+2, y-2]
				board[x+2, y-2] = 0
	if (board[x, y] == 3 and x > 0):
		if (y < 6):
			if (board[x-1, y+1] < 0 and board[x-2, y+2] == 0):
				board[x-2, y+2] = board[x, y]
				board[x, y] = 0
				temp = board[x-1, y+1]
				board[x-1, y+1] = 0
				count += num_branches(board, x-2, y+2) + 1
				board[x-1, y+1] = temp
				board[x, y] = board[x-2, y+2]
				board[x-2, y+2] = 0
		if (y > 1):
			if (board[x-1, y-1] < 0 and board[x-2, y-2] == 0):
				board[x-2, y-2] = board[x, y]
				board[x, y] = 0
				temp = board[x-1, y-1]
				board[x-1, y-1] = 0
				count += num_branches(board, x-2, y-2) + 1
				board[x-1, y-1] = temp
				board[x, y] = board[x-2, y-2]
				board[x-2, y-2] = 0
	return count

def possible_moves(board):
	count = 0
	for i in range(0, 8):
		for j in range(0, 8):
			if (board[i, j] > 0):
				count += num_branches(board, i, j)
	if (count > 0):
		return count
	for i in range(0, 8):
		for j in range(0, 8):
			if (board[i, j] >= 1 and i < 7):
				if (j < 7):
					count += (board[i+1, j+1] == 0)
				if (j > 0):
					count += (board[i+1, j-1] == 0)
			if (board[i, j] == 3 and i > 0):
				if (j < 7):
					count += (board[i-1, j+1] == 0)
				elif (j > 0):
					count += (board[i-1, j-1] == 0)
	return count


def game_winner(board):
	if (np.sum(board < 0) == 0):
		return 1
	elif (np.sum(board > 0) == 0):
		return -1
	if (possible_moves(board) == 0):
		return -1
	elif (possible_moves(reverse(board)) == 0):
		return 1
	else:
		return 0

def at_enemy(board):
	count = 0
	for i in range(5, 8):
		count += np.sum(board[i] == 1) + np.sum(board[i] == 3)
	return count

def at_middle(board):
	count = 0
	for i in range(3, 5):
		count += np.sum(board[i] == 1) + np.sum(board[i] == 3)
	return count

def num_men(board):
	return np.sum(board == 1)

def num_kings(board):
	return np.sum(board == 3)

def capturables(board): # possible number of unsupported enemies
	count = 0
	for i in range(1, 7):
		for j in range(1, 7):
			if (board[i, j] < 0):
				count += (board[i+1, j+1] >= 0 and board[i+1, j-1] >= 0 and  board[i-1, j+1] >= 0 and board[i-1, j-1] >= 0)
	return count

def semicapturables(board): # number of own units with at least one support
	return (12 - uncapturables(board) - capturables(reverse(board)))

def uncapturables(board): # number of own units that can't be captured
	count = 0
	for i in range(1, 7):
		for j in range(1, 7):
			if (board[i, j] > 0):
				count += ((board[i+1, j+1] > 0 < board[i+1, j-1]) or (board[i-1, j+1] > 0 < board[i-1, j-1]) or (board[i+1, j+1] > 0 < board[i-1, j+1]) or (board[i+1, j-1] > 0 < board[i-1, j-1]))
	count += np.sum(board[0] == 1) + np.sum(board[0] == 3) + np.sum(board[1:7, 0] == 1) + np.sum(board[1:7, 0] == 3) + np.sum(board[7] == 1) + np.sum(board[7] == 3) + np.sum(board[1:7, 7] == 1) + np.sum(board[1:7, 7] == 3)
	return count

def reverse(board):
	b = -board
	b = np.fliplr(b)
	b = np.flipud(b)
	return b

def get_metrics(board): # returns [label, 10 labeling metrics]
	b = expand(board)

	capped = num_captured(b)
	potential = possible_moves(b) - possible_moves(reverse(b))
	men = num_men(b) - num_men(-b)
	kings = num_kings(b) - num_kings(-b)
	caps = capturables(b) - capturables(reverse(b))
	semicaps = semicapturables(b)
	uncaps = uncapturables(b) - uncapturables(reverse(b))
	mid = at_middle(b) - at_middle(-b)
	far = at_enemy(b) - at_enemy(reverse(b))
	won = game_winner(b)

	score = capped + potential + men + kings + caps + semicaps + uncaps + mid + far + won
	if (score < 0):
		return np.array([-1, capped, potential, men, kings, caps, semicaps, uncaps, mid, far, won])
	else:
		return np.array([1, capped, potential, men, kings, caps, semicaps, uncaps, mid, far, won])

def np_board():
	return np.array(get_board())

def get_board():
	return [1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  0, 0, 0, 0,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1]

def expand(board):
	b = np.zeros((8, 8), dtype='b')
	for i in range(0, 8):
		if (i%2 == 0):
			b[i] = np.array([0, board[i*4], 0, board[i*4 + 1], 0, board[i*4 + 2], 0, board[i*4 + 3]])
		else:
			b[i] = np.array([board[i*4], 0, board[i*4 + 1], 0, board[i*4 + 2], 0, board[i*4 + 3], 0])
	return b

def compress(board):
	b = np.zeros((1,32), dtype='b')
	for i in range(0, 8):
		if (i%2 == 0):
			b[0, i*4 : i*4+4] = np.array([board[i, 1], board[i, 3], board[i, 5], board[i, 7]])
		else:
			b[0, i*4 : i*4+4] = np.array([board[i, 0], board[i, 2], board[i, 4], board[i, 6]])
	return b

def generate_branches(board, x, y):
	bb = compress(board)
	if (board[x, y] >= 1 and x < 6):
		temp_1 = board[x, y]
		if (y < 6):
			if (board[x+1, y+1] < 0 and board[x+2, y+2] == 0):
				board[x+2, y+2] = board[x, y]
				if (x+2 == 7):
					board[x+2, y+2] = 3
				temp = board[x+1, y+1]
				board[x+1, y+1] = 0
				if (board[x, y] != board[x+2, y+2]):
					board[x, y] = 0
					bb = np.vstack((bb, compress(board)))
				else:
					board[x, y] = 0
					bb = np.vstack((bb, generate_branches(board, x+2, y+2)))
				board[x+1, y+1] = temp
				board[x, y] = temp_1
				board[x+2, y+2] = 0
		if (y > 1):
			if (board[x+1, y-1] < 0 and board[x+2, y-2] == 0):
				board[x+2, y-2] = board[x, y]
				if (x+2 == 7):
					board[x+2, y-2] = 3
				temp = board[x+1, y-1]
				board[x+1, y-1] = 0
				if (board[x, y] != board[x+2, y-2]):
					board[x, y] = 0
					bb = np.vstack((bb, compress(board)))
				else:
					board[x, y] = 0
				bb = np.vstack((bb, generate_branches(board, x+2, y-2)))
				board[x+1, y-1] = temp
				board[x, y] = temp_1
				board[x+2, y-2] = 0
	if (board[x, y] == 3 and x > 0):
		if (y < 6):
			if (board[x-1, y+1] < 0 and board[x-2, y+2] == 0):
				board[x-2, y+2] = board[x, y]
				board[x, y] = 0
				temp = board[x-1, y+1]
				board[x-1, y+1] = 0
				bb = np.vstack((bb, generate_branches(board, x-2, y+2)))
				board[x-1, y+1] = temp
				board[x, y] = board[x-2, y+2]
				board[x-2, y+2] = 0
		if (y > 1):
			if (board[x-1, y-1] < 0 and board[x-2, y-2] == 0):
				board[x-2, y-2] = board[x, y]
				board[x, y] = 0
				temp = board[x-1, y-1]
				board[x-1, y-1] = 0
				bb = np.vstack((bb, generate_branches(board, x-2, y-2)))
				board[x-1, y-1] = temp
				board[x, y] = board[x-2, y-2]
				board[x-2, y-2] = 0
	return bb

def generate_next(board):
	bb = np.array([get_board()])
	for i in range(0, 8):
		for j in range(0, 8):
			if (board[i, j] > 0):
				bb = np.vstack((bb, generate_branches(board, i, j)[1:]))
	if (len(bb) > 1):
		return bb[1:]
	for i in range(0, 8):
		for j in range(0, 8):
			if (board[i, j] >= 1 and i < 7):
				temp = board[i, j]
				if (j < 7):
					if (board[i+1, j+1] == 0):
						board[i+1, j+1] = board[i, j]
						if (i+1 == 7):
							board[i+1, j+1] = 3
						board[i, j] = 0
						bb = np.vstack((bb, compress(board)))
						board[i, j] = temp
						board[i+1, j+1] = 0
				if (j > 0):
					if (board[i+1, j-1] == 0):
						board[i+1, j-1] = board[i, j]
						if (i+1 == 7):
							board[i+1, j-1] = 3
						board[i, j] = 0
						bb = np.vstack((bb, compress(board)))
						board[i, j] = temp
						board[i+1, j-1] = 0
			if (board[i, j] == 3 and i > 0):
				if (j < 7):
					if (board[i-1, j+1] == 0):
						board[i-1, j+1] = board[i, j]
						board[i, j] = 0
						bb = np.vstack((bb, compress(board)))
						board[i, j] = board[i-1, j+1]
						board[i-1, j+1] = 0
				elif (j > 0):
					if (board[i-1, j-1] == 0):
						board[i-1, j-1] = board[i, j]
						board[i, j] = 0
						bb = np.vstack((bb, compress(board)))
						board[i, j] = board[i-1, j-1]
						board[i-1, j-1] = 0
	return bb[1:]

json_file = open('rl.json', 'r') # disc = rl
disc_json = json_file.read()
json_file.close()

meta_model = model_from_json(disc_json)
meta_model.load_weights('rl.h5')
meta_model.compile(optimizer='nadam', loss='mean_squared_error')

base_model = model_from_json(disc_json)
base_model.load_weights('rl.h5')
base_model.compile(optimizer='nadam', loss='mean_squared_error')

data = np.zeros((1, 32))
labels = np.zeros(1)
win = lose = draw = 0
for gen in range(0, 150):
	for game in range(0, 200):
		temp_data = np.zeros((1, 32))
		board = expand(np_board())
		player = np.sign(np.random.random() - 0.5)
		turn = 0
		while (True):
			moved = False
			boards = np.zeros((0, 32))
			if (player == 1):
				boards = generate_next(board)
			else:
				boards = generate_next(reverse(board))

			score = meta_model.predict_on_batch(boards) if (player == 1)  else base_model.predict_on_batch(boards)

			max = [0, score[0, 0]]
			for i, x in enumerate(score[:, 0]):
				if (x >= max[1] and np.random.random() > 0.1) or not moved:
					moved = True
					max = [i, x]

			if (player == 1):
				board = expand(boards[max[0]])
				temp_data = np.vstack((temp_data, compress(board)))
			else:
				board = reverse(expand(boards[max[0]]))

			if (game == 199 and gen > 145):
				print('\nTurn player: %d - Confidence = %f' %(player, max[1]))
				print(board)
			player = -player

			# punish losers more for short games, reward winners more for longer games
			winner = game_winner(board)
			if (winner == 1):
				win = win + 1
				temp_labels = meta_model.predict_on_batch(temp_data[1:]) + (np.reshape(np.linspace(0.0001, 0.001, temp_data.shape[0] - 1), (temp_data.shape[0]-1, 1)) * (turn))
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
			meta_model.fit(data[1:], labels[1:], epochs=32, batch_size=256, verbose=0)
			if ((game+1) % 10 == 0 and np.random.random() > 0.5):
				data = np.zeros((1, 32))
				labels = np.zeros(1)

	meta_model.save_weights('rl.h5')

	base_model.load_weights('rl.h5')
	base_model.compile(optimizer='nadam', loss='mean_squared_error')