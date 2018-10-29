import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.models import model_from_json

# American checkers: wikipedia.org/wiki/English_draughts
# 	on a 8x8 checkerboard, both players start with 12 men
#	Black plays the first move
#	all pieces can only move and capture diagonally
#	men can only move/capture diagonally forward
#	kings can move/capture in any diagonal direction
#	if a man reaches the other side of the board, the turn ends and it becomes a king
#	captures are made by moving any piece diagonally over an opponent's
#	if a capture can be made, it must be taken
#	mutliple captures can be made in a single turn and with a single piece
#	the game ends when a players captures all the opponent's pieces
#	a player also whens when the opponent can not make a legal move

#	example board: 
#	/b/b/b/b	b/w = Black/White man {1, -1}
#	b/b/b/b/	B/W = Black/White king {3, -3}
#	/b/b/b/b	_ = empty square {0}
#	_/_/_/_/	/ = unusable square
#	/_/_/_/_
#	w/w/w/w/
#	/w/w/w/w
#	w/w/w/w/	* since pieces only mmove diagonally, only 32 squares are used

# number of opponent pieces captured (max = 12)
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

	score = 4*capped + potential + men + 3*kings + caps + 2*semicaps + 3*uncaps + 2*mid + 3*far + 100*won
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

def random_board():
	b = get_board()
	promote = 0.9
	remove = 0.4
	add = 0
	for piece in b:
		# randomly promote, remove, or add piece
		rand = np.random.random()
		if piece is not 0 and rand > promote:
			piece = piece * 3
			promote = promote + 0.005
		elif piece is not 0 and rand < remove:
			piece = 0
			remove = remove - 0.025
			add = add + 0.05
		elif piece is 0 and rand < add:
			if np.random.random() > 0.5:
				piece = 1
			else:
				piece = -1
	return b

if __name__ == '__main__':
	# generative model, which only looks at heuristic scoring metrics used for labeling
	gen_model = Sequential()
	gen_model.add(Dense(32, input_dim=10)) 
	gen_model.add(Dense(4, activation='relu',  kernel_regularizer=regularizers.l2(0.01)))

	# output is passed to relu() because labels are binary
	gen_model.add(Dense(1, activation='sigmoid',  kernel_regularizer=regularizers.l2(0.01)))
	gen_model.compile(optimizer='adam', loss='binary_crossentropy')



	# generate 5 sets of 1000 game states, used to train generative model
	for i in range(0, 5):
		board_0 = expand(random_board())
		boards_1 = generate_next(board_0)
		boards_2 = np.zeros((0,32))
		counter_1 = counter_2 = 0
		while (len(boards_1) + len(boards_2) < 1000):
			temp = counter_1
			for counter_1 in range(temp, min(temp + 10, len(boards_1))):
				if (possible_moves(reverse(expand(boards_1[counter_1]))) > 0):
					boards_2 = np.vstack((boards_2, generate_next(reverse(expand(boards_1[counter_1])))))
			temp = counter_2
			for counter_2 in range(temp, min(temp + 10, len(boards_2))):
				if (possible_moves(expand(boards_2[counter_2])) > 0):
					boards_1 = np.vstack((boards_1, generate_next(expand(boards_2[counter_2]))))

		# concat 1000 game states
		data = np.vstack((boards_1, boards_2))
		boards_2 = np.zeros((0, 32))
		counter_2 = 0
		boards_1 = np.vstack((boards_1[-10:], generate_next(board_0)))
		counter_1 = len(boards_1) - 1
		metrics = np.zeros((0, 11))

		# calculate/save heuristic metrics for each game state
		for board in iter(data):
			metrics = np.vstack((metrics, get_metrics(board)))

		# pass to generative model
		gen_model.fit(metrics[:, 1:], metrics[:, 0], epochs=32, batch_size=64, verbose=0)

	# discriminative model
	disc_model = Sequential()

	# input dimensions is 32 board position values (and 10 heuristic metrics - removed)
	disc_model.add(Dense(32 , activation='relu', input_dim=32))

	# use regularizers, to prevent fitting noisy labels
	disc_model.add(Dense(16 , activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	disc_model.add(Dense(8 , activation='relu', kernel_regularizer=regularizers.l2(0.01))) # 16
	disc_model.add(Dense(4 , activation='relu', kernel_regularizer=regularizers.l2(0.01))) # 8

	# output isn't squashed, because it might lose information
	disc_model.add(Dense(1 , activation='linear', kernel_regularizer=regularizers.l2(0.01)))
	disc_model.compile(optimizer='adam', loss='binary_crossentropy')


	# generative 32 sets of 1000 game states, used to train discriminative model
	for i in range(0, 32):
		boards_1 = generate_next(expand(random_board()))
		boards_2 = np.zeros((0,32))
		counter_1 = counter_2 = 0
		while (len(boards_1) + len(boards_2) < 1000):
			temp = counter_1
			for counter_1 in range(temp, min(temp + 10, len(boards_1))):
				if (possible_moves(reverse(expand(boards_1[counter_1]))) > 0):
					boards_2 = np.vstack((boards_2, generate_next(reverse(expand(boards_1[counter_1])))))
			temp = counter_2
			for counter_2 in range(temp, min(temp + 10, len(boards_2))):
				if (possible_moves(expand(boards_2[counter_2])) > 0):
					boards_1 = np.vstack((boards_1, generate_next(expand(boards_2[counter_2]))))

		data = np.vstack((boards_1, boards_2))
		boards_2 = np.zeros((0, 32))
		counter_2 = 0
		boards_1 = np.vstack((boards_1[-10:], generate_next(board_0)))
		counter_1 = len(boards_1) - 1

		# calculate heuristic metric for data
		metrics = np.zeros((0, 11))
		for board in iter(data):
			metrics = np.vstack((metrics, get_metrics(board)))

		# calculate probilistic (noisy) labels
		probabilistic = gen_model.predict_on_batch(metrics[:, 1:])

		# calculate confidence score for each probabilistic label using error between probabilistic and weak label
		confidence = 1/(1 + np.absolute(metrics[:, 0] - probabilistic[:, 0]))

		# fit labels to {-1, 1}
		probabilistic = np.sign(probabilistic)

		# concat board position data with heurstic metric and pass for training - removed
		# data = np.hstack((data, metrics[:, 1:]))
		disc_model.fit(data, probabilistic, epochs=16, batch_size=64, sample_weight=confidence, verbose=0)

	# save models
	gen_json = gen_model.to_json()
	with open('gen.json', 'w') as json_file:
		json_file.write(gen_json)
	gen_model.save_weights('gen.h5')

	disc_json = disc_model.to_json()
	with open('disc.json', 'w') as json_file:
		json_file.write(disc_json)
	disc_model.save_weights('disc.h5')

	print('Checkers Model saved to: gen.json/h5 and disc.json/h5')