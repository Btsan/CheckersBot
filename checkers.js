var agent = tf.sequential();
agent.add(tf.layers.dense({units: 64, useBias: true, activation: 'relu', inputShape:[32]}));
agent.add(tf.layers.dense({units: 32, useBias: true, activation: 'relu'}));
agent.add(tf.layers.dense({units: 16, useBias: true, activation: 'relu'}));
agent.add(tf.layers.dense({units: 8, useBias: true, activation: 'relu'}));
agent.add(tf.layers.dense({units: 1, useBias: true}));
agent.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
agent.layers[0].setWeights([tf.tensor2d(model["w_0"], [32, 64]), tf.tensor1d(model["bias_0"])]);
agent.layers[1].setWeights([tf.tensor2d(model["w_1"], [64, 32]), tf.tensor1d(model["bias_1"])]);
agent.layers[2].setWeights([tf.tensor2d(model["w_2"], [32, 16]), tf.tensor1d(model["bias_2"])]);
agent.layers[3].setWeights([tf.tensor2d(model["w_3"], [16, 8]), tf.tensor1d(model["bias_3"])]);
agent.layers[4].setWeights([tf.tensor2d(model["w_4"], [8, 1]), tf.tensor1d(model["bias_4"])]);

var board = [[0, 1, 0, 1, 0, 1, 0, 1],
			 [1, 0, 1, 0, 1, 0, 1, 0],
			 [0, 1, 0, 1, 0, 1, 0, 1],
			 [0, 0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0, 0],
			 [-1, 0, -1, 0, -1, 0, -1, 0],
			 [0, -1, 0, -1, 0, -1, 0, -1],
			 [-1, 0, -1, 0, -1, 0, -1, 0]];

var c = document.getElementById("checkers");
var ctx = c.getContext("2d");
ctx.lineWidth = 5;
var length = 720;
var square = length/8;

var player = (Math.random() > 0.5)? true:false;
var toMove = {row: 0, col: 0};
c.addEventListener("click", userInterface);

function to_input(b){
	var t = [b[0][1], b[0][3], b[0][5], b[0][7],
			 b[1][0], b[1][2], b[1][4], b[1][6],
			 b[2][1], b[2][3], b[2][5], b[2][7],
			 b[3][0], b[3][2], b[3][4], b[3][6],
			 b[4][1], b[4][3], b[4][5], b[4][7],
			 b[5][0], b[5][2], b[5][4], b[5][6],
			 b[6][1], b[6][3], b[6][5], b[6][7],
			 b[7][0], b[7][2], b[7][4], b[7][6]];
	return tf.tensor2d([t], [1, 32]);
}
function clear_board(){
	ctx.clearRect(0, 0, length, length);
	ctx.fillStyle = "white";
	for (var i = 0; i < 8; ++i){
		row = i*square;
		if (i%2 == 0)
			for (var j = 0; j < 7; j += 2){
				col = j*square;
				ctx.fillRect(col, row, square, square);
			}
		else
			for (var j = 1; j < 8; j += 2){
				col = j*square;
				ctx.fillRect(col, row, square, square);
			}
	}
}
function draw_board(b){
	clear_board();
	for (var i = 0; i < 8; ++i){
		y = i*square + square/2;
		if (i%2 == 0)
			for (var j = 1; j < 8; j += 2){
				x = j*square + square/2;
				if (b[i][j] != 0){
					ctx.beginPath();
					ctx.arc(x, y, square/2 - 4, 0, 2*Math.PI);
					if (b[i][j] > 0){ // Black
						if (b[i][j] == 3){
							ctx.fillStyle = "red";
							ctx.fill();
						}
						else
						{
							ctx.strokeStyle = "red";
							ctx.stroke();
						}
					}
					else{ // White
						if (b[i][j] == -3){
							ctx.fillStyle = "yellow";
							ctx.fill();
						}
						else
						{
							ctx.strokeStyle = "yellow";
							ctx.stroke();
						}
					}
				}
			}
		else
			for (var j = 0; j < 7; j += 2){
				x = j*square + square/2;
				if (b[i][j] != 0){
					ctx.beginPath();
					ctx.arc(x, y, square/2 - 4, 0, 2*Math.PI);
					if (b[i][j] > 0){ // Black
						if (b[i][j] == 3){
							ctx.fillStyle = "red";
							ctx.fill();
						}
						else
						{
							ctx.strokeStyle = "red";
							ctx.stroke();
						}
					}
					else{ // White
						if (b[i][j] == -3){
							ctx.fillStyle = "yellow";
							ctx.fill();
						}
						else
						{
							ctx.strokeStyle = "yellow";
							ctx.stroke();
						}
					}
				}
			}
	}
}
function paint_blue(targ){
	requestAnimationFrame(function(){
	ctx.fillStyle = "blue";
	ctx.fillRect(targ.col*square, targ.row*square, square, square);});
}
function userInterface(ev){
	if (!player) return;
	var rect = c.getBoundingClientRect();
	var click = {x: ev.clientX - rect.left, y: ev.clientY - rect.top};
	var targ = {row: Math.floor(click.y/square), col: Math.floor(click.x/square)};

	var ate = false;
	var must = false;
	for (var i = 0; i < 8; ++i){
		for (var j = 0; j < 8; ++j){
			if (has_capture({row: i, col: j})){
				must = true;
				break;
			}
		}
	}
	if (board[targ.row][targ.col] < 0) draw_board(board);
	switch (board[targ.row][targ.col]){
		case -3:
			if (targ.row < 7 && targ.col > 0 && board[targ.row+1][targ.col-1] == 0){ // bot left
				if (!must) paint_blue({row: targ.row+1, col: targ.col-1});
			}
			else if (targ.row < 6 && targ.col > 1 && board[targ.row+1][targ.col-1] > 0 && board[targ.row+2][targ.col-2] == 0){ // bot left
				paint_blue({row: targ.row+2, col: targ.col-2});
			}
			if (targ.row < 7 && targ.col < 7 && board[targ.row+1][targ.col+1] == 0){ // bot right
				if (!must) paint_blue({row: targ.row+1, col: targ.col+1});
			}
			else if (targ.row < 6 && targ.col < 6 && board[targ.row+1][targ.col+1] > 0 && board[targ.row+2][targ.col+2] == 0){ // bot right
				paint_blue({row: targ.row+2, col: targ.col+2});
			}
		case -1:
			if (targ.row > 0 && targ.col > 0 && board[targ.row-1][targ.col-1] == 0){ // top left
				if (!must) paint_blue({row: targ.row-1, col: targ.col-1});
			}
			else if (targ.row > 1 && targ.col > 1 && board[targ.row-1][targ.col-1] > 0 && board[targ.row-2][targ.col-2] == 0){ // top left
				paint_blue({row: targ.row-2, col: targ.col-2});
			}
			if (targ.row > 0 && targ.col < 7 && board[targ.row-1][targ.col+1] == 0){ // top right
				if (!must) paint_blue({row: targ.row-1, col: targ.col+1});
			}
			else if (targ.row > 1 && targ.col < 6 && board[targ.row-1][targ.col+1] > 0 && board[targ.row-2][targ.col+2] == 0){ // top right
				paint_blue({row: targ.row-2, col: targ.col+2});
			}
			if (!must || (must && has_capture(targ))) toMove = targ;
			break;
		case 0:
			if (toMove.row == 0 && toMove.col == 0) break;
			if (board[toMove.row][toMove.col] == -3 && targ.row > toMove.row){ // down
				if (targ.row == toMove.row+2){ // capture
					if (targ.col == toMove.col+2 && board[toMove.row+1][toMove.col+1] > 0){ // right
						board[toMove.row+1][toMove.col+1] = 0;
						board[targ.row][targ.col] = board[toMove.row][toMove.col];
						board[toMove.row][toMove.col] = 0;
						ate = true;
					}
					else if(targ.col == toMove.col-2 && board[toMove.row+1][toMove.col-1] > 0){ // left
						board[toMove.row+1][toMove.col-1] = 0;
						board[targ.row][targ.col] = board[toMove.row][toMove.col];
						board[toMove.row][toMove.col] = 0;
						ate = true;
					}
				}
				else if (!must && targ.row == toMove.row+1){ // move
					if (targ.col == toMove.col+1 || targ.col == toMove.col-1){
						board[targ.row][targ.col] = board[toMove.row][toMove.col];
						board[toMove.row][toMove.col] = 0;
					}
				}
			}
			else if (targ.row < toMove.row){ // up
				if (targ.row == toMove.row-2){ // capture
					if (targ.col == toMove.col+2 && board[toMove.row-1][toMove.col+1] > 0){ // right
						board[toMove.row-1][toMove.col+1] = 0;
						board[targ.row][targ.col] = board[toMove.row][toMove.col];
						board[toMove.row][toMove.col] = 0;
						ate = true;
					}
					else if(targ.col == toMove.col-2 && board[toMove.row-1][toMove.col-1] > 0){ // left
						board[toMove.row-1][toMove.col-1] = 0;
						board[targ.row][targ.col] = board[toMove.row][toMove.col];
						board[toMove.row][toMove.col] = 0;
						ate = true;
					}
				}
				else if (!must && targ.row == toMove.row-1){ // move
					if (targ.col == toMove.col+1 || targ.col == toMove.col-1){
						board[targ.row][targ.col] = board[toMove.row][toMove.col];
						board[toMove.row][toMove.col] = 0;
					}
				}
			}
			if (targ.row == 0 && board[targ.row][targ.col] == -1){
				board[targ.row][targ.col] = -3;
				requestAnimationFrame(function(){draw_board(board);});
				player = false;
				toMove = {row: 0, col: 0};
				computer_move();
			}
			if (board[toMove.row][toMove.col] == 0){
				requestAnimationFrame(function(){draw_board(board);});
				if ((ate && !has_capture(targ)) || !ate){
					player = false;
					toMove = {row: 0, col: 0};
					computer_move();
				}
				else{
					toMove = targ;
				}
			}
			break;
	}
}
function has_capture(targ){
	var cap = false;
	switch (board[targ.row][targ.col]){
		case -3:
			if (targ.row < 6 && targ.col > 1 && board[targ.row+1][targ.col-1] > 0 && board[targ.row+2][targ.col-2] == 0){ // bot left
				paint_blue({row: targ.row+2, col: targ.col-2});
				cap = true;
			}
			if (targ.row < 6 && targ.col < 6 && board[targ.row+1][targ.col+1] > 0 && board[targ.row+2][targ.col+2] == 0){ // bot right
				paint_blue({row: targ.row+2, col: targ.col+2});
				cap = true;
			}
		case -1:
			if (targ.row > 1 && targ.col > 1 && board[targ.row-1][targ.col-1] > 0 && board[targ.row-2][targ.col-2] == 0){ // top left
				paint_blue({row: targ.row-2, col: targ.col-2});
				cap = true;
			}
			if (targ.row > 1 && targ.col < 6 && board[targ.row-1][targ.col+1] > 0 && board[targ.row-2][targ.col+2] == 0){ // top right
				paint_blue({row: targ.row-2, col: targ.col+2});
				cap = true;
			}
			break;
	}
	return cap;
}
function computer_move(){
	try{
		var boards = generate_next(board); // array of board tensors
		if (ai_pcs(board) < 6){ // play more careful at the end
			var min = capturable(boards[0]);
			for (var i = 1; i < boards.length; ++i){
				var temp_min = capturable(boards[i]);
				if ((temp_min) < min){
					min = temp_min;
					i = 0;
				}
				else if (temp_min == min){
					continue;
				}
				else{
					boards.splice(i, 1);
					i -= 1;
				}
			}
		}
		var next = {score: 0, index: 0};
		next.score = agent.predict(minmax(to_board(reverse(minmax(to_board(reverse(boards[0])))))));
		for (var i = 1; i < boards.length; ++i){
			var score = agent.predict(minmax(to_board(reverse(minmax(to_board(reverse(boards[i])))))));
			if (score >= next.score){
				next.score = score;
				next.index = i;
			}
		}
		board = to_board(boards[next.index]);
		requestAnimationFrame(function(){draw_board(board);});
		player = true;
	}
	catch(err){
		board = [[0, 1, 0, 1, 0, 1, 0, 1],
			[1, 0, 1, 0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0, 1, 0, 1],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0],
			[-1, 0, -1, 0, -1, 0, -1, 0],
			[0, -1, 0, -1, 0, -1, 0, -1],
			[-1, 0, -1, 0, -1, 0, -1, 0]];
		player = (Math.random() > 0.5)? true:false;
		draw_board(board);
		if (!player) computer_move();
	}
}
function capturable(t){
	b = to_board(reverse(t));
	var count = 0;
	for (var i = 0; i < 8; ++i)
		for (var j = 0; j < 8; ++j)
			if (b[i][j] > 0)
				count += count_branches(b, i, j);
	return count;
}

function ai_pcs(b){
	var count = 0;
	for (var i = 0; i < 8; ++i){
		for (var j = 0; j < 8; ++j){
			if (b[i][j] > 0) count++;
		}
	}
	return count;
}

function count_branches(b, x, y){
	var count = 0;
	if (b[x][y] >= 1 && x < 6){
		var temp_1 = b[x][y];
		if (y < 6){
			if (b[x+1][y+1] < 0 && b[x+2][y+2] == 0){
				b[x+2][y+2] = b[x][y];
				if (x+2 == 7) b[x+2][y+2] = 3;
				var temp = b[x+1][y+1];
				b[x+1][y+1] = 0;
				if (b[x][y] != b[x+2][y+2]){
					b[x][y] = 0;
					count += 1
				}
				else{
					b[x][y] = 0;
					count = count_branches(b, x+2, y+2) + 1;
				}
				b[x+1][y+1] = temp;
				b[x][y] = temp_1;
				b[x+2][y+2] = 0;
			}
		}
		if (y > 1){
			if (b[x+1][y-1] < 0 && b[x+2][y-2] == 0){
				b[x+2][y-2] = b[x][y];
				if (x+2 == 7) b[x+2][y-2] = 3;
				var temp = b[x+1][y-1];
				b[x+1][y-1] = 0;
				if (b[x][y] != b[x+2][y-2]){
					b[x][y] = 0;
					count += 1;
				}
				else{
					b[x][y] = 0;
					count = count_branches(b, x+2, y-2) + 1;
				}
				b[x+1][y-1] = temp;
				b[x][y] = temp_1;
				b[x+2][y-2] = 0;
			}
		}
	}
	if (b[x][y] == 3 && x > 1){
		if (y < 6){
			if (b[x-1][y+1] < 0 && b[x-2][y+2] == 0){
				b[x-2][y+2] = b[x][y];
				b[x][y] = 0;
				var temp = b[x-1][y+1];
				b[x-1][y+1] = 0;
				count = count_branches(b, x-2, y+2) + 1;
				b[x-1][y+1] = temp;
				b[x][y] = b[x-2][y+2];
				b[x-2][y+2] = 0;
			}
		}
		if (y > 1){
			if (b[x-1][y-1] < 0 && b[x-2][y-2] == 0){
				b[x-2][y-2] = b[x][y];
				b[x][y] = 0;
				var temp = b[x-1][y-1];
				b[x-1][y-1] = 0;
				count = count_branches(b, x-2, y-2) + 1;
				b[x-1][y-1] = temp;
				b[x][y] = b[x-2][y-2];
				b[x-2][y-2] = 0;
			}
		}
	}
	return count;
}

function minmax(b){
	var boards = generate_next(b); // array of board tensors
	var next = {score: 0, index: 0};
	var reversed = generate_next(to_board(reverse(boards[0])));
	if (reversed.length < 1) return boards[0];
	next.score = agent.predict(reverse(reversed[0]));
	for (var i = 1; i < reversed.length; ++i){
		var temp_min = agent.predict(reverse(reversed[i]));
		if (temp_min < next.score){
			next.score = temp_min;
		}
	}
	for (var i = 1; i < boards.length; ++i){
		reversed = generate_next(to_board(reverse(boards[i])));
		if (reversed.length < 1) return boards[i];
		var score = agent.predict(reverse(reversed[0]));
		for (var j = 1; j < reversed.length; ++j){
			var temp_min = agent.predict(reverse(reversed[j]));
			if (temp_min < score){
				score = temp_min;
			}
		}
		if (score >= next.score){
			next.score = score;
			next.index = i;
		}
	}
	return boards[next.index];
}
function reverse(t){ // board tensor -> board tensor
	var b = [[0, -t.get(31-0), 0, -t.get(31-1), 0, -t.get(31-2), 0, -t.get(31-3)],
			 [-t.get(31-4), 0, -t.get(31-5), 0, -t.get(31-6), 0, -t.get(31-7), 0],
			 [0, -t.get(31-8), 0, -t.get(31-9), 0, -t.get(31-10), 0, -t.get(31-11)],
			 [-t.get(31-12), 0, -t.get(31-13), 0, -t.get(31-14), 0, -t.get(31-15), 0],
			 [0, -t.get(31-16), 0, -t.get(31-17), 0, -t.get(31-18), 0, -t.get(31-19)],
			 [-t.get(31-20), 0, -t.get(31-21), 0, -t.get(31-22), 0, -t.get(31-23), 0],
			 [0, -t.get(31-24), 0, -t.get(31-25), 0, -t.get(31-26), 0, -t.get(31-27)],
			 [-t.get(31-28), 0, -t.get(31-29), 0, -t.get(31-30), 0, -t.get(31-31), 0]];
	return to_input(b);
}

function to_board(t){
	var b = [[0, t.get(0), 0, t.get(1), 0, t.get(2), 0, t.get(3)],
			 [t.get(4), 0, t.get(5), 0, t.get(6), 0, t.get(7), 0],
			 [0, t.get(8), 0, t.get(9), 0, t.get(10), 0, t.get(11)],
			 [t.get(12), 0, t.get(13), 0, t.get(14), 0, t.get(15), 0],
			 [0, t.get(16), 0, t.get(17), 0, t.get(18), 0, t.get(19)],
			 [t.get(20), 0, t.get(21), 0, t.get(22), 0, t.get(23), 0],
			 [0, t.get(24), 0, t.get(25), 0, t.get(26), 0, t.get(27)],
			 [t.get(28), 0, t.get(29), 0, t.get(30), 0, t.get(31), 0]];
	return b;
}

function generate_branches(b, x, y){
	var boards = [to_input(b)];
	if (b[x][y] >= 1 && x < 6){
		var temp_1 = b[x][y];
		if (y < 6){
			if (b[x+1][y+1] < 0 && b[x+2][y+2] == 0){
				b[x+2][y+2] = b[x][y];
				if (x+2 == 7) b[x+2][y+2] = 3;
				var temp = b[x+1][y+1];
				b[x+1][y+1] = 0;
				if (b[x][y] != b[x+2][y+2]){
					b[x][y] = 0;
					boards = boards.concat([to_input(b)]);
				}
				else{
					b[x][y] = 0;
					boards = boards.concat(generate_branches(b, x+2, y+2));
				}
				b[x+1][y+1] = temp;
				b[x][y] = temp_1;
				b[x+2][y+2] = 0;
			}
		}
		if (y > 1){
			if (b[x+1][y-1] < 0 && b[x+2][y-2] == 0){
				b[x+2][y-2] = b[x][y];
				if (x+2 == 7) b[x+2][y-2] = 3;
				var temp = b[x+1][y-1];
				b[x+1][y-1] = 0;
				if (b[x][y] != b[x+2][y-2]){
					b[x][y] = 0;
					boards = boards.concat([to_input(b)]);
				}
				else{
					b[x][y] = 0;
					boards = boards.concat(generate_branches(b, x+2, y-2));
				}
				b[x+1][y-1] = temp;
				b[x][y] = temp_1;
				b[x+2][y-2] = 0;
			}
		}
	}
	if (b[x][y] == 3 && x > 1){
		if (y < 6){
			if (b[x-1][y+1] < 0 && b[x-2][y+2] == 0){
				b[x-2][y+2] = b[x][y];
				b[x][y] = 0;
				var temp = b[x-1][y+1];
				b[x-1][y+1] = 0;
				boards = boards.concat(generate_branches(b, x-2, y+2));
				b[x-1][y+1] = temp;
				b[x][y] = b[x-2][y+2];
				b[x-2][y+2] = 0;
			}
		}
		if (y > 1){
			if (b[x-1][y-1] < 0 && b[x-2][y-2] == 0){
				b[x-2][y-2] = b[x][y];
				b[x][y] = 0;
				var temp = b[x-1][y-1];
				b[x-1][y-1] = 0;
				boards = boards.concat(generate_branches(b, x-2, y-2));
				b[x-1][y-1] = temp;
				b[x][y] = b[x-2][y-2];
				b[x-2][y-2] = 0;
			}
		}
	}
	return boards;
}

function generate_next(b){
	var boards = [];
	for (var i = 0; i < 8; ++i)
		for (var j = 0; j < 8; ++j)
			if (b[i][j] > 0)
				boards = boards.concat(generate_branches(b, i, j).slice(1)); // returns array of board tensors
	if (boards.length > 0)
		return boards;
	for (var i = 0; i < 8; ++i)
		for (var j = 0; j < 8; ++j){
			if (b[i][j] >= 1 && i < 7){
				var temp = b[i][j];
				if (j < 7){
					if (b[i+1][j+1] == 0){
						b[i+1][j+1] = b[i][j];
						if (i+1 == 7) b[i+1][j+1] = 3;
						b[i][j] = 0;
						boards = boards.concat([to_input(b)]);
						b[i][j] = temp;
						b[i+1][j+1] = 0;
					}
				}
				if (j > 0){
					if (b[i+1][j-1] == 0){
						b[i+1][j-1] = b[i][j];
						if (i+1 == 7) b[i+1][j-1] = 3;
						b[i][j] = 0;
						boards = boards.concat([to_input(b)]);
						b[i][j] = temp;
						b[i+1][j-1] = 0;
					}
				}
			}
			if (b[i][j] == 3 && i > 0){
				if (j < 7){
					if (b[i-1][j+1] == 0){
						b[i-1][j+1] = b[i][j];
						b[i][j] = 0;
						boards = boards.concat([to_input(b)]);
						b[i][j] = b[i-1][j+1];
						b[i-1][j+1] = 0;
					}
				}
				else if (j > 0){
					if (b[i-1][j-1] == 0){
						b[i-1][j-1] = b[i][j];
						b[i][j] = 0;
						boards = boards.concat([to_input(b)]);
						b[i][j] = b[i-1][j-1];
						b[i-1][j-1] = 0;
					}
				}
			}
		}
	return boards
}
