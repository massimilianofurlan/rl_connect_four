using DataStructures
using ProgressMeter

# set board size: a regular board is 6 x 7 (too many states)
const n_cols = 7 # regular is 7
const n_rows = 6 # regular is 6
const last_turn = n_cols * n_rows

# Q-learning hyperparameters
Q = DefaultDict(()->zeros(Float16,n_cols)) 	# the Q matrix is a DefaultDict indexed with boards
const n_episodes = 20000000					# each episode is a self-play training match
const alpha = Float16(0.5)					# learning rate
const gamma = Float16(0.99)					# discount factor
# exploration decay: e_t = beta * e_{t-1} for t \in 0,..,T where T = n_episodes and e_0 = epsilon0
const epsilon0 = 1/2
const beta = (0.1/epsilon0)^(1/n_episodes)	# I want e_T = 0.1
const epsilon = [epsilon0 * beta^episode for episode in 1:n_episodes]	

# generic functions

function argmax_(A::AbstractArray; inds = Array{Int8,1}(undef,n_cols))
	# return indices of array A's set of maxima
	maxval = typemin(eltype(A))
	n = 0
	@inbounds for i in keys(A)
		Ai = A[i]
		Ai < maxval && continue
		if Ai > maxval
			maxval = Ai
			n = 1
			inds[n] = i
		else
			inds[n+=1] = i
		end
	end
	return view(inds,1:n)
end

function get_bit_board(board)
	# codify board using 2-bits information for each disk (0:none,1:yellow,2:red) to reduce allocations
	bit_board = BitMatrix(undef,n_rows*2,n_cols)		# 2x1 blocks of this array are 2-bits
	bit_board .= false									# initially set everything to 0
	for j in 1:n_cols	
		for i in 1:n_rows
			if board[i,j] == Int8(1)					# if 1:yellow, then codify [1, 0]					
				bit_board[i*2-1,j] = true 	
				bit_board[i*2,j] = false
			elseif board[i,j] == Int8(2)				# if 2:red, then codify [0, 1]
				bit_board[i*2-1,j] = false	
				bit_board[i*2,j] = true
			end
		end
	end
	return bit_board
end

function turn_board(board)
	# 'turn' board to get next or preceding player POV
	for j in 1:n_cols	
		for i in 1:n_rows
			if board[i,j] == Int8(1)
				board[i,j] = Int8(2)
			elseif board[i,j] == Int8(2)
				board[i,j] = Int8(1)
			end
		end
	end
	return board
end

function flip_board(board)
	# flip board around the vertical axis
	return view(board,:,n_cols:-1:1)
end

function isterminal(board, turn)
	# check if board is terminal, that is, if somebody won 
	turn < 7 && return false			# the minimum number of turns to win is seven
	# check rows
	for j in 1:n_cols-3
		for i in 1:n_rows
			board[i,j] == 1 || continue
			all( ==(1), view(board,i,j:j+3)) && return true
		end
	end
	# check cols  
	for j in 1:n_cols
		for i in 1:n_rows-3
			board[i,j] == 1 || continue
			all( ==(1), view(board,i:i+3,j)) && return true
		end
	end
	# check upward diagonals 
	for j in 1:n_cols-3	
		for i in 1:n_rows-3
			board[i,j] == 1 || continue
			all( ==(1), [board[i+k,j+k] for k in 0:3]) && return true
		end
	end
	# check donward diagonals 
	for j in 1:n_cols-3
		for i in 4:n_rows
			board[i,j] == 1 || continue
			all( ==(1), [board[i-k,j+k] for k in 0:3]) && return true
		end
	end
	return false
end

# Q-learning functions

function get_best_actions(Q, board, n_disks)
	# get set of best actions in current board
	bit_board = get_bit_board(board)						# get bit board
	legal_actions = (1:n_cols)[n_disks .< n_rows] 			# get set of feasible actions in current board
	if haskey(Q,bit_board)
		# if there is a known best action in the current board, bo it 
		return legal_actions[argmax_(view(Q[bit_board],n_disks .< n_rows))]
	end
	flipped_bit_board = flip_board(bit_board)	
	if haskey(Q,flipped_bit_board)
		# if there is a known best action in the current board flipped, bo it flipped
		return legal_actions[argmax_(view(Q[flipped_bit_board][n_cols:-1:1],n_disks .< n_rows))]
	end
	# if there is no best known action in the current board and current board flipped, randomize
	return legal_actions
end

function get_action(Q, board, n_disks, epsilon)
	# get action give board according to an epsilon greedy strategy 
	if rand() <= epsilon
		return rand((1:n_cols)[n_disks .< n_rows])			# explore
	else
		return rand(get_best_actions(Q,board,n_disks))		# exploit
	end
end

function update_q(Q, alpha, gamma, board, new_board, a, reward)
	# check if board and new_board are known 
	bit_board = get_bit_board(board)			# use bit_boards as indeces to reduce allocations
	bit_new_board = get_bit_board(new_board)
	if haskey(Q,bit_new_board)
		# if new_board is known, full update
		Q[bit_board][a] = (1 - alpha) * Q[bit_board][a] + alpha * (reward + gamma * maximum(Q[bit_new_board]))
		return nothing	# update done, quit function
	elseif haskey(Q,bit_board)
		# if new_board is unkwnow but board is known, partial update, maximum(Q[bit_new_board] = 0
		Q[bit_board][a] = (1 - alpha) * Q[bit_board][a] + alpha * reward
		return nothing
	end 
	# if neither bit_new_board and bit_board are known, check if they are known flipped
	flipped_bit_board = flip_board(bit_board)	
	flipped_bit_new_board = flip_board(bit_new_board)
	flipped_a = (n_cols:-1:1)[a]
	if haskey(Q,flipped_bit_new_board)
		# if flipped_new_board is known, full update
		Q[flipped_bit_board][flipped_a] = (1 - alpha) * Q[flipped_bit_board][flipped_a] + alpha * (reward + gamma * maximum(Q[flipped_bit_new_board]))
		return nothing
	elseif haskey(Q,flipped_bit_board)
		# if flipped_new_board is unkwnow but flipped_board is known, partial update,  maximum(Q[flipped_bit_new_board] = 0
		Q[flipped_bit_board][flipped_a] = (1 - alpha) * Q[flipped_bit_board][flipped_a]  + alpha * reward
		return nothing
	end
	if reward != 0
		# if board and new_board are unknown, both flipped and not as is, and reward != 0, partial update: Q[bit_board][a] = 0, maximum(Q[bit_new_board] = 0
		# everything starts here: first time winning/losing this way 
		Q[bit_board][a] = alpha * reward 
	end
end

# main functions

function self_play(Q, alpha, gamma, epsilon)
	# training self-play environment
	pre_board = zeros(Int8, n_rows, n_cols)		# board before current player move 
	board = zeros(Int8, n_rows, n_cols)			# current board
	post_board = zeros(Int8, n_rows, n_cols) 	# board after current player move
	n_disks = zeros(Int8, n_cols)				# number of disks per column
	a, pre_a = undef, undef						# initialize curerent action and preceding player action
	turn = 1 									# turn number
	for turn in 1:last_turn
		a = get_action(Q, board, n_disks, epsilon)					# get e-greedy action given current board
		n_disks[a] += 1 											# store disk
		post_board[n_disks[a],a] = Int8(1)							# update post_board (agent always plays 1:yellow)
		if isterminal(post_board, turn)								# check if current player won
			# current player gets a 100 reward
			update_q(Q, alpha, 0.0, board, post_board, a, Float16(100.0))
			# preceding player gets a -100 reward
			post_board = turn_board(post_board)						# preceding player POV
			update_q(Q, alpha, gamma, pre_board, post_board, pre_a, Float16(-100.0))
			# end of game
			break
		end 
		# non-winning move gives 0 reward: note, sequential moves (s_i, a_i) -> (s_j, a_j) -> (s'_i, ...)
		post_board = turn_board(post_board)							# preceding player POV
		turn != 1 && update_q(Q, alpha, gamma, pre_board, post_board, pre_a, Float16(0.0))
		# transition game to next player environment
		pre_a = copy(a)
		pre_board = copy(board)
		board = copy(post_board)
	end
	return Q
end

# begin execution 

println("Wait, I am training ... (around 15 minutes, Apple Silicon M1 Pro)")
# training sessions
progress = Progress(n_episodes, color=:white, showspeed=true)
@time for episode in 1:n_episodes
	# self-play training, epsilon decays according to e_t = beta * e_{t-1} each episode
	global Q = self_play(Q, alpha, gamma, epsilon[episode])
	next!(progress,; showvalues = [(:episode,episode),(:epsilon,epsilon[episode]),(:known_boards, length(Q))])
end

print("Training finished, press enter to continue ... ")
readline()

function get_stats(board, Q, n_disks)
	# print some nice stats 
	bit_board = get_bit_board(board)
	flipped_bit_board = flip_board(bit_board)
	if haskey(Q,bit_board)
		println("\n\nThe agent knows this board ", Q[bit_board])
		println("The known best action is ", get_best_actions(Q,board,n_disks))
		println("The continuation value is ", unique(Q[bit_board][get_best_actions(Q,board,n_disks)]))
	elseif haskey(Q,flipped_bit_board)
		println("\n\nThe agent knows this board: ", Q[flipped_bit_board][end:-1:1])
		println("The known best action is ", get_best_actions(Q,board,n_disks))
		println("The continuation value is ", unique(Q[flipped_bit_board][get_best_actions(Q,board,n_disks)]))
	else	
		println("\n\nThis board is unknown")
	end
end

function play_against_agent(Q; players = ["agent","human"])
	# ugly function to play against the agent: one might want to use gtk...
	board = zeros(Int8, n_rows, n_cols)
	n_disks = zeros(Int8, n_cols)
	for turn in 1:last_turn
		if players[1] == "agent"
			run(`clear`)
			println("Agent turn ...\n")
			display("text/plain", board[end:-1:1,:])
			get_stats(board, Q, n_disks)
			a = rand(get_best_actions(Q, board, n_disks))
			println("Agent took action $a \n")
			n_disks[a] += 1
			board[n_disks[a],a] = Int8(1)
			isterminal(board, turn) && break
		else 
			println("Human turn ...\n")
			display("text/plain", board[end:-1:1,:])
			while true
				print("\n\nHuman, select row: ")
				a = parse(Int,readline())
				a in 1:n_cols && n_disks[a] < n_rows && break
			end
			n_disks[a] += 1
			board[n_disks[a],a] = Int8(2)
			human_won = isterminal(turn_board(board), turn) 
			turn_board(board)
			human_won && break
		end
		players = players[end:-1:1]
	end 
	println()
	display("text/plain", board[end:-1:1,:])
	println("\n\n ### The ",players[1]," won ### \n")
end

players = ["agent","human"]					# first match, agent moves first
while true
	global players
	play_against_agent(Q; players = players)
	print("Want to play again? [y,n]: ")	# default is y
	readline() != "n" || break				# exit game if 'n', play again otherwise
	players = players[end:-1:1]				# alternating first mover
end







