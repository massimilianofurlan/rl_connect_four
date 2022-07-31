
using DataStructures
using ProgressMeter
using JLD

const n_cols = 7
const n_rows = 6

Q = DefaultDict(()->zeros(n_cols))
alpha = 0.5
delta = 0.99

function isterminal(board)
	# check rows
	for i in 1:n_rows
		for j in 1:n_cols-3
			board[i,j] != 0 || continue
			all( ==(board[i,j]), board[i,j:j+3]) && return true
		end
	end
	# check jumn  
	for j in 1:n_cols
		for i in 1:n_rows-3
			board[i,j] != 0 || continue
			all( ==(board[i,j]), board[i:i+3,j]) && return true
		end
	end
	# check upward diagonals 
	for i in 1:n_rows-3
		for j in 1:n_cols-3
			board[i,j] != 0 || continue
			all( ==(board[i,j]), [board[i+k,j+k] for k in 0:3]) && return true
		end
	end
	# check donward diagonals 
	for i in 4:n_rows
		for j in 1:4
			board[i,j] != 0 || continue
			all( ==(board[i,j]), [board[i-k,j+k] for k in 0:3]) && return true
		end
	end
	return false
end


function get_action(Q,board,n_tokens)
	legal_actions = (1:n_cols)[n_tokens .< 6] 
	if rand() <= 1/3 || !haskey(Q,board)
		return rand(legal_actions)
	else 
		return legal_actions[rand(findall(Q[board][n_tokens .< 6] .== maximum(Q[board][n_tokens .< 6])))]
	end
end


function update_q(Q, alpha, delta, board, new_board, a; reward = 0.0)
	if haskey(Q,new_board)
		Q[board][a] = (1 - alpha) * Q[board][a] + alpha * (reward + delta * maximum(Q[new_board]))
	elseif haskey(Q,board)
		Q[board][a] = (1 - alpha) * Q[board][a] 
	elseif reward != 0
		Q[board][a] = (1 - alpha) * Q[board][a] + alpha * reward 
	end
	new_board_ = flip_board(new_board)
	board_ = flip_board(board)
	a_ = (n_cols:-1:1)[a]
	if haskey(Q,new_board_)
		Q[board_][a_] = (1 - alpha) * Q[board_][a_] + alpha * (reward + delta * maximum(Q[new_board_]))
	elseif haskey(Q,board_)
		Q[board_][a_] = (1 - alpha) * Q[board_][a_] 
	elseif reward != 0
		Q[board_][a_] = (1 - alpha) * Q[board_][a_] + alpha * reward 
	end
end


function turn_board(board)
	for i in 1:n_rows			#n_tokens[j]
		for j in 1:n_cols
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
	return board[:,end:-1:1]
end

function self_play(Q, alpha, delta)
	board2 = zeros(Int8, n_rows, n_cols)
	board1 = zeros(Int8, n_rows, n_cols)
	board = zeros(Int8, n_rows, n_cols)
	n_tokens = zeros(Int8, n_cols)
	turn = 1
	winner = 0
	a = 0
	a1 = 0
	# begin game
	a1 = get_action(Q,board2,n_tokens)	
	n_tokens[a1] += 1
	board1[n_tokens[a1],a1] = Int8(2)	# next player sees this as the -i move
	board[n_tokens[a1],a1] = Int8(2)	# next player sees this as the -i move
	while turn < n_rows*n_cols
		a = get_action(Q,board1,n_tokens)
		n_tokens[a] += 1
		board[n_tokens[a],a] = Int8(1)
		#
		isterminal(board) && break
		# Q-update from preceding move
		board = turn_board(board)								# preceding player perspective
		update_q(Q, alpha, delta, board2, board, a1)	# Q update
		#
		a1 = copy(a)
		board2 = copy(board1)
		board1 = copy(board)
		#
		turn += 1
	end
	if turn != n_rows*n_cols
		update_q(Q, alpha, 0.0, board1, board, a; reward = 100.0)
		update_q(Q, alpha, delta, board2, turn_board(board), a1; reward = -100.0)
	else
		update_q(Q, alpha, 0.0, board1, board, a)					# last mover has no future
		update_q(Q, alpha, delta, board2, turn_board(board), a1)	# 
	end
	return Q
end

n_sessions = 10000000
progress = Progress(n_sessions, color=:white, showspeed=true)
@time for i in 1:n_sessions
	global Q = self_play(Q, alpha, delta)
	next!(progress)
end
#save(Q,"Q.jld")


for key in keys(Q)
	display("text/plain", key[end:-1:1,:])
end


function game_human_cpu(Q)
	board = zeros(Int8, n_rows, n_cols)
	n_tokens = zeros(Int8, n_cols)
	turn = 1
	winner = 0
	a1 = 0
	a2 = 0

	while !isterminal(board)
		display("text/plain", board[end:-1:1,:])
		if isodd(turn)
			run(`clear`)
			if haskey(Q,board)
				println("This board is known: ", round.(Q[board],digits = 4))
			else 
				println("This board is unknown")
			end
			while true
				a1 = rand(findall(Q[board] .== maximum(Q[board][n_tokens .< 6])))
				n_tokens[a1] < 6 && break
			end
			n_tokens[a1] += 1
			board[n_tokens[a1],a1] = Int8(1)
			
		else 
			while true
				print("\n1 player select row: ")
				a2 = parse(Int,readline())
				n_tokens[a2] < 6 && break
			end
			n_tokens[a2] += 1
			board[n_tokens[a2],a2] = Int8(2)
		end
		turn += 1
	end 
	println("\nPlayer $winner wins")
end

@profilehtml for i in 1:100
	global Q = self_play(Q, alpha, delta)
end

