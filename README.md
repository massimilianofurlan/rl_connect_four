# Reinforcement learning algorithms applied to the connect four game

Reinforcement learning algorithms are cool, and I want to know more about them. At the moment, I only know the Q-learning algorithm. Each time I learn a new reinforcement learning algorithm, I will apply it to the connect four game. 

## 1. Q-learning 

The game of connect four with a standard 6 x 7 board can display [4,531,985,219,092](https://oeis.org/A212693) different game configurations. Tabular methods such as Q-learning are knowingly unsuitable when the state-action space is this large: it might take a tremendous amount of time to get an accurate estimate for each state-action pair, and storing the Q-matrix might require a lot of memory.

I give it a try on a reduced 5 x 4 board, which has [1,706,255](https://tromp.github.io/c4/c4.html) different game configurations. I exploit the symmetric properties of the game to halve the state space and I store the board using 2-bits information for each disk (00: none, 10: yellow, 01:red) to reduce memory allocations. With enough training (say around 10^7 self-play episodes), the Q-agent becomes quite an opponent, especially if he plays first. You can win by being creative, finding those lone boards it has never seen before. With 2 x 10^7 self-play episodes the Q-agent is definitely good, even though it only knows 55% of all possible game configurations.

Parameters can be changed directly from the first few lines of  ```ql_connect_four.jl```. The code should work with boards of sizes 4x4:8, 5x4:7, 6x4:7, 7x4:5, and 8x4. 

Run ```julia --check-bounds=no ql_connect_four.jl``` in terminal. The Q-learning agent will train with self-play. When the training phase is finished, you will be able to play against it.

## 2. Deep Q-learning

 