# Created by: kaan
# Created on: 5.03.2022

# Against denominator:
#   Null, mu1-mu2 = 0
# Independent samples t-test
library(BayesFactor)
library(Dict)


library("rjson")
games = c('logic_game') #, 'change_agent_game', 'contingency_game', 'contingency_game_shuffled_1'
agents = c('self_class')
game_datas <- c()
all_stats <- c()
for (game in games) {
  filename <- paste("/Users/kaan/Documents/GitHub/probabilisticSelf2/stats/data_", game, ".json", sep = "", collapse = NULL)
  game_data <- fromJSON(file = filename)
  game_datas[[game]] <- game_data

  print(paste("******", game, "*******"))
  bfs <- c()
  ts <- c()
  bf_corr <- c()
  for (agent in agents) {
    levels <- 1:100
    print(paste("--------- FIRST 100: Human vs. ", agent, " ---------"))
    for (level in levels) {
      result <- 1 / ttestBF(x = game_datas[[game]]$human[[level]], y = game_datas[[game]]$self_class_first_100[[level]])  # Favors Alternative Hypothesis (mu =/= 0)
      result_welsh <- t.test(x = game_datas[[game]]$human[[level]], y = game_datas[[game]]$self_class_first_100[[level]], var.equal = FALSE)


      result_bf <- exp(result@bayesFactor$bf)

      bf_corr <- append(bf_corr, result_bf)
      ts <- append(ts, 1/result_welsh$statistic)

      if (result_bf < 1.0) { # Bayes factor below 1.0 means: They are different
        result_bf <- 0
      } else {
        result_bf <- 1
      }
      bfs <- append(bfs, result_bf)


      #t.test(game_datas[[game]]$human[[i]] ~ game_datas[[game]]$self_class_first_100[[i]], paired=FALSE, var.equal=FALSE)
    }
    all_stats[[game]] <- c('Bayes Factors'=bf_corr, 't-values'=ts)
  }
  plot(ts)
}

if (FALSE) {
  # LOGIC GAME
  print("**** 0 ****")
  # Level 0 (x=Humans, y=Self Class):
  ttestBF(x = c(9, 27, 7, 31, 19, 15, 39, 10, 8, 14, 5, 13, 9, 9, 11, 16, 32, 18),
          y = c(6, 8, 7, 8, 7, 6, 6, 8, 6, 8))

  print("**** 1 ****")
  # Level 1 (x=Humans, y=Self Class):
  ttestBF(x = c(5, 5, 12, 6, 10, 6, 5, 13, 5, 8, 20, 6, 6, 43, 7, 5, 12, 9),
          y = c(8, 7, 6, 6, 8, 7, 7, 6, 6, 8))

  print("**** 2 ****")
  # Level 2 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 6, 5, 6, 5, 7, 6, 5, 8, 37, 16, 8, 5, 5, 6, 6, 12, 6),
          y = c(7, 7, 8, 7, 7, 6, 6, 8, 6, 6))

  print("**** 3 ****")
  # Level 3 (x=Humans, y=Self Class):
  ttestBF(x = c(6, 7, 6, 5, 8, 11, 7, 17, 7, 5, 6, 8, 5, 5, 5, 9, 5),
          y = c(8, 8, 7, 6, 8, 8, 7, 8, 6, 6))

  print("**** 4 ****")
  # Level 4 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 5, 7, 5, 6, 8, 9, 8, 5, 5, 9, 5, 7, 5, 6, 8, 5, 9),
          y = c(6, 6, 7, 7, 7, 8, 7, 6, 8, 7))

  print("**** 5 ****")
  # Level 5 (x=Humans, y=Self Class):
  ttestBF(x = c(5, 5, 5, 8, 6, 8, 7, 9, 6, 9, 5, 5, 6, 10, 6, 5, 6, 7),
          y = c(8, 6, 8, 6, 6, 7, 6, 6, 6, 6))

  print("**** 6 ****")
  # Level 6 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 5, 5, 7, 6, 7, 8, 5, 5, 5, 11, 5, 5, 10, 5, 5, 7, 7),
          y = c(8, 6, 6, 8, 6, 6, 6, 6, 6, 8))

  print("**** 7 ****")
  # Level 7 (x=Humans, y=Self Class):
  ttestBF(x = c(5, 6, 5, 5, 5, 8, 5, 5, 6, 7, 7, 7, 5, 11, 7, 5, 8, 7),
          y = c(7, 7, 6, 6, 6, 8, 8, 6, 7, 6))

  print("**** 8 ****")
  # Level 8 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 5, 6, 5, 7, 8, 5, 7, 5, 7, 5, 6, 5, 5, 6, 7, 7, 8),
          y = c(6, 6, 6, 6, 6, 7, 8, 8, 6, 6))

  print("**** 9 ****")
  # Level 9 (x=Humans, y=Self Class):
  ttestBF(x = c(6, 7, 5, 5, 5, 10, 7, 7, 7, 9, 11, 5, 7, 6, 7, 6, 5, 9),
          y = c(8, 7, 7, 7, 6, 8, 8, 7, 6, 8))


  # Compare human (Level 100) to each baseline (Level 2000).
  print("Compare human (Level 100) to each baseline (Level 2000)")
  print("**** Human vs. Self Class ****")
  ttestBF(x = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5),
          y = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6))

  print("**** Human vs. Random ****")
  ttestBF(x = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5),
          y = c(38, 62, 240, 20, 83, 28, 39, 26, 58, 82))

  print("**** Human vs. DQN ****")
  ttestBF(x = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5),
          y = c(50, 17, 6, 6, 46, 36, 11, 10, 6, 32))

  print("**** Human vs. ACER ****")
  ttestBF(x = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5),
          y = c(7, 14, 14, 8, 18, 44, 10, 8, 6, 6))

  print("**** Human vs. TRPO ****")
  ttestBF(x = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5),
          y = c(8, 11, 11, 8, 12, 18, 23, 12, 7, 12))

  print("**** Human vs. A2C ****")
  ttestBF(x = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5),
          y = c(9, 50, 8, 35, 37, 17, 11, 15, 6, 20))

  print("**** Human vs. PPO2 ****")
  ttestBF(x = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5),
          y = c(8, 8, 21, 9, 18, 7, 7, 11, 6, 6))


  print("Compare self-class to each baseline (Level 2000)")
  print("**** Self vs DQN ****")
  ttestBF(x = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
          y = c(50, 17, 6, 6, 46, 36, 11, 10, 6, 32))

  print("**** Self vs ACER ****")
  ttestBF(x = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
          y = c(7, 14, 14, 8, 18, 44, 10, 8, 6, 6))

  print("**** Self vs TRPO ****")
  ttestBF(x = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
          y = c(8, 11, 11, 8, 12, 18, 23, 12, 7, 12))

  print("**** Self vs A2C ****")
  ttestBF(x = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
          y = c(9, 50, 8, 35, 37, 17, 11, 15, 6, 20))

  print("**** Self vs PPO2 ****")
  ttestBF(x = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
          y = c(8, 8, 21, 9, 18, 7, 7, 11, 6, 6))

  print("**** Self vs Human ****")
  ttestBF(x = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
          y = c(6, 5, 7, 5, 5, 6, 10, 6, 7, 5, 7, 7, 6, 5, 7, 7, 5, 5))

  print("**** Self vs Random ****")
  ttestBF(x = c(7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
          y = c(38, 62, 240, 20, 83, 28, 39, 26, 58, 82))


  # CONTINGENCY GAME
  # Compare human (Level 100) to each baseline (Level 2000).
  print("CONTINGENCY - Compare human (Level 100) to each baseline (Level 2000)")
  print("*** CONTINGENCY * Human vs. Self Class ****")
  ttestBF(x = c(7, 7, 11, 7, 7, 7, 9, 7, 9, 15, 13, 13, 9, 15, 20, 11, 17, 7, 13, 13),
          y = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8))

  print("*** CONTINGENCY * Human vs. Random ****")
  ttestBF(x = c(7, 7, 11, 7, 7, 7, 9, 7, 9, 15, 13, 13, 9, 15, 20, 11, 17, 7, 13, 13),
          y = c(915, 726, 764, 1055, 2011, 78, 1355, 2543, 2543, 2543))

  print("*** CONTINGENCY * Human vs. DQN ****")
  ttestBF(x = c(7, 7, 11, 7, 7, 7, 9, 7, 9, 15, 13, 13, 9, 15, 20, 11, 17, 7, 13, 13),
          y = c(10, 8, 8, 8, 8, 12, 8, 10, 8, 10))

  print("*** CONTINGENCY * Human vs. ACER ****")
  ttestBF(x = c(7, 7, 11, 7, 7, 7, 9, 7, 9, 15, 13, 13, 9, 15, 20, 11, 17, 7, 13, 13),
          y = c(14, 14, 12, 16, 22, 14, 14, 10, 39, 70))

  print("*** CONTINGENCY * Human vs. TRPO ****")
  ttestBF(x = c(7, 7, 11, 7, 7, 7, 9, 7, 9, 15, 13, 13, 9, 15, 20, 11, 17, 7, 13, 13),
          y = c(24, 46, 28, 10, 16, 10, 8, 22, 14, 10))

  print("*** CONTINGENCY * Human vs. A2C ****")
  ttestBF(x = c(7, 7, 11, 7, 7, 7, 9, 7, 9, 15, 13, 13, 9, 15, 20, 11, 17, 7, 13, 13),
          y = c(8, 10, 8, 10, 10, 8, 8, 8, 12, 10))

  print("*** CONTINGENCY * Human vs. PPO2 ****")
  ttestBF(x = c(7, 7, 11, 7, 7, 7, 9, 7, 9, 15, 13, 13, 9, 15, 20, 11, 17, 7, 13, 13),
          y = c(8, 8, 10, 12, 10, 10, 30, 10, 74, 10))


  print("CONTINGENCY - Compare self-class to each baseline (Level 2000)")
  print("*** CONTINGENCY * Self vs. Self Class ****")
  ttestBF(x = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8),
          y = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8))

  print("*** CONTINGENCY * Self vs. Random ****")
  ttestBF(x = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8),
          y = c(915, 726, 764, 1055, 2011, 78, 1355, 2543, 2543, 2543))

  print("*** CONTINGENCY * Self vs. DQN ****")
  ttestBF(x = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8),
          y = c(10, 8, 8, 8, 8, 12, 8, 10, 8, 10))

  print("*** CONTINGENCY * Self vs. ACER ****")
  ttestBF(x = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8),
          y = c(14, 14, 12, 16, 22, 14, 14, 10, 39, 70))

  print("*** CONTINGENCY * Self vs. TRPO ****")
  ttestBF(x = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8),
          y = c(24, 46, 28, 10, 16, 10, 8, 22, 14, 10))

  print("*** CONTINGENCY * Self vs. A2C ****")
  ttestBF(x = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8),
          y = c(8, 10, 8, 10, 10, 8, 8, 8, 12, 10))

  print("*** CONTINGENCY * Self vs. PPO2 ****")
  ttestBF(x = c(10, 10, 10, 10, 10, 10, 10, 10, 8, 8),
          y = c(8, 8, 10, 12, 10, 10, 30, 10, 74, 10))


  print("*** CONT * 0 ****")
  # Level 0 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 20, 9, 80, 9, 21, 21, 31, 33, 7, 43, 41, 69, 94, 15, 7, 587, 19, 31, 7),
          y = c(8, 10, 12, 10, 10, 10, 10, 10, 10, 8))

  print("*** CONT * 1 ****")
  # Level 1 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 23, 7, 46, 21, 35, 13, 73, 7, 39, 14, 9, 7, 11, 107, 7, 75, 9, 17, 9),
          y = c(8, 10, 8, 12, 10, 10, 8, 12, 8, 10))

  print("*** CONT * 2 ****")
  # Level 2 (x=Humans, y=Self Class):
  ttestBF(x = c(11, 19, 7, 22, 21, 20, 7, 11, 7, 15, 9, 7, 7, 38, 7, 7, 7, 41, 9, 13),
          y = c(10, 8, 12, 10, 10, 8, 8, 10, 8, 8))

  print("*** CONT * 3 ****")
  # Level 3 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 7, 7, 7, 17, 23, 7, 23, 9, 27, 46, 15, 22, 17, 52, 13, 28, 7, 7, 7),
          y = c(12, 8, 10, 12, 10, 10, 8, 8, 10, 10))

  print("*** CONT * 4 ****")
  # Level 4 (x=Humans, y=Self Class):
  ttestBF(x = c(9, 7, 20, 7, 13, 13, 7, 7, 26, 11, 18, 21, 9, 17, 11, 7, 17, 11, 11, 7),
          y = c(8, 10, 10, 8, 8, 12, 10, 8, 8, 10))

  print("*** CONT * 5 ****")
  # Level 5 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 7, 19, 7, 7, 29, 7, 42, 9, 9, 19, 22, 21, 21, 7, 31, 20, 7, 18, 7),
          y = c(10, 8, 12, 8, 10, 8, 10, 10, 10, 8))

  print("*** CONT * 6 ****")
  # Level 6 (x=Humans, y=Self Class):
  ttestBF(x = c(9, 17, 11, 7, 21, 17, 11, 7, 13, 19, 9, 11, 7, 7, 7, 21, 19, 7, 11, 7),
          y = c(8, 10, 8, 10, 8, 10, 8, 8, 8, 8))

  print("*** CONT * 7 ****")
  # Level 7 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 7, 7, 40, 19, 7, 11, 7, 9, 7, 13, 9, 19, 7, 27, 87, 7, 38, 11, 7),
          y = c(8, 12, 8, 10, 10, 8, 8, 8, 10, 10))

  print("*** CONT * 8 ****")
  # Level 8 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 11, 23, 7, 7, 17, 13, 9, 9, 7, 11, 15, 7, 7, 28, 28, 22, 28, 18, 7),
          y = c(10, 12, 12, 12, 10, 12, 8, 10, 8, 10))

  print("*** CONT * 9 ****")
  # Level 9 (x=Humans, y=Self Class):
  ttestBF(x = c(11, 7, 9, 18, 17, 7, 7, 20, 7, 19, 7, 7, 7, 11, 22, 24, 7, 17, 18, 7),
          y = c(8, 12, 8, 8, 10, 10, 8, 8, 10, 10))

  # CONTINGENCY GAME
  # Compare human (Level 100) to each baseline (Level 2000).
  print("SWITCHING EMBODIMENTS - Compare human (Level 100) to each baseline (Level 2000)")
  print("*** SE * Human vs. Self Class ****")
  ttestBF(x = c(37, 15, 33, 23, 79, 62, 61, 79, 61, 11, 39, 17, 17, 81, 31, 39, 12, 33),
          y = c(26, 13, 19, 28, 48, 20, 20, 34, 18, 41))

  print("*** SE * Human vs. Random ****")
  ttestBF(x = c(37, 15, 33, 23, 79, 62, 61, 79, 61, 11, 39, 17, 17, 81, 31, 39, 12, 33),
          y = c(126, 120, 44, 329, 264, 778, 32, 1253, 42, 267))

  print("*** SE * Human vs. DQN ****")
  ttestBF(x = c(37, 15, 33, 23, 79, 62, 61, 79, 61, 11, 39, 17, 17, 81, 31, 39, 12, 33),
          y = c(82, 40, 20, 88, 81, 40, 104, 30, 133, 14))

  print("*** SE * Human vs. ACER ****")
  ttestBF(x = c(37, 15, 33, 23, 79, 62, 61, 79, 61, 11, 39, 17, 17, 81, 31, 39, 12, 33),
          y = c(34, 18, 102, 28, 44, 22, 21, 20, 30, 73))

  print("*** SE * Human vs. TRPO ****")
  ttestBF(x = c(37, 15, 33, 23, 79, 62, 61, 79, 61, 11, 39, 17, 17, 81, 31, 39, 12, 33),
          y = c(18, 28, 180, 27, 144, 28, 154, 30, 18, 272))

  print("*** SE * Human vs. A2C ****")
  ttestBF(x = c(37, 15, 33, 23, 79, 62, 61, 79, 61, 11, 39, 17, 17, 81, 31, 39, 12, 33),
          y = c(40, 83, 33, 12, 46, 84, 36, 26, 219, 14))

  print("*** SE * Human vs. PPO2 ****")
  ttestBF(x = c(37, 15, 33, 23, 79, 62, 61, 79, 61, 11, 39, 17, 17, 81, 31, 39, 12, 33),
          y = c(34, 26, 36, 49, 140, 95, 34, 14, 18, 26))


  print("SWITCHING EMBODIMENTS - Compare self-class to each baseline (Level 2000)")

  print("*** SE * Self vs. DQN ****")
  ttestBF(x = c(26, 13, 19, 28, 48, 20, 20, 34, 18, 41),
          y = c(82, 40, 20, 88, 81, 40, 104, 30, 133, 14))

  print("*** SE * Self vs. ACER ****")
  ttestBF(x = c(26, 13, 19, 28, 48, 20, 20, 34, 18, 41),
          y = c(34, 18, 102, 28, 44, 22, 21, 20, 30, 73))

  print("*** SE * Self vs. TRPO ****")
  ttestBF(x = c(26, 13, 19, 28, 48, 20, 20, 34, 18, 41),
          y = c(18, 28, 180, 27, 144, 28, 154, 30, 18, 272))

  print("*** SE * Self vs. A2C ****")
  ttestBF(x = c(26, 13, 19, 28, 48, 20, 20, 34, 18, 41),
          y = c(40, 83, 33, 12, 46, 84, 36, 26, 219, 14))

  print("*** SE * Self vs. PPO2 ****")
  ttestBF(x = c(26, 13, 19, 28, 48, 20, 20, 34, 18, 41),
          y = c(34, 26, 36, 49, 140, 95, 34, 14, 18, 26))


  print("*** SE * 0 ****")
  # Level 0 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 20, 9, 80, 9, 21, 21, 31, 33, 7, 43, 41, 69, 94, 15, 7, 587, 19, 31, 7),
          y = c(12, 14, 46, 86, 77, 48, 118, 26, 28, 56))

  print("*** SE * 1 ****")
  # Level 1 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 23, 7, 46, 21, 35, 13, 73, 7, 39, 14, 9, 7, 11, 107, 7, 75, 9, 17, 9),
          y = c(44, 18, 88, 14, 51, 56, 28, 14, 91, 20))

  print("*** SE * 2 ****")
  # Level 2 (x=Humans, y=Self Class):
  ttestBF(x = c(11, 19, 7, 22, 21, 20, 7, 11, 7, 15, 9, 7, 7, 38, 7, 7, 7, 41, 9, 13),
          y = c(16, 28, 32, 20, 20, 20, 14, 14, 20, 42))

  print("*** SE * 3 ****")
  # Level 3 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 7, 7, 7, 17, 23, 7, 23, 9, 27, 46, 15, 22, 17, 52, 13, 28, 7, 7, 7),
          y = c(44, 34, 32, 168, 12, 20, 48, 24, 14, 28))

  print("*** SE * 4 ****")
  # Level 4 (x=Humans, y=Self Class):
  ttestBF(x = c(9, 7, 20, 7, 13, 13, 7, 7, 26, 11, 18, 21, 9, 17, 11, 7, 17, 11, 11, 7),
          y = c(14, 14, 16, 20, 42, 34, 60, 20, 18, 83))

  print("*** SE * 5 ****")
  # Level 5 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 7, 19, 7, 7, 29, 7, 42, 9, 9, 19, 22, 21, 21, 7, 31, 20, 7, 18, 7),
          y = c(38, 42, 34, 34, 12, 28, 16, 16, 34, 14))

  print("*** SE * 6 ****")
  # Level 6 (x=Humans, y=Self Class):
  ttestBF(x = c(9, 17, 11, 7, 21, 17, 11, 7, 13, 19, 9, 11, 7, 7, 7, 21, 19, 7, 11, 7),
          y = c(18, 12, 14, 30, 26, 114, 38, 16, 34, 20))

  print("*** SE * 7 ****")
  # Level 7 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 7, 7, 40, 19, 7, 11, 7, 9, 7, 13, 9, 19, 7, 27, 87, 7, 38, 11, 7),
          y = c(18, 34, 33, 52, 20, 68, 14, 26, 51, 26))

  print("*** SE * 8 ****")
  # Level 8 (x=Humans, y=Self Class):
  ttestBF(x = c(7, 11, 23, 7, 7, 17, 13, 9, 9, 7, 11, 15, 7, 7, 28, 28, 22, 28, 18, 7),
          y = c(82, 32, 40, 42, 14, 40, 18, 24, 46, 26))

  print("*** SE * 9 ****")
  # Level 9 (x=Humans, y=Self Class):
  ttestBF(x = c(11, 7, 9, 18, 17, 7, 7, 20, 7, 19, 7, 7, 7, 11, 22, 24, 7, 17, 18, 7),
          y = c(40, 24, 28, 34, 20, 42, 12, 20, 26, 42))
}
