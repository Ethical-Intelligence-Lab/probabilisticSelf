if (!require(pacman)) { install.packages(pacman) }
library(BayesFactor)
library(Dict)
pacman::p_load('effsize')
pacman::p_load('rjson')

# Manually enter directory path if you are not using Rstudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

games = c('logic_game', 'contingency_game', 'change_agent_game', 'contingency_game_shuffled_1')

agents = c('self_class')
game_datas <- c()
all_stats <- c()

## BAYESIAN ANALYSIS
for (game in games) {
  filename <- paste("./data_", game, ".json", sep = "", collapse = NULL)
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
      # Favors Alternative Hypothesis (mu =/= 0)
      result <- 1 / ttestBF(x = game_datas[[game]]$human[[level]], y = game_datas[[game]]$self_class_first_100[[level]])

      var_result <- var.test(game_datas[[game]]$human[[level]], game_datas[[game]]$self_class_first_100[[level]])
      result_ttest <- t.test(x = game_datas[[game]]$human[[level]], y = game_datas[[game]]$self_class_first_100[[level]], var.equal = var_result$p.value > 0.05)

      result_bf <- exp(result@bayesFactor$bf)

      bf_corr <- append(bf_corr, result_bf)
      ts <- append(ts, 1 / result_ttest$statistic)

      if (result_bf < 1.0) { # Bayes factor below 1.0 means: They are different
        result_bf <- 0
      } else {
        result_bf <- 1
      }
      bfs <- append(bfs, result_bf)


      #t.test(game_datas[[game]]$human[[level]] ~ game_datas[[game]]$self_class_first_100[[level]], paired=FALSE, var.equal=FALSE)
    }
    all_stats[[game]] <- c('Bayes Factors' = bf_corr, 't-values' = ts)
  }

}

all_stats

## Compare First and Last Level
human_first <- game_datas[[game]]$human[[1]]
human_last <- game_datas[[game]]$human[[100]]

self_first <- game_datas[[game]]$self_class_first_100[[1]]
self_last <- game_datas[[game]]$self_class_first_100[[100]]

for (game in games) {
  print(paste("*-*-*-*-*-*-*-*-*-*", game, "*-*-*-*-*-*-*-*-*-*"))

  print("Human vs DQN (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$dqn_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$human[[1]], game_datas[[game]]$dqn_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$dqn_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)

  print("Self vs DQN (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$dqn_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$self_class_first_100[[1]], game_datas[[game]]$dqn_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$dqn_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)

  print("Human vs PPO2 (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$ppo2_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$human[[1]], game_datas[[game]]$ppo2_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$ppo2_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)

  print("Self vs. PPO2 (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$ppo2_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$self_class_first_100[[1]], game_datas[[game]]$ppo2_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$ppo2_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)

  print("Human vs TRPO (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$trpo_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$human[[1]], game_datas[[game]]$trpo_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$trpo_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)


  print("Self vs TRPO (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$trpo_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$self_class_first_100[[1]], game_datas[[game]]$trpo_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$trpo_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)


  print("Human vs ACER (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$acer_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$human[[1]], game_datas[[game]]$acer_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$acer_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)


  print("Self vs ACER (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$acer_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$self_class_first_100[[1]], game_datas[[game]]$acer_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$acer_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)

  print("Human vs A2C (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$a2c_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$human[[1]], game_datas[[game]]$a2c_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$human[[1]], y = game_datas[[game]]$a2c_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)


  print("Self vs A2C (First):")

  result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$a2c_training_first_100[[1]])
  var_result <- var.test(game_datas[[game]]$self_class_first_100[[1]], game_datas[[game]]$a2c_training_first_100[[1]])
  result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[1]], y = game_datas[[game]]$a2c_training_first_100[[1]], var.equal = var_result$p.value > 0.05)

  summary(result)
  print(result_ttest)

  print("**** LAST *****")
  print("Human vs DQN (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$dqn_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$human[[100]], game_datas[[game]]$dqn_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$dqn_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)

    print("Self vs DQN (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$dqn_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$self_class_first_100[[100]], game_datas[[game]]$dqn_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$dqn_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)

    print("Human vs PPO2 (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$ppo2_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$human[[100]], game_datas[[game]]$ppo2_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$ppo2_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)

    print("Self vs. PPO2 (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$ppo2_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$self_class_first_100[[100]], game_datas[[game]]$ppo2_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$ppo2_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)

    print("Human vs TRPO (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$trpo_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$human[[100]], game_datas[[game]]$trpo_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$trpo_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)


    print("Self vs TRPO (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$trpo_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$self_class_first_100[[100]], game_datas[[game]]$trpo_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$trpo_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)


    print("Human vs ACER (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$acer_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$human[[100]], game_datas[[game]]$acer_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$acer_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)



    print("Self vs ACER (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$acer_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$self_class_first_100[[100]], game_datas[[game]]$acer_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$acer_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)

    print("Human vs A2C (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$a2c_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$human[[100]], game_datas[[game]]$a2c_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$human[[100]], y = game_datas[[game]]$a2c_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)



    print("Self vs A2C (Last):")

    result <- 1 / ttestBF(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$a2c_training_last_100[[100]])
    var_result <- var.test(game_datas[[game]]$self_class_first_100[[100]], game_datas[[game]]$a2c_training_last_100[[100]])
    result_ttest <- t.test(x = game_datas[[game]]$self_class_first_100[[100]], y = game_datas[[game]]$a2c_training_last_100[[100]], var.equal = var_result$p.value > 0.05)

    summary(result)
    print(result_ttest)
}


## Check whether the distributions of steps for the two games are
## different from each other using independent samples t-test

# Contingency Game
human_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$human))
self_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$self_class_first_100))

vr <- var.test(human_means_cont, self_means_cont)
t.test(human_means_cont, self_means_cont, var.equal = vr$p.value > 0.05)
cohen.d(human_means_cont, self_means_cont)

# Switching Mappings Game
human_means_sm <- colMeans(do.call(rbind, game_datas$contingency_game_shuffled_1$human))
self_means_sm <- colMeans(do.call(rbind, game_datas$contingency_game_shuffled_1$self_class_first_100))

vr <- var.test(human_means_sm, self_means_sm)
t.test(human_means_sm, self_means_sm, var.equal = vr$p.value > 0.05)
cohen.d(human_means_sm, self_means_sm)

