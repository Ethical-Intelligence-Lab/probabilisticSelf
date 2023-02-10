if (!require(pacman)) { install.packages(pacman) }

pacman::p_load('effsize')
pacman::p_load('rjson')
pacman::p_load('BayesFactor')
pacman::p_load('Dict')
pacman::p_load('ggpubr')

# Manually enter directory path if you are not using Rstudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

games = c('logic_game')

agents = c('self_class')
game_datas <- c()
all_stats <- c()
all_bfs_binary <- c()

# Note. ‘.’= p < .1, ‘**’ = p < .01, ‘***’ = p < .001.
print_sig <- function(p) {
    if (p < 0.001) {
        print("Significance: ***")
    } else if (p < 0.01) {
        print("Significance: **")
    } else if (p < 0.05) {
        print("Significance: *")
    } else if (p < 0.1) {
        print("Significance: .")
    } else {
        print("not significant")
    }
}

## Human v. Self Class for the First Hundred Levels -- BAYESIAN ANALYSIS
for (game in games) {
    filename <- paste("./data_", game, ".json", sep = "", collapse = NULL)
    game_data <- fromJSON(file = filename)
    game_datas[[game]] <- game_data
    bfs_binary <- c()
    ts <- c()
    ps <- c()
    bf_corr <- c()
    for (agent in agents) {
        levels <- 1:150
        for (level in levels) {
            # Favors Alternative Hypothesis (mu =/= 0)
            x <- game_datas[[game]]$human_extended[[level]]
            y <- game_datas[[game]]$self_class_first_150[[level]]
            result <- 1 / ttestBF(x = x, y = y)

            var_result <- var.test(x, y)
            result_ttest <- t.test(x = x, y = y, var.equal = var_result$p.value > 0.05)

            result_bf <- exp(result@bayesFactor$bf)

            bf_corr <- append(bf_corr, result_bf)
            ts <- append(ts, result_ttest$statistic)
            ps <- append(ps, result_ttest$p.value)

            if (result_bf < 1.0) { # Bayes factor below 1.0 means: They are different
                result_bf <- 0
            } else {
                result_bf <- 1
            }
            bfs_binary <- append(bfs_binary, result_bf)

        }
        all_stats[[game]][['Bayes Factors']] <- bf_corr
        all_stats[[game]][['Binary Bayes Factors']] <- bfs_binary
        all_stats[[game]][['t-values']] <- ts
        all_stats[[game]][['p-values']] <- ps

        all_bfs_binary[[game]] <- bfs_binary
    }

}

for (game in games) {
    print(paste("*-*-*-*-*-*-*-*-*-*", game, "*-*-*-*-*-*-*-*-*-*"))

    print("**** LAST *****")
    print("Human vs DQN (LAST HUNDRED):")

    game_d <- game_datas[[game]]

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$dqn_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$dqn_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$dqn_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Self vs DQN (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[100]], y = game_d$dqn_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_100[[100]], game_d$dqn_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[100]], y = game_d$dqn_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs PPO2 (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$ppo2_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$ppo2_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$ppo2_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Self vs. PPO2 (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[100]], y = game_d$ppo2_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_100[[100]], game_d$ppo2_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[100]], y = game_d$ppo2_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs TRPO (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$trpo_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$trpo_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$trpo_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs TRPO (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[100]], y = game_d$trpo_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_100[[100]], game_d$trpo_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[100]], y = game_d$trpo_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Human vs ACER (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$acer_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$acer_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$acer_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs ACER (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[100]], y = game_d$acer_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_100[[100]], game_d$acer_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[100]], y = game_d$acer_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs A2C (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$a2c_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$a2c_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$a2c_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs A2C (LAST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[100]], y = game_d$a2c_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_100[[100]], game_d$a2c_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[100]], y = game_d$a2c_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("***** FIRST HUNDRED ******")
    print("Human vs DQN (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$dqn_training_first_100[[1]])
    var_result <- var.test(game_d$human[[1]], game_d$dqn_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$human[[1]], y = game_d$dqn_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Self vs DQN (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[1]], y = game_d$dqn_training_first_100[[1]])
    var_result <- var.test(game_d$self_class_first_100[[1]], game_d$dqn_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[1]], y = game_d$dqn_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs PPO2 (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$ppo2_training_first_100[[1]])
    var_result <- var.test(game_d$human[[1]], game_d$ppo2_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$human[[1]], y = game_d$ppo2_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Self vs. PPO2 (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[1]], y = game_d$ppo2_training_first_100[[1]])
    var_result <- var.test(game_d$self_class_first_100[[1]], game_d$ppo2_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[1]], y = game_d$ppo2_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs TRPO (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$trpo_training_first_100[[1]])
    var_result <- var.test(game_d$human[[1]], game_d$trpo_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$human[[1]], y = game_d$trpo_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs TRPO (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[1]], y = game_d$trpo_training_first_100[[1]])
    var_result <- var.test(game_d$self_class_first_100[[1]], game_d$trpo_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[1]], y = game_d$trpo_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Human vs ACER (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$acer_training_first_100[[1]])
    var_result <- var.test(game_d$human[[1]], game_d$acer_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$human[[1]], y = game_d$acer_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs ACER (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[1]], y = game_d$acer_training_first_100[[1]])
    var_result <- var.test(game_d$self_class_first_100[[1]], game_d$acer_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[1]], y = game_d$acer_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs A2C (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$a2c_training_first_100[[1]])
    var_result <- var.test(game_d$human[[1]], game_d$a2c_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$human[[1]], y = game_d$a2c_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs A2C (FIRST HUNDRED):")

    result <- 1 / ttestBF(x = game_d$self_class_first_100[[1]], y = game_d$a2c_training_first_100[[1]])
    var_result <- var.test(game_d$self_class_first_100[[1]], game_d$a2c_training_first_100[[1]])
    result_ttest <- t.test(x = game_d$self_class_first_100[[1]], y = game_d$a2c_training_first_100[[1]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)
}


## Check whether the distributions of steps for the two games are
## different from each other using independent samples t-test

# Contingency Game
# On average, humans took significantly more steps than the self-class to solve each level over the first 100 levels
human_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$human))
self_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$self_class_first_100))

vr <- var.test(human_means_cont, self_means_cont)
t.test(human_means_cont, self_means_cont, var.equal = vr$p.value > 0.05)
cohen.d(human_means_cont, self_means_cont)

# Switching Mappings Game
human_means_sm <- colMeans(do.call(rbind, game_datas$contingency_game_shuffled_1$human))
self_means_sm <- colMeans(do.call(rbind, game_datas$
    contingency_game_shuffled_1$
    self_class_first_100))

vr <- var.test(human_means_sm, self_means_sm)
t.test(human_means_sm, self_means_sm, var.equal = vr$p.value > 0.05)
cohen.d(human_means_sm, self_means_sm)


################## SELF FINDING STUDIES ##################

for (game in c('contingency_game', 'contingency_game_shuffled_1')) {
    print(paste("*-*-*-*-*-*-*-*-*", game, "*-*-*-*-*-*-*-*-*"))
    ## Check if participants performed differently in the self-finding game compared to the original run
    human_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$human))
    human_sf_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$data_sf))

    vr <- var.test(human_lvl_means_cont, human_sf_lvl_means_cont)
    print(t.test(human_lvl_means_cont, human_sf_lvl_means_cont, var.equal = vr$p.value > 0.05))
    print(cohen.d(human_lvl_means_cont, human_sf_lvl_means_cont))

    # Correlating no. steps until self-orienting with no. steps to complete a level, for each participant
    self_orienting_steps <- read.csv(paste0('self_orienting_', game, '.csv'))$level_means
    print(cor.test(human_sf_lvl_means_cont, self_orienting_steps))

    # Creating the plot
    plot <- ggplot(data = data.frame("step_count" = human_sf_lvl_means_cont, "self_orienting" = self_orienting_steps),
                   aes(x = step_count, y = self_orienting)) +
        geom_point(alpha = 0.2) +
        xlab("Total Step Count") +
        ylab("No. Steps Until Self Orienting") +
        #theme_bw(base_size = 28) +
        geom_point() +
        coord_fixed() +
        geom_smooth(method = 'lm', se = T, size = 1, alpha = 0.2, color = 'red') +
        theme_bw(base_size = 20) +
        theme(legend.position = "none", axis.line = element_line(color = 'black'), panel.grid.minor = element_blank(), panel.border = element_blank())

    if (game == 'contingency_game') {
        plot <- plot +
            coord_cartesian(ylim = c(2, 7), xlim = c(8, 20)) +
            stat_cor(method = "pearson", label.x = 12.6, label.y = 7, p.digits = 3, size = 5, cor.coef.name = "r")
    } else if (game == 'contingency_game_shuffled_1') {
        plot <- plot +
            stat_cor(method = "pearson", label.x = 35, label.y = 45, p.digits = 3, size = 5, cor.coef.name = "r") +
            coord_cartesian(ylim = c(5, 45), xlim = c(20, 65))
    }

    ggsave(paste0("scatter_self_orient_", game, ".png"), plot = plot, units = "cm")

}
