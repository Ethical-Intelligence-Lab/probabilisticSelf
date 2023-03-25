rm(list = ls())
if (!require(pacman)) { install.packages(pacman) }

pacman::p_load('effsize')
pacman::p_load('rjson')
pacman::p_load('BayesFactor')
pacman::p_load('Dict')
pacman::p_load('ggpubr')

# Manually enter directory path if you are not using Rstudio
setwd(getwd())

games = c('change_agent_game')

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

# *-*-*-*-*-*-*-*-* BAYESIAN TESTS *-*-*-*-*-*-*-*-* #
## Human v. Self Class for the First Level Levels
for (game in games) {
    filename <- paste("./data_", game, ".json", sep = "", collapse = NULL)
    game_data <- fromJSON(file = filename)
    game_datas[[game]] <- game_data

    if(game != 'contingency_game_shuffled_1') {  # Switching mappings does not have perturbation task
        perturbation_c <- c(TRUE, FALSE)
    } else {
        perturbation_c <- c(FALSE)
    }

    for(perturbation in perturbation_c) { # Two separate tests for perturbation task and regular
        bfs_binary <- c()
        ts <- c()
        ps <- c()
        bf_corr <- c()

        if(perturbation) { levels <- 1:150 } else { levels <- 1:100 }

        if (game == "change_agent_game" && perturbation) { levels <- 1:30 } # 30 Levels in change_agent_game perturbation
        
        for (level in levels) {
            # Favors Alternative Hypothesis (mu =/= 0)
            if(perturbation) {
                x <- game_datas[[game]]$human_extended[[level]]
            } else {
                x <- game_datas[[game]]$human[[level]]
            }
            
            y <- game_datas[[game]]$self_class_first_150[[level]]
            result <- 1 / ttestBF(x = x, y = y)

            var_result <- var.test(x, y)
            result_ttest <- t.test(x = x, y = y, var.equal = var_result$p.value > 0.05)
            print(level)
            print(result_ttest)

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

            if(perturbation) {
                perturbation_s <- "perturbated"
            } else {
                perturbation_s <- "normal"
            }
            all_stats[[game]][[perturbation_s]][["Bayes Factors"]] <- bf_corr
            all_stats[[game]][[perturbation_s]][['Binary Bayes Factors']] <- bfs_binary
            all_stats[[game]][[perturbation_s]][['t-values']] <- ts
            all_stats[[game]][[perturbation_s]][['p-values']] <- ps

            all_bfs_binary[[game]][[perturbation_s]] <- bfs_binary
       }
    }
}


### Save binary bayes results
json_data <- toJSON(all_bfs_binary)
write(json_data, paste0(game, "_bfs.json"))


### Did humans do more steps after the perturbation?
print("Did humans do more steps after the perturbation?")
if( game %in% c('contingency_game', 'change_agent_game') ) {
    if (game == 'change_agent_game') {
        level_means <- lapply(game_datas[[game]][['human_extended']], mean)
        var_result <- var.test(unlist(level_means[0:19]), unlist(level_means[20:30]))
        result_ttest <- t.test(x = unlist(level_means[0:19]), y = unlist(level_means[20:30]), var.equal = var_result$p.value > 0.05)
        print(result_ttest)
        print(cohen.d(unlist(level_means[0:19]), unlist(level_means[20:30])))
    } else {
        level_means <- lapply(game_datas[[game]][['human_extended']], mean)
        var_result <- var.test(unlist(level_means[0:99]), unlist(level_means[100:150]))
        result_ttest <- t.test(x = unlist(level_means[0:99]), y = unlist(level_means[100:150]), var.equal = var_result$p.value > 0.05)
        print(result_ttest)
        print(cohen.d(unlist(level_means[0:99]), unlist(level_means[100:150])))
    }
}

################ T-Tests ################
for (game in games) {
    print(paste("*-*-*-*-*-*-*-*-*-*", game, "*-*-*-*-*-*-*-*-*-*"))

    print("**** LAST *****")
    print("Human vs OC (Last Level):")

    game_d <- game_datas[[game]]

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$option_critic_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$option_critic_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$option_critic_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Self vs OC (Last Level):")

    result <- 1 / ttestBF(x = game_d$self_class_first_150[[150]], y = game_d$option_critic_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_150[[150]], game_d$option_critic_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_150[[150]], y = game_d$option_critic_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs DQN (Last Level):")

    game_d <- game_datas[[game]]

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$dqn_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$dqn_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$dqn_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Self vs DQN (Last Level):")

    result <- 1 / ttestBF(x = game_d$self_class_first_150[[150]], y = game_d$dqn_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_150[[150]], game_d$dqn_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_150[[150]], y = game_d$dqn_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs PPO2 (Last Level):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$ppo2_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$ppo2_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$ppo2_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Self vs. PPO2 (Last Level):")

    result <- 1 / ttestBF(x = game_d$self_class_first_150[[150]], y = game_d$ppo2_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_150[[150]], game_d$ppo2_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_150[[150]], y = game_d$ppo2_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs TRPO (Last Level):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$trpo_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$trpo_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$trpo_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs TRPO (Last Level):")

    result <- 1 / ttestBF(x = game_d$self_class_first_150[[150]], y = game_d$trpo_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_150[[150]], game_d$trpo_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_150[[150]], y = game_d$trpo_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Human vs ACER (Last Level):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$acer_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$acer_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$acer_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs ACER (Last Level):")

    result <- 1 / ttestBF(x = game_d$self_class_first_150[[150]], y = game_d$acer_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_150[[150]], game_d$acer_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_150[[150]], y = game_d$acer_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)

    print("Human vs A2C (Last Level):")

    result <- 1 / ttestBF(x = game_d$human[[100]], y = game_d$a2c_training_last_100[[100]])
    var_result <- var.test(game_d$human[[100]], game_d$a2c_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$human[[100]], y = game_d$a2c_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    print("Self vs A2C (Last Level):")

    result <- 1 / ttestBF(x = game_d$self_class_first_150[[150]], y = game_d$a2c_training_last_100[[100]])
    var_result <- var.test(game_d$self_class_first_150[[150]], game_d$a2c_training_last_100[[100]])
    result_ttest <- t.test(x = game_d$self_class_first_150[[150]], y = game_d$a2c_training_last_100[[100]], var.equal = var_result$p.value > 0.05)
    print_sig(result_ttest$p.value)

    summary(result)
    print(result_ttest)


    if(FALSE) {
        print("***** First Level ******")
        print("Human vs DQN (First Level):")

        result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$dqn_training_first_150[[1]])
        var_result <- var.test(game_d$human[[1]], game_d$dqn_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$human[[1]], y = game_d$dqn_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)

        print("Self vs DQN (First Level):")

        result <- 1 / ttestBF(x = game_d$self_class_first_150[[1]], y = game_d$dqn_training_first_150[[1]])
        var_result <- var.test(game_d$self_class_first_150[[1]], game_d$dqn_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$self_class_first_150[[1]], y = game_d$dqn_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)

        print("Human vs PPO2 (First Level):")

        result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$ppo2_training_first_150[[1]])
        var_result <- var.test(game_d$human[[1]], game_d$ppo2_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$human[[1]], y = game_d$ppo2_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)

        print("Self vs. PPO2 (First Level):")

        result <- 1 / ttestBF(x = game_d$self_class_first_150[[1]], y = game_d$ppo2_training_first_150[[1]])
        var_result <- var.test(game_d$self_class_first_150[[1]], game_d$ppo2_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$self_class_first_150[[1]], y = game_d$ppo2_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)

        print("Human vs TRPO (First Level):")

        result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$trpo_training_first_150[[1]])
        var_result <- var.test(game_d$human[[1]], game_d$trpo_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$human[[1]], y = game_d$trpo_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)


        print("Self vs TRPO (First Level):")

        result <- 1 / ttestBF(x = game_d$self_class_first_150[[1]], y = game_d$trpo_training_first_150[[1]])
        var_result <- var.test(game_d$self_class_first_150[[1]], game_d$trpo_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$self_class_first_150[[1]], y = game_d$trpo_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)


        print("Human vs ACER (First Level):")

        result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$acer_training_first_150[[1]])
        var_result <- var.test(game_d$human[[1]], game_d$acer_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$human[[1]], y = game_d$acer_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)


        print("Self vs ACER (First Level):")

        result <- 1 / ttestBF(x = game_d$self_class_first_150[[1]], y = game_d$acer_training_first_150[[1]])
        var_result <- var.test(game_d$self_class_first_150[[1]], game_d$acer_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$self_class_first_150[[1]], y = game_d$acer_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)

        print("Human vs A2C (First Level):")

        result <- 1 / ttestBF(x = game_d$human[[1]], y = game_d$a2c_training_first_150[[1]])
        var_result <- var.test(game_d$human[[1]], game_d$a2c_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$human[[1]], y = game_d$a2c_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)


        print("Self vs A2C (First Level):")

        result <- 1 / ttestBF(x = game_d$self_class_first_150[[1]], y = game_d$a2c_training_first_150[[1]])
        var_result <- var.test(game_d$self_class_first_150[[1]], game_d$a2c_training_first_150[[1]])
        result_ttest <- t.test(x = game_d$self_class_first_150[[1]], y = game_d$a2c_training_first_150[[1]], var.equal = var_result$p.value > 0.05)
        print_sig(result_ttest$p.value)

        summary(result)
        print(result_ttest)
    }
    
}


## Check whether the distributions of steps for the two games are
## different from each other using independent samples t-test

if(FALSE) {
    # Contingency Game
    # On average, humans took significantly more steps than the self-class to solve each level over the first 100 levels
    human_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$human))
    self_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$self_class_first_150))

    vr <- var.test(human_means_cont, self_means_cont)
    t.test(human_means_cont, self_means_cont, var.equal = vr$p.value > 0.05)
    cohen.d(human_means_cont, self_means_cont)

    # Switching Mappings Game
    human_means_sm <- colMeans(do.call(rbind, game_datas$contingency_game_shuffled_1$human))
    self_means_sm <- colMeans(do.call(rbind, game_datas$
        contingency_game_shuffled_1$
        self_class_first_150))

    vr <- var.test(human_means_sm, self_means_sm)
    t.test(human_means_sm, self_means_sm, var.equal = vr$p.value > 0.05)
    cohen.d(human_means_sm, self_means_sm)
}


################## SELF FINDING STUDIES ##################

if (game %in% c('contingency_game', 'contingency_game_shuffled_1')) { #'contingency_game'
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
        theme_bw(base_size = 22) +
        theme(legend.position = "none", axis.line = element_line(color = 'black', size=0.25), panel.grid.minor = element_blank(), panel.grid.major = element_blank(), panel.border = element_blank(), axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"), axis.ticks.length=unit(0.013,"inch"), axis.ticks=element_line(size=unit(0.2,"inch"))) +
        scale_x_continuous(breaks = function(x) unique(floor(pretty(seq(0, (max(x) + 1) * 1.1)))))

    if (game == 'contingency_game') {
        plot <- plot +
            coord_cartesian(ylim = c(2, 7), xlim = c(8, 20)) +
            stat_cor(method = "pearson", label.x = 11.7, label.y = 7, p.digits = 3, size = 7, cor.coef.name = "r")
    } else if (game == 'contingency_game_shuffled_1') {
        plot <- plot +
            stat_cor(method = "pearson", label.x = 35, label.y = 45, p.digits = 3, size = 7, cor.coef.name = "r") +
            coord_cartesian(ylim = c(5, 45), xlim = c(20, 65))
    }

    ggsave(paste0("scatter_self_orient_", game, ".png"), plot = plot, units = "cm", width=21.8, height=12.1, dpi=1000)

}
