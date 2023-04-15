rm(list = ls())
if (!require(pacman)) { install.packages(pacman) }

pacman::p_load('effsize')
pacman::p_load('rjson')
pacman::p_load('BayesFactor')
pacman::p_load('Dict')
pacman::p_load('ggpubr')

# Set working directors
if(!grepl('stats', getwd())) {
    setwd(paste0(getwd(), '/stats'))
}

# Generate bayes factor for the given game
generate_bfs <- function(game) {
    game_datas <- c()
    all_stats <- c()
    all_bfs_binary <- c()

    filename <- paste("./data_", game, ".json", sep = "", collapse = NULL)
    game_datas[[game]] <- fromJSON(file = filename)

    # *-*-*-*-*-*-*-*-* BAYESIAN TESTS *-*-*-*-*-*-*-*-* #
    ## Human v. Self Class for the First Level Levels
    if(!(game %in% c('contingency_game_shuffled_1', 'logic_game', 'change_agent_game'))) {  # Switching mappings and logic does not have perturbation task
        perturbation_c <- c(TRUE, FALSE)
    } else {
        perturbation_c <- c(FALSE)
    }

    if(game == "change_agent_game_harder") {
        perturbation_c <- c(TRUE)
    }

    for(perturbation in perturbation_c) { # Two separate tests for perturbation task and regular
        if(perturbation) {
            print(paste0("Generating bayes factors for ", game, " perturbated"))
        } else {
            print(paste0("Generating bayes factors for ", game))
        }
        
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
                y <- game_datas[[game]]$self_classshort_first_150[[level]]
            } else {
                x <- game_datas[[game]]$human[[level]]
                y <- game_datas[[game]]$self_class_first_150[[level]]
            }
            
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


    ### Save binary bayes results
    json_data <- toJSON(all_bfs_binary)
    write(json_data, paste0(game, "_bfs.json"))

}

## *-*-*-* GENERATE BAYES FACTORS *-*-*-* ##
for(game in c('logic_game', 'contingency_game', 'contingency_game_shuffled_1', 'change_agent_game', 'change_agent_game_harder')) {
    generate_bfs(game)
}

################ T-Tests Comparing Last Level of Humans v. RL Agents ################
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

for (game in c('change_agent_game')) { #'logic_game', 'contingency_game', 'contingency_game_shuffled_1', 'change_agent_game'
    game_datas <- c()
    filename <- paste("./data_", game, ".json", sep = "", collapse = NULL)
    game_datas[[game]] <- fromJSON(file = filename)

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

    if(game == 'contingency_game') {
        print("On average, humans took significantly more steps than the self-class to solve each level over the first 100 levels")
        human_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$human))
        self_means_cont <- colMeans(do.call(rbind, game_datas$contingency_game$self_class_first_150))

        vr <- var.test(human_means_cont, self_means_cont)
        print(t.test(human_means_cont, self_means_cont, var.equal = vr$p.value > 0.05))
        print(cohen.d(human_means_cont, self_means_cont))
    }

    if(game == 'contingency_game_shuffled_1') {
        # Switching Mappings Game
        print("On average, humans took significantly more steps than the self-class to solve each level over the first 100 levels")
        human_means_sm <- colMeans(do.call(rbind, game_datas$contingency_game_shuffled_1$human))
        self_means_sm <- colMeans(do.call(rbind, game_datas$
            contingency_game_shuffled_1$
            self_class_first_150))

        vr <- var.test(human_means_sm, self_means_sm)
        print(t.test(human_means_sm, self_means_sm, var.equal = vr$p.value > 0.05))
        print(cohen.d(human_means_sm, self_means_sm))
    }
}

################## SELF FINDING STUDIES ##################

game_datas <- c()
for(game in c('logic_game')) { #, 'logic_game', 'contingency_game', 'contingency_game_shuffled_1'
    filename <- paste("./data_", game, ".json", sep = "", collapse = NULL)
    game_datas[[game]] <- fromJSON(file = filename)

    print(paste0("*-*-*-*-*-*-*-*", game, "*-*-*-*-*-*-*-*"))
    ## Check if participants performed differently in the self-finding game compared to the original run
    if(game != 'logic_game') {
        print("Checking if participants performed differently in the self-finding game compared to the original run")
        human_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$human))
        human_sf_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$data_sf))

        vr <- var.test(human_lvl_means_cont, human_sf_lvl_means_cont)
        print(t.test(human_lvl_means_cont, human_sf_lvl_means_cont, var.equal = vr$p.value > 0.05))
        print(cohen.d(human_lvl_means_cont, human_sf_lvl_means_cont))
    }
    
    # Correlations for artificial agents:
    if(game == 'logic_game') {
        human_sf_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$human))
        for(ai in c('dqn_training', 'a2c_training', 'trpo_training', 'acer_training', 'ppo2_training', 'option_critic', 'random', 'self_class')) {
            print(paste("*-*-*-* ", ai, " *-*-*-*"))
            print(cor.test(game_datas[game][[1]][[paste0(ai, '_all')]][1900:2000], read.csv(paste0('self_orienting_', game, '.csv'))[[paste0(ai, '_m')]][1900:2000]))

            plot <- ggplot(data = data.frame("step_count" = game_datas[game][[1]][[paste0(ai, '_all')]][1900:2000], "self_orienting" = read.csv(paste0('self_orienting_', game, '.csv'))[[paste0(ai, '_m')]][1900:2000]),
                aes(x = step_count, y = self_orienting)) +
            geom_point(alpha = 0.2) +
            xlab(paste0("Total Step Count ", ai)) +
            ylab("No. Steps Until Self Orienting") +
            #theme_bw(base_size = 28) +
            geom_point() +
            coord_fixed() +
            geom_smooth(method = 'lm', se = T, size = 1, alpha = 0.2, color = 'red') +
            theme_bw(base_size = 22) +
            theme(legend.position = "none", axis.line = element_line(color = 'black', size=0.25), panel.grid.minor = element_blank(), panel.grid.major = element_blank(), panel.border = element_blank(), axis.text.x=element_text(colour="black"), axis.text.y=element_text(colour="black"), axis.ticks.length=unit(0.013,"inch"), axis.ticks=element_line(size=unit(0.2,"inch"))) +
            scale_x_continuous(breaks = function(x) unique(floor(pretty(seq(0, (max(x) + 1) * 1.1))))) +
            stat_cor(method = "pearson", r.digits=2, p.digits = 3, size = 7, cor.coef.name = "r")

            print(plot)
        }
    }

    ############ Scatter plot ############
    self_orienting_steps <- read.csv(paste0('self_orienting_', game, '.csv'))$level_means
    self_orienting_steps <- self_orienting_steps[!is.na(self_orienting_steps)]
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
    } else {
        plot <- plot +
            stat_cor(method = "pearson", label.x = 6.5, label.y = 3.5, p.digits = 3, size = 7, cor.coef.name = "r")  +
            coord_cartesian(ylim = c(1, 3), xlim=c(5.5, 8.2))
    }

    ggsave(paste0("scatter_self_orient_", game, ".png"), plot = plot, units = "cm", width=21.8, height=12.1, dpi=1000)

}



########################################## PERTURBATIONS ##########################################

# After-Perturbation Comparisons
games <- c("change_agent_game_harder") # "contingency_game", "change_agent_game_harder"
game_datas_pert <- c()
for(game in games) {
    filename <- paste("./data_", game, "_after_perturbation.json", sep = "", collapse = NULL)
    game_data_pert <- fromJSON(file = filename)
    game_datas_pert[[game]] <- game_data_pert
}


for(game in games) {
    print(paste("*-*-*-*-", game, "-*-*-*-*"))
    print("Did humans perform similarly before vs. after perturbation: ")
    var_result <- var.test(game_datas_pert[[game]][["human"]][1:100], game_datas_pert[[game]][["human"]][101:150])
    result_ttest <- t.test(game_datas_pert[[game]][["human"]][1:100], game_datas_pert[[game]][["human"]][101:150], var.equal = var_result$p.value > 0.05)
    print(result_ttest)
    print(cohen.d(game_datas_pert[[game]][["human"]][1:100], game_datas_pert[[game]][["human"]][101:150]))

    for(agent_type in c("a2c_training", "ppo2_training", "acer_training", "dqn_training", "trpo_training", "option_critic")) {
        if(agent_type == "option_critic" & game == "change_agent_game_harder") {
            next
        }

        if(FALSE) {
            print(paste0("Comparing to just 50 levels after-perturbation: ", agent_type, " vs. human"))
            var_result <- var.test(game_datas_pert[[game]][[paste0(agent_type, "_all")]][2001:2050], game_datas_pert[[game]][["human"]][101:150])
            result_ttest <- t.test(game_datas_pert[[game]][[paste0(agent_type, "_all")]][2001:2050], game_datas_pert[[game]][["human"]][101:150], var.equal = var_result$p.value > 0.05)
            print(result_ttest)
            print(cohen.d(game_datas_pert[[game]][[paste0(agent_type, "_all")]][2001:2050], game_datas_pert[[game]][["human"]][101:150]))
        }

        if(TRUE) {
            print(paste0("Comparing to all 2000 levels averaged at every 40-level interval: ", agent_type, " vs. human"))
            var_result <- var.test(game_datas_pert[[game]][[paste0(agent_type, "_averaged")]][51:100], game_datas_pert[[game]][["human"]][101:150])
            result_ttest <- t.test(x=game_datas_pert[[game]][[paste0(agent_type, "_averaged")]][51:100], y=game_datas_pert[[game]][["human"]][101:150], var.equal = var_result$p.value > 0.05)
            print(result_ttest)
            print(cohen.d(game_datas_pert[[game]][[paste0(agent_type, "_averaged")]][51:100], game_datas_pert[[game]][["human"]][101:150]))
        }
    }

    for(agent_type in c("self_class", "human", "a2c_training", "ppo2_training", "acer_training", "dqn_training", "trpo_training", "option_critic")) {
        if(agent_type == "option_critic" & game == "change_agent_game_harder") {
            next
        }

        #### Comparing 50 levels before perturbation to all levels after perturbation (averaged)
        if(FALSE) {
            if(agent_type != "human") {
                print(paste0("Comparing 50 levels before-perturbation to after-perturbation: ", agent_type))
                var_result <- var.test(game_datas_pert[[game]][[paste0(agent_type, "_all")]][1951:2000], game_datas_pert[[game]][[paste0(agent_type, "_all")]][2001:2050])
                result_ttest <- t.test(game_datas_pert[[game]][[paste0(agent_type, "_all")]][1951:2000], game_datas_pert[[game]][[paste0(agent_type, "_all")]][2001:2050], var.equal = var_result$p.value > 0.05)
                print(result_ttest)
                print(cohen.d(game_datas_pert[[game]][[paste0(agent_type, "_all")]][1951:2000], game_datas_pert[[game]][[paste0(agent_type, "_all")]][2001:2050]))
            } else {
                print(paste0("Comparing 50 levels before-perturbation to after-perturbation: ", agent_type))
                var_result <- var.test(game_datas_pert[[game]][[agent_type]][51:100], game_datas_pert[[game]][[agent_type]][101:150])
                result_ttest <- t.test(game_datas_pert[[game]][[agent_type]][51:100], game_datas_pert[[game]][[agent_type]][101:150], var.equal = var_result$p.value > 0.05)
                print(result_ttest)
                print(cohen.d(game_datas_pert[[game]][[agent_type]][51:100], game_datas_pert[[game]][[agent_type]][101:150]))
            }
        }
        
    }
}
