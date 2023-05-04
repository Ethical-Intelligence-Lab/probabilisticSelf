rm(list = ls())
if (!require(pacman)) { install.packages(pacman) }

pacman::p_load('effsize')
pacman::p_load('rjson')
pacman::p_load('BayesFactor')
pacman::p_load('Dict')
pacman::p_load('ggplot2')
pacman::p_load('lsr')

#####################################################################################################
##### Most of the data files that we use here are generated in the "plotter_curves.ipynb" file. #####
#####################################################################################################

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

# ********** t-tests For Comparing the Last Level of the Self Class and Human vs. the RL algorithms **********
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
# Create plotter function
corr_violin_plotter <- function(df_plot, game, plot_type = "regular") {
    p <- ggplot(df_plot, aes(x=index, y=corr)) + 
    geom_violin(size=1.25, adjust = .5) + geom_jitter(shape=16, position=position_jitter(0.2), size=2) +
    #geom_point(alpha = 0.2) +
    xlab("") +
    ylab("Correlation") +
    #theme_bw(base_size = 28) +
    coord_fixed() +
    theme_bw(base_size = 22) +
    stat_summary(fun.data = "mean_se", color = "black",
            size = 0.4, fun.args = list(mult = 1),
            position = position_dodge(width = 0.9)) +
    stat_summary(fun.data = mean_cl_boot, color = "black",
            geom = "errorbar", position=position_dodge(0.9), width = 0.2) +
            geom_hline(yintercept=0, size=0.25) 
            
    if(game == 'change_agent_game') {
        p <- p + ylim(-0.06, 1)
    } else {
        p <- p + ylim(0, 1)
    }

    
    p <- p + theme(legend.position = "none", axis.line = element_line(color = 'black', size=0.25), panel.grid.minor = element_blank(), panel.grid.major = element_blank(), panel.border = element_blank(), axis.text.y=element_text(colour="black"), axis.ticks.length=unit(0.013,"inch"), axis.ticks.y=element_line(size=unit(0.2,"inch")), axis.line.x.bottom=element_line(size=0), axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())


    if(FALSE) {
        if(game != "change_agent_game") {
            p <- p + theme(legend.position = "none", axis.line = element_line(color = 'black', size=0.25), panel.grid.minor = element_blank(), panel.grid.major = element_blank(), panel.border = element_blank(), axis.text.y=element_text(colour="black"), axis.ticks.length=unit(0.013,"inch"), axis.ticks.y=element_line(size=unit(0.2,"inch")), axis.line.x.bottom=element_line(size=0), axis.title.x=element_blank(),
                            axis.text.x=element_blank(),
                            axis.ticks.x=element_blank())
        } else {
                p <- p + theme_bw(base_size = 30) + theme(legend.position = "none", axis.line = element_line(color = 'black', size=0.25), panel.grid.minor = element_blank(), panel.grid.major = element_blank(), panel.border = element_blank(), axis.text.y=element_text(colour="black"), axis.ticks.length=unit(0.013,"inch"), axis.ticks.y=element_line(size=unit(0.2,"inch")), axis.line.x.bottom=element_line(size=0), axis.title.x=element_blank(),
                axis.ticks.x=element_blank())
        }
    }
    
    return(p)
}

### Calculate correlation for all participants
# Create function
get_participant_correlation <- function(self_orient_data, participant_id, agent_type = "") {
    c_participant <- self_orient_data[self_orient_data$participant == participant_id,]

    if(game != "logic_game") {
        c_participant <- c_participant[!is.na(c_participant$self_finding_steps),]
    }

    if(game == "change_agent_game") {
        return( list(cor( na.omit(c_participant[c_participant$level, 'self_finding_steps']), na.omit(c_participant[c_participant$level, 'steps']) ), 
                        cor( na.omit(c_participant[c_participant$level, 'prop_selected_correctly']), na.omit(c_participant[c_participant$level, 'steps']) )) )
    } else if(game == "logic_game") {
        if(agent_type == "human") {
            return(list(cor( na.omit(c_participant[c_participant$level < 100, paste0(agent_type, '_self_finding_steps')]), na.omit(c_participant[c_participant$level < 100, paste0(agent_type, '_total_steps')] ))))
        } else if(agent_type %in% c("random", "self_class")) {
            return(list(cor( na.omit(c_participant[, paste0(agent_type, '_self_finding_steps')]),
                                na.omit(c_participant[, paste0(agent_type, '_total_steps')] ))))
        } else {
            return(list(cor( na.omit(c_participant[c_participant$level > 1900, paste0(agent_type, '_self_finding_steps')]),
                                na.omit(c_participant[c_participant$level > 1900, paste0(agent_type, '_total_steps')] ))))
        }
    } else {
        return(list(cor( na.omit(c_participant$self_finding_steps), na.omit(c_participant$steps) )))
    }
}



game_datas <- c()
for(game in c('change_agent_game')) { #, 'logic_game', 'contingency_game', 'contingency_game_shuffled_1', 'change_agent_game'
    filename <- paste("./data_", game, ".json", sep = "", collapse = NULL)
    game_datas[[game]] <- fromJSON(file = filename)

    print(paste0("*-*-*-*-*-*-*-*", game, "*-*-*-*-*-*-*-*"))
    ## Check if participants performed differently in the self-finding game compared to the original run
    if(game != 'logic_game') {
        print("Checking if participants performed differently in the self-finding game compared to the original run")

        if(game == 'change_agent_game') {
            # Before perturbation
            human_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$human))[1:34]
            human_sf_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$data_sf))[1:34]

            vr <- var.test(human_lvl_means_cont, human_sf_lvl_means_cont)
            print(t.test(human_lvl_means_cont, human_sf_lvl_means_cont, var.equal = vr$p.value > 0.05))
            print(cohen.d(human_lvl_means_cont, human_sf_lvl_means_cont))
        } else {
            human_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$human))
            human_sf_lvl_means_cont <- colMeans(do.call(cbind, game_datas[game][[1]]$data_sf))

            vr <- var.test(human_lvl_means_cont, human_sf_lvl_means_cont)
            print(t.test(human_lvl_means_cont, human_sf_lvl_means_cont, var.equal = vr$p.value > 0.05))
            print(cohen.d(human_lvl_means_cont, human_sf_lvl_means_cont))
        }
        
    }
    
    ###### SCATTER PLOTS ######
    self_orient_data <- read.csv(paste0('self_orienting_', game, '.csv'))

    # Get correlation values for each participant
    if(game == "change_agent_game") { self_orient_data['cor_prop_selected'] <- NA } # Only in last game

    if(game != "logic_game") {
        self_orient_data['cor_sf_steps'] <- NA
    }

    n_participant <- 20
    if(game == "change_agent_game") {
        n_participant <- 19
    }

    if(game == "logic_game") {
        for(agent in c('random', 'human', 'dqn_training', 'a2c_training', 'trpo_training', 'acer_training', 'ppo2_training', 'option_critic', 'self_class')) {
            self_orient_data[paste0('cor_sf_steps_', agent)] <- NA
            print(paste("*-*-*-*-*-* ", agent, " *-*-*-*-*-*"))

            for(participant in 0: n_participant) {
                corrs <- get_participant_correlation(self_orient_data, participant, agent)
                self_orient_data[self_orient_data$participant == participant, paste0('cor_sf_steps_', agent)] <- corrs[[1]]
            }

            # Chance level (assuming zero for no relationship)
            chance_level <- mean(na.omit(self_orient_data[self_orient_data$level == 0, paste0('cor_sf_steps_random')]))

            print("Comparing self finding step correlations against chance: ")
            if(agent != "self_class") { # t-test gives error on self class since all is same, so skip that
                print(t.test(na.omit(self_orient_data[self_orient_data$level == 0, paste0('cor_sf_steps_', agent)]), mu=chance_level))
                print(cohensD(na.omit(self_orient_data[self_orient_data$level == 0, paste0('cor_sf_steps_', agent)]), mu=chance_level))
                print(wilcox.test(na.omit(self_orient_data[self_orient_data$level == 0, paste0('cor_sf_steps_', agent)]), mu = chance_level))
            }
        }
    } else {
        for(participant in 0: n_participant) {
            corrs <- get_participant_correlation(self_orient_data, participant)

            self_orient_data[self_orient_data$participant == participant, 'cor_sf_steps'] <- corrs[[1]]
            if(game == "change_agent_game") { self_orient_data[self_orient_data$participant == participant, 'cor_prop_selected'] <- corrs[[2]] }
        }
        
        # Chance level (assuming zero for no relationship)
        chance_level <- 0

        print("Comparing self finding step correlations against chance: ")
        print(t.test(na.omit(self_orient_data[self_orient_data$level == 0, 'cor_sf_steps']), mu = chance_level))
        print(cohensD(na.omit(self_orient_data[self_orient_data$level == 0, 'cor_sf_steps']), mu = chance_level))

        # Perform the one-sample t-test
        if(game == "change_agent_game") {
            print("Comparing select proportion correlations against chance: ")
            print(t.test(na.omit(self_orient_data[self_orient_data$level == 0, 'cor_prop_selected']), mu = chance_level))
            print(cohensD(na.omit(self_orient_data[self_orient_data$level == 0, 'cor_prop_selected']), mu = chance_level))
        }
    }

    #################### Violin Plot ####################
    # Violin plot of the correlations
    if(game == "change_agent_game") {
        #d <- na.omit(self_orient_data[self_orient_data$level == 0, 'cor_prop_selected'])
        #d2 <- na.omit(self_orient_data[self_orient_data$level == 0, 'cor_sf_steps'])    
        #df_plot <- data.frame("corr" = c(d, d2), "index" = factor(c(rep("Self Orienting\nAccuracy", length(d)), rep("Self Orienting\nSteps", length(d2)))))

        d <- na.omit(self_orient_data[self_orient_data$level == 0, 'cor_sf_steps'])
        df_plot <- data.frame("corr" = d, "index" = rep(0, length(d)))
    } else if(game == "logic_game") {
        d <- na.omit(self_orient_data[self_orient_data$level == 0, 'cor_sf_steps_human'])
        df_plot <- data.frame("corr" = d, "index" = rep(0, length(d)))
    } else {
        d <- na.omit(self_orient_data[self_orient_data$level == 0, 'cor_sf_steps'])
        df_plot <- data.frame("corr" = d, "index" = rep(0, length(d)))
    }

    p <- corr_violin_plotter(df_plot, game)
    ggsave(paste0("scatter_self_orient_", game, ".png"), plot = p, units = "cm", width=13.5, height=11.1, dpi=1000)

    # Taking mean of each participant across all levels
    average_levels <- aggregate(self_orient_data, list(self_orient_data$level), mean, na.rm=TRUE)
    
    if(game == 'change_agent_game') {
        # Compare the propotion of correct self finding to that of the proximity algorithm
        selected_correct_proximity <- read.csv('keep_close_control_prop.csv')$keep_close_control_prop

        # Comparing before perturbation and after perturbation to the proximity algorithm
        # Before perturbation
        print("Human vs. keep close algorithm before perturbation")
        vr <- var.test(average_levels$prop_selected_correctly[1:34], selected_correct_proximity[1:34])
        print(t.test(average_levels$prop_selected_correctly[1:34], selected_correct_proximity[1:34], var.equal = vr$p.value > 0.05))
        print(cohen.d(average_levels$prop_selected_correctly[1:34], selected_correct_proximity[1:34]))

        # After perturbation
        print("Human vs. keep close algorithm after perturbation")
        vr <- var.test(average_levels$prop_selected_correctly[35:53], selected_correct_proximity[35:53])
        print(t.test(average_levels$prop_selected_correctly[35:53], selected_correct_proximity[35:53], var.equal = vr$p.value > 0.05))
        print(cohen.d(average_levels$prop_selected_correctly[35:53], selected_correct_proximity[35:53]))

        # Humans before vs. after perturbation
        print("Humans before vs. after perturbation for accuracy on finding real self")
        vr <- var.test(average_levels$prop_selected_correctly[1:34], average_levels$prop_selected_correctly[35:53])
        print(t.test(average_levels$prop_selected_correctly[1:34], average_levels$prop_selected_correctly[35:53], var.equal = vr$p.value > 0.05))
        print(cohen.d(average_levels$prop_selected_correctly[1:34], average_levels$prop_selected_correctly[35:53]))
    }
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
