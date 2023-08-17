library(dplyr)
#includes all runs made, so also those on an older hardware configuration
df_all = data.frame(model=factor(c("original","original","original", "growingConst", "growingConst", "growingVar", "growingVar", "growingVar", "growingConst", "growingConst", "growingConst", "growingConst", "original")), 
                rounds=factor(c("10", "10", "10", 
                         "5,5,5,5", "5,5,5,5", "10,4,4,4", "10,4,4,4", "10,4,4,4", "5,5,5,5", "10,10,10,10", "10,10,10,10", "20,20,20,20", "10")), 
                date_of_execution = c("2023_08_10 23:08:34","2023_08_03 14:08:48", "2023_08_03 10:57:02",
                                      "2023_08_11 11:09:44", "2023_08_11 08:31:07", "2023_08_10 20:30:28", "2023_08_10 18:02:15","2023_08_10 15:38:58", "2023_08_07 10:37:39", "2022_12_30 09:00:00", "2023_08_08 08:58:56", "2022_12_30 09:00:00", "2022_12_30 09:00:00"),
                auc_pr=c(0.8892, 0.8935, 0.8937, 0.8774, 0.8276, 0.8609, 0.8455, 0.8748, 0.8510, 0.9012, 0.8754, 0.8750, 0.8321), 
                auc_roc=c(0.9710, 0.9730, 0.9731, 0.9670, 0.9450, 0.9591, 0.9539, 0.9600, 0.9540, 0.9762, 0.9664, 0.9712, 0.9557),
                dice_coeff=c(0.8032, 0.8059, 0.8075, 0.7897, 0.7404, 0.7760, 0.7651, 0.7912, 0.7684, 0.8150, 0.7604, 0.7852, 0.7384),
                acc=c(0.9510, 0.9493, 0.9503, 0.9492, 0.9409, 0.9460, 0.9401, 0.9502, 0.9452, 0.9524, 0.9471, 0.9473, 0.9261),
                sensitivity=c(0.7855, 0.8263, 0.8178, 0.7487, 0.6612, 0.7343, 0.7659, 0.7414, 0.7136, 0.8238, 0.6584, 0.7560, 0.8191),
                specificity=c(0.9751, 0.9672, 0.9697, 0.9784, 0.9817, 0.9769, 0.9655, 0.9806, 0.9790, 0.9711, 0.9893, 0.9753, 0.9417))
df_all$date_of_execution = as.POSIXct(df_all$date_of_execution, format="%Y_%m_%d %H:%M:%S")
new_runs_date = as.POSIXct("2023_07_10 15:38:58", format="%Y_%m_%d %H:%M:%S")
df = df_all%>%filter(date_of_execution >= new_runs_date)
df$rounds=factor(df$rounds)

write.csv(df_all, "runs_all.csv", row.names = FALSE)
write.csv(df, "runs_new.csv", row.names = FALSE)
par(cex=1.5)
boxplot(df$acc~df$model, ylab="Accuracy", xlab="Training procedure")
boxplot(df$acc~df$rounds, ylab="Accuracy", xlab="Number of rounds")
df%>% group_by(rounds)%>%filter(acc == max(acc, na.rm=TRUE))

