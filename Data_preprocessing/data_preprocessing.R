dataset = read.csv('Data.csv')

dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN= function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN= function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)
dataset$Country = factor(dataset$Country, c('France', 'Spain', 'Germany'), c(1,2,3))
dataset$Purchased = factor(dataset$Purchased, c('No', 'Yes'), c(0,1))

#split training testing set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)

training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

View(training_set)

#feature scaling
# attention factor in R is not numeric
training_set[, 2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3])








