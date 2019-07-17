

library(tidyverse)  
library(glmnet)
library(plotmo)


train <- read_csv("train.csv")


col_types <- sapply(train,class)

to_factor <- names(col_types[col_types == "character"])


#For character columns replace na with "NA"
#FOr numeric, repace na with 0

train_f <- train[to_factor]
train_n <- train%>%
  select(-to_factor)

train_f[is.na(train_f)] <- "Data_Missing" 
train_n[is.na(train_n)] <- 0

train_processed <- cbind(train_f, train_n)

train_processed[to_factor] <- lapply(train_processed[to_factor], as.factor)

x = model.matrix(SalePrice ~ ., train_processed) 
y = train_processed$SalePrice

print(paste("There are",dim(x)[2], "features w/ dummy variables"))

#Find best lambda and visualize through cross validation
glm_fit <- cv.glmnet(x,y, alpha = 1)

plot(glm_fit)

bestlambda = glm_fit$lambda.min

print(paste("best log lambda value:",round(bestlambda,2)))


#Create glm model with best lmabda from cross validation
glm_model <- glmnet(x,y, alpha = 1, lambda = bestlambda)

plot_glmnet(glm_model)+ abline(v = bestlambda, lwd = 2)



#get R2 of best model (is this correct!?!?)
#glm_fit$cvm is the mean squared error, var(y) is the 
#glm_fit$cvm is the 
R2 <- 1 - min(glm_fit$cvm/var(y))
CVK <- sqrt(min(glm_fit$cvm))

print(paste("R Squared value of best model",round(R2,2)))
print(paste("Cross validated MSE",round(CVK,2)))


#Predict ON TRAINING DATA
y_hat <- predict(glm_model,x, s = 100)

library(mltools)

mse(y, y_hat)

cor(y, y_hat)^2












