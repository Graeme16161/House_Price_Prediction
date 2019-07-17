

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


glm_fit <- cv.glmnet(x,y, alpha = 1)

plot(glm_fit)

bestlambda = log(glm_fit$lambda.min)

print(paste("best log lambda value:",round(bestlambda,2)))


glm_model <- glmnet(x,y, alpha = 1, nlambda = 100)

plot_glmnet(glm_model)+ abline(v = bestlambda, lwd = 2)



#get R2
R2 <- 1 - min(glm_fit$cvm/var(y))

print(paste("R Squared value of best model",round(R2,2)))
















