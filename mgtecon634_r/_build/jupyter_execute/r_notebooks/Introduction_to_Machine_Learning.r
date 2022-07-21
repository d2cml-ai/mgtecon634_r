# Deleting all current variables
rm(list=ls())

# Ensuring consistent random values in the bookdown version (this can be ignored).
set.seed(2, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")

# loading relevant packages
# if you need to install a new package, 
# use e.g., install.packages("grf")
library(grf)
library(rpart)
library(glmnet)
library(splines)
library(lmtest)
library(MASS)
library(sandwich)
library(ggplot2)
library(reshape2)

# Simulating data

# Sample size
n <- 500

# Generating covariate X ~ Unif[-4, 4]
x <- runif(n, -4, 4)

# Generate outcome
# if x < 0:
#   y = cos(2*x) + N(0, 1)
# else:
#   y = 1-sin(x) + N(0, 1)
mu <- ifelse(x < 0, cos(2*x), 1-sin(x)) 
y <- mu + 1 * rnorm(n)

# collecting observations in a data.frame object
data <- data.frame(x=x, y=y)

# outcome variable name
outcome <- "y"

# covariate names
covariates <- c("x")

plot(x, y, col="black", ylim=c(-4, 4), pch=21, bg="red", ylab = "Outcome y")
lines(x[order(x)], mu[order(x)], col="black", lwd=3, type="l")
legend("bottomright", legend=c("Ground truth E[Y|X=x]", "Data"), cex=.8, lty=c(1, NA), col="black",  pch=c(NA, 21), pt.bg=c(NA, "red"))

# Note: this code assumes that the first covariate is continuous.
# Fitting a flexible model on very little data

# selecting only a few data points
subset <- 1:30

# formula for a high-dimensional polynomial regression
# y ~ 1 + x1 + x1^2 + x1^3 + .... + x1^q
fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ", 10)"))

# linear regression using only a few observations
ols <- lm(fmla, data = data, subset=subset)

# compute a grid of x1 values we'll use for prediction
x <- data[,covariates[1]]
x.grid <- seq(min(x), max(x), length.out=1000)
new.data <- data.frame(x.grid)
colnames(new.data) <- covariates[1]

# predict
y.hat <- predict(ols, newdata = new.data)

# plotting observations (in red) and model predictions (in green)
plot(data[subset, covariates[1]], data[subset, outcome], pch=21, bg="red", xlab=covariates[1], ylab="Outcome y", main="Example of overfitting")
lines(x.grid, y.hat, col="green", lwd=2, ylim=c(-3, 3))
legend("bottomright", legend=c("Estimate", "Data"), col = c("green", "black"),pch = c(NA, 21), pt.bg = c(NA, "red"), lty = c(1, NA), lwd = c(2, NA),cex = .8)

# Note: this code assumes that the first covariate is continuous
# Fitting a very simply model on very little data

# only a few data points
subset <- 1:25

# formula for a linear regression (without taking polynomials of x1)
# y ~ 1 + x1
fmla <- formula(paste0(outcome, "~", covariates[1]))

# linear regression
ols <- lm(fmla, data[subset,])

# compute a grid of x1 values we'll use for prediction
x <- data[,covariates[1]]
x.grid <- seq(min(x), max(x), length.out=1000)
new.data <- data.frame(x.grid)
colnames(new.data) <- covariates[1]

# predict
y.hat <- predict(ols, newdata = new.data)

# plotting observations (in red) and model predictions (in green)
plot(data[subset, covariates[1]], data[subset, outcome], pch=21, bg="red", xlab=covariates[1], ylab="Outcome y", main="Example of underfitting")
lines(x.grid, y.hat, col="green", lwd=2)
legend("bottomright", legend=c("Estimate", "Data"), col = c("green", "black"),pch = c(NA, 21), pt.bg = c(NA, "red"), lty = c(1, NA), lwd = c(2, NA),cex = .8)

# polynomial degrees that we'll loop over
poly.degree <- seq(3, 20)

# training data observations: 1 to (n/2)
train <- 1:(n/2)

# looping over each polynomial degree
mse.estimates <- lapply(poly.degree, function(q) {

  # formula y ~ 1 + x1 + x1^2 + ... + x1^q
  fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ",", q,")"))

  # linear regression using the formula above
  # note we're fitting only on the training data observations
  ols <- lm(fmla, data=data[train,])

  # predicting on the training subset
  # (no need to pass a dataframe)
  y.hat.train <- predict(ols)
  y.train <- data[train, outcome]
  
  # predicting on the validation subset
  # (the minus sign in "-train" excludes observations in the training data)
  y.hat.test <- predict(ols, newdata=data[-train,])
  y.test <- data[-train, outcome]
  
  # compute the mse estimate on the validation subset and output it
  data.frame(
    mse.train=mean((y.hat.train - y.train)^2),
    mse.test=mean((y.hat.test - y.test)^2))
  })
mse.estimates <- do.call(rbind, mse.estimates)

matplot(poly.degree, mse.estimates, type="l",  main="MSE estimates (train-test split)", ylab="MSE estimate", xlab="Polynomial degree")
text(poly.degree[2], .9*max(mse.estimates), pos=4, "<-----\nHigh bias\nLow variance") 
text(max(poly.degree), .9*max(mse.estimates), pos=2, "----->\nLow bias\nHigh variance") 
legend("top", legend=c("Training", "Validation"), bty="n", lty=1:2, col=1:2, cex=.7)

# number of folds (K)
n.folds <- 5

# polynomial degrees that we'll loop over to select
poly.degree <- seq(4, 20)

# list of indices that will be left out at each step
indices <- split(seq(n), sort(seq(n) %% n.folds))

# looping over polynomial degrees (q)
mse.estimates <- sapply(poly.degree, function(q) {

    # formula y ~ 1 + x1 + x1^2 + ... + x1^q
    fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ",", q,")"))

    # loop over folds get cross-validated predictions
    y.hat <- lapply(indices, function(fold.idx) {

        # fit on K-1 folds, leaving out observations in fold.idx
        # (the minus sign in -fold.idx excludes those observations)
        ols <- lm(fmla, data=data[-fold.idx,])

        # predict on left-out kth fold
        predict(ols, newdata=data[fold.idx,])
    })
    # concatenate all the cross-validated predictions
    y.hat <- unname(unlist(y.hat))

    # cross-validated mse estimate
    mean((y.hat - data[, outcome])^2)
})

# plot
plot(poly.degree, mse.estimates, main="MSE estimates (K-fold cross-validation)", ylab="MSE estimate", xlab="Polynomial degree", type="l", lty=2, col=2)
legend("top", legend=c("Cross-validated MSE"), bty="n", lty=2, col=2, cex=.7)

# Note this code assumes that the first covariate is continuous
# Fitting a flexible model on a lot of data

# now using much more data
subset <- 1:n

# formula for high order polynomial regression
# y ~ 1 + x1 + x1^2 + ... + x1^q
fmla <- formula(paste0(outcome, "~ poly(", covariates[1], ", 15)"))

# linear regression
ols <- lm(fmla, data, subset=subset)

# compute a grid of x1 values we'll use for prediction
x <- data[,covariates[1]]
x.grid <- seq(min(x), max(x), length.out=1000)
new.data <- data.frame(x.grid)
colnames(new.data) <- covariates[1]

# predict
y.hat <- predict(ols, newdata = new.data)

# plotting observations (in red) and model predictions (in green)
plot(data[subset, covariates[1]], data[subset, outcome], pch=21, bg="red", xlab=covariates[1], ylab="Outcome")
lines(x[order(x)], mu[order(x)], lwd=2, col="black")
lines(x.grid, y.hat, col="green", lwd=2)
legend("bottomright", lwd=2, lty=c(1, 1), col=c("black", "green"), legend=c("Ground truth", "Estimate"))

# load dataset
data <- read.csv("https://docs.google.com/uc?id=1qHr-6nN7pCbU8JUtbRDtMzUKqS9ZlZcR&export=download")

# outcome variable name
outcome <- "LOGVALUE"

# covariates
true.covariates <- c('LOT','UNITSF','BUILT','BATHS','BEDRMS','DINING','METRO','CRACKS','REGION','METRO3','PHONE','KITCHEN','MOBILTYP','WINTEROVEN','WINTERKESP','WINTERELSP','WINTERWOOD','WINTERNONE','NEWC','DISH','WASH','DRY','NUNIT2','BURNER','COOK','OVEN','REFR','DENS','FAMRM','HALFB','KITCH','LIVING','OTHFN','RECRM','CLIMB','ELEV','DIRAC','PORCH','AIRSYS','WELL','WELDUS','STEAM','OARSYS')
p.true <- length(true.covariates)

# noise covariates added for didactic reasons
p.noise <- 20
noise.covariates <- paste0('noise', seq(p.noise))
covariates <- c(true.covariates, noise.covariates)
X.noise <- matrix(rnorm(n=nrow(data)*p.noise), nrow(data), p.noise)
colnames(X.noise) <- noise.covariates
data <- cbind(data, X.noise)

# sample size
n <- nrow(data)

# total number of covariates
p <- length(covariates)

round(cor(data[,covariates[1:8]]), 3)

# A formula of type "~ x1 + x2 + ..." (right-hand side only) to
# indicate how covariates should enter the model. If you'd like to add, e.g.,
# third-order polynomials in x1, you could do so here by modifying the formula
# to be something like  "~ poly(x1, 3) + x2 + ..."
fmla <- formula(paste(" ~ 0 + ", paste0(covariates, collapse=" + ")))

# Use this formula instead if you'd like to fit on piecewise polynomials
# fmla <- formula(paste(" ~ 0 + ", paste0("bs(", covariates, ", df=5)", collapse=" + ")))

# Function model.matrix selects the covariates according to the formula
# above and expands the covariates accordingly. In addition, if any column
# is a factor, then this creates dummies (one-hot encoding) as well.
XX <- model.matrix(fmla, data)
Y <- data[, outcome]

# Fit a lasso model.
# Note this automatically performs cross-validation.
lasso <- cv.glmnet(
  x=XX, y=Y,
  family="gaussian", # use 'binomial' for logistic regression
  alpha=1. # use alpha=0 for ridge, or alpha in (0, 1) for elastic net
)

plot(lasso)

# Estimated coefficients at the lambda value that minimized cross-validated MSE
coef(lasso, s = "lambda.min")[1:5,]  # showing only first coefficients
print(paste("Number of nonzero coefficients at optimal lambda:", lasso$nzero[which.min(lasso$cvm)], "out of", length(coef(lasso))))

# Retrieve predictions at best lambda regularization parameter
y.hat <- predict(lasso, newx=XX, s="lambda.min", type="response")

# Get k-fold cross validation
mse.glmnet <- lasso$cvm[lasso$lambda == lasso$lambda.min]
print(paste("glmnet MSE estimate (k-fold cross-validation):", mse.glmnet))

plot(lasso$glmnet.fit, xvar="lambda")

# Generating some data 
# y = 1 + 2*x1 + 3*x2 + noise, where corr(x1, x2) = .5
# note the sample size is very large -- this isn't solved by big data!
x <- mvrnorm(100000, mu=c(0,0), Sigma=diag(c(.5,.5)) + 1)
y <- 1 + 2*x[,1] + 3*x[,2] + rnorm(100000)
data.sim <- data.frame(x=x, y=y)

print("Correct model")
lm(y ~ x.1 + x.2, data.sim)

print("Model with omitted variable bias")
lm(y ~ x.1, data.sim)

# prepare data
fmla <- formula(paste0(outcome, "~", paste0(covariates, collapse="+")))
XX <- model.matrix(fmla, data)[,-1]  # [,-1] drops the intercept
Y <- data[,outcome]

# fit ols, lasso and ridge models
ols <- lm(fmla, data)
lasso <- cv.glmnet(x=XX, y=Y, alpha=1.)  # alpha = 1 for lasso
ridge <- cv.glmnet(x=XX, y=Y, alpha=0.)  # alpha = 0 for ridge

# retrieve ols, lasso and ridge coefficients
lambda.grid <- c(0, sort(lasso$lambda))
ols.coefs <- coef(ols)
lasso.coefs <- as.matrix(coef(lasso, s=lambda.grid))
ridge.coefs <- as.matrix(coef(ridge, s=lambda.grid))

# loop over lasso coefficients and re-fit OLS to get post-lasso coefficients
plasso.coefs <- apply(lasso.coefs, 2, function(beta) {

    # which slopes are non-zero
    non.zero <- which(beta[-1] != 0)  # [-1] excludes intercept

    # if there are any non zero coefficients, estimate OLS
    fmla <- formula(paste0(outcome, "~", paste0(c("1", covariates[non.zero]), collapse="+")))
    beta <- rep(0, ncol(XX) + 1)

    # populate post-lasso coefficients
    beta[c(1, non.zero + 1)] <- coef(lm(fmla, data))

    beta
  })

selected <- 'BATHS'
k <- which(rownames(lasso.coefs) == selected) # index of coefficient to plot
coefs <- cbind(postlasso=plasso.coefs[k,],  lasso=lasso.coefs[k,], ridge=ridge.coefs[k,], ols=ols.coefs[k])
matplot(lambda.grid, coefs, col=1:4, type="b", pch=1:4, lwd=2, main=paste("Coefficient estimate on", selected))
abline(h = 0, lty="dashed", col="gray")

legend("bottomleft",
  legend = colnames(coefs),
  bty="n", col=1:4,  pch=1:4, inset=c(.05, .05), lwd=2)

covs <- which(covariates %in% c('UNITSF', 'BEDRMS',  'DINING'))
matplot(lambda.grid, t(lasso.coefs[covs+1,]), type="l", lwd=2)
legend("topright", legend = covariates[covs], bty="n", col=1:p,  lty=1:p, inset=c(.05, .05), lwd=2, cex=.6)

# Fixing lambda. This choice is not very important; the same occurs any intermediate lambda value.
selected.lambda <- lasso$lambda.min
n.folds <- 10
foldid <- (seq(n) %% n.folds) + 1
coefs <- sapply(seq(n.folds), function(k) {
  lasso.fold <- glmnet(XX[foldid == k,], Y[foldid == k])
  as.matrix(coef(lasso.fold, s=selected.lambda))
})
heatmap(1*(coefs != 0), Rowv = NA, Colv = NA, cexCol = 1, scale="none", col=gray(c(1,0)), margins = c(3, 1), xlab="Fold", labRow=c("Intercept", covariates), main="Non-zero coefficient estimates")

# Number of data-driven subgroups.
num.groups <- 4

# Fold indices
n.folds <- 5
foldid <- (seq(n) %% n.folds) + 1

fmla <- formula(paste(" ~ 0 + ", paste0("bs(", covariates, ", df=3)", collapse=" + ")))

# Function model.matrix selects the covariates according to the formula
# above and expands the covariates accordingly. In addition, if any column
# is a factor, then this creates dummies (one-hot encoding) as well.
XX <- model.matrix(fmla, data)
Y <- data[, outcome]

# Fit a lasso model.
# Passing foldid argument so we know which observations are in each fold.
lasso <- cv.glmnet(x=XX, y=Y, foldid = foldid, keep=TRUE)

y.hat <- predict(lasso, newx = XX, s = "lambda.min")

# Ranking observations.
ranking <- lapply(seq(n.folds), function(i) {

    # Extract cross-validated predictions for remaining fold.
    y.hat.cross.val <- y.hat[foldid == i]

    # Find the relevant subgroup break points
    qs <- quantile(y.hat.cross.val, probs = seq(0, 1, length.out=num.groups + 1))

    # Rank observations into subgroups depending on their predictions
    cut(y.hat.cross.val, breaks = qs, labels = seq(num.groups))
  })
ranking <- factor(do.call(c, ranking))

# Estimate expected covariate per subgroup
avg.covariate.per.ranking <- mapply(function(x.col) {
  fmla <- formula(paste0(x.col, "~ 0 + ranking"))
  ols <- lm(fmla, data=transform(data, ranking=ranking))
  t(lmtest::coeftest(ols, vcov=vcovHC(ols, "HC2"))[, 1:2])
}, covariates, SIMPLIFY = FALSE)

avg.covariate.per.ranking[1:2]

df <- mapply(function(covariate) {
      # Looping over covariate names
      # Compute average covariate value per ranking (with correct standard errors)
      fmla <- formula(paste0(covariate, "~ 0 + ranking"))
      ols <- lm(fmla, data=transform(data, ranking=ranking))
      ols.res <- coeftest(ols, vcov=vcovHC(ols, "HC2"))
    
      # Retrieve results
      avg <- ols.res[,1]
      stderr <- ols.res[,2]
      
      # Tally up results
      data.frame(covariate, avg, stderr, ranking=paste0("G", seq(num.groups)), 
                 # Used for coloring
                 scaling=pnorm((avg - mean(avg))/sd(avg)), 
                 # We will order based on how much variation is 'explain' by the averages
                 # relative to the total variation of the covariate in the data
                 variation=sd(avg) / sd(data[,covariate]),
                 # String to print in each cell in heatmap below
                 # Note: depending on the scaling of your covariates, 
                 # you may have to tweak these formatting parameters a little.
                 labels=paste0(formatC(avg),  " (", formatC(stderr, digits = 2, width = 2), ")"))
}, covariates, SIMPLIFY = FALSE)
df <- do.call(rbind, df)

# a small optional trick to ensure heatmap will be in decreasing order of 'variation'
df$covariate <- reorder(df$covariate, order(df$variation))
df <- df[order(df$variation, decreasing=TRUE),]

# plot heatmap
ggplot(df[1:(9*num.groups),]) +  # showing on the first few results (ordered by 'variation')
    aes(ranking, covariate) +
    geom_tile(aes(fill = scaling)) + 
    geom_text(aes(label = labels), size=3) +  # 'size' controls the fontsize inside cell
    scale_fill_gradient(low = "#E1BE6A", high = "#40B0A6") +
    ggtitle(paste0("Average covariate values within group (based on prediction ranking)")) +
    theme_minimal() + 
    ylab("") + xlab("") +
    theme(plot.title = element_text(size = 10, face = "bold"),
          legend.position="bottom")

# Fit tree without pruning first
fmla <- formula(paste(outcome, "~", paste(covariates, collapse=" + ")))
tree <- rpart(fmla, data=data, cp=0, method="anova")  # use method="class" for classification

plot(tree, uniform=TRUE)

plotcp(tree)

# Retrieves the optimal parameter
cp.min <- which.min(tree$cptable[,"xerror"]) # minimum error
cp.idx <- which(tree$cptable[,"xerror"] - tree$cptable[cp.min,"xerror"] < tree$cptable[,"xstd"])[1]  # at most one std. error from minimum error
cp.best <- tree$cptable[cp.idx,"CP"]

# Prune the tree
pruned.tree <- prune(tree, cp=cp.best)

plot(pruned.tree, uniform=TRUE, margin = .05)
text(pruned.tree, cex=.7)

# Retrieve predictions from pruned tree
y.hat <- predict(pruned.tree)

# Compute mse for pruned tree (using cross-validated predictions)
mse.tree <- mean((xpred.rpart(tree)[,cp.idx] - data[,outcome])^2, na.rm=TRUE)
print(paste("Tree MSE estimate (cross-validated):", mse.tree))

y.hat <- predict(pruned.tree)

# Number of leaves should equal the number of distinct prediction values.
# This should be okay for most applications, but if an exact answer is needed use
# predict.rpart.leaves from package treeCluster
num.leaves <- length(unique(y.hat))

# Leaf membership, ordered by increasing prediction value
leaf <- factor(y.hat, ordered = TRUE, labels = seq(num.leaves))

# Looping over covariates
avg.covariate.per.leaf <- mapply(function(covariate) {
  
  # Coefficients on linear regression of covariate on leaf 
  #  are the average covariate value in each leaf.
  # covariate ~ leaf.1 + ... + leaf.L 
  fmla <- formula(paste0(covariate, "~ 0 + leaf"))
  ols <- lm(fmla, data=transform(data, leaf=leaf))
  
  # Heteroskedasticity-robust standard errors
  t(coeftest(ols, vcov=vcovHC(ols, "HC2"))[,1:2])
}, covariates, SIMPLIFY = FALSE)

print(avg.covariate.per.leaf[1:2])  # Showing only first few

df <- mapply(function(covariate) {
      # Looping over covariate names
      # Compute average covariate value per ranking (with correct standard errors)
      fmla <- formula(paste0(covariate, "~ 0 + leaf"))
      ols <- lm(fmla, data=transform(data, leaf=leaf))
      ols.res <- coeftest(ols, vcov=vcovHC(ols, "HC2"))
    
      # Retrieve results
      avg <- ols.res[,1]
      stderr <- ols.res[,2]
      
      # Tally up results
      data.frame(covariate, avg, stderr, 
                 ranking=factor(seq(num.leaves)), 
                 # Used for coloring
                 scaling=pnorm((avg - mean(avg))/sd(avg)), 
                 # We will order based on how much variation is 'explain' by the averages
                 # relative to the total variation of the covariate in the data
                 variation=sd(avg) / sd(data[,covariate]),
                 # String to print in each cell in heatmap below
                 # Note: depending on the scaling of your covariates, 
                 # you may have to tweak these  formatting parameters a little.
                 labels=paste0(formatC(avg),"\n(", formatC(stderr, digits = 2, width = 2), ")"))
}, covariates, SIMPLIFY = FALSE)
df <- do.call(rbind, df)

# a small optional trick to ensure heatmap will be in decreasing order of 'variation'
df$covariate <- reorder(df$covariate, order(df$variation))
df <- df[order(df$variation, decreasing=TRUE),]

# plot heatmap
ggplot(df[1:(8*num.leaves),]) +  # showing on the first few results (ordered by 'variation')
    aes(ranking, covariate) +
    geom_tile(aes(fill = scaling)) + 
    geom_text(aes(label = labels), size=2.5) +  # 'size' controls the fontsize inside cell
    scale_fill_gradient(low = "#E1BE6A", high = "#40B0A6") +
    ggtitle(paste0("Average covariate values within leaf")) +
    theme_minimal() + 
    ylab("") + xlab("Leaf (ordered by prediction, low to high)") +
    labs(fill="Normalized\nvariation") +
    theme(plot.title = element_text(size = 12, face = "bold", hjust = .5),
          axis.title.x = element_text(size=9),
          legend.title = element_text(hjust = .5, size=9))

X <- data[,covariates]
Y <- data[,outcome]

# Fitting the forest
# We'll use few trees for speed here. 
# In a practical application please use a higher number of trees.
forest <- regression_forest(X=X, Y=Y, num.trees=200)  

# There usually isn't a lot of benefit in tuning forest parameters, but the next code does so automatically (expect longer training times)
# forest <- regression_forest(X=X, Y=Y, tune.parameters="all")

# Retrieving forest predictions
y.hat <- predict(forest)$predictions

# Evaluation (out-of-bag mse)
mse.oob <- mean(predict(forest)$debiased.error)
print(paste("Forest MSE (out-of-bag):", mse.oob))

var.imp <- variable_importance(forest)
names(var.imp) <- covariates
sort(var.imp, decreasing = TRUE)[1:10] # showing only first few






