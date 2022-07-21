# Deleting all current variables
rm(list=ls())

# Ensuring consistent random values in the bookdown version (this can be ignored).
set.seed(1, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")

# Note: bookdown seems to require explicitly calling these packages.
library(reshape2)
library(DiagrammeR)

# use e.g., install.packages("grf") to install any of the following packages.
library(grf)
library(glmnet)
library(splines)
library(policytree)
library(ggplot2)
library(lmtest)
library(sandwich)

# A randomized setting.
n <- 1000
p <- 4
X <- matrix(runif(n*p), n, p)
e <- .5   # fixed, known probability
W <- rbinom(n, prob = e, size = 1)
Y <- .5*(X[,1] - .5) + (X[,2] - .5)*W + .1 * rnorm(n) 
data <- data.frame(x=X, y=Y, w=W)

outcome <- "y"
covariates <- paste0("x.", seq(p))
treatment <- "w"

# Preparing to run a regression with splines (\\piecewise polynomials).
# Note that if we have a lot of data we should increase the argument `df` below.
# The optimal value of `df` can be found by cross-validation
# i.e., check if the value of the policy, estimated below, increases or decreases as `df` varies. 
fmla.xw <- formula(paste0("~", paste0("bs(", covariates, ", df=5) *", treatment, collapse="+")))
XW <- model.matrix(fmla.xw, data)
Y <- data[,outcome]

# Data-splitting
# Define training and evaluation sets
train <- 1:(n/2)
test <- (n/2 + 1):n

# Fitting the outcome model on the *training* data
model.m <- cv.glmnet(XW[train,], Y[train])  

# Predict outcome E[Y|X,W=w] for w in {0, 1} on the *test* data
data.0 <- data[test,] 
data.0[,treatment] <- 0
XW0 <- model.matrix(fmla.xw, data.0)
mu.hat.0 <- predict(model.m, XW0, s="lambda.min")

data.1 <- data[test,]  
data.1[,treatment] <- 1
XW1 <- model.matrix(fmla.xw, data.1)
mu.hat.1 <- predict(model.m, XW1, s="lambda.min")

# Computing the CATE estimate tau.hat
tau.hat <- mu.hat.1 - mu.hat.0

# Assignment if tau.hat is positive (or replace by non-zero cost if applicable)
pi.hat <- as.numeric(tau.hat > 0)

# Estimate assignment probs e(x). 
# (This will be useful for evaluation via AIPW scores a little later)

# Uncomment/comment the next lines as appropriate
# In randomized settings assignment probabilities are fixed and known.
e.hat <- rep(0.5, length(test))  
# In observational setttings the assignment probability is ty\\pically unknown.
# fmla.x <- formula(paste0("~", paste0("bs(", covariates, ", df=3, degree=3)", collapse="+")))
# XX <- model.matrix(fmla.x, data)
# model.e <- cv.glmnet(XX[train,], W[train], family="binomial")
# e.hat <- predict(model.e, XX[test,], s="lambda.min", type="response")


# Only valid in randomized settings.
A <- pi.hat == 1
Y <- data[test, outcome]
W <- data[test, treatment]
value.estimate <- mean(Y[A & (W==1)]) * mean(A) + mean(Y[!A & (W==0)]) * mean(!A)
value.stderr <- sqrt(var(Y[A & (W==1)]) / sum(A & (W==1)) * mean(A)^2 + var(Y[!A & (W==0)]) / sum(!A & W==0) * mean(!A)^2)
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))


# Valid in randomized settings and observational settings with unconfoundedness and overlap.
Y <- data[test, outcome]
W <- data[test, treatment]
gamma.hat.1 <- mu.hat.1 + W / e.hat * (Y - mu.hat.1)
gamma.hat.0 <- mu.hat.0 + (1 - W) / (1 - e.hat) * (Y - mu.hat.0)
gamma.hat.pi <- pi.hat * gamma.hat.1 + (1 - pi.hat) * gamma.hat.0

value.estimate <- mean(gamma.hat.pi)
value.stderr <- sd(gamma.hat.pi) / sqrt(length(gamma.hat.pi))
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))


# Using the entire data
X <- data[,covariates]
Y <- data[,outcome]
W <- data[,treatment]

# Comment / uncomment as approriate
# Randomized setting with known assignment probability (here, 0.5)
forest <- causal_forest(X, Y, W, W.hat=.5)
# Observational setting with unconfoundedness and overlap.
# forest <- causal_forest(X, Y, W)

# Get "out-of-bag" predictions
tau.hat.oob <- predict(forest)$predictions
pi.hat <- as.numeric(tau.hat.oob > 0)

# Only valid in randomized settings.

# We can use the entire data because predictions are out-of-bag
A <- pi.hat == 1
value.estimate <- mean(Y[A & (W==1)]) * mean(A) + mean(Y[!A & (W==0)]) * mean(!A)
value.stderr <- sqrt(var(Y[A & (W==1)]) / sum(A & (W==1)) * mean(A)^2 + var(Y[!A & (W==0)]) / sum(!A & W==0) * mean(!A)^2)
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Valid in randomized settings and observational settings with unconfoundedness and overlap.
tau.hat <- predict(forest)$predictions

# Retrieve relevant quantities.
e.hat <- forest$W.hat # P[W=1|X]
mu.hat.1 <- forest$Y.hat + (1 - e.hat) * tau.hat  # E[Y|X,W=1] = E[Y|X] + (1 - e(X)) * tau(X)
mu.hat.0 <- forest$Y.hat - e.hat * tau.hat        # E[Y|X,W=0] = E[Y|X] - e(X) * tau(X)

# Compute AIPW scores.
gamma.hat.1 <- mu.hat.1 + W / e.hat * (Y - mu.hat.1)
gamma.hat.0 <- mu.hat.0 + (1-W) / (1-e.hat) * (Y - mu.hat.0)
gamma.hat.pi <- pi.hat * gamma.hat.1 + (1 - pi.hat) * gamma.hat.0

# Value estimates.
value.estimate <- mean(gamma.hat.pi)
value.stderr <- sd(gamma.hat.pi) / sqrt(length(gamma.hat.pi))
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Randomized setting: pass the known treatment assignment as an argument.
forest <- causal_forest(X, Y, W, W.hat=.5)
# Observational settting with unconfoundedness+overlap: let the assignment probabilities be estimated.
# forest <- causal_forest(X, Y, W)

# from policytree package
gamma.matrix <- double_robust_scores(forest)  

# Note: the function double_robust_scores is equivalent to the following:
# tau.hat <- predict(forest)$predictions
# mu.hat.1 <- forest$Y.hat + (1 - forest$W.hat) * tau.hat
# mu.hat.0 <- forest$Y.hat - forest$W.hat * tau.hat
# gamma.hat.1 <- mu.hat.1 + W/forest$W.hat * (Y - mu.hat.1)
# gamma.hat.0 <- mu.hat.0 + (1-W)/(1-forest$W.hat) * (Y - mu.hat.0)
# gamma.matrix <- cbind(gamma.hat.0, gamma.hat.1)

# Divide data into train and test sets
train <- 1:(n/2)
test <- (n/2 + 1):n

# Train on a portion of the data
# The argument `depth` controls the depth of the tree.
# Depth k means that we can partition the data into up to 2^(k+1) regions.  
policy <- policy_tree(X[train,], gamma.matrix[train,], depth=2)

# Predict on remaining portion
# Note policytree recodes the treatments to 1,2
# We substract one to get back to our usual encoding 0,1.
pi.hat <- predict(policy, X[test,]) - 1

print(policy)

plot(policy, leaf.labels = c("control", "treatment"))

# only valid for randomized setting!
A <- pi.hat == 1 
Y <- data[test, outcome]
W <- data[test, treatment]
value.estimate <- mean(Y[A & (W==1)]) * mean(A) + mean(Y[!A & (W==0)]) * mean(!A)
value.stderr <- sqrt(var(Y[A & (W==1)]) / sum(A & (W==1)) * mean(A)^2 + var(Y[!A & (W==0)]) / sum(!A & W==0) * mean(!A)^2)
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Valid in randomized settings and observational settings with unconfoundedness and overlap.
gamma.hat.pi <- pi.hat * gamma.matrix[test,2] + (1 - pi.hat)  * gamma.matrix[test,1]
value.estimate <- mean(gamma.hat.pi)
value.stderr <- sd(gamma.hat.pi) / sqrt(length(gamma.hat.pi))
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))


# Read in data
data <- read.csv("https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download")
n <- nrow(data)

## NOTE: invert treatment and control, compared to the ATE and HTE chapters.
data$w <- 1 - data$w

# Treatment is the wording of the question:
# 'does the the gov't spend too much on 'assistance to the poor' (control: 0)
# 'does the the gov't spend too much on "welfare"?' (treatment: 1)
treatment <- "w"

# Outcome: 1 for 'yes', 0 for 'no'
outcome <- "y"

# Additional covariates
covariates <- c("age", "polviews", "income", "educ", "marital", "sex")

# Prepare data
X <- data[,covariates]
Y <- data[,outcome]
W <- data[,treatment]
cost <- .3

Sys.sleep(10)
# Fit a policy tree on forest-based AIPW scores
forest <- causal_forest(X, Y, W)


gamma.matrix <- double_robust_scores(forest)
gamma.matrix[,2] <- gamma.matrix[,2] - cost  # Subtracting cost of treatment

# Divide data into train and evaluation sets
train <- 1:(.8*n)
test <- (.8*n):n

# Fit policy on training subset
policy <- policy_tree(X[train,], gamma.matrix[train,], depth = 2, min.node.size=1)

# Predicting treatment on test set
pi.hat <- predict(policy, X[test,]) - 1

# Predicting leaves (useful later)
leaf <- predict(policy, X[test,], type = "node.id")
num.leaves <- length(unique(leaf))


print(policy)

plot(policy, leaf.labels = c("control", "treatment"))

A <- pi.hat == 1
Y.test <- data[test, outcome]
W.test <- data[test, treatment]

# Only valid for randomized setting.
# Note the -cost here!
value.avg.estimate <- (mean(Y.test[A & (W.test==1)]) - cost) * mean(A) + mean(Y.test[!A & (W.test==0)]) * mean(!A)
value.avg.stderr <- sqrt(var(Y.test[A & (W.test==1)]) / sum(A & (W.test==1)) * mean(A)^2 + var(Y.test[!A & (W.test==0)]) / sum(!A & W.test==0) * mean(!A)^2)
print(paste("Estimate [sample avg]:", value.avg.estimate, "(", value.avg.stderr, ")"))

# Valid in both randomized and obs setting with unconf + overlap.
gamma.hat.1 <- gamma.matrix[test,2]
gamma.hat.0 <- gamma.matrix[test,1]
gamma.hat.pi <- pi.hat * gamma.hat.1 + (1 - pi.hat)  * gamma.hat.0
value.aipw.estimate <- mean(gamma.hat.pi)
value.aipw.stderr <- sd(gamma.hat.pi) / sqrt(length(gamma.hat.pi))
print(paste("Estimate [AIPW]:", value.aipw.estimate, "(", value.aipw.stderr, ")"))

# Only valid for randomized setting.
diff.estimate <- (mean(Y.test[A & (W.test==1)]) - cost - mean(Y.test[A & (W.test==0)])) * mean(A)
diff.stderr <- sqrt(var(Y.test[A & (W.test==1)]) / sum(A & (W.test==1)) + var(Y.test[A & (W.test==0)]) / sum(A & W.test==0)) * mean(A)^2
print(paste("Difference estimate [sample avg]:", diff.estimate, "Std. Error:", diff.stderr))

# Valid in both randomized and obs setting with unconf + overlap.
gamma.hat.pi.diff <- gamma.hat.pi - gamma.hat.0
diff.estimate <- mean(gamma.hat.pi.diff)
diff.stderr <- sd(gamma.hat.pi.diff) / sqrt(length(gamma.hat.pi.diff))
print(paste("Difference estimate [aipw]:", diff.estimate, "Std. Error:", diff.stderr))

# Only valid in randomized settings
fmla <- formula(paste0(outcome, "~ 0 + pi.hat + w:pi.hat"))
ols <- lm(fmla, data=transform(data[test,], pi.hat=factor(pi.hat)))
coefs <- coeftest(ols, vcov=vcovHC(ols, 'HC2'))[3:4,1:2] 
coefs[,1] <- coefs[,1] - cost  # subtracting cost
coefs

# Valid in randomized settings and observational settings with unconfoundedness+overlap
ols <- lm(gamma.hat.1 - gamma.hat.0 ~ 0 + factor(pi.hat))
coeftest(ols, vcov=vcovHC(ols, 'HC2'))[1:2,]

# Only valid in randomized settings.
fmla <- paste0(outcome, ' ~ 0 + leaf +  w:leaf')
ols <- lm(fmla, data=transform(data[test,], leaf=factor(leaf)))
coefs <- coeftest(ols, vcov=vcovHC(ols, 'HC2'))[,1:2]
interact <- grepl(":", rownames(coefs))
coefs[interact,1] <- coefs[interact,1] - cost # subtracting cost
coefs[interact,]

# Valid in randomized settings and observational settings with unconfoundedness+overlap.
gamma.hat.diff <- gamma.hat.1 - gamma.hat.0
ols <- lm(gamma.hat.1 - gamma.hat.0 ~ 0 + factor(leaf))
coeftest(ols, vcov=vcovHC(ols, 'HC2'))[,1:2]

df <- lapply(covariates, function(covariate) {
  fmla <- formula(paste0(covariate, " ~ 0 + factor(pi.hat)"))
  ols <- lm(fmla, data=transform(data[test,], pi.hat=pi.hat))
  ols.res <- coeftest(ols, vcov=vcovHC(ols, "HC2"))
    
  # Retrieve results
  avg <- ols.res[,1]
  stderr <- ols.res[,2]
  
  # Tally up results
  data.frame(
    covariate, avg, stderr, pi.hat=factor(c('control', 'treatment')),
    # Used for coloring
    scaling=pnorm((avg - mean(avg))/sd(avg)), 
    # We will order based on how much variation is 'explained' by the averages
    # relative to the total variation of the covariate in the data
    variation=sd(avg) / sd(data[,covariate]),
    # String to print in each cell in heatmap below
    labels=paste0(signif(avg, 3), "\n", "(", signif(stderr, 3), ")"))
})
df <- do.call(rbind, df)

#a small optional trick to ensure heatmap will be in decreasing order of 'variation'
df$covariate <- reorder(df$covariate, order(df$variation))

# plot heatmap
ggplot(df) +
    aes(pi.hat, covariate) +
    geom_tile(aes(fill = scaling)) + 
    geom_text(aes(label = labels)) +
    scale_fill_gradient(low = "#ccc7a5", high = "#235b2b") +
    ggtitle(paste0("Average covariate values within each leaf")) +
    theme_minimal() + 
    ylab("") + xlab("") +
    theme(plot.title = element_text(size = 12, face = "bold"),
          axis.text=element_text(size=11))

# Creating random costs.
data$costs <- C <- ifelse(data$w == 1, rexp(n=n, 1 * (data$age + data$polviews)), 0)

# Preprating data
X <- data[,covariates]
Y <- data[,outcome]
W <- data[,treatment]
C <- data[,'costs']

# Sample splitting.
# Note that we can't simply rely on out-of-bag observations here, because we're ranking *across* observations.
# This is the same issue we encountered when ranking observations according to predicted CATE in the HTE chapter.
train <- 1:(n / 2)
test <- (n / 2 + 1):n
train.treat <- which(W[train] == 1)

# Because they will be useful to save computational time later,
# we'll estimate the outcome model and propensity score model separately.

# Propensity score model.
# Comment / uncomment as appropriate.
# Randomized setting with fixed and known assignment probability (here: 0.5)
W.hat.train <- 0.5
# Observational settings with unconfoundedness and overlap.
# e.forest <- regression_forest(X = X[train,], Y = W[train])
# W.hat.train <- predict(e.forest)$predictions

# Outcome model.
m.forest <- regression_forest(X = X[train,], Y = Y[train])
Y.hat.train <- predict(m.forest)$predictions

# Estimating the numerator.
tau.forest <- causal_forest(X = X[train,], Y = Y[train], W = W[train], W.hat = W.hat.train, Y.hat = Y.hat.train)

# Estimating the denominator.
# Because costs for untreated observations are known to be zero, we're only after E[C(1)|X].
# Under unconfoundedness, this can be estimated by regressing C on X using only the treated units.
gamma.forest <- regression_forest(X = X[train.treat,], Y = C[train.treat])
gamma.hat.train <- predict(m.forest)$predictions
# If costs were not zero, we could use the following.
# gamma.forest <- causal_forest(X = X[train,], Y = C[train], W = W[train], W.hat = W.hat.train, Y.hat = Y.hat.train)

# Compute predictions on test set
tau.hat <- predict(tau.forest, X[test,])$predictions
gamma.hat <- predict(gamma.forest, X[test,])$predictions

# Rankings
rank.ignore <- order(tau.hat, decreasing = TRUE)
rank.direct <- order(tau.hat / gamma.hat, decreasing = TRUE)

# IPW-based estimates of (normalized) treatment and cost
W.hat.test <- .5
# W.hat.test <- predict(e.forest, X[test,])$predictions
n.test <- length(test)
treatment.ipw <- 1 / n.test * (W[test]/W.hat.test - (1 - W[test])/(1 - W.hat.test)) * Y[test]
cost.ipw <-  1 / n.test * W[test] / W.hat.test * C[test]

# Cumulative benefit and cost of treatment (normalized) for a policy that ignores costs.
treatment.value.ignore <- cumsum(treatment.ipw[rank.ignore]) / sum(treatment.ipw)
treatment.cost.ignore <- cumsum(cost.ipw[rank.ignore]) / sum(cost.ipw)

# Cumulative benefit and cost of treatment (normalized) for a policy that uses the ratio, estimated separately.
treatment.value.direct <- cumsum(treatment.ipw[rank.direct]) / sum(treatment.ipw)
treatment.cost.direct <- cumsum(cost.ipw[rank.direct]) / sum(cost.ipw)

# Plotting
plot(treatment.cost.ignore, treatment.value.ignore, col=rgb(0.2,0.4,0.1,0.7), lwd = 3, type = "l", main="Cost curves", xlab="(Normalized) cumulative cost", ylab="(Normalized) cumulative value")
lines(treatment.cost.direct, treatment.value.direct, col=rgb(0.6,0.4,0.1,0.7), lwd = 3, type = "l")
abline(a = 0, b = 1, lty = 2)
legend("bottomright", legend = c("Ignoring costs", "Direct ratio"), col = c(rgb(0.2,0.4,0.1,0.7), rgb(0.8,0.4,0.1,0.7)), lwd=3)

# Estimating rho(x) directly via instrumental forests.
# In observational settings, remove the argument W.hat.
iv.forest <- instrumental_forest(X = X[train,],
                                 Y = Y[train],
                                 W = C[train],   # cost as 'treatment'
                                 Z = W[train],   # treatment as 'instrument'
                                 Y.hat = Y.hat.train,
                                 W.hat = NULL,   # If costs are nonzero: predict(gamma.forest)$predictions,
                                 Z.hat = tau.forest$W.hat)

# Predict and compute and estimate of the ranking on a test set.
rho.iv <- predict(iv.forest, X[test,])$predictions
rank.iv <- order(rho.iv, decreasing = TRUE)

# Cumulative benefit and cost of treatment (normalized) for a policy based on the IV analogy.
treatment.value.iv <- cumsum(treatment.ipw[rank.iv]) / sum(treatment.ipw)
treatment.cost.iv <- cumsum(cost.ipw[rank.iv]) / sum(cost.ipw)

# Plotting
plot(treatment.cost.ignore, treatment.value.ignore, col=rgb(0.2,0.4,0.1,0.7), lwd = 3, type = "l", main="Cost curves", xlab="(Normalized) cumulative cost", ylab="(Normalized) cumulative value")
lines(treatment.cost.direct, treatment.value.direct, col=rgb(0.6,0.4,0.1,0.7), lwd = 3, type = "l")
abline(a = 0, b = 1, lty = 2)
lines(treatment.cost.iv, treatment.value.iv, col=rgb(1,0,0,0.7), lwd = 3, type = "l")
abline(a = 0, b = 1, lty = 2)
legend("bottomright", legend = c("Ignoring costs", "Direct ratio", "Sun, Du, Wager (2021)"), col = c(rgb(0.2,0.4,0.1,0.7), rgb(0.8,0.4,0.1,0.7), rgb(1,0,0,0.7)), lwd=3)

plot(gamma.hat, tau.hat, 
     xlab="Estimated cost (normalized)", ylab="Estimated CATE (normalized)")

auc <- data.frame(
  ignore=sum((treatment.value.ignore - treatment.cost.ignore) * diff((c(0, treatment.cost.ignore)))),
  ratio=sum((treatment.value.direct - treatment.cost.direct) * diff((c(0, treatment.cost.direct)))),
  iv=sum((treatment.value.iv - treatment.cost.iv) * diff((c(0, treatment.cost.iv))))
)
auc

