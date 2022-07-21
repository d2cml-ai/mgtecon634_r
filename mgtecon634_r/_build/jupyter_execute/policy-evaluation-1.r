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
library(ggplot2)

# Note: bookdown seems to require explicitly calling these packages.
library(reshape2)
library(DiagrammeR)

n <- 1000
p <- 4
X <- matrix(runif(n*p), n, p)
W <- rbinom(n, prob = .5, size = 1)  # independent from X and Y
Y <- .5*(X[,1] - .5) + (X[,2] - .5)*W + .1 * rnorm(n) 

y.norm <- 1-(Y - min(Y))/(max(Y)-min(Y)) # just for plotting

plot(X[,1], X[,2], pch=ifelse(W, 21, 23), cex=1.5, bg=gray(y.norm), xlab="X1", ylab="X2")

par(mfrow=c(1,2))
for (w in c(0, 1)) {
  plot(X[W==w,1], X[W==w,2], pch=ifelse(W[W==w], 21, 23), cex=1.5,
       bg=gray(y.norm[W==w]), main=ifelse(w, "Treated", "Untreated"),
       xlab="X1", ylab="X2")
}

col2 <- rgb(0.250980, 0.690196, 0.650980, .35)
col1 <- rgb(0.9960938, 0.7539062, 0.0273438, .35)
plot(X[,1], X[,2], pch=ifelse(W, 21, 23), cex=1.5, bg=gray(y.norm), xlab="X1", ylab="X2")
rect(-.1, -.1, .5, 1.1, density = 250, angle = 45, col = col1, border = NA)
rect(.5, -.1, 1.1, .5, density = 250, angle = 45, col = col1, border = NA)
rect(.5, .5, 1.1, 1.1, density = 250, angle = 45, col = col2, border = NA)
text(.75, .75, labels = "TREAT (A)", cex = 1.8)
text(.25, .25, labels = expression(DO ~ NOT ~ TREAT ~ (A^C)), cex = 1.8, adj = .25)

# Only valid in randomized setting.
A <- (X[,1] > .5) & (X[,2] > .5)
value.estimate <- mean(Y[A & (W==1)]) * mean(A) + mean(Y[!A & (W==0)]) * mean(!A)
value.stderr <- sqrt(var(Y[A & (W==1)]) / sum(A & (W==1)) * mean(A)^2 + var(Y[!A & (W==0)]) / sum(!A & W==0) * mean(!A)^2)
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Only valid in randomized setting.
p <- .75 # for example
value.estimate <- p * mean(Y[(W==1)]) + (1 - p) * mean(Y[(W==0)])
value.stderr <- sqrt(var(Y[(W==1)]) / sum(W==1) * p^2 + var(Y[(W==0)]) / sum(W==0) * (1-p)^2)
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Only valid in randomized settings.
A <- (X[,1] > .5) & (X[,2] > .5)
diff.estimate <- (mean(Y[A & (W==1)]) - mean(Y[A & (W==0)])) * mean(A)
diff.stderr <- sqrt(var(Y[A & (W==1)]) / sum(A & (W==1)) + var(Y[A & (W==0)]) / sum(A & W==0)) * mean(A)^2
print(paste("Difference estimate:", diff.estimate, "Std. Error:", diff.stderr))

# Only valid in randomized settings.
A <- (X[,1] > .5) & (X[,2] > .5)
diff.estimate <- (mean(Y[A & (W==1)]) - mean(Y[A & (W==0)])) * mean(A) / 2 +
                 (mean(Y[!A & (W==0)]) - mean(Y[!A & (W==1)])) * mean(!A) / 2
diff.stderr <- sqrt((mean(A)/2)^2 * (var(Y[A & (W==1)])/sum(A & W==1) + var(Y[A & (W==0)])/sum(A & W==0)) + 
                    (mean(!A)/2)^2 * (var(Y[!A & (W==1)])/sum(!A & W==1) + var(Y[!A & (W==0)])/sum(!A & W==0)))
print(paste("Difference estimate:", diff.estimate, "Std. Error:", diff.stderr))

# An "observational" dataset satisfying unconfoundedness and overlap.
n <- 1000
p <- 4
X <- matrix(runif(n*p), n, p)
e <- 1/(1+exp(-2*(X[,1]-.5)-2*(X[,2]-.5)))  # not observed by the analyst.
W <- rbinom(n, prob = e, size = 1)
Y <- .5*(X[,1] - .5) + (X[,2] - .5)*W + .1 * rnorm(n) 

y.norm <- (Y - min(Y))/(max(Y)-min(Y))
par(mfrow=c(1,2))
for (w in c(0, 1)) {
  plot(X[W==w,1], X[W==w,2], pch=ifelse(W[W==w], 21, 23), cex=1.5,
       bg=gray(y.norm[W==w]), main=ifelse(w, "Treated", "Untreated"),
       xlab="X1", ylab="X2")
}

# Valid in observational and randomized settings

# Randomized settings: use the true assignment probability:
# forest <- causal_forest(X, Y, W, W.hat=.5, num.trees=100)  # replace .5 with true assign prob
# Observational settings with unconfoundedness + overlap:
forest <- causal_forest(X, Y, W)

# Estimate a causal forest
tau.hat <- predict(forest)$predictions

# Estimate outcome model for treated and propen
mu.hat.1 <- forest$Y.hat + (1 - forest$W.hat) * tau.hat # E[Y|X,W=1] = E[Y|X] + (1-e(X))tau(X)
mu.hat.0 <- forest$Y.hat - forest$W.hat * tau.hat  # E[Y|X,W=0] = E[Y|X] - e(X)tau(X)

# Compute AIPW scores
gamma.hat.1 <- mu.hat.1 + W/forest$W.hat * (Y - mu.hat.1)
gamma.hat.0 <- mu.hat.0 + (1-W)/(1-forest$W.hat) * (Y - mu.hat.0)

# If you have the package policytree installed, the following is equivalent
# gamma.hat.matrix <- policytree::double_robust_scores(forest)
# gamma.hat.1 <- gamma.hat.matrix[,2]
# gamma.hat.0 <- gamma.hat.matrix[,1]

# # Valid randomized data and observational data with unconfoundedness+overlap.
# # Note: read the comments below carefully. 
# # In randomized settings, do not estimate forest.e and e.hat; use known assignment probs.
#  data <- data.frame(x=X, w=W, y=Y)
# covariates <- paste0("x.", seq(ncol(X)))
# treatment <- "w"
# outcome <- "y"
# 
# fmla.xw <- formula(paste0("~", paste0("bs(", covariates, ", df=3, degree=3) *", treatment, collapse="+")))
# fmla.x <- formula(paste0("~", paste0("bs(", covariates, ", df=3, degree=3)", collapse="+")))
# XW <- model.matrix(fmla.xw, data)
# XX <- model.matrix(fmla.x, data)
# Y <- data[,outcome]
# 
# n.folds <- 5
# indices <- split(seq(n), sort(seq(n) %% n.folds))
# 
# gamma.hat.1 <- rep(NA, n)
# gamma.hat.0 <- rep(NA, n)
# for (idx in indices) {
# 
#   # Fitting the outcome model on training folds
#   model.m <- cv.glmnet(XW[-idx,], Y[-idx])  
#   
#   # Predict outcome E[Y|X,W=w] for w in {0, 1} on the held-out fold 
#   data.0 <- data[idx,]
#   data.0[,treatment] <- 0
#   XW0 <- model.matrix(fmla.xw, data.0)
#   mu.hat.0 <- predict(model.m, XW0, s="lambda.min")
#   
#   data.1 <- data[idx,]
#   data.1[,treatment] <- 1
#   XW1 <- model.matrix(fmla.xw, data.1)
#   mu.hat.1 <- predict(model.m, XW1, s="lambda.min")
#   
#   # Fitting the propensity score model
#   # Comment / uncomment the lines below as appropriate.
#   # OBSERVATIONAL SETTING (with unconfoundedness+overlap):
#   # model.e <- cv.glmnet(XX[-idx,], W[-idx], family="binomial")  
#   # e.hat <- predict(model.e, XX[idx,], s="lambda.min", type="response")
#   # RANDOMIZED SETTING
#   e.hat <- rep(0.5, length(idx))    # assuming 0.5 is known assignment prob.
#   
#   # Compute AIPW scores
#   gamma.hat.1[idx] <- mu.hat.1 +  W[idx] / e.hat * (Y[idx] -  mu.hat.1)
#   gamma.hat.0[idx] <- mu.hat.0 + (1 - W[idx]) / (1 - e.hat) * (Y[idx] -  mu.hat.0)
# }

# Valid in observational and randomized settings
pi <- (X[,1] > .5) & (X[,2] > .5)
gamma.hat.pi <- pi * gamma.hat.1 + (1 - pi) * gamma.hat.0
value.estimate <- mean(gamma.hat.pi)
value.stderr <- sd(gamma.hat.pi) / sqrt(length(gamma.hat.pi))
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Valid in observational and randomized settings
pi.random <- .75
gamma.hat.pi<- pi.random * gamma.hat.1 + (1 - pi.random) * gamma.hat.0
value.estimate <- mean(gamma.hat.pi)
value.stderr <- sd(gamma.hat.pi) / sqrt(length(gamma.hat.pi))
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Valid in randomized settings and observational settings with unconfoundedness + overlap

# AIPW scores associated with first policy
pi <- (X[,1] > .5) & (X[,2] > .5)
gamma.hat.pi <- pi * gamma.hat.1 + (1 - pi) * gamma.hat.0

# AIPW scores associated with second policy
pi.never <- 0
gamma.hat.pi.never <- pi.never * gamma.hat.1 + (1 - pi.never) * gamma.hat.0

# Difference
diff.scores <- gamma.hat.pi - gamma.hat.pi.never 
diff.estimate <- mean(diff.scores)
diff.stderr <- sd(diff.scores) / sqrt(length(diff.scores))
print(paste("diff estimate:", diff.estimate, "Std. Error:", diff.stderr))

# Read in data
data <- read.csv("https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download")
n <- nrow(data)

# NOTE: We'll invert treatment and control, compared to previous chapters
data$w <- 1 - data$w

# Treatment is the wording of the question:
# 'does the the gov't spend too much on 'assistance to the poor' (control: 0)
# 'does the the gov't spend too much on "welfare"?' (treatment: 1)
treatment <- "w"

# Outcome: 1 for 'yes', 0 for 'no'
outcome <- "y"

# Additional covariates
covariates <- c("age", "polviews", "income", "educ", "marital", "sex")

# Only valid in randomized setting
X <- data[,covariates]
Y <- data[,outcome]
W <- data[,treatment]
  
pi <- (X[,"polviews"] <= 4) | (X[,"age"] > 50)
A <- pi == 1
value.estimate <- mean(Y[A & (W==1)]) * mean(A) + mean(Y[!A & (W==0)]) * mean(!A)
value.stderr <- sqrt(var(Y[A & (W==1)]) / sum(A & (W==1)) * mean(A)^2 + var(Y[!A & (W==0)]) / sum(!A & W==0) * mean(!A)^2)
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Valid in randomized settings and observational settings with unconfoundedness + overlap
fmla <- formula(paste0("~", paste(covariates, collapse="+")))
X <- model.matrix(fmla, data)
Y <- data[,outcome]
W <- data[,treatment]
  
# Estimate a causal forest
# Important: comment/uncomment as appropriate.
# Randomized setting (known, fixed assignment probability):
forest <- causal_forest(X, Y, W, W.hat=.5)
# Observational setting with unknown probability:
# forest <- causal_forest(X, Y, W)

# Estimate a causal forest
tau.hat <- predict(forest)$predictions

# Estimate outcome model for treated and propensity scores
mu.hat.1 <- forest$Y.hat + (1 - forest$W.hat) * tau.hat  # E[Y|X,W=1] = E[Y|X] + (1-e(X))tau(X)
mu.hat.0 <- forest$Y.hat - forest$W.hat * tau.hat  # E[Y|X,W=0] = E[Y|X] - e(X)tau(X)

# Compute AIPW scores
gamma.hat.1 <- mu.hat.1 + W / forest$W.hat * (Y - mu.hat.1)
gamma.hat.0 <- mu.hat.0 + (1-W) / (1-forest$W.hat) * (Y - mu.hat.0)

# Valid in randomized settings and observational settings with unconfoundedness + overlap
gamma.hat.pi <- pi * gamma.hat.1 + (1 - pi) * gamma.hat.0
value.estimate <- mean(gamma.hat.pi)
value.stderr <- sd(gamma.hat.pi) / sqrt(length(gamma.hat.pi))
print(paste("Value estimate:", value.estimate, "Std. Error:", value.stderr))

# Valid in randomized settings and observational settings with unconfoundedness + overlap
pi.2 <- .5
gamma.hat.pi.1 <- pi * gamma.hat.1 + (1 - pi) * gamma.hat.0
gamma.hat.pi.2 <- pi.2 * gamma.hat.1 + (1 - pi.2) * gamma.hat.0
gamma.hat.pi.diff <- gamma.hat.pi.1 - gamma.hat.pi.2
diff.estimate <- mean(gamma.hat.pi.diff)
diff.stderr <- sd(gamma.hat.pi.diff) / sqrt(length(gamma.hat.pi.diff))
print(paste("Difference estimate:", diff.estimate, "Std. Error:", diff.stderr))
