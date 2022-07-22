library(lmtest)
library(sandwich)
library(grf)
library(glmnet)
library(splines)
library(ggplot2)
library(reshape2)

# Read in data
data <- read.csv("https://docs.google.com/uc?id=1kSxrVci_EUcSr_Lg1JKk1l7Xd5I9zfRC&export=download")
n <- nrow(data)


# Treatment: does the the gov't spend too much on "welfare" (1) or "assistance to the poor" (0)
treatment <- "w"

# Outcome: 1 for 'yes', 0 for 'no'
outcome <- "y"

# Additional covariates
covariates <- c("age", "polviews", "income", "educ", "marital", "sex")

# Only valid in the randomized setting. Do not use in observational settings.
Y <- data[,outcome]
W <- data[,treatment]
ate.est <- mean(Y[W==1]) - mean(Y[W==0])
ate.se <- sqrt(var(Y[W == 1]) / sum(W == 1) + var(Y[W == 0]) / sum(W == 0))
ate.tstat <- ate.est / ate.se
ate.pvalue <- 2*(pnorm(1 - abs(ate.est/ate.se)))
ate.results <- c(estimate=ate.est, std.error=ate.se, t.stat=ate.tstat, pvalue=ate.pvalue)
print(ate.results)

fmla <- formula(paste(outcome, '~', treatment))  # y ~ w
t.test(fmla, data=data)

# Do not use! standard errors are not robust to heteroskedasticity! (See below)
fmla <- formula(paste0(outcome, '~', treatment))
ols <- lm(fmla, data=data)
coef(summary(ols))[2,]

# Use this instead. Standard errors are heteroskedasticity-robust.
# Only valid in randomized setting.
fmla <- formula(paste0(outcome, '~', treatment))
ols <- lm(fmla, data=data)
coeftest(ols, vcov=vcovHC(ols, type='HC2'))[2,]

# Probabilistically dropping observations in a manner that depends on x

# copying old dataset, just in case
data.exp <- data

# defining group that we will be dropped with some high probaibility
grp <- ((data$w == 1) &  # if treated AND...
        (
            (data$age > 45) |     # belongs an older group OR
            (data$polviews < 5)   # more conservative
        )) | # OR
        ((data$w == 0) &  # if untreated AND...
        (
            (data$age < 45) |     # belongs a younger group OR
            (data$polviews > 4)   # more liberal
        )) 

# Individuals in the group above have a small chance of being kept in the sample
prob.keep <- ifelse(grp, .15, .85)
keep.idx <- as.logical(rbinom(n=nrow(data), prob=prob.keep, size = 1))

# Dropping
data <- data[keep.idx,]

X <- model.matrix(formula("~ 0 + age + polviews"), data.exp)  # old 'experimental' dataset
W <- data.exp$w
Y <- data.exp$y
par(mfrow=c(1,2))
for (w in c(0, 1)) {
  plot(X[W==w,1] + rnorm(n=sum(W==w), sd=.1), X[W==w,2] + rnorm(n=sum(W==w), sd=.1), 
       pch=ifelse(Y, 23, 21), cex=1, col=ifelse(Y, rgb(1,0,0,1/4), rgb(0,0,1,1/4)),
       bg=ifelse(Y, rgb(1,0,0,1/4), rgb(0,0,1,1/4)), main=ifelse(w, "Treated", "Untreated"), xlab="age", ylab="polviews")
}

X <- model.matrix(formula("~ 0 + age + polviews"), data)
W <- data$w
Y <- data$y
par(mfrow=c(1,2))
for (w in c(0, 1)) {
  plot(X[W==w,1] + rnorm(n=sum(W==w), sd=.1), X[W==w,2] + rnorm(n=sum(W==w), sd=.1), 
       pch=ifelse(Y, 23, 21), cex=1, col=ifelse(Y, rgb(1,0,0,1/4), rgb(0,0,1,1/4)),
       bg=ifelse(Y, rgb(1,0,0,1/4), rgb(0,0,1,1/4)), main=ifelse(w, "Treated", "Untreated"), xlab="age", ylab="polviews")
}

# Do not use in observational settings.
# This is only to show how the difference-in-means estimator is biased in that case.
fmla <- formula(paste0(outcome, '~', treatment))
ols <- lm(fmla, data=data)
coeftest(ols, vcov=vcovHC(ols, type='HC2'))[2,]

# Do not use! We'll see a better estimator below.

# Fitting some model of E[Y|X,W]
fmla <- as.formula(paste0(outcome, "~ ", paste("bs(", covariates, ", df=3)", "*", treatment, collapse="+")))
model <- lm(fmla, data=data)  

# Predicting E[Y|X,W=w] for w in {0, 1}
data.1 <- data
data.1[,treatment] <- 1
data.0 <- data
data.0[,treatment] <- 0
muhat.treat <- predict(model, newdata=data.1)
muhat.ctrl <- predict(model, newdata=data.0)

# Averaging predictions and taking their difference
ate.est <- mean(muhat.treat) - mean(muhat.ctrl)
print(ate.est)

# Simulating the scenario above a large number of times
A.mean <- 60
B.mean <- 70
pop.mean <- .5 * A.mean + .5 * B.mean  # both schools have the same size

# simulating the scenario about a large number of times
sample.means <- replicate(1000, {
    school <- sample(c("A", "B"), p=c(.5, .5), size=100, replace=TRUE)
    treated <- ifelse(school == 'A', rbinom(100, 1, .05), rbinom(100, 1, .4))
    grades <- ifelse(school == 'A', rnorm(100, A.mean, 5), rnorm(100, B.mean, 5))
    mean(grades[treated == 1])  # average grades among treated students, without taking school into account
  })
hist(sample.means, freq=F, main="Sample means of treated students' grades", xlim=c(55, 75), col=rgb(0,0,1,1/8), ylab="", xlab="")
abline(v=pop.mean, lwd=3, lty=2)
legend("topleft", "truth", lwd=3, lty=2, bty="n")

# simulating the scenario about a large number of times
agg.means <- replicate(1000, {
    school <- sample(c("A", "B"), p=c(.5, .5), size=100, replace=TRUE)
    treated <- ifelse(school == 'A', rbinom(100, 1, .05), rbinom(100, 1, .4))
    grades <- ifelse(school == 'A', rnorm(100, A.mean, 5), rnorm(100, B.mean, 5))
    # average grades among treated students in each school
    mean.treated.A <- mean(grades[(treated == 1) & (school == 'A')])
    mean.treated.B <- mean(grades[(treated == 1) & (school == 'B')])
    # probability of belonging to each school
    prob.A <- mean(school == 'A')
    prob.B <- mean(school == 'B')
    prob.A * mean.treated.A + prob.B * mean.treated.B
  })
hist(agg.means, freq=F, main="Aggregated sample means of treated students' grades", xlim=c(50, 75), col=rgb(0,0,1,1/8), ylab="", xlab="")
abline(v=pop.mean, lwd=3, lty=2)
legend("topright", "truth", lwd=3, lty=2, bty="n")

# Available in randomized settings and observational settings with unconfoundedness+overlap

# Estimate the propensity score e(X) via logistic regression using splines
fmla <- as.formula(paste0("~", paste0("bs(", covariates, ", df=3)", collapse="+")))
W <- data[,treatment]
Y <- data[,outcome]
XX <- model.matrix(fmla, data)
logit <- cv.glmnet(x=XX, y=W, family="binomial")
e.hat <- predict(logit, XX, s = "lambda.min", type="response")

# Using the fact that
z <- Y * (W/e.hat - (1-W)/(1-e.hat))
ate.est <- mean(z)
ate.se <- sd(z) / sqrt(length(z))
ate.tstat <- ate.est / ate.se
ate.pvalue <- 2*(pnorm(1 - abs(ate.est/ate.se)))
ate.results <- c(estimate=ate.est, std.error=ate.se, t.stat=ate.tstat, pvalue=ate.pvalue)
print(ate.results)

# Available in randomized settings and observational settings with unconfoundedness+overlap

# A list of vectors indicating the left-out subset
n <- nrow(data)
n.folds <- 5
indices <- split(seq(n), sort(seq(n) %% n.folds))

# Preparing data
W <- data[,treatment]
Y <- data[,outcome]

# Matrix of (transformed) covariates used to estimate E[Y|X,W]
fmla.xw <- formula(paste("~ 0 +", paste0("bs(", covariates, ", df=3)", "*", treatment, collapse=" + ")))
XW <- model.matrix(fmla.xw, data)
# Matrix of (transformed) covariates used to predict E[Y|X,W=w] for each w in {0, 1}
data.1 <- data
data.1[,treatment] <- 1
XW1 <- model.matrix(fmla.xw, data.1)  # setting W=1
data.0 <- data
data.0[,treatment] <- 0
XW0 <- model.matrix(fmla.xw, data.0)  # setting W=0

# Matrix of (transformed) covariates used to estimate and predict e(X) = P[W=1|X]
fmla.x <- formula(paste(" ~ 0 + ", paste0("bs(", covariates, ", df=3)", collapse=" + ")))
XX <- model.matrix(fmla.x, data)

# (Optional) Not penalizing the main effect (the coefficient on W)
penalty.factor <- rep(1, ncol(XW))
penalty.factor[colnames(XW) == treatment] <- 0

# Cross-fitted estimates of E[Y|X,W=1], E[Y|X,W=0] and e(X) = P[W=1|X]
mu.hat.1 <- rep(NA, n)
mu.hat.0 <- rep(NA, n)
e.hat <- rep(NA, n)
for (idx in indices) {
  # Estimate outcome model and propensity models
  # Note how cross-validation is done (via cv.glmnet) within cross-fitting! 
  outcome.model <- cv.glmnet(x=XW[-idx,], y=Y[-idx], family="gaussian", penalty.factor=penalty.factor)
  propensity.model <- cv.glmnet(x=XX[-idx,], y=W[-idx], family="binomial")

  # Predict with cross-fitting
  mu.hat.1[idx] <- predict(outcome.model, newx=XW1[idx,], type="response")
  mu.hat.0[idx] <- predict(outcome.model, newx=XW0[idx,], type="response")
  e.hat[idx] <- predict(propensity.model, newx=XX[idx,], type="response")
}

# Commpute the summand in AIPW estimator
aipw.scores <- (mu.hat.1 - mu.hat.0
                + W / e.hat * (Y -  mu.hat.1)
                - (1 - W) / (1 - e.hat) * (Y -  mu.hat.0))

# Tally up results
ate.aipw.est <- mean(aipw.scores)
ate.aipw.se <- sd(aipw.scores) / sqrt(n)
ate.aipw.tstat <- ate.aipw.est / ate.aipw.se
ate.aipw.pvalue <- 2*(pnorm(1 - abs(ate.aipw.tstat)))
ate.aipw.results <- c(estimate=ate.aipw.est, std.error=ate.aipw.se, t.stat=ate.aipw.tstat, pvalue=ate.aipw.pvalue)
print(ate.aipw.results)

head(data)

# Available in randomized settings and observational settings with unconfoundedness+overlap

# Input covariates need to be numeric. 
XX <- model.matrix(formula(paste0("~", paste0(covariates, collapse="+"))), data=data)

# Estimate a causal forest.
forest <- causal_forest(
              X=XX,  
              W=data[,treatment],
              Y=data[,outcome],
              #W.hat=...,  # In randomized settings, set W.hat to the (known) probability of assignment
              num.trees = 100)
forest.ate <- average_treatment_effect(forest)
print(forest.ate)

# Here, adding covariates and their interactions, though there are many other possibilities.
fmla <- formula(paste("~ 0 +", paste(apply(expand.grid(covariates, covariates), 1, function(x) paste0(x, collapse="*")), collapse="+")))

# Using the propensity score estimated above
e.hat <- forest$W.hat

XX <- model.matrix(fmla, data)
W <- data[,treatment]
pp <- ncol(XX)

# Unadjusted covariate means, variances and standardized abs mean differences
means.treat <- apply(XX[W == 1,], 2, mean)
means.ctrl <- apply(XX[W == 0,], 2, mean)
abs.mean.diff <- abs(means.treat - means.ctrl)

var.treat <- apply(XX[W == 1,], 2, var)
var.ctrl <- apply(XX[W == 0,], 2, var)
std <- sqrt(var.treat + var.ctrl)

# Adjusted covariate means, variances and standardized abs mean differences
means.treat.adj <- apply(XX*W/e.hat, 2, mean)
means.ctrl.adj <- apply(XX*(1-W)/(1-e.hat), 2, mean)
abs.mean.diff.adj <- abs(means.treat.adj - means.ctrl.adj)

var.treat.adj <- apply(XX*W/e.hat, 2, var)
var.ctrl.adj <- apply(XX*(1-W)/(1-e.hat), 2, var)
std.adj <- sqrt(var.treat.adj + var.ctrl.adj)

# Plotting
par(oma=c(0,4,0,0))
plot(-2, xaxt="n", yaxt="n", xlab="", ylab="", xlim=c(-.01, 1.3), ylim=c(0, pp+1), main="Standardized absolute mean differences")
axis(side=1, at=c(-1, 0, 1), las=1)
lines(abs.mean.diff / std, seq(1, pp), type="p", col="blue", pch=19)
lines(abs.mean.diff.adj / std.adj, seq(1, pp), type="p", col="orange", pch=19)
legend("topright", c("Unadjusted", "Adjusted"), col=c("blue", "orange"), pch=19)
abline(v = seq(0, 1, by=.25), lty = 2, col = "grey", lwd=.5)
abline(h = 1:pp,  lty = 2, col = "grey", lwd=.5)
mtext(colnames(XX), side=2, cex=0.7, at=1:pp, padj=.4, adj=1, col="black", las=1, line=.3)
abline(v = 0)

abs.mean.diff / std

# change this if working on different data set
XX <- model.matrix(fmla, data)[,c("age", "polviews", "age:polviews", "educ")] 
W <- data[,treatment]
pp <- ncol(XX)

plot.df <- data.frame(XX, W = as.factor(W), IPW = ifelse(W == 1, 1 / e.hat, 1 / (1 - e.hat)))
head(plot.df)

plot.df <- reshape(plot.df, varying = list(1:pp), direction = "long", v.names = "X",
                   times = factor(colnames(XX), levels = colnames(XX)))

ggplot(plot.df, aes(x = X, fill = W)) +
  geom_histogram(alpha = 0.5, position = "identity", bins=30) +
  facet_wrap( ~ time, ncol = 2, scales = "free") +
  ggtitle("Covariate histograms (unajusted)")

ggplot(plot.df, aes(x = X, weight=IPW, fill = W)) +
  geom_histogram(alpha = 0.5, position = "identity", bins=30) +
  facet_wrap( ~ time, ncol = 2, scales = "free") +
  ggtitle("Covariate histograms (adjusted)")

e.hat <- forest$W.hat
hist(e.hat, main="Estimated propensity scores (causal forest)", breaks=100, freq=FALSE, xlab="", ylab="", xlim=c(-.1, 1.1))
