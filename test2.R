alpha = c(0.5, 0.5, 0.5, 0, -0.2, -0.1, 0, 0.1, 0.2, 0.3)
beta = c(0, 0.5, 0.5, 0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.3)

psi1 <- function(x){
  x
}

# generate dataset

n = 1000 # sample size
max_k = 15
min_k = 3
set.seed(1000)

X <- matrix(nrow=n, ncol=10)
X[,1] <- rbinom(n, 1, 1/5)
X[,2] <- rbinom(n, 1, 2/5)
X[,3] <- rbinom(n, 1, 3/5)
X[,4] <- rbinom(n, 1, 4/5)
X[,5] <- rnorm(n, 0, 0.5)
X[(X[,5]>=1), 5] <- 1
X[(X[,5]<=-1), 5] <- -1
X[,6] <- rnorm(n, 0, 0.5)
X[(X[,6]>=1), 6] <- 1
X[(X[,6]<=-1), 6] <- -1
X[,7] <- rnorm(n, 0, 0.5)
X[(X[,7]>=1), 7] <- 1
X[(X[,7]<=-1), 7] <- -1
X[,8] <- rnorm(n, 0, 0.5)
X[(X[,8]>=1), 8] <- 1
X[(X[,8]<=-1), 8] <- -1
X[,9] <- rnorm(n, 0, 1)
X[(X[,9]>=2), 9] <- 2
X[(X[,9]<=-2), 9] <- -2
X[,10] <- rnorm(n, 0, 1)
X[(X[,10]>=2), 10] <- 2
X[(X[,10]<=-2), 10] <- -2

A <- rbinom(n, 1, 0.5)
#A <- rbinom(n, 1, 0.5) * 2 - 1

t_surv = rexp(n, rate = 1) / exp(X%*%alpha + A*psi1(X%*%beta))
t_cens = runif(n, 0, 5)
censoring = as.numeric(t_surv<=t_cens) # censoring = 1 <-> failure time observed 
t_obs = t_surv
t_obs[which(t_surv>t_cens)] = t_cens[which(t_surv>t_cens)]

min_t <- min(t_obs[which(censoring == 1)])
X_inquantile <- X[which(t_obs>=min_t),] 

# test is the rcpp package I wrote for the algorithm part
library(test)
library(quadprog)
library(splines)

est_a <- matrix(nrow = 10, ncol = max_k - min_k + 1)
est_b <- matrix(nrow = 10, ncol = max_k - min_k + 1)
est_c <- matrix(nrow = max_k + 1, ncol = max_k - min_k + 1)
type_conv <- rep(0, max_k - min_k + 1)
num_conv <- rep(0, max_k - min_k + 1)
likelihood <- rep(0, max_k - min_k + 1)
knots <- matrix(nrow = max_k + 1, ncol = max_k - min_k + 1)

i <- 0
for (k in min_k:max_k){
  i <- i+1
  s <- solvealg(X, X_inquantile, t_obs, censoring, A, k, 100)
  est_a[,i] <- s$a
  est_b[,i] <- s$b
  est_c[1:(k+1),i] <- s$c
  likelihood[i] <- s$likelihood
  num_conv[i] <- s$iteration
  type_conv[i] <- s$conv_type
}

print(est_a)
print(est_b)
print(likelihood)
print(num_conv)
print(type_conv)
likelihood[1:13] + c(min_k:max_k) * 2 / n



###########################
library(test)
library(quadprog)
library(splines)
# true value - parameter and index function
alpha = c(1, -0.6, 0, 0.5, 0, 0.2, -0.4, 0.1, -0.1, 0)
beta = c(0, 0.2, 0.2, -0.5, 0.3, 0.4, 0.8, 0, 0, -1)

psi2 <- function(x){
  x^3
}

# generate dataset
n = 1000 # sample size
max_k = 15
min_k = 3
set.seed(1000)

X <- matrix(nrow=n, ncol=10)
X[,1] <- rbinom(n, 1, 1/5)
X[,2] <- rbinom(n, 1, 2/5)
X[,3] <- rbinom(n, 1, 3/5)
X[,4] <- rbinom(n, 1, 4/5)
X[,5] <- rnorm(n, (5-7)/10, 0.5)
X[,6] <- rnorm(n, (6-7)/10, 0.5)
X[,7] <- rnorm(n, (7-7)/10, 0.5)
X[,8] <- rnorm(n, (8-7)/10, 0.5)
X[,9] <- rnorm(n, 1-9*9/80, 1)
X[,10] <- rnorm(n, 10*10/80, 1)

A <- rbinom(n, 1, 0.5)
#A <- rbinom(n, 1, 0.5) * 2 - 1

t_surv = rexp(n, rate = 1) / exp(X%*%alpha + A*psi2(X%*%beta))
t_cens = runif(n, 0, 5)
censoring = as.numeric(t_surv<=t_cens) # censoring = 1 <-> failure time observed 
t_obs = t_surv
t_obs[which(t_surv>t_cens)] = t_cens[which(t_surv>t_cens)]

min_t <- min(t_obs[which(censoring == 1)])
X_inquantile <- X[which(t_obs>=min_t),]

k <- 3
s_test <- solvealg(X, X_inquantile, t_obs, censoring, A, k, 1)
s_test$a
s_test$b
s_test$c
a_new <- s_test$a
b_new <- s_test$b
c_new <- s_test$c
# calculate the knots
all_knots <- X_inquantile %*% b_new
all_knots <- unique(sort(all_knots))
length <- length(all_knots)
knots <- all_knots[which((1:length)%%(length%/%(k-1)) == 1 )]
knots[k] <- all_knots[length]
knots <- c(knots[1], knots[1], knots, knots[k], knots[k])  
# calculate b-spline basis function values and derivatives
b_func <- spline.des(knots, X%*%b_new, ord = 3)
b_deri <- spline.des(knots, knots[3:(k+2)], ord = 3, derivs = rep(1, k))
bs_der_all <- spline.des(knots, X%*%b_new, ord = 3, derivs = rep(1, n)) 

# alpha, gamma
result1 <- opt_ac(a_new, c_new, b_func$design, b_deri$design, 
                  A, X, censoring, t_obs)
a_new <- a_new + result1[1:length(a_new)]
c_new <- c_new + result1[(length(a_new)+1):length(result1)]
a_new
c_new

# beta
result2 <- opt_b_test(a_new, b_new, c_new, b_func$design, bs_der_all$design,
                      A, X, censoring, t_obs)
result2$s0[1:10]