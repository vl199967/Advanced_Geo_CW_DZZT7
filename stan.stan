//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// data block
data {
  int<lower=0> N;                                                        // number of spatial units or neighbourhoods in London
  int<lower=0> N_edges;                                                  // number of edges connecting adjacent areas using Queens contiguity
  array[N_edges] int<lower=1, upper=N> node1;                            // list of index areas showing which spatial units are neighbours
  array[N_edges] int<lower=1, upper=N> node2;                            // list of neighbouring areas showing the connection to index spatial unit
  array[N] int<lower=0> Y;                                               // dependent variable i.e., number of Fire incidents involving electric vehicles
  int<lower=1> K;                                                        // number of independent variables i.e., K = 2
  matrix[N, K] X;                                                        // independent variables in matrix form i.e., Traffic_volume, Total_vehicles
  vector<lower=0>[N] Offset;                                             //  offset variable
}

transformed data {
    vector[N] log_Offset = log(Offset);       // use the expected cases as an offset and add to the regression model
}
//We are going to include a transformed data block. 
//Here, we are simply changing the expected numbers by taking its log() and creating another vector called log_offset.
//This will be added to the poisson_log() sampling statement in our likelihood function of the spatial model to account for the reference population in England.

parameters {
  real alpha;                                   // intercept
  vector[K] beta;                                    // covariates
  real<lower=0> sigma;                          // overall standard deviation
  real<lower=0, upper=1> rho;                 // proportion unstructured vs. spatially structured variance
  vector[N] theta;                            // unstructured random effects
  vector[N] phi;                            // structured spatial random effects
}

transformed parameters {
  vector[N] combined;                                                    // values derived from adding the unstructure and structured effect of each area
  combined = sqrt(1 - rho) * theta + sqrt(rho) * phi;                    // formulation for the combined random effect
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  Y ~ poisson_log(log_Offset + alpha + X * beta + combined * sigma);    // likelihood function: multivariable Poisson ICAR regression model
                                                                        // setting priors
  alpha ~ normal(0.0, 1.0);                                             // prior for alpha: weakly informative
  beta ~ normal(0.0, 1.0);                                              // prior for betas: weakly informative
  theta ~ normal(0.0, 1.0);                                             // prior for theta: weakly informative
  sigma ~ normal(0.0, 1.0);                                             // prior for sigma: weakly informative
  rho ~ beta(0.5, 0.5);                                                 // prior for rho
  target += -0.5 * dot_self(phi[node1] - phi[node2]);                   // calculates the spatial weights
  sum(phi) ~ normal(0, 0.001 * N);                                      // priors for phi
}

//Lastly, we instruct Stan on the parameters we want to report.
//We want them as relative risk ratio (RR). 
//We can use the generated quantities block to obtain these estimates by exponentiation of the ICAR regression model

generated quantities {
  vector[N] eta = alpha + X * beta + combined * sigma;                  // compute eta and exponentiate into mu                   
  vector[N] rr_mu = exp(eta);                                           // output the neighbourhood-specific relative risks in mu
  vector[K] rr_beta = exp(beta);                                             // output the risk ratios for each coefficient
  real rr_alpha = exp(alpha);                                           // output the risk ratios for the intercept
}




