#include <RcppArmadillo.h>
#include <Rmath.h>
using namespace Rcpp;


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::vec opt_ac(const arma::vec& a_old, const arma::vec& c_old, 
                 const arma::mat& bs, const arma::mat& bs_der, 
                 const arma::vec& A, const arma::mat& X, 
                 const arma::vec& delta, const arma::vec& tobs){
  // initialization
  int sample = delta.n_elem;
  int size = a_old.n_elem + c_old.n_elem;
  arma::vec param(size);
  param.head(a_old.n_elem) = a_old;
  param.tail(c_old.n_elem) = c_old;
  arma::mat matrix(sample, size);
  matrix.cols(0, a_old.n_elem - 1) = X;
  for (int i = a_old.n_elem; i < size; i++){
    matrix.col(i) = bs.col(i - a_old.n_elem) % A;
  }  
  Rcpp::Environment quadprog("package:quadprog");
  Rcpp::Function solve_qp = quadprog["solve.QP"];
  
  // so, s1, s2
  arma::vec s0 = arma::zeros<arma::vec>(sample);
  arma::mat s1 = arma::zeros<arma::mat>(sample, size);
  arma::cube s2 = arma::zeros<arma::cube>(size, size, sample);
  for (int j = 0; j < sample; j++) {
    double temp = exp(arma::as_scalar(matrix.row(j) * param));
    for (int i = 0; i < sample; i++) {
      if ((delta(i) == 1) && (tobs(j) >= tobs(i))) {
        s0(i) += temp;
				s1.row(i) += temp * matrix.row(j);
        s2.slice(i) += temp * matrix.row(j).t() * matrix.row(j);
      }
    }
  }
  
  // create D, d, A0, b0 to be used in the solve.qp 
  // the problem is of the format min (1/2b^tDb - d^tb) s.t. (A0^tb >= b0)
  arma::mat D = arma::zeros<arma::mat>(size, size);
  arma::vec d = arma::zeros<arma::vec>(size);
  arma::mat A0 = arma::zeros<arma::mat>(size, bs_der.n_rows);
  arma::vec b0 = arma::zeros<arma::vec>(bs_der.n_rows);
	for (int i = 0; i < sample; i++) {
		if (delta[i] == 1) {
      D += s2.slice(i)/s0(i) - s1.row(i).t() * s1.row(i) / (s0(i) * s0(i));
      d += matrix.row(i).t() - s1.row(i).t() / s0(i);
 		}
	}
  A0.rows(a_old.n_elem, size - 1) = bs_der.t();
  b0 = -bs_der * c_old;
  
  // call solve_qp to solve the quadratic programming problem
  List sol = solve_qp(D, d, A0, b0);
  arma::vec increase = as<arma::vec>(sol["solution"]);

  // return increase
  return increase;
/*  return Rcpp::List::create(Rcpp::Named("D") = D,
                            Rcpp::Named("s0") = s0,
                            Rcpp::Named("s1") = s1,
                            Rcpp::Named("s2") = s2,
                            Rcpp::Named("increase") = increase);
*/
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::vec opt_b(const arma::vec& a_new, const arma::vec& b_old, 
                const arma::vec& c_new, const arma::mat& bs, 
                const arma::mat& bs_der_all, const arma::vec& A, 
                const arma::mat& X, const arma::vec& delta, 
                const arma::vec& tobs){
// using the one step Newton directly instead of QP
  
  // initiation
  arma::vec psi = bs * c_new;
  arma::vec diff_psi = bs_der_all * c_new;
	int sample = delta.n_elem;
	int size = b_old.n_elem;
  
  // s0, s1, s2
  arma::vec s0 = arma::zeros<arma::vec>(sample);
	arma::mat s1 = arma::zeros<arma::mat>(sample, size);
  arma::cube s2 = arma::zeros<arma::cube>(size, size, sample);

  for (int j = 0; j < sample; j++) {
    double temp = exp(arma::as_scalar(X.row(j) * a_new) + psi(j) * A(j));
    for (int i = 0; i < sample; i++) {
      if ((delta(i) == 1) && (tobs(j) >= tobs(i))) {
  			s0(i) += temp;
				s1.row(i) += temp * A(j) * diff_psi(j) * X.row(j);
        s2.slice(i) += temp * A(j) * A(j) * diff_psi(j) * diff_psi(j)
                        * X.row(j).t() * X.row(j);
      }
    }
  }
	
  // create D_tilde, d_tilde to be used in the one step Newton algorithm  
  // the problem is of the format min (1/2b^tDb - d^tb) s.t. (b[9] = 0)
  // which is equivalent to min (1/2b[0:9]^tD[0:9,0:9]b[0:9] - d[0:9]^tb[0:9])
  // and b[9] = 0
  arma::mat D = arma::zeros<arma::mat>(size, size);
  arma::vec d = arma::zeros<arma::vec>(size);
  for (int i = 0; i < sample; i++) {
		if (delta[i] == 1) {
      D += s2.slice(i) / s0(i) - s1.row(i).t() * s1.row(i) / (s0(i) * s0(i)) ;
      d += A(i) * diff_psi(i) * X.row(i).t() - s1.row(i).t() / s0(i);
 		}
	}
  arma::mat D_tilde = D.submat(0, 0, 8, 8);
  arma::vec d_tilde = d.head(9);
  
  // one step Newton for the quadratic problem  
  arma::vec increase = arma::zeros<arma::vec>(size);
  increase.head(9) = solve(D_tilde, d_tilde);

  // return increase
  return increase;
}

                               
// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
double likeli(const arma::vec& a, const arma::vec& b, const arma::vec& c, 
              const arma::vec& A, const arma::mat& X, const arma::vec& delta, 
              const arma::vec& tobs, const arma::vec& knots, int order){
  // initialization
  Rcpp::Environment splines("package:splines");
  Rcpp::Function splines_des = splines["spline.des"]; 
  int sample = delta.n_elem;
  arma::vec s_tilde = arma::zeros<arma::vec>(sample);
  arma::vec der = arma::zeros<arma::vec>(sample);
  List r_bs = splines_des(knots, X*b, order, der, 1);
  arma::mat bs = as<arma::mat>(r_bs["design"]);
  arma::vec psi = bs * c; 

  // s_tilda
  for (int j = 0; j < sample; j++) {
    double temp = exp(arma::as_scalar(X.row(j) * a) + psi(j) * A(j));
    for (int i = 0; i < sample; i++) {
      if ((delta(i) == 1) && (tobs(j) >= tobs(i))) {
    		s_tilde(i) += temp;
     }
    }
  }

  // profile log-likelihood
  double lik = 0;
  for (int i = 0; i < sample; i++) {
  	if (delta(i) == 1) {
      lik += arma::as_scalar(X.row(i) * a) + psi(i) * A(i) - log(s_tilde(i));
 		}
	}

  return lik;
}


template <class T> const T& max (const T& a, const T& b) {
  return (a<b)?b:a;     
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
SEXP solvealg(arma::mat X, arma::mat X_inquantile, arma::vec t_obs, 
           arma::vec censoring, arma::vec A, int k, int max_n){
  // environmental parameters
  Rcpp::Environment splines("package:splines");
  Rcpp::Function splines_des = splines["spline.des"];  
  Rcpp::Environment base("package:base");
  Rcpp::Function sort = base["sort"];
  Rcpp::Function unique = base["unique"];
  const double thre_alpha = 1e-5;
  const double thre_beta = 1e-5;
  const double thre_gamma = 1e-5;
	const double thre_lik = 1e-5;
  int max_count = max_n; // maximum number of iterations allowed
  const int order = 3; // quadratic splines

  
  // initiation
  arma::vec a_new = arma::zeros<arma::vec>(10);
  arma::vec a_old = a_new;
  arma::vec b_new = arma::zeros<arma::vec>(10);
  b_new(9) = -1;
  arma::vec b_old = b_new;
  arma::vec c_new = arma::zeros<arma::vec>(k + 1);
  arma::vec c_old = c_new;
  double lik_old = -HUGE_VAL;
	double lik_new = -HUGE_VAL;
	double a_diff, b_diff, c_diff, lik_diff;
  int converge = 0;
	int count = 0;
  arma::vec knots(k);
  int conv_type = 0;
	
  // start the optimization algorithm
  while (converge == 0){
    count++;
    // calculate the knots
    arma::vec all_knots = X_inquantile * b_new;
    NumericVector r_sort_knots = sort(all_knots);
    NumericVector r_uniq_knots = unique(r_sort_knots);
    arma::vec uniq_knots= as<arma::vec>(r_uniq_knots);
    arma::vec k_knots(k+4);
    int length = uniq_knots.n_elem;
    k_knots[0] = uniq_knots[0];
    k_knots[1] = uniq_knots[0];
    k_knots[k+2] = uniq_knots[length - 1];
    k_knots[k+3] = k_knots[k+2];
    k_knots[k+1] = k_knots[k+2]; 
    for (int i = 2; i <= k; i++){
      k_knots[i] = uniq_knots[(i-2)*(length/(k-1))];
    }  
 
    // calculate b-spline basis function values and derivatives
    arma::vec first(k);
    first.fill(1);
    List r_bs = splines_des(k_knots, X*b_new, order);
    arma::mat bs = as<arma::mat>(r_bs["design"]);
    arma::vec interior_knots = k_knots(arma::span(2, k+1));
    List r_bs_der = splines_des(k_knots, interior_knots, order, first);
    arma::mat bs_der = as<arma::mat>(r_bs_der["design"]);    
    arma::vec first_all(X.n_rows);
    first_all.fill(1);
    List r_bs_der_all = splines_des(k_knots, X*b_new, order, first_all);
    arma::mat bs_der_all = as<arma::mat>(r_bs_der_all["design"]);    
                     
    // optimize over alpha and gamma
    arma::vec change_ac = opt_ac(a_old, c_old, bs, bs_der, A, X, 
                                 censoring, t_obs);
    a_new += change_ac.head(a_new.n_elem);
    c_new += change_ac.tail(c_new.n_elem);
  
    // optimize over beta
    // assume we know beta[9] < 0 in advance - need to modify this part
    arma::vec change_b = opt_b(a_new, b_old, c_new, bs, bs_der_all, 
                               A, X, censoring, t_obs);
    b_new += change_b;

    // calculate likelihood
    lik_new = likeli(a_new, b_new, c_new, A, X,
                     censoring, t_obs, k_knots, order);
    
    // determine convergence, update old to new
    a_diff = arma::norm(a_new - a_old) / max(arma::norm(a_old), 1e-4);
    b_diff = arma::norm(b_new - b_old) / max(arma::norm(b_old), 1e-4);
    c_diff = arma::norm(c_new - c_old) / max(arma::norm(c_old), 1e-4);
    lik_diff = (lik_new - lik_old) / max(lik_old, 1e-4);
    if((a_diff <= thre_alpha) && (b_diff <= thre_beta) && (c_diff <= thre_gamma)
       && (lik_diff <= thre_lik)){
      converge = 1;
      knots = k_knots;
      conv_type = 1;
    }
    else if(count >= max_count){
      converge = 1;
      knots = k_knots;
      conv_type = 2;

    }
    else{
      a_old = a_new;
      b_old = b_new;
      c_old = c_new;
      lik_old = lik_new;
    } 
  }
  
  return Rcpp::List::create(Rcpp::Named("a") = a_new,
                          Rcpp::Named("b") = b_new,
                          Rcpp::Named("c") = c_new,
//                          Rcpp::Named("bs") = bs,
//                          Rcpp::Named("bs_der") = bs_der,
//                          Rcpp::Named("uniq_knots") = uniq_knots,
                          Rcpp::Named("knots") = knots,
                          Rcpp::Named("likelihood") = lik_new,
                          Rcpp::Named("iteration") = count,
                          Rcpp::Named("conv_type") = conv_type
                          );
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
SEXP test_splines(arma::mat X, arma::mat X_inquantile, arma::vec beta, 
                       arma::vec knots, int dev, int order, int k){

  Rcpp::Environment splines("package:splines");
  Rcpp::Function spline_des = splines["spline.des"];  
  Rcpp::Environment base("package:base");
  Rcpp::Function sort = base["sort"];
  Rcpp::Function unique = base["unique"];
  
  arma::vec b_new = arma::zeros<arma::vec>(10);
  b_new[9] = -1;
  arma::vec all_knots = X_inquantile * b_new;
  NumericVector r_sort_knots = sort(all_knots);
  NumericVector r_uniq_knots = unique(r_sort_knots);
  arma::vec uniq_knots= as<arma::vec>(r_uniq_knots);
  arma::vec k_knots(k+4);
  int length = uniq_knots.n_elem;
  k_knots[0] = uniq_knots[0];
  k_knots[1] = uniq_knots[0];
  k_knots[k+2] = uniq_knots[length - 1];
  k_knots[k+3] = k_knots[k+2];
  k_knots[k+1] = k_knots[k+2]; 
  for (int i = 2; i <= k; i++){
    k_knots[i] = uniq_knots[(i-2)*(length/(k-1))];
  }  

// NumericMatrix r_bs = spline_des(k_knots, X*b_new, order);
  arma::vec derivatives(X.n_rows);
  derivatives.fill(dev);
  List r_bs = spline_des(k_knots, X*b_new, order, derivatives);
  arma::mat bs = as<arma::mat>(r_bs["design"]);

  return Rcpp::List::create(Rcpp::Named("k_knots") = k_knots,
                            Rcpp::Named("uniq_knots") = uniq_knots,
                            Rcpp::Named("x") = X*b_new,
                            Rcpp::Named("bs") = bs);
}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
SEXP opt_ac_test(const arma::vec& a_old, const arma::vec& c_old, 
                 const arma::mat& bs, const arma::mat& bs_der, 
                 const arma::vec& A, const arma::mat& X, 
                 const arma::vec& delta, const arma::vec& tobs){
  // initialization
  int sample = delta.n_elem;
  int size = a_old.n_elem + c_old.n_elem;
  arma::vec param(size);
  param.head(a_old.n_elem) = a_old;
  param.tail(c_old.n_elem) = c_old;
  arma::mat matrix(sample, size);
  matrix.cols(0, a_old.n_elem - 1) = X;
  for (int i = a_old.n_elem; i < size; i++){
    matrix.col(i) = bs.col(i - a_old.n_elem) % A;
  }  
  Rcpp::Environment quadprog("package:quadprog");
  Rcpp::Function solve_qp = quadprog["solve.QP"];
  
  // so, s1, s2
  arma::vec s0 = arma::zeros<arma::vec>(sample);
  arma::mat s1 = arma::zeros<arma::mat>(sample, size);
  arma::cube s2 = arma::zeros<arma::cube>(size, size, sample);
  for (int j = 0; j < sample; j++) {
    double temp = exp(arma::as_scalar(matrix.row(j) * param));
    for (int i = 0; i < sample; i++) {
      if ((delta(i) == 1) && (tobs(j) >= tobs(i))) {
        s0(i) += temp;
  			s1.row(i) += temp * matrix.row(j);
        s2.slice(i) += temp * matrix.row(j).t() * matrix.row(j);
      }
    }
  }
  
  // create D, d, A0, b0 to be used in the solve.qp 
  // the problem is of the format min (1/2b^tDb - d^tb) s.t. (A0^tb >= b0)
  arma::mat D = arma::zeros<arma::mat>(size, size);
  arma::vec d = arma::zeros<arma::vec>(size);
  arma::mat A0 = arma::zeros<arma::mat>(size, bs_der.n_rows);
  arma::vec b0 = arma::zeros<arma::vec>(bs_der.n_rows);
	for (int i = 0; i < sample; i++) {
		if (delta[i] == 1) {
      D += s2.slice(i)/s0(i) - s1.row(i).t() * s1.row(i) / (s0(i) * s0(i));
      d += matrix.row(i).t() - s1.row(i).t() / s0(i);
 		}
	}
  A0.rows(a_old.n_elem, size - 1) = bs_der.t();
  b0 = -bs_der * c_old;
  
  // call solve_qp to solve the quadratic programming problem
//  List sol = solve_qp(D, d, A0, b0);
//  arma::vec increase = as<arma::vec>(sol["solution"]);

  // return increase
//  return increase;
  return Rcpp::List::create(Rcpp::Named("D") = D,
                            Rcpp::Named("s0") = s0,
                            Rcpp::Named("s1") = s1,
                            Rcpp::Named("s2") = s2);
                           // Rcpp::Named("increase") = increase);

}


// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
SEXP opt_b_test(const arma::vec& a_new, const arma::vec& b_old, 
                const arma::vec& c_new, const arma::mat& bs, 
                const arma::mat& bs_der_all, const arma::vec& A, 
                const arma::mat& X, const arma::vec& delta, 
                const arma::vec& tobs){
  // initiation
  arma::vec psi = bs * c_new;
	arma::vec diff_psi = bs_der_all * c_new;
	int sample = delta.n_elem;
	int size = b_old.n_elem;
  Rcpp::Environment quadprog("package:quadprog");
  Rcpp::Function solve_qp = quadprog["solve.QP"];
  
  // s0, s1, s2
  arma::vec s0 = arma::zeros<arma::vec>(sample);
	arma::mat s1 = arma::zeros<arma::mat>(sample, size);
  arma::cube s2 = arma::zeros<arma::cube>(size, size, sample);

  for (int j = 0; j < sample; j++) {
    double temp = exp(arma::as_scalar(X.row(j) * a_new) + psi(j) * A(j));
    for (int i = 0; i < sample; i++) {
      if ((delta(i) == 1) && (tobs(j) >= tobs(i))) {
  			s0(i) += temp;
				s1.row(i) += temp * A(j) * diff_psi(j) * X.row(j);
        s2.slice(i) += temp * A(j) * A(j) * diff_psi(j) * diff_psi(j)
                        * X.row(j).t() * X.row(j);
      }
    }
  }
	
  // create D, d, A0, b0 to be used in the solve.qp 
  // the problem is of the format min (1/2b^tDb - d^tb) s.t. (A0^tb = b0 = 0)
  arma::mat D = arma::zeros<arma::mat>(size, size);
  arma::vec d = arma::zeros<arma::vec>(size);
  arma::vec A0 = arma::zeros<arma::vec>(size);
  double b0 = 0;
  for (int i = 0; i < sample; i++) {
		if (delta[i] == 1) {
      D += s2.slice(i) / s0(i) - s1.row(i).t() * s1.row(i) / (s0(i) * s0(i)) ;
      d += A(i) * diff_psi(i) * X.row(i).t() - s1.row(i).t() / s0(i);
 		}
	}
  A0(size - 1) = 1;
  
  // call solve_qp to solve the quadratic programming problem
//  List sol = solve_qp(D, d, A0, b0, 1); // meq = 1 <-> equality constraint
//  arma::vec increase = as<arma::vec>(sol["solution"]);

//  return increase;
  return Rcpp::List::create(Rcpp::Named("D") = D,
                            Rcpp::Named("s0") = s0,
                            Rcpp::Named("s1") = s1,
                            Rcpp::Named("s2") = s2,
                            Rcpp::Named("psi") = psi,
                            Rcpp::Named("diff_psi") = diff_psi);
                           // Rcpp::Named("increase") = increase);

}



// [[Rcpp::depends("RcppArmadillo")]]
// [[Rcpp::export]]
arma::vec opt_b_QP(const arma::vec& a_new, const arma::vec& b_old, 
                const arma::vec& c_new, const arma::mat& bs, 
                const arma::mat& bs_der_all, const arma::vec& A, 
                const arma::mat& X, const arma::vec& delta, 
                const arma::vec& tobs){
  // initiation
  arma::vec psi = bs * c_new;
  arma::vec diff_psi = bs_der_all * c_new;
  int sample = delta.n_elem;
	int size = b_old.n_elem;
  Rcpp::Environment quadprog("package:quadprog");
  Rcpp::Function solve_qp = quadprog["solve.QP"];
  
  // s0, s1, s2
  arma::vec s0 = arma::zeros<arma::vec>(sample);
	arma::mat s1 = arma::zeros<arma::mat>(sample, size);
  arma::cube s2 = arma::zeros<arma::cube>(size, size, sample);

  for (int j = 0; j < sample; j++) {
    double temp = exp(arma::as_scalar(X.row(j) * a_new) + psi(j) * A(j));
    for (int i = 0; i < sample; i++) {
      if ((delta(i) == 1) && (tobs(j) >= tobs(i))) {
  			s0(i) += temp;
				s1.row(i) += temp * A(j) * diff_psi(j) * X.row(j);
        s2.slice(i) += temp * A(j) * A(j) * diff_psi(j) * diff_psi(j)
                        * X.row(j).t() * X.row(j);
      }
    }
  }
	
  // create D, d, A0, b0 to be used in the solve.qp 
  // the problem is of the format min (1/2b^tDb - d^tb) s.t. (A0^tb = b0 = 0)
  arma::mat D = arma::zeros<arma::mat>(size, size);
  arma::vec d = arma::zeros<arma::vec>(size);
  arma::vec A0 = arma::zeros<arma::vec>(size);
  double b0 = 0;
  for (int i = 0; i < sample; i++) {
		if (delta[i] == 1) {
      D += s2.slice(i) / s0(i) - s1.row(i).t() * s1.row(i) / (s0(i) * s0(i)) ;
      d += A(i) * diff_psi(i) * X.row(i).t() - s1.row(i).t() / s0(i);
 		}
	}
  A0(size - 1) = 1;
  
  // call solve_qp to solve the quadratic programming problem
  List sol = solve_qp(D, d, A0, b0, 1); // meq = 1 <-> equality constraint
  arma::vec increase = as<arma::vec>(sol["solution"]);

  // return increase
  return increase;
}
