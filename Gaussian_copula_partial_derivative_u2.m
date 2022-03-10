function conditional_CDF = Gaussian_copula_partial_derivative_u2(u1, u2, rho)
    conditional_CDF = normcdf((norminv(u1) - rho.*norminv(u2))./(sqrt(1-rho^2)));
end