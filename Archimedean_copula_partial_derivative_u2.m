function conditional_CDF = Archimedean_copula_partial_derivative_u2(copula, u1, u2, theta_param)
    if strcmp(copula, 'Clayton')
        conditional_CDF = 1./(u2.^(theta_param + 1).*(1./u1.^theta_param + 1./u2.^theta_param - 1).^(1/theta_param + 1));
    elseif strcmp(copula, 'Gumbel')
        conditional_CDF = (exp(-((-log(u2)).^theta - log(u1).^theta).^(1/theta)).*(-log(u2)).^(theta - 1).*((-log(u2)).^theta - log(u1).^theta).^(1/theta - 1))./u2;
    end
end