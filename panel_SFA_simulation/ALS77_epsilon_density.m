function den = ALS77_epsilon_density(eps, sigma, lambda)
    den = (2/sigma)*normpdf(eps./sigma).*(1 - normcdf((eps.*lambda)./sigma));
end