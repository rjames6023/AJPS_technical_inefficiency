function [S1_avg, S2_avg] = Murphy_Diagram_scores(u, u_hat_1, u_hat_2, sigma2_u)
    n = length(u);
    %Find a range for theta
    max_tmp = sqrt(sigma2_u).*norminv((0.999+1)/2, 0, 1); %set the max theta to the 99.9th quantile of the distribution of technical inefficiency
    min_tmp = 0; %theoretical minimum of the technical inefficiency predictions
    
    tmp = [min_tmp-0.1*(max_tmp - min_tmp), max_tmp + 0.1*(max_tmp - min_tmp)];
    theta = linspace(tmp(1), tmp(2), 501);
    
    %Matrices for elementary scores at each i = 1,..,n for all theta
    S1 = zeros(501, n);
    S2 = zeros(501, n);
        %row-wise maxima
    max1 = max([u_hat_1, u], [], 2);
    max2 = max([u_hat_2, u], [], 2);
    min1 = min([u_hat_1, u], [], 2);
    min2 = min([u_hat_2, u], [], 2);
    
    for j=1:n
        S1(:,j) = abs(u(j)-theta).*(max1(j) > theta).*(min1(j) <= theta); %see pg.506 Ehm et al. 2016
        S2(:,j) = abs(u(j)-theta).*(max2(j) > theta).*(min2(j) <= theta); %see pg.506 Ehm et al. 2016
    end
    
    S1_avg = mean(S1, 2);
    S2_avg = mean(S2, 2);
end
