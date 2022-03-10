function Sigma = CholeskiToMatrix(CholOfSigma, n_inputs) 
    SigmaChol = zeros(n_inputs,n_inputs); %Construct empty covariance matrix
    SigmaChol(itril(size(SigmaChol))) = CholOfSigma; %extract and insert lower triangular cholesky decomposition
    Sigma = SigmaChol*SigmaChol'; 
end