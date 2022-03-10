function CholOfSigma = MatrixToCholeski(Sigma)
SigmaChol = chol(Sigma)'; % is PD and SigmaChol*SigmaChol'=Sigma
CholOfSigma = SigmaChol(itril(size(SigmaChol)))'; %elements of lower triangle of the choleski
end