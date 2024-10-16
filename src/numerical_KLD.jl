function numerical_KLD(d, logp; numsamples = 10_000)

    sample = Distributions.rand(d, numsamples)

    l = [Distributions.logpdf(d, sample[:,i]) - logp(sample[:,i]) for i=1:numsamples]

    return Distributions.mean(l), Distributions.std(l)/sqrt(numsamples)

end
