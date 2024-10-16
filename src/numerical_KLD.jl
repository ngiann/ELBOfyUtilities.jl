function numerical_KLD(d, logp; numsamples = 10_000)

    sample = rand(d, numsamples)

    l = [logpdf(d, sample[:,i]) - logp(sample[:,i]) for i=1:numsamples]

    return mean(l), std(l)/sqrt(numsamples)

end
