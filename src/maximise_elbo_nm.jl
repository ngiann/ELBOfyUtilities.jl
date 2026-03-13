function maximise_elbo_nm(elbo, params::Vector{T}, options::Optim.Options) where T<:Real

    negative_elbo(params) = -elbo(params)

    optimize(negative_elbo, params, NelderMead(), options).minimizer

end