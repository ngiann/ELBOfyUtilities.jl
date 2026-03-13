module ELBOfyExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Random, LinearAlgebra



    function ELBOfyUtilities.maximise_elbo_diagonal_nm(logp, params, S, options, rng::AbstractRNG = Random.default_rng()) 

        D = length(params)
        
        elbo = ELBOfy.elbofy_diag(logp, D, S, gradlogp = nothing, rng = rng)
        
        extparams = [params; ones(D)] # add D parameters for the diagonal covariance matrix

        paramopt = ELBOfyUtilities.maximise_elbo_nm(elbo, extparams, options)

        posterior(elbo, paramopt)

    end

    function ELBOfyUtilities.maximise_elbo_full_nm(logp, params, S, options, rng::AbstractRNG = Random.default_rng()) 

        D = length(params)

        elbo = ELBOfy.elbofy_full(logp, D, S, gradlogp = nothing, rng = rng)
        
        extparams = [params; vec(Diagonal(ones(D)))] # add parameters for the diagonal covariance matrix

        paramopt = ELBOfyUtilities.maximise_elbo_nm(elbo, extparams, options)

        posterior(elbo, paramopt)

    end

    function ELBOfyUtilities.maximise_elbo_mvi_nm(logp, params, S, options, rng::AbstractRNG = Random.default_rng()) 

        D = length(params)

        m, = ELBOfyUtilities.getmode_nm(logp, params)

        V = ELBOfyUtilities.geteigenvectors(logp, m)

        elbo = ELBOfy.elbofy_mvi(logp, V, S, gradlogp = nothing, rng = rng)
        
        extparams = [params; ones(D)] # add parameters for the "stretching" parameters

        paramopt = ELBOfyUtilities.maximise_elbo_nm(elbo, extparams, options)

        posterior(elbo, paramopt)

    end

    
end