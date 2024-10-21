module ELBOfyExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Optim

    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.maximise_elbo(elbo::T, params; iterations = 1000, show_trace = true, g_tol=1e-4) where T<:ELBOfy.AbstractElbo
    #-------------------------------------------------------------------------------------------------------------------------------------

        opt = Optim.Options(show_trace = show_trace, show_every=2, iterations = iterations, allow_f_increases = false, g_tol = g_tol)

        if has_logp_gradient(elbo)

            # display("gradient opt")

            gradhelper!(st, param) = copyto!(st, -ELBOfy.grad(elbo, param))
        
            # params = optimize(x-> -elbo(x), params, ParticleSwarm(), Optim.Options(iterations=1000, show_every=100, show_trace=true)).minimizer

            return optimize(x-> -elbo(x), params, ConjugateGradient(), opt)

        end

        return optimize(x-> -elbo(x), params, NelderMead(), opt)

    end


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVIExt, param; minimumeigenvalue = 1e-6)
    #-------------------------------------------------------------------------------------------------------------------------------------
  
        μ, Cprv, ψ = ELBOfy.getμCψ(elbo, param)
    
        Σnew = getcovariance(elbo.logp, μ; minimumeigenvalue = minimumeigenvalue)
    
        Vnew, = ELBOfy.eigendecomposition(Σnew)
    
        elbonew = ELBOfy.ElboMVIExt(elbo.Z, elbo.D, elbo.d, elbo.S, elbo.logp, elbo.gradlogp, elbo.howtorun, Vnew, Cprv)
    
        elbonew, [μ; zeros(elbo.D); 1.0; ψ] # set mean μ to current mean
                                            # set eigenvalues to zero, this makes the contribution of the new covariance zero
                                            # set t to 1, this retains the previous solution
    end

end # module