module ELBOfyExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Optim, Printf

    include("trackElbo.jl")

    include("get_callback_and_track_elbo_for_tracking_test_evidence.jl")


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.maximise_elbo(elbo::T, res::Optim.OptimizationResults; iterations = 1000, iteration_test = 0, show_trace = true, g_tol=1e-6) where T<:ELBOfy.AbstractElbo
    #-------------------------------------------------------------------------------------------------------------------------------------

        params = res.minimizer

        @printf("Resuming optimisation starting with approximate elbo of %f\n", elbo(res.minimizer))

        ELBOfyUtilities.maximise_elbo(elbo, params; iterations = iterations, iteration_test = iteration_test, show_trace = show_trace, g_tol = g_tol)

    end


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.maximise_elbo(elbo::T; iterations = 1000, iteration_test = 0, show_trace = true, g_tol=1e-6) where T<:ELBOfy.AbstractElbo
    #-------------------------------------------------------------------------------------------------------------------------------------
    
        ELBOfyUtilities.maximise_elbo(elbo, randn(ELBOfy.numparam(elbo)); iterations = iterations, iteration_test = iteration_test, show_trace = show_trace, g_tol = g_tol)

    end


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.maximise_elbo(elbo::T, params; iterations = 1000, iteration_test = 0, show_trace = true, g_tol=1e-6) where T<:ELBOfy.AbstractElbo
    #-------------------------------------------------------------------------------------------------------------------------------------
    
        trackelbo, cb = get_callback_and_track_elbo_for_tracking_test_evidence(elbo, iteration_test)

        opt = Optim.Options(callback = cb, show_trace = show_trace, show_every=2, iterations = iterations, allow_f_increases = false, g_tol = g_tol)


        if has_logp_gradient(elbo)

            gradhelper!(st, param) = copyto!(st, -ELBOfy.grad(trackelbo, param))
        
            return optimize(x-> -trackelbo(x), params, LBFGS(), opt)

        end

        return optimize(x-> -trackelbo(x), params, NelderMead(), opt)

    end

    
    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVIExt, param)
    #-------------------------------------------------------------------------------------------------------------------------------------
  
        μ, Cprv, ψ = ELBOfy.getμCψ(elbo, param)
    
        # Σnew = getcovariance(elbo.logp, μ; minimumeigenvalue = minimumeigenvalue)
    
        # Vnew, = ELBOfy.eigendecomposition(Σnew)

        Vnew = geteigenvectors(elbo.logp, μ)
    
        elbonew = ELBOfy.ElboMVIExt(elbo.Z, elbo.D, elbo.d, elbo.S, elbo.logp, elbo.gradlogp, elbo.parallel, Vnew, Cprv)
    
        elbonew, [μ; zeros(elbo.D); 1.0; ψ] # set mean μ to current mean
                                            # set eigenvalues to zero, this makes the contribution of the new covariance zero
                                            # set t to 1, this retains the previous solution
    end


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVIExt, res::Optim.OptimizationResults)
    #-------------------------------------------------------------------------------------------------------------------------------------
       
        params = res.minimizer

        elbonew, paramsnew = ELBOfyUtilities.updatecovariance(elbo, params)

        res.minimizer = paramsnew
        
        return elbonew, res

    end

end # module