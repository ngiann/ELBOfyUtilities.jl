module ELBOfyExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Optim, Printf

    include("trackElbo.jl")


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.maximise_elbo(elbo::T, res::Optim.OptimizationResults; iterations = 1000, iteration_test = 0, show_trace = true, g_tol=1e-4) where T<:ELBOfy.AbstractElbo
    #-------------------------------------------------------------------------------------------------------------------------------------

        params = res.minimizer

        @printf("Resuming optimisation starting with approximate elbo of %f\n", elbo(res.minimizer))

        ELBOfyUtilities.maximise_elbo(elbo, params; iterations = iterations, iteration_test = iteration_test, show_trace = show_trace, g_tol = g_tol)

    end


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.maximise_elbo(elbo::T; iterations = 1000, iteration_test = 0, show_trace = true, g_tol=1e-4) where T<:ELBOfy.AbstractElbo
    #-------------------------------------------------------------------------------------------------------------------------------------
    
        ELBOfyUtilities.maximise_elbo(elbo, randn(ELBOfy.numparam(elbo)); iterations = iterations, iteration_test = iteration_test, show_trace = show_trace, g_tol = g_tol)

    end


    #-------------------------------------------------------------------------------------------------------------------------------------
    function ELBOfyUtilities.maximise_elbo(elbo::T, params; iterations = 1000, iteration_test = 0, show_trace = true, g_tol=1e-4) where T<:ELBOfy.AbstractElbo
    #-------------------------------------------------------------------------------------------------------------------------------------

        trackelbo = trackElbo(elbo)

        function cb(_)

            incrementcounter(trackelbo)
            
            if iteration_test > 0 && mod(getcounter(trackelbo), iteration_test) == 1

                testlogevidence = testelbo(trackelbo, trackelbo.bestsolutionsofar)
                
                @printf("\t Test evidence is %f\n", testlogevidence)

            end
            
            false
            
        end


        opt = Optim.Options(callback = cb, show_trace = show_trace, show_every=2, iterations = iterations, allow_f_increases = false, g_tol = g_tol)


        if has_logp_gradient(elbo)

            # display("gradient opt")

            gradhelper!(st, param) = copyto!(st, -ELBOfy.grad(trackelbo, param))
        
            # params = optimize(x-> -elbo(x), params, ParticleSwarm(), Optim.Options(iterations=1000, show_every=100, show_trace=true)).minimizer

            return optimize(x-> -trackelbo(x), params, ConjugateGradient(), opt)

        end

        return optimize(x-> -trackelbo(x), params, NelderMead(), opt)

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