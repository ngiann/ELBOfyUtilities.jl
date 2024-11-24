#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.maximise_elbo(elbo::T, res::Optim.OptimizationResults; iterations = 1000, iteration_test = 0, show_every = 1, g_tol=1e-8, Method=BFGS()) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    params = res.minimizer

    ELBOfyUtilities.maximise_elbo(elbo, params; iterations = iterations, iteration_test = iteration_test, show_every = show_every, g_tol = g_tol, Method = Method)

end


#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.maximise_elbo(elbo::T; iterations = 1000, iteration_test = 0, show_every = 1, g_tol=1e-8, Method = BFGS()) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    ELBOfyUtilities.maximise_elbo(elbo, randn(ELBOfy.numparam(elbo)); iterations = iterations, iteration_test = iteration_test, show_every = show_every, g_tol = g_tol, Method = Method)

end


#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.maximise_elbo(elbo::T, params; iterations = 1000, iteration_test = 0, show_every = 1, g_tol=1e-8, Method = BFGS()) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    @printf("Resuming optimisation starting with approximate elbo of %f\n", elbo(params))
    
    # trackelbo, cb = get_callback_and_track_elbo_for_tracking_test_evidence(elbo, iteration_test)

    opt = Optim.Options(show_trace = show_every > 0, show_every = max(show_every,1), iterations = iterations, allow_f_increases = false, g_tol = g_tol)

    g!(storage, params) = copyto!(storage, -1*Zygote.gradient(elbo, params)[1])

    # if has_logp_gradient(elbo)

    #     gradhelper!(st, param) = copyto!(st, -ELBOfy.grad(trackelbo, param))
    
    #     return optimize(x-> -trackelbo(x), params, LBFGS(), opt)

    # end

    return optimize(x-> -elbo(x), g!, params, Method, opt)

end

    
