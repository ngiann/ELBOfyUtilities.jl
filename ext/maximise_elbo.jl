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

    
