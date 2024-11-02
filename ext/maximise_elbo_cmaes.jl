#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.cmaesmaximise_elbo(elbo::T, res::Evolutionary.OptimizationResults; iterations = 1000, iteration_test = 0, rng::AbstractRNG = Xoshiro(1)) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    params = Evolutionary.minimizer(res)

    ELBOfyUtilities.cmaesmaximise_elbo(elbo, params; iterations = iterations, iteration_test = iteration_test, rng = rng)

end


#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.cmaesmaximise_elbo(elbo::T; iterations = 1000, iteration_test = 0, rng::AbstractRNG = Xoshiro(1), kwargs...) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    ELBOfyUtilities.cmaesmaximise_elbo(elbo, randn(ELBOfy.numparam(elbo));  iterations = iterations, iteration_test = iteration_test, rng = rng, )

end


#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.cmaesmaximise_elbo(elbo::T, params; iterations = 1000, iteration_test = 0, rng::AbstractRNG = Xoshiro(1)) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    @printf("Resuming optimisation starting with approximate elbo of %f\n", elbo(params))

    trackelbo, cb = get_callback_and_track_elbo_for_tracking_test_evidence(elbo, iteration_test)

    @assert(length(params) == numparam(elbo))

    Evolutionary.optimize(x-> -trackelbo(x), params, CMAES(), Evolutionary.Options(iterations = iterations, rng = rng, show_trace = true))

end

    
