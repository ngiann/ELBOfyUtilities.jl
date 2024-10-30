#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.bbmaximise_elbo(elbo::T, res::BlackBoxOptim.OptimizationResults; iterations = 1000, iteration_test = 0) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    params = best_candidate(res)

    @printf("Resuming optimisation starting with approximate elbo of %f\n", elbo(params))

    ELBOfyUtilities.bbmaximise_elbo(elbo, params; Method = Symbol(res.method), iterations = iterations, iteration_test = iteration_test)

end


#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.bbmaximise_elbo(elbo::T; Method = :adaptive_de_rand_1_bin_radiuslimited, bound = 10.0, iterations = 1000, iteration_test = 0) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    ELBOfyUtilities.bbmaximise_elbo(elbo, randn(ELBOfy.numparam(elbo)); Method = Method, bound = bound, iterations = iterations, iteration_test = iteration_test)

end


#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.bbmaximise_elbo(elbo::T, params; Method = :adaptive_de_rand_1_bin_radiuslimited, bound = maximum(abs.(params)), iterations = 1000, iteration_test = 0) where T<:ELBOfy.AbstractElbo
#-------------------------------------------------------------------------------------------------------------------------------------

    trackelbo, cb = get_callback_and_track_elbo_for_tracking_test_evidence(elbo, iteration_test)

    @assert(length(params) == numparam(elbo))

    bboptimize(x-> -trackelbo(x), params; Method = Method, SearchRange = (-abs(bound), abs(bound)), NumDimensions = numparam(elbo), MaxFuncEvals = iterations)

end

    
