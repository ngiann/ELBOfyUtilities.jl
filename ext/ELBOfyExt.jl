module ELBOfyExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Optim, Printf
    using BlackBoxOptim

    ELBOfyUtilities.getsolution(res::Vector) = res
  
    ELBOfyUtilities.getminimum(res::BlackBoxOptim.OptimizationResults) = BlackBoxOptim.best_fitness(res)
    ELBOfyUtilities.getsolution(res::BlackBoxOptim.OptimizationResults) = BlackBoxOptim.best_candidate(res)
  
    ELBOfyUtilities.getminimum(res::Optim.OptimizationResults) = Optim.minimum(res)
    ELBOfyUtilities.getsolution(res::Optim.OptimizationResults) = Optim.minimizer(res)   

    include("trackElbo.jl")

    include("maximise_elbo.jl")

    include("maximise_elbo_blackboxoptim.jl")

    include("get_callback_and_track_elbo_for_tracking_test_evidence.jl")

    include("updatecovariance.jl")

end # module