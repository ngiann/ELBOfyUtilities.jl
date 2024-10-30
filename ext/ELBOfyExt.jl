module ELBOfyExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Optim, Printf, BlackBoxOptim

    include("trackElbo.jl")

    include("maximise_elbo.jl")

    include("get_callback_and_track_elbo_for_tracking_test_evidence.jl")

    include("updatecovariance.jl")

end # module