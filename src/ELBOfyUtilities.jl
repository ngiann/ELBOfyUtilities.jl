module ELBOfyUtilities

    using Distributions,LinearAlgebra, OnlineStats, Random
    using Optim, BlackBoxOptim, Evolutionary
    using ForwardDiff
    using Printf
    
    include("getmode.jl")

    include("getcovariance.jl")

    include("geteigenvectors.jl")

    include("maximise_elbo_nm.jl")

    include("nearestposdef.jl")

    include("numerical_KLD.jl")

    include("convert_parameters.jl")
    
    maximise_elbo_diagonal_nm(_...) = error("ELBOfy must be independently loaded!")
    maximise_elbo_full_nm(_...) = error("ELBOfy must be independently loaded!")
    maximise_elbo_mvi_nm(_...) = error("ELBOfy must be independently loaded!")

    export maximise_elbo_diagonal_nm, maximise_elbo_full_nm, maximise_elbo_mvi_nm
    export numerical_KLD
end
