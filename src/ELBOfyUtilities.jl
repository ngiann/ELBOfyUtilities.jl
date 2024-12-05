module ELBOfyUtilities

    using LinearAlgebra, Distributions, FiniteDiff, Random
    using Optim, BlackBoxOptim, Evolutionary
    using OnlineStats
    using Printf
    
    include("updatecovariance.jl")
    
    include("getmode.jl")

    include("getcovariance.jl")

    include("geteigenvectors.jl")

    include("maximise_elbo.jl")

    include("nearestposdef.jl")

    include("numerical_KLD.jl")

    include("convert_parameters.jl")

    export updatecovariance
    
    export getmode, getcovariance, geteigenvectors
    
    export maximise_elbo, bbmaximise_elbo, cmaesmaximise_elbo

    export getminimum, getsolution
    
    export nearestposdef

    export numerical_KLD

    export diagonal_parameters, full_parameters, mvi_parameters

end
