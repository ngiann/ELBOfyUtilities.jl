module ELBOfyUtilities

    using Optim, LinearAlgebra, Distributions, FiniteDiff

    include("updatecovariance.jl")
    
    include("getmode.jl")

    include("getcovariance.jl")

    include("geteigenvectors.jl")

    include("maximise_elbo.jl")

    include("nearestposdef.jl")

    include("numerical_KLD.jl")

    export updatecovariance
    
    export getmode, getcovariance, geteigenvectors
    
    export maximise_elbo, bbmaximise_elbo, cmaesmaximise_elbo

    export getminimum, getsolution
    
    export nearestposdef

    export numerical_KLD

end
