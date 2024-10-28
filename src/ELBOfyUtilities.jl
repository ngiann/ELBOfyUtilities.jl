module ELBOfyUtilities

    using Optim, ForwardDiff, LinearAlgebra, Distributions
    # using BlackBoxOptim

    include("updatecovariance.jl")
    
    include("getmode.jl")

    include("getcovariance.jl")

    include("geteigenvectors.jl")

    include("maximise_elbo.jl")

    include("nearestposdef.jl")

    include("numerical_KLD.jl")

    export updatecovariance
    
    export getmode, getcovariance, geteigenvectors
    
    export maximise_elbo
    
    export nearestposdef

    export numerical_KLD

end
