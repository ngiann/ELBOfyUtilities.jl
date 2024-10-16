module ELBOfyUtilities

    using Optim, ForwardDiff, LinearAlgebra

    
    include("getmode.jl")

    include("getcovariance.jl")

    include("maximise_elbo.jl")

    include("nearestposdef.jl")

    include("numerical_KLD.jl")

    
    export getmode, getcovariance
    
    export maximise_elbo
    
    export nearestposdef

    export numerical_KLD

end
