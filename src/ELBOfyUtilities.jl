module ELBOfyUtilities

    using Optim, ForwardDiff, LinearAlgebra

    include("getmode.jl")

    include("getcovariance.jl")

    include("maximise_elbo.jl")

    export getmode, getcovariance, maximise_elbo

end
