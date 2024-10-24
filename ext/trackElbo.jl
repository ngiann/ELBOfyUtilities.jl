# Mitigates current deficiency of Optim to keep track of 
# currect best solution accross all optimisers in a consistent manner
mutable struct trackElbo{T1<:ELBOfy.AbstractElbo,T3}
    
    elbo::T1
    bestvaluesofar::T3
    bestsolutionsofar::Vector{T3}
    counter::Int64

end

function trackElbo(elbo::ELBOfy.AbstractElbo)#; S = elbo.S, rng::AbstractRNG = Xoshiro(1))

    trackElbo(elbo, -Inf, zeros(numparam(elbo)), 0)

end


function (e::trackElbo)(params) 
    
    aux = e.elbo(params)

    if aux > e.bestvaluesofar
        e.bestvaluesofar = aux
        copyto!(e.bestsolutionsofar, params)
    end

    aux

end

testelbo(e::trackElbo, p) = ELBOfy.testelbo(e.elbo, p)

# Counter related
resetcounter!(e::trackElbo) = e.counter = 0
getcounter(e::trackElbo) = e.counter
incrementcounter(e::trackElbo) = e.counter += 1
