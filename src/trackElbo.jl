# Mitigates current deficiency of Optim to keep track of 
# currect best solution accross all optimisers in a consistent manner
mutable struct ObjectiveTracker{T1,T2}
    
    objective::T1
    bestvaluesofar::T2
    bestsolutionsofar::Vector{T2}
    counter::Int64

end

function ObjectiveTracker(elbo::ELBOfy.AbstractElbo)#; S = elbo.S, rng::AbstractRNG = Xoshiro(1))

    ObjectiveTracker(elbo, -Inf, zeros(numparam(elbo)), 0)

end


function (e::ObjectiveTracker)(params) 
    
    aux = e.objective(params)

    if aux > e.bestvaluesofar
        e.bestvaluesofar = aux
        copyto!(e.bestsolutionsofar, params)
    end

    aux

end


# Counter related
resetcounter!(e::ObjectiveTracker) = e.counter = 0
getcounter(e::ObjectiveTracker) = e.counter
incrementcounter!(e::ObjectiveTracker) = e.counter += 1
