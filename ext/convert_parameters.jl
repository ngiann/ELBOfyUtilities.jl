########
# DIAG #
########

function ELBOfyUtilities.diagonal_parameters(elbo::T, p) where T<:ELBOfy.ElboSphere

    D = elbo.D

    @assert(length(p) == D + 1) # D parameters for the mean
                                # and 1 parameter for the spherical covariance matrix

    newp = [p[1:D]; p[D+1]*ones(D)]

    @assert(length(newp) == 2D)

    return newp
    
end


# function ELBOfyUtilities.diagonal_parameters(elbo::ELBOfy.ElboMixture{T}, p) where T<:ELBOfy.ElboSphere

#     K = length(elbo)
    
#     np = numparam(elbo.comp[1]) # assume all components are of the same type!
    
#     logω = p[1:K] 

#     return [logω; reduce(vcat, [diagonal_parameters(elbo.comp[k], p[(K + 1 + (k-1)*np):(K + k*np)]) for k in 1:K])]

# end


# function ELBOfyUtilities.diagonal_parameters(elbo::ELBOfy.ElboUniformMixture{T}, p) where T<:ELBOfy.ElboSphere

#     K = length(elbo)
    
#     np = numparam(elbo.comp[1]) # assume all components are of the same type!
    
#     logω = log.(ones(K)/K) 

#     return [logω; reduce(vcat, [diagonal_parameters(elbo.comp[k], p[(1 + (k-1)*np):(k*np)]) for k in 1:K])]

# end



########
# FULL #
########

function ELBOfyUtilities.full_parameters(elbo::T, p) where T<:ELBOfy.ElboSphere

    D = elbo.D

    @assert(length(p) == D + 1) # D parameters for the mean
                                # and 1 parameter for the spherical covariance matrix

    newp = [p[1:D]; p[D+1] * vec(1.0 * Matrix(I, D, D))]

    @assert(length(newp) == D + D*D)

    return newp
    
end


# function ELBOfyUtilities.full_parameters(elbo::ELBOfy.ElboMixture{T}, p) where T<:ELBOfy.ElboSphere

#     K = length(elbo)
    
#     np = numparam(elbo.comp[1]) # assume all components are of the same type!
    
#     logω = p[1:K] 

#     return [logω; reduce(vcat, [full_parameters(elbo.comp[k], p[(K + 1 + (k-1)*np):(K + k*np)]) for k in 1:K])]

# end


# function ELBOfyUtilities.full_parameters(elbo::ELBOfy.ElboUniformMixture{T}, p) where T<:ELBOfy.ElboSphere

#     K = length(elbo)
    
#     np = numparam(elbo.comp[1]) # assume all components are of the same type!
    
#     logω = log.(ones(K)/K) 

#     return [logω; reduce(vcat, [full_parameters(elbo.comp[k], p[(1 + (k-1)*np):(k*np)]) for k in 1:K])]

# end


#######
# MVI #
#######

function ELBOfyUtilities.mvi_parameters(elbo::T, p) where T<:ELBOfy.ElboSphere

    D = elbo.D

    @assert(length(p) == D + 1) # D parameters for the mean
                                # and 1 parameter for the spherical covariance matrix

    newp = [p[1:D]; p[D+1]*ones(D); 1]

    @assert(length(newp) == 2D + 1)

    return newp
    
end

# function ELBOfyUtilities.mvi_parameters(elbo::ELBOfy.ElboMixture{T}, p) where T<:ELBOfy.ElboSphere

#     K = length(elbo)
    
#     np = numparam(elbo.comp[1]) # assume all components are of the same type!
    
#     logω = p[1:K] 

#     return [logω; reduce(vcat, [mvi_parameters(elbo.comp[k], p[(K + 1 + (k-1)*np):(K + k*np)]) for k in 1:K])]

# end


# function ELBOfyUtilities.mvi_parameters(elbo::ELBOfy.ElboUniformMixture{T}, p) where T<:ELBOfy.ElboSphere

#     K = length(elbo)
    
#     np = numparam(elbo.comp[1]) # assume all components are of the same type!
    
#     logω = log.(ones(K)/K)  

#     return [logω; reduce(vcat, [mvi_parameters(elbo.comp[k], p[(1 + (k-1)*np):(k*np)]) for k in 1:K])]

# end