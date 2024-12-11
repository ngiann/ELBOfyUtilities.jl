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