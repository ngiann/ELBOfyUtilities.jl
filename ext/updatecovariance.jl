#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVI, param::Vector)
#-------------------------------------------------------------------------------------------------------------------------------------

    μ, Esqrt, t = ELBOfy.unpack(elbo, param)

    # Vold = ELBOfy.interpolateV(elbo, t)
    Vold = zeros(elbo.D, elbo.D)
    elbo.SOinterp(Vold, t)
    
    Veig = geteigenvectors(elbo.logp, μ)

    ELBOfy.makespecialorthogonal!(Veig)
   
    # # Dflip is an orthogonal matrix representing a reflection or inversion.
    # # Its role is to make Veig a proper orthogonal matrix with determinant equal to 1.
    # # We only change the orientation of the basis.
    # # The transformation flips the overall orientation to match a positively oriented space.

    # Dflip = Diagonal(ones(elbo.D))
    # if det(Veig)<0 
    #     Dflip[elbo.D, elbo.D] = -1 # any diagonal element will do
    # end
    # Veig = Veig*Dflip


    # # need the 'real' below because occassionaly due to numerics tiny imaginary values may occur
    # # Theoretically, logR is a real skewed symmetric matrix
    # logR = real.(log(Veig'*Vold)) 

    elbonew = ELBOfy.ElboMVI(elbo.Z, elbo.D, elbo.S, elbo.logp, 
                            elbo.gradlogp, Veig,
                            ELBOfy.create_elbo_mvi_buffer(elbo.D, elbo.gradlogp), # <--- create new buffer to avoid aliasing
                            ELBOfy.create_interpolator(Veig, Vold))
    

    elbonew, [μ; Esqrt; 1] # set mean μ to current mean
                           # set eigenvalues to zero, this makes the contribution of the new covariance zero
                           # set t to 1, this retains the previous solution
end


ELBOfyUtilities.updatecovariance(elbo, res::Optim.OptimizationResults) = ELBOfyUtilities.updatecovariance(elbo, getsolution(res))

ELBOfyUtilities.updatecovariance(elbo, res::BlackBoxOptim.OptimizationResults) = ELBOfyUtilities.updatecovariance(elbo, getsolution(res))


# #-------------------------------------------------------------------------------------------------------------------------------------
# function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboSkewDiagExt, param::Vector)
# #-------------------------------------------------------------------------------------------------------------------------------------

#     μ, Cprv, δ, ψ = ELBOfy.getμCδψ(elbo, param)

#     Vnew = geteigenvectors(elbo.logp, μ)

#     elbonew = ELBOfy.ElboSkewDiagExt(elbo.Z, elbo.D, elbo.d, elbo.S, elbo.logp, elbo.gradlogp, elbo.parallel, Vnew, Cprv)

#     elbonew, [μ; zeros(elbo.D); δ; 1.0; ψ] # set mean μ to current mean
#                                            # set eigenvalues to zero, this makes the contribution of the new covariance zero
#                                            # set t to 1, this retains the previous solution
# end
    

# #-------------------------------------------------------------------------------------------------------------------------------------
# function ELBOfyUtilities.updatecovariance(mix::ELBOfy.ElboMixture{T}, param::Vector) where T<:ELBOfy.ElboMVIExt
# #-------------------------------------------------------------------------------------------------------------------------------------

#     K = length(mix)

#     ω, p = ELBOfy.unpack(mix, param) 

#     elboarray = map(1:K) do k

#         ELBOfyUtilities.updatecovariance(mix.comp[k], p[k])

#     end

#     newparams = [log.(ω); reduce(vcat, [e[2] for e in elboarray])]

#     return ELBOfy.ElboMixture([e[1] for e in elboarray]), newparams


# end


# #-------------------------------------------------------------------------------------------------------------------------------------
# function ELBOfyUtilities.updatecovariance(mix, param::Vector)
# #-------------------------------------------------------------------------------------------------------------------------------------

#     @printf("Nothing to update, returning arguments.")

#     mix, param

# end