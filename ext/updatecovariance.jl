#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVI, param::Vector)
#-------------------------------------------------------------------------------------------------------------------------------------

    μ, Cprv = ELBOfy.getμC(elbo, param)

    # Σnew = getcovariance(elbo.logp, μ; minimumeigenvalue = minimumeigenvalue)

    # Vnew, = ELBOfy.eigendecomposition(Σnew)

    Vnew = geteigenvectors(elbo.logp, μ)

    elbonew = ELBOfy.ElboMVI(elbo.Z, elbo.D, elbo.S, elbo.logp, 
                            elbo.gradlogp, Vnew, copy(Cprv), # <--- copy to avoid aliasing
                            ELBOfy.create_elbo_mvi_buffer(D, elbo.gradlogp)) # <--- create new buffer to avoid aliasing

    elbonew, [μ; zeros(elbo.D); 1.0] # set mean μ to current mean
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