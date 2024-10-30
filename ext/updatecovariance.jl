#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVIExt, param)
#-------------------------------------------------------------------------------------------------------------------------------------

    μ, Cprv, ψ = ELBOfy.getμCψ(elbo, param)

    # Σnew = getcovariance(elbo.logp, μ; minimumeigenvalue = minimumeigenvalue)

    # Vnew, = ELBOfy.eigendecomposition(Σnew)

    Vnew = geteigenvectors(elbo.logp, μ)

    elbonew = ELBOfy.ElboMVIExt(elbo.Z, elbo.D, elbo.d, elbo.S, elbo.logp, elbo.gradlogp, elbo.parallel, Vnew, Cprv)

    elbonew, [μ; zeros(elbo.D); 1.0; ψ] # set mean μ to current mean
                                        # set eigenvalues to zero, this makes the contribution of the new covariance zero
                                        # set t to 1, this retains the previous solution
end


#-------------------------------------------------------------------------------------------------------------------------------------
function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVIExt, res::Optim.OptimizationResults)
#-------------------------------------------------------------------------------------------------------------------------------------
    
    params = res.minimizer

    elbonew, paramsnew = ELBOfyUtilities.updatecovariance(elbo, params)

    res.minimizer = paramsnew
    
    return elbonew, res

end