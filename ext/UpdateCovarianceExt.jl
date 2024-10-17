module UpdateCovarianceExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy

    function ELBOfyUtilities.updatecovariance(elbo::ElboMVIExt, param)

        minimumeigenvalue = 1e-6
        
        μ, Cprv = ELBOfy.getμC(elbo, param)
    
        Σnew = getcovariance(logposterior, μ; minimumeigenvalue = minimumeigenvalue)
    
        Vnew, = ELBOfy.eigendecomposition(Σnew)
    
        elbonew = ELBOfy.ElboMVIExt(elbo.Z, elbo.D, elbo.S, elbo.logp, elbo.gradlogp, elbo.howtorun, Vnew, Cprv)
    
        elbonew, [μ; zeros(elbo.D); 1.0]
    end


end # module