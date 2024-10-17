module ELBOfyExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Optim

    function ELBOfyUtilities.maximise_elbo(elbo::T, params; iterations = 1000, show_trace = true, g_tol=1e-4) where T<:ELBOfy.AbstractElbo

        opt = Optim.Options(show_trace = show_trace, iterations = iterations, allow_f_increases = false, g_tol = g_tol)

        if has_logp_gradient(elbo)

            gradhelper!(st, param) = copyto!(st, -ELBOfy.grad(elbo, param))
        
            return optimize(x-> -elbo(x), params, ConjugateGradient(), opt)

        end

        return optimize(x-> -elbo(x), params, NelderMead(), opt)

    end

    function ELBOfyUtilities.maximise_elbo(elbo::T; iterations = 1000, show_trace = true, g_tol = 1e-4) where T<:ELBOfy.AbstractElbo

        ELBOfyUtilities.maximise_elbo(elbo, randn(numparam(elbo)); iterations = iterations, show_trace = show_trace, g_tol = g_tol)

    end


    function ELBOfyUtilities.updatecovariance(elbo::ELBOfy.ElboMVIExt, param; minimumeigenvalue = 1e-6)
        
        μ, Cprv = ELBOfy.getμC(elbo, param)
    
        Σnew = getcovariance(elbo.logp, μ; minimumeigenvalue = minimumeigenvalue)
    
        Vnew, = ELBOfy.eigendecomposition(Σnew)
    
        elbonew = ELBOfy.ElboMVIExt(elbo.Z, elbo.D, elbo.S, elbo.logp, elbo.gradlogp, elbo.howtorun, Vnew, Cprv)
    
        elbonew, [μ; zeros(elbo.D); 1.0]
    end

end # module