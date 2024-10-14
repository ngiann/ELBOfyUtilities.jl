module MaximiseElboExt # Should be same name as the file (just like a normal package)

    using ELBOfyUtilities, ELBOfy, Optim

    function ELBOfyUtilities.maximise_elbo(elbo::T, params; iterations = 1000, show_trace = true) where T<:ELBOfy.AbstractElbo

        optimize(x-> -elbo(x), params, Optim.Options(show_trace = show_trace, iterations = iterations))

    end

    function ELBOfyUtilities.maximise_elbo(elbo::T; iterations = 1000, show_trace = true) where T<:ELBOfy.AbstractElbo

        ELBOfyUtilities.maximise_elbo(elbo, randn(numparam(elbo)); iterations = iterations, show_trace = show_trace)

    end


end # module