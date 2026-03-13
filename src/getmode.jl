function getmode_lbfgs(logl, x; iterations = 1_000_000, show_trace = false)

    opt = Optim.Options(iterations = iterations, show_trace = show_trace, show_every = 1, g_tol=1e-10)

    result = optimize(x -> -logl(x),  x, LBFGS(), opt, autodiff=:forward)

    Optim.minimizer(result), Optim.minimum(result)
    
end


function getmode_nm(logl, x; iterations = 1_000_000, show_trace = false)

    opt = Optim.Options(iterations = iterations, show_trace = show_trace, show_every = 1, g_tol=1e-10)

    result = optimize(x -> -logl(x),  x, NelderMead(), opt)

    Optim.minimizer(result), Optim.minimum(result)
    
end
