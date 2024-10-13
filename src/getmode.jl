function getmode(logl, x; iterations = 10000)

    opt = Optim.Options(iterations = iterations, show_trace = true, show_every = 1, g_tol=1e-10)

    # backend =  getbackend()

    # gradhelper!(storage, x) = DifferentiationInterface.gradient!(x->-logl(x), storage, backend, x)
    
    result = optimize(x -> -logl(x),  x, LBFGS(), opt, autodiff=:forward)

    Optim.minimizer(result), Optim.minimum(result)
    
end

