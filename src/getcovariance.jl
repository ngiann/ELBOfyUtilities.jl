function getcovariance(logposterior, mode)

   H = -ForwardDiff.hessian(logposterior, mode)

   Symmetric(inv(H))

end