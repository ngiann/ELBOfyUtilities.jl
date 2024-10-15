function getcovariance(logposterior, mode; minimumeigenvalue = 1e-6)

   H = -ForwardDiff.hessian(logposterior, mode)

   inv(nearestposdef(H; minimumeigenvalue = minimumeigenvalue))

end