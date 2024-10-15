"""
    nearestposdef(A; minimumeigenvalue = 1e-6)

Nearest positive definite matrix in Frobenious norm.
Also called the projection of the matrix onto the cone of positive definite matrices.
See: https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/
```
"""
function nearestposdef(A; minimumeigenvalue = 1e-6)

    local Evalues, Evector = eigen(A)

    local newA = Evector * Diagonal(max.(Evalues, minimumeigenvalue)) * Evector'

    Symmetric(newA)

end