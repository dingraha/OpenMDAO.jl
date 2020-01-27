include("composite_vector.jl")
inputs = [VarData("x", (1, 1, 1), 2.0), VarData("y", (1, 1, 1), 3.0)]
@show inputs
ncv = NamedCompositeVector(inputs)
@show ncv

outputs = [VarData("z1", (1, 2, 3), 1.5), VarData("z2", (3, 2, 3), 8.0)]
@show outputs

ncv = NamedCompositeVector(outputs)
@show ncv

ncm = NamedCompositeMatrix(inputs, outputs)
@show ncm
@show size(ncm)
@show ncm.name2idx
