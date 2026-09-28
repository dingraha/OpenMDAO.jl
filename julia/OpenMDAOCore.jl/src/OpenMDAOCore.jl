module OpenMDAOCore

using ADTypes: ADTypes
using ComponentArrays: ComponentArray, ComponentVector, ComponentMatrix, getaxes, getdata
using DifferentiationInterface: DifferentiationInterface
using Random: rand!

include("utils.jl")
export get_rows_cols, ca2strdict, rcdict2strdict

include("interface.jl")
export AbstractComp, AbstractExplicitComp, AbstractImplicitComp
export has_setup_partials
export has_compute_partials, has_compute_jacvec_product
export has_apply_nonlinear, has_solve_nonlinear, has_linearize, has_apply_linear, has_solve_linear, has_guess_nonlinear 

include("var_data.jl")
export VarData

include("partials_data.jl")
export PartialsData

include("abstract_ad.jl")
export get_callback, get_input_ca, get_output_ca, get_jacobian_ca, get_units, get_backend, get_prep

include("dense_ad.jl")
export DenseADExplicitComp

include("matrix_free_ad.jl")
export MatrixFreeADExplicitComp, get_dinput_ca, get_doutput_ca

# Sparse types are declared in the main package so they can be imported and
# dispatched on without the extension loaded. The sparse *functionality*
# (constructors, `compute_partials!`, sparsity detection, etc.) is provided by
# the `OpenMDAOCoreSparseMatrixColoringsExt` extension, which loads when both
# `SparseArrays` and `SparseMatrixColorings` are available.
include("sparse_ad.jl")
export SparseADExplicitComp, get_rows_cols_dict, PerturbedDenseSparsityDetector

end # module
