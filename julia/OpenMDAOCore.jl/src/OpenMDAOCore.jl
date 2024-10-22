module OpenMDAOCore

using ADTypes: ADTypes
using ComponentArrays: ComponentArray, ComponentVector, ComponentMatrix, getaxes, getdata
using DifferentiationInterface: DifferentiationInterface
using Random: rand!
using SparseArrays: sparse, findnz, nonzeros, AbstractSparseArray
using SparseMatrixColorings: SparseMatrixColorings

include("utils.jl")
export get_rows_cols, get_rows_cols_dict_from_sparsity, ca2strdict, ca2strdict_sparse, rcdict2strdict, PerturbedDenseSparsityDetector

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

include("sparse_ad.jl")
export SparseADExplicitComp, get_rows_cols_dict

include("matrix_free_ad.jl")
export MatrixFreeADExplicitComp, get_dinput_ca, get_doutput_ca

end # module
