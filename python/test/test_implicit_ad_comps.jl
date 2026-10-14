using ADTypes: ADTypes
using ComponentArrays: ComponentVector
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using SparseMatrixColorings: SparseMatrixColorings
using SparseArrays: SparseArrays
using OpenMDAOCore: OpenMDAOCore

# Implicit system used for the end-to-end tests:
#
#     R.z1[i] = a*x^2 + y[i]^2 - z1[i]
#     R.z2    = a*x + sum(y) - z2
#
# The exact solution is
#
#     z1[i] = a*x^2 + y[i]^2
#     z2    = a*x + sum(y)
#
# so the states, the partial derivatives, and the total derivatives are all
# known analytically.

function f_implicit_ad!(R, Y, X, params)
    a = params.a
    x = only(X[:x])
    y = @view X[:y]
    z1 = @view Y[:z1]
    z2 = only(Y[:z2])
    r_z1 = @view R[:z1]

    @. r_z1 = a * x^2 + y^2 - z1
    # Note: `z2` is a scalar component, so `R[:z2]` returns a scalar; use plain
    # assignment instead of broadcasting.
    R[:z2] = a * x + sum(y) - z2

    return nothing
end

function f_implicit_ad(Y, X, params)
    a = params.a
    x = only(X[:x])
    y = X[:y]
    z1 = Y[:z1]
    z2 = only(Y[:z2])

    R = ComponentVector(z1=a * x^2 .+ y .^ 2 .- z1, z2=a * x + sum(y) - z2)
    return R
end

# Residual for the shape_by_conn test component:
#
#     R.z[i] = scale*y[i]^2 + a - z[i]
#
# where `scale` and `a` are component parameters. The exact solution is
# z = scale*y.^2 .+ a.
function f_implicit_sbc!(R, Y, X, params)
    a = params.a
    y = @view X[:y]
    z = @view Y[:z]
    r_z = @view R[:z]

    @. r_z = params.scale * y^2 + a - z

    return nothing
end

function get_dense_implicit_test_comp(; n, a, in_place)
    ad_backend = ADTypes.AutoForwardDiff()
    Y_ca = ComponentVector(z1=zeros(n), z2=0.0)
    X_ca = ComponentVector(x=2.0, y=ones(n))
    params = (a=a,)
    if in_place
        return OpenMDAOCore.create_implicit_component(
            OpenMDAOCore.DenseFlavor(), Val(true), ad_backend, f_implicit_ad!, Y_ca, X_ca; params)
    else
        return OpenMDAOCore.create_implicit_component(
            OpenMDAOCore.DenseFlavor(), Val(false), ad_backend, f_implicit_ad, Y_ca, X_ca; params)
    end
end

function get_sparse_implicit_test_comp(; n, a)
    sparse_atol = 1e-10
    sparsity_detector = OpenMDAOCore.PerturbedDenseSparsityDetector(
        ADTypes.AutoForwardDiff(); atol=sparse_atol, method=:direct)
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    ad_backend = ADTypes.AutoSparse(ADTypes.AutoForwardDiff();
                                    sparsity_detector=sparsity_detector,
                                    coloring_algorithm=coloring_algorithm)
    Y_ca = ComponentVector(z1=zeros(n), z2=0.0)
    X_ca = ComponentVector(x=2.0, y=ones(n))
    return OpenMDAOCore.create_implicit_component(
        OpenMDAOCore.SparseFlavor(), Val(true), ad_backend, f_implicit_ad!, Y_ca, X_ca; params=(a=a,))
end

function get_matrix_free_implicit_test_comp(; n, a, reverse)
    Y_ca = ComponentVector(z1=zeros(n), z2=0.0)
    X_ca = ComponentVector(x=2.0, y=ones(n))
    if reverse
        ad_backend = ADTypes.AutoReverseDiff()
        flavor = OpenMDAOCore.MatrixFreeReverseFlavor()
    else
        ad_backend = ADTypes.AutoForwardDiff()
        flavor = OpenMDAOCore.MatrixFreeForwardFlavor()
    end
    return OpenMDAOCore.create_implicit_component(
        flavor, Val(true), ad_backend, f_implicit_ad!, Y_ca, X_ca; params=(a=a,))
end

function get_matrix_free_implicit_fwd_rev_test_comp(; n, a)
    # Forward+reverse matrix-free flavor: JVPs via AutoForwardDiff, VJPs via
    # AutoReverseDiff. Both `apply_linear` directions are supported.
    Y_ca = ComponentVector(z1=zeros(n), z2=0.0)
    X_ca = ComponentVector(x=2.0, y=ones(n))
    return OpenMDAOCore.create_implicit_component(
        OpenMDAOCore.MatrixFreeForwardReverseFlavor(), Val(true),
        ADTypes.AutoForwardDiff(), ADTypes.AutoReverseDiff(), f_implicit_ad!, Y_ca, X_ca;
        params=(a=a,))
end

function get_dense_implicit_units_tags_test_comp(; n, a)
    # `z2` and `y` are deliberately left out of the dicts so the defaults
    # ("unitless" units and no tags) are exercised too.
    units_dict = Dict(:x=>"m", :z1=>"kg")
    tags_dict = Dict(:x=>["tag_x"], :z1=>["tag_z1a", "tag_z1b"])
    ad_backend = ADTypes.AutoForwardDiff()
    Y_ca = ComponentVector(z1=zeros(n), z2=0.0)
    X_ca = ComponentVector(x=2.0, y=ones(n))
    return OpenMDAOCore.create_implicit_component(
        OpenMDAOCore.DenseFlavor(), Val(true), ad_backend, f_implicit_ad!, Y_ca, X_ca;
        params=(a=a,), units_dict=units_dict, tags_dict=tags_dict)
end

function get_dense_implicit_shape_by_conn_test_comp(; n, a, scale)
    ad_backend = ADTypes.AutoForwardDiff()
    # `y` (input) and `z` (state) get their shapes from connections; the sizes
    # here are placeholders that `setup_partials`/`_update_prep` will replace.
    Y_ca = ComponentVector(z=0.0)
    X_ca = ComponentVector(y=0.0)
    shape_by_conn_dict = Dict(:y=>true)
    copy_shape_dict = Dict(:z=>:y)
    return OpenMDAOCore.create_implicit_component(
        OpenMDAOCore.DenseFlavor(), Val(true), ad_backend, f_implicit_sbc!, Y_ca, X_ca;
        params=(a=a, scale=scale), shape_by_conn_dict=shape_by_conn_dict,
        copy_shape_dict=copy_shape_dict)
end
