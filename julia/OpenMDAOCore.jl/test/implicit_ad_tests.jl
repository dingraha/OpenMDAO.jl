@testitem "ADImplicitComp DenseFlavor in-place" begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays
    using ForwardDiff
    using Test

    # Simple implicit function: R = Y^2 - X = 0, so Y = sqrt(X)
    function f_implicit!(R, Y, X, params)
        # Note: `y` is a scalar component, so `R[:y]` returns a scalar, not a
        # 0-dimensional array; use plain assignment instead of broadcasting.
        R[:y] = only(Y[:y])^2 - only(X[:x])
        return nothing
    end

    # Create ComponentVectors
    Y_ca = ComponentVector(y=1.0)
    X_ca = ComponentVector(x=4.0)

    # Create ADImplicitComp with DenseFlavor
    comp = ADImplicitComp(DenseFlavor(), Val(true), ADTypes.AutoForwardDiff(), f_implicit!, Y_ca, X_ca)

    # Test apply_nonlinear!
    inputs = Dict("x" => 4.0)
    outputs = Dict("y" => 2.0)
    residuals = Dict("y" => 0.0)
    
    OpenMDAOCore.apply_nonlinear!(comp, inputs, outputs, residuals)
    @test residuals["y"] ≈ 0.0 atol=1e-12

    # Test with a different output value
    outputs["y"] = 3.0
    OpenMDAOCore.apply_nonlinear!(comp, inputs, outputs, residuals)
    @test residuals["y"] ≈ 5.0 atol=1e-12

    # Test linearize!
    # Pre-populate the partials dict: `_scatter_implicit_partials!` only writes
    # to keys that are already present.
    partials = Dict(("y", "y") => zeros(1, 1), ("y", "x") => zeros(1, 1))
    OpenMDAOCore.linearize!(comp, inputs, outputs, partials)
    
    # For R = Y^2 - X, we have:
    # dR/dY = 2*Y = 2*3.0 = 6.0
    # dR/dX = -1.0
    @test only(partials["y", "y"]) ≈ 6.0 atol=1e-12
    @test only(partials["y", "x"]) ≈ -1.0 atol=1e-12
end

@testitem "ADImplicitComp DenseFlavor out-of-place" begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays
    using ForwardDiff
    using Test

    # Simple implicit function: R = Y^2 - X = 0, so Y = sqrt(X)
    function f_implicit(Y, X, params)
        R = ComponentVector(y=only(Y[:y])^2 - only(X[:x]))
        return R
    end

    # Create ComponentVectors
    Y_ca = ComponentVector(y=1.0)
    X_ca = ComponentVector(x=4.0)

    # Create ADImplicitComp with DenseFlavor
    comp = ADImplicitComp(DenseFlavor(), Val(false), ADTypes.AutoForwardDiff(), f_implicit, Y_ca, X_ca)

    # Test apply_nonlinear!
    inputs = Dict("x" => 4.0)
    outputs = Dict("y" => 2.0)
    residuals = Dict("y" => 0.0)
    
    OpenMDAOCore.apply_nonlinear!(comp, inputs, outputs, residuals)
    @test residuals["y"] ≈ 0.0 atol=1e-12

    # Test with a different output value
    outputs["y"] = 3.0
    OpenMDAOCore.apply_nonlinear!(comp, inputs, outputs, residuals)
    @test residuals["y"] ≈ 5.0 atol=1e-12

    # Test linearize!
    # Pre-populate the partials dict: `_scatter_implicit_partials!` only writes
    # to keys that are already present.
    partials = Dict(("y", "y") => zeros(1, 1), ("y", "x") => zeros(1, 1))
    OpenMDAOCore.linearize!(comp, inputs, outputs, partials)
    
    # For R = Y^2 - X, we have:
    # dR/dY = 2*Y = 2*3.0 = 6.0
    # dR/dX = -1.0
    @test only(partials["y", "y"]) ≈ 6.0 atol=1e-12
    @test only(partials["y", "x"]) ≈ -1.0 atol=1e-12
end

@testitem "ADImplicitComp MatrixFreeForwardFlavor" begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays
    using ForwardDiff
    using Test

    # Simple implicit function: R = Y^2 - X = 0, so Y = sqrt(X)
    function f_implicit!(R, Y, X, params)
        # Note: `y` is a scalar component, so `R[:y]` returns a scalar, not a
        # 0-dimensional array; use plain assignment instead of broadcasting.
        R[:y] = only(Y[:y])^2 - only(X[:x])
        return nothing
    end

    # Create ComponentVectors
    Y_ca = ComponentVector(y=1.0)
    X_ca = ComponentVector(x=4.0)

    # Create ADImplicitComp with MatrixFreeForwardFlavor
    comp = ADImplicitComp(MatrixFreeForwardFlavor(), Val(true), ADTypes.AutoForwardDiff(), f_implicit!, Y_ca, X_ca)

    # Test apply_nonlinear!
    inputs = Dict("x" => 4.0)
    outputs = Dict("y" => 2.0)
    residuals = Dict("y" => 0.0)
    
    OpenMDAOCore.apply_nonlinear!(comp, inputs, outputs, residuals)
    @test residuals["y"] ≈ 0.0 atol=1e-12

    # Test apply_linear! (JVP)
    d_inputs = Dict("x" => 1.0)
    d_outputs = Dict("y" => 0.0)
    d_residuals = Dict("y" => 0.0)
    
    OpenMDAOCore.apply_linear!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, "fwd")
    # For R = Y^2 - X, dR = 2*Y*dY - dX
    # With dY = 0.0 and dX = 1.0, we get dR = -1.0
    @test d_residuals["y"] ≈ -1.0 atol=1e-12

    # Test with dY = 1.0
    d_outputs["y"] = 1.0
    d_residuals["y"] = 0.0
    OpenMDAOCore.apply_linear!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, "fwd")
    # dR = 2*Y*dY - dX = 2*2.0*1.0 - 1.0 = 3.0
    @test d_residuals["y"] ≈ 3.0 atol=1e-12
end

@testitem "ADImplicitComp MatrixFreeReverseFlavor" begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays
    using ReverseDiff
    using Test

    # Simple implicit function: R = Y^2 - X = 0, so Y = sqrt(X)
    function f_implicit!(R, Y, X, params)
        # Note: `y` is a scalar component, so `R[:y]` returns a scalar, not a
        # 0-dimensional array; use plain assignment instead of broadcasting.
        R[:y] = only(Y[:y])^2 - only(X[:x])
        return nothing
    end

    # Create ComponentVectors
    Y_ca = ComponentVector(y=1.0)
    X_ca = ComponentVector(x=4.0)

    # Create ADImplicitComp with MatrixFreeReverseFlavor
    comp = ADImplicitComp(MatrixFreeReverseFlavor(), Val(true), ADTypes.AutoReverseDiff(), f_implicit!, Y_ca, X_ca)

    # Test apply_nonlinear!
    inputs = Dict("x" => 4.0)
    outputs = Dict("y" => 2.0)
    residuals = Dict("y" => 0.0)
    
    OpenMDAOCore.apply_nonlinear!(comp, inputs, outputs, residuals)
    @test residuals["y"] ≈ 0.0 atol=1e-12

    # Test apply_linear! (VJP)
    d_inputs = Dict("x" => 0.0)
    d_outputs = Dict("y" => 0.0)
    d_residuals = Dict("y" => 1.0)  # Seed
    
    OpenMDAOCore.apply_linear!(comp, inputs, outputs, d_inputs, d_outputs, d_residuals, "rev")
    # For R = Y^2 - X, the VJP is:
    # dY += dR * dR/dY = 1.0 * 2*Y = 1.0 * 4.0 = 4.0
    # dX += dR * dR/dX = 1.0 * (-1.0) = -1.0
    @test d_outputs["y"] ≈ 4.0 atol=1e-12
    @test d_inputs["x"] ≈ -1.0 atol=1e-12
end

@testitem "create_implicit_component convenience functions" begin
    using OpenMDAOCore
    using ADTypes: ADTypes
    using ComponentArrays
    using ForwardDiff
    # Needed so the DifferentiationInterface ReverseDiff extension gets loaded
    # before `create_implicit_component` prepares a pullback with AutoReverseDiff.
    using ReverseDiff
    using Test

    # Simple implicit function: R = Y^2 - X = 0, so Y = sqrt(X)
    function f_implicit!(R, Y, X, params)
        # Note: `y` is a scalar component, so `R[:y]` returns a scalar, not a
        # 0-dimensional array; use plain assignment instead of broadcasting.
        R[:y] = only(Y[:y])^2 - only(X[:x])
        return nothing
    end

    function f_implicit(Y, X, params)
        R = ComponentVector(y=only(Y[:y])^2 - only(X[:x]))
        return R
    end

    # Create ComponentVectors
    Y_ca = ComponentVector(y=1.0)
    X_ca = ComponentVector(x=4.0)

    # Test in-place convenience constructor
    comp1 = create_implicit_component(DenseFlavor(), Val(true), ADTypes.AutoForwardDiff(), f_implicit!, Y_ca, X_ca)
    @test typeof(comp1) <: ADImplicitComp

    # Test out-of-place convenience constructor
    comp2 = create_implicit_component(DenseFlavor(), Val(false), ADTypes.AutoForwardDiff(), f_implicit, Y_ca, X_ca)
    @test typeof(comp2) <: ADImplicitComp

    # Test matrix-free forward convenience constructor
    comp3 = create_implicit_component(MatrixFreeForwardFlavor(), Val(true), ADTypes.AutoForwardDiff(), f_implicit!, Y_ca, X_ca)
    @test typeof(comp3) <: ADImplicitComp

    # Test matrix-free reverse convenience constructor
    comp4 = create_implicit_component(MatrixFreeReverseFlavor(), Val(true), ADTypes.AutoReverseDiff(), f_implicit!, Y_ca, X_ca)
    @test typeof(comp4) <: ADImplicitComp
end
