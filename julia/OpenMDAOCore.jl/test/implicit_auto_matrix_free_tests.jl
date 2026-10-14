@testitem "implicit matrix-free, in-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_forward_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false))
end

@testitem "implicit matrix-free, out-of-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place_forward_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false))
end

@testitem "implicit matrix-free, in-place, reverse, reversediff, fsp=false" setup=[ADCallbacks] begin
    doit_in_place_reverse_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "reversediff", false))
end

@testitem "implicit matrix-free, out-of-place, reverse, reversediff, fsp=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "reversediff", false))
end

@testitem "implicit matrix-free, in-place, forward, forwarddiff, shape_by_conn" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeImplicitShapeByConnTestPrep(4, 3, "forwarddiff"))
end

@testitem "implicit matrix-free, in-place, reverse, reversediff, shape_by_conn" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeImplicitShapeByConnTestPrep(4, 3, "reversediff"))
end

@testitem "implicit matrix-free, in-place, fwd+rev, forwarddiff+reversediff" setup=[ADCallbacks] begin
    doit_in_place_fwd_rev_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false), "reversediff")
end

@testitem "implicit matrix-free, out-of-place, fwd+rev, forwarddiff+reversediff" setup=[ADCallbacks] begin
    doit_out_of_place_fwd_rev_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false), "reversediff")
end

@testitem "implicit matrix-free, in-place, fwd+rev, forwarddiff+reversediff, shape_by_conn" setup=[ADCallbacks] begin
    doit_in_place_fwd_rev_sbc_implicit(AutoMatrixFreeImplicitShapeByConnTestPrep(4, 3, "forwarddiff"), "reversediff")
end
