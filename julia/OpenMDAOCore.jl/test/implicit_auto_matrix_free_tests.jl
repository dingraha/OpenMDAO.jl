# ── in-place forward ──────────────────────────────────────────────────────────

@testitem "implicit matrix-free, in-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_forward_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false))
end

# ── out-of-place forward ──────────────────────────────────────────────────────

@testitem "implicit matrix-free, out-of-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_out_of_place_forward_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "forwarddiff", false))
end

# ── in-place reverse ──────────────────────────────────────────────────────────

@testitem "implicit matrix-free, in-place, reverse, reversediff, fsp=false" setup=[ADCallbacks] begin
    doit_in_place_reverse_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "reversediff", false))
end

# ── out-of-place reverse ──────────────────────────────────────────────────────

@testitem "implicit matrix-free, out-of-place, reverse, reversediff, fsp=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse_implicit(AutoMatrixFreeImplicitTestPrep(4, 3, "reversediff", false))
end
