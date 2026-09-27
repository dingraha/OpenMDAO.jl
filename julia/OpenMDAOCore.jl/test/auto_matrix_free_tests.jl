@testitem "auto matrix-free, in-place, forward, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, in-place, forward, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeTestPrep(4, 3, "enzymeforward", false))
end

@testitem "auto matrix-free, in-place, forward, shape_by_conn, forwarddiff" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeShapeByConnTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, in-place, forward, shape_by_conn, enzymeforward" setup=[ADCallbacks] begin
    doit_in_place_forward(AutoMatrixFreeShapeByConnTestPrep(4, 3, "enzymeforward", false))
end

@testitem "auto matrix-free, in-place, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, in-place, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, in-place, reverse, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeTestPrep(4, 3, "enzymereverse", false))
end

@testitem "auto matrix-free, in-place, shape_by_conn, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, in-place, shape_by_conn, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, in-place, shape_by_conn, reverse, enzymereverse" setup=[ADCallbacks] begin
    doit_in_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "enzymereverse", false))
end

@testitem "auto matrix-free, out-of-place, forward, forwarddiff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_forward(AutoMatrixFreeTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, out-of-place, forward, shape_by_conn, forwarddiff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_forward(AutoMatrixFreeShapeByConnTestPrep(4, 3, "forwarddiff", false))
end

@testitem "auto matrix-free, out-of-place, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, out-of-place, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, out-of-place, reverse, zygote, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "zygote", false))
end

@testitem "auto matrix-free, out-of-place, reverse, zygote, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeTestPrep(4, 3, "zygote", true))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, reversediff, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", false))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, reversediff, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "reversediff", true))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, zygote, disable_prep=false" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "zygote", false))
end

@testitem "auto matrix-free, out-of-place, shape_by_conn, reverse, zygote, disable_prep=true" setup=[ADCallbacks] begin
    doit_out_of_place_reverse(AutoMatrixFreeShapeByConnTestPrep(4, 3, "zygote", true))
end
