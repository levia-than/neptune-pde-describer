from sympy import symbols, Function, Eq, diff

x, t = symbols('x t')
alpha = symbols('alpha')
u = Function('u')(x, t)
# PDE: u_t = alpha * u_xx
pde = Eq(diff(u, t), alpha * diff(u, x, 2))

def derive_fd(pde_expr, alpha_val, dt, dx):
    """
    Pseudo-derivation: 从 PDE 符号表达返回显式差分系数。
    我们硬编码为向前时间中心空间差分：
      u_new = u + k*(u_ip1 - 2*u + u_im1),  k = alpha * dt/dx^2

    返回一个 dict: {'u_im1': a, 'u_i': b, 'u_ip1': c}
    """
    k = alpha_val * dt / (dx*dx)
    a = k
    b = 1.0 - 2.0 * k
    c = k
    return {'u_im1': a, 'u_i': b, 'u_ip1': c}

def emit_field_ref(varname, idx_name, memref):
    return f"\t\t\t%{varname} = neptune_ir.field.ref %u_old[{idx_name}]: {memref} -> !neptune_ir.field_elem<etype=f64>\n"


def emit_scale(src, tmpname, scalar):
    return f"\t\t\t%{tmpname} = neptune_ir.field.scale %{src} {{ scalar = {scalar} }} : !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>\n"


def emit_add(lhs, rhs, tmpname):
    return f"\t\t\t%{tmpname} = neptune_ir.field.add %{lhs}, %{rhs} : !neptune_ir.field_elem<etype=f64>, !neptune_ir.field_elem<etype=f64> -> !neptune_ir.field_elem<etype=f64>\n"


def gen_linear_expr_ops(coeff_map):
    ops = []
    tmp_index = 0
    temp_names = []
    mapping = {'u_im1': 'fe_im1', 'u_i': 'fe_i', 'u_ip1': 'fe_ip1'}

    # 生成 scale（如果系数不是 1）或直接使用 fe_*
    for sym in ['u_im1', 'u_i', 'u_ip1']:
        a = coeff_map.get(sym, 0.0)
        if abs(a) < 1e-15:
            continue
        src = mapping[sym]
        if abs(a - 1.0) < 1e-15:
            temp_names.append(src)
        else:
            tmp = f"s{tmp_index}"
            ops.append(emit_scale(src, tmp, round(a, 12)))
            temp_names.append(tmp)
            tmp_index += 1

    if not temp_names:
        final = 'fe_i'
    else:
        cur = temp_names[0]
        for i in range(1, len(temp_names)):
            nxt = temp_names[i]
            tmp = f"a{tmp_index}"
            ops.append(emit_add(cur, nxt, tmp))
            cur = tmp
            tmp_index += 1
        final = cur

    return ops, final


def gen_boundary_kernel(name, idx, memref):
    hdr = f"\tfunc.func @{name}(%u_old: {memref}, %u_new:{memref}) {{\n"
    body = f"\t\t\t%c{idx} = arith.constant {idx} : index\n"
    body += f"\t\t\t%expr = neptune_ir.field.ref %u_old[%c{idx}]: {memref} -> !neptune_ir.field_elem<etype=f64>\n"
    body += f"\t\t\tneptune_ir.evaluate %u_new, %expr: {memref}, !neptune_ir.field_elem<etype=f64>\n"
    body += "\t\treturn\n\t}\n\n"
    return hdr + body


def gen_interior_kernel_from_coeffs(memref, coeffs, nx):
    hdr = f"\tfunc.func @heat_sim_single_step_kernel(%u_old: {memref}, %u_new: {memref}){{\n"
    hdr += "\t\t\t%c0 = arith.constant 0 : index\n\t\t\t%c1 = arith.constant 1 : index\n"
    hdr += f"\t\t\t%c{nx-1} = arith.constant {nx-1} : index\n"
    hdr += "\t\t\t%start = arith.addi %c0, %c1 : index\n\t\t\t%end = arith.subi %c15, %c1 : index\n\n".replace('%c15', f'%c{nx-1}')
    hdr += "\t\t\tscf.for %i = %start to %end step %c1 : index {\n"
    hdr += "\t\t\t\t%ni_1 = arith.addi %i, %c1 : index\n\t\t\t\t%ni_2 = arith.subi %i, %c1 : index\n"
    hdr += emit_field_ref('fe_i', '%i', memref)
    hdr += emit_field_ref('fe_ip1', '%ni_1', memref)
    hdr += emit_field_ref('fe_im1', '%ni_2', memref)

    ops, final = gen_linear_expr_ops(coeffs)
    for op in ops:
        hdr += op
    hdr += f"\t\t\t\tneptune_ir.evaluate %u_new, %{final} : {memref}, !neptune_ir.field_elem<etype=f64>\n"
    hdr += "\t\t\t\tscf.yield\n\t\t\t}\n\t\t\treturn\n\t}\n\n"
    return hdr


def gen_heat_sim_main(memref):
    hdr = f"\tfunc.func @heat_sim(%input: {memref}, %total_time: index) -> {memref} {{\n"
    body = "\t\t\t%c0 = arith.constant 0 : index\n\t\t\t%c1 = arith.constant 1 : index\n\t\t\t%c2 = arith.constant 2 : index\n"
    body += f"\t\t\t%ping_buf = memref.alloc() : {memref}\n\t\t\t%pong_buf = memref.alloc() : {memref}\n\n"
    body += f"\t\t\tmemref.copy %input, %ping_buf : {memref} to {memref}\n\n"
    body += "\t\t\tscf.for %time_step = %c0 to %total_time step %c1 : index {\n"
    body += "\t\t\t\t%is_even_step_idx = arith.remui %time_step, %c2 : index\n\t\t\t\t%is_even_step = arith.cmpi eq, %is_even_step_idx, %c0 : index\n\n"
    body += "\t\t\t\t%u_old = scf.if %is_even_step -> (" + memref + ") {\n\t\t\t\t\tscf.yield %ping_buf : " + memref + "\n\t\t\t\t} else {\n\t\t\t\t\tscf.yield %pong_buf : " + memref + "\n\t\t\t\t}\n\n"
    body += "\t\t\t\t%u_new = scf.if %is_even_step -> (" + memref + ") {\n\t\t\t\t\tscf.yield %pong_buf : " + memref + "\n\t\t\t\t} else {\n\t\t\t\t\tscf.yield %ping_buf : " + memref + "\n\t\t\t\t}\n\n"
    body += "\t\t\t\tfunc.call @heat_sim_single_step_kernel_broder_1(%u_old, %u_new) : (" + memref + ", " + memref + ") -> ()\n"
    body += "\t\t\t\tfunc.call @heat_sim_single_step_kernel_broder_2(%u_old, %u_new) : (" + memref + ", " + memref + ") -> ()\n"
    body += "\t\t\t\tfunc.call @heat_sim_single_step_kernel(%u_old, %u_new) : (" + memref + ", " + memref + ") -> ()\n"
    body += "\t\t\t}\n\n"
    body += "\t\t\t%is_total_time_odd_idx = arith.remui %total_time, %c2 : index\n\t\t\t%is_total_time_odd = arith.cmpi eq, %is_total_time_odd_idx, %c0 : index\n\n"
    body += "\t\t\t%final_result = scf.if %is_total_time_odd -> (" + memref + ") {\n\t\t\t\tscf.yield %pong_buf : " + memref + "\n\t\t\t} else {\n\t\t\t\tscf.yield %ping_buf : " + memref + "\n\t\t\t}\n\n"
    body += "\t\t\tscf.if %is_total_time_odd {\n\t\t\t\tmemref.dealloc %ping_buf : " + memref + "\n\t\t\t} else {\n\t\t\t\tmemref.dealloc %pong_buf : " + memref + "\n\t\t\t}\n\n"
    body += "\t\t\treturn %final_result : " + memref + "\n\t}\n\n"
    return hdr + body

def main():
    NX = 16
    alpha_val = 0.5
    dt = 0.1
    dx = 0.2
    memref = f"memref<{NX}xf64>"

    # 伪推导得到系数
    coeffs = derive_fd(pde, alpha_val, dt, dx)

    # 生成 module
    s = "module {\n\n"
    s += gen_boundary_kernel('heat_sim_single_step_kernel_broder_1', 0, memref)
    s += gen_boundary_kernel('heat_sim_single_step_kernel_broder_2', NX-1, memref)
    s += gen_interior_kernel_from_coeffs(memref, coeffs, NX)
    s += gen_heat_sim_main(memref)
    s += "}\n"

    print(s)

if __name__ == '__main__':
    main()
