//! Compile mirdata `FnBody` directly to Cranelift IR.
//!
//! Self-contained compilation path that works directly with `ra_mir_types::FnBody`
//! — no `HirDatabase`, `StoredTy`, or `MirBody` imports. Only depends on
//! `ra_mir_types`, `cranelift_*`, `rustc_abi`, and our existing `layout`/`pointer`
//! modules.
//!
//! Uses the shared `FunctionCx` with `MirSource::Mirdata` variant, which gives
//! access to the full `CPlace`/`CValue` infrastructure without duplication.

use cranelift_codegen::ir::{AbiParam, InstBuilder, Signature, Value};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{FuncId, Linkage, Module};
use ra_mir_types::{Body, ConstKind, Operand, Rvalue, Statement, Terminator};
use rustc_abi::{BackendRepr, Float, Primitive, TargetDataLayout};
use triomphe::Arc as TArc;

use hir_ty::layout::Layout;

use crate::pointer::Pointer;
use crate::value_and_place::{CPlace, CValue};
use crate::{FunctionCx, MirSource, pointer_ty, scalar_to_clif_type};

type LayoutArc = TArc<Layout>;

// ---------------------------------------------------------------------------
// Layout resolution
// ---------------------------------------------------------------------------

fn local_layout(
    local: &ra_mir_types::Local,
    layouts: &[LayoutArc],
) -> Result<LayoutArc, String> {
    let idx = local.layout.ok_or("local missing layout index")?;
    Ok(layouts[idx as usize].clone())
}

// ---------------------------------------------------------------------------
// Signature building
// ---------------------------------------------------------------------------

fn build_mirdata_fn_sig(
    isa: &dyn TargetIsa,
    dl: &TargetDataLayout,
    body: &Body,
    layouts: &[LayoutArc],
) -> Result<Signature, String> {
    let mut sig = Signature::new(isa.default_call_conv());
    let ptr_ty = pointer_ty(dl);

    // Return type: locals[0]
    let ret_layout = local_layout(&body.locals[0], layouts)?;
    match ret_layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
        }
        BackendRepr::ScalarPair(a, b) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
        }
        _ if ret_layout.is_zst() => {}
        _ => {
            sig.params.push(AbiParam::special(
                ptr_ty,
                cranelift_codegen::ir::ArgumentPurpose::StructReturn,
            ));
        }
    }

    // Parameters: locals[1..=arg_count]
    for i in 1..=body.arg_count as usize {
        let param_layout = local_layout(&body.locals[i], layouts)?;
        match param_layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
            }
            BackendRepr::ScalarPair(a, b) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
            }
            _ if param_layout.is_zst() => {}
            _ => {
                sig.params.push(AbiParam::new(ptr_ty));
            }
        }
    }

    Ok(sig)
}

// ---------------------------------------------------------------------------
// Operand / Rvalue codegen
// ---------------------------------------------------------------------------

fn codegen_md_operand(
    fx: &mut FunctionCx<'_, impl Module>,
    operand: &Operand,
    dest_layout: &LayoutArc,
) -> CValue {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            assert!(place.projections.is_empty(), "projections not yet supported");
            let local_place = fx.local_place_idx(place.local as usize).clone();
            local_place.to_cvalue(fx)
        }
        Operand::Constant(c) => match &c.kind {
            ConstKind::Scalar(bits, _size) => {
                let BackendRepr::Scalar(scalar) = dest_layout.backend_repr else {
                    panic!("Scalar constant with non-Scalar layout");
                };
                let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
                let val = match scalar.primitive() {
                    Primitive::Float(Float::F32) => {
                        fx.bcx.ins().f32const(f32::from_bits(*bits as u32))
                    }
                    Primitive::Float(Float::F64) => {
                        fx.bcx.ins().f64const(f64::from_bits(*bits as u64))
                    }
                    _ => fx.bcx.ins().iconst(clif_ty, *bits as i64),
                };
                CValue::by_val(val, dest_layout.clone())
            }
            ConstKind::ZeroSized => CValue::zst(dest_layout.clone()),
            _ => panic!("unsupported constant kind: {:?}", c.kind),
        },
    }
}

fn codegen_md_rvalue(
    fx: &mut FunctionCx<'_, impl Module>,
    rvalue: &Rvalue,
    dest_layout: &LayoutArc,
) -> CValue {
    match rvalue {
        Rvalue::Use(operand) => codegen_md_operand(fx, operand, dest_layout),
        _ => panic!("unsupported rvalue: {:?}", rvalue),
    }
}

// ---------------------------------------------------------------------------
// Statement / Terminator codegen
// ---------------------------------------------------------------------------

fn codegen_md_statement(fx: &mut FunctionCx<'_, impl Module>, stmt: &Statement) {
    match stmt {
        Statement::Assign(place, rvalue) => {
            assert!(place.projections.is_empty(), "projections not yet supported");
            let dest = fx.local_place_idx(place.local as usize).clone();
            let val = codegen_md_rvalue(fx, rvalue, &dest.layout);
            dest.write_cvalue(fx, val);
        }
        Statement::StorageLive(_) | Statement::StorageDead(_) | Statement::Nop => {}
        _ => panic!("unsupported statement: {:?}", stmt),
    }
}

fn codegen_md_terminator(fx: &mut FunctionCx<'_, impl Module>, term: &Terminator) {
    match term {
        Terminator::Return => {
            let ret_place = fx.local_place_idx(0).clone();
            if ret_place.layout.is_zst() {
                fx.bcx.ins().return_(&[]);
            } else {
                let cval = ret_place.to_cvalue(fx);
                match ret_place.layout.backend_repr {
                    BackendRepr::Scalar(_) => {
                        let val = cval.load_scalar(fx);
                        fx.bcx.ins().return_(&[val]);
                    }
                    BackendRepr::ScalarPair(_, _) => {
                        let (a, b) = cval.load_scalar_pair(fx);
                        fx.bcx.ins().return_(&[a, b]);
                    }
                    _ => panic!("unsupported return value repr"),
                }
            }
        }
        Terminator::Goto(target) => {
            let block = fx.clif_block_idx(*target as usize);
            fx.bcx.ins().jump(block, &[]);
        }
        Terminator::Unreachable => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());
        }
        _ => panic!("unsupported terminator: {:?}", term),
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

pub fn compile_mirdata_fn(
    module: &mut impl Module,
    isa: &dyn TargetIsa,
    dl: &TargetDataLayout,
    fn_body: &ra_mir_types::FnBody,
    layouts: &[LayoutArc],
    fn_name: &str,
    linkage: Linkage,
) -> Result<FuncId, String> {
    let body = &fn_body.body;
    let pointer_type = pointer_ty(dl);
    let sig = build_mirdata_fn_sig(isa, dl, body, layouts)?;

    let func_id = module
        .declare_function(fn_name, linkage, &sig)
        .map_err(|e| format!("declare_function: {e}"))?;

    let mut func = cranelift_codegen::ir::Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
        sig.clone(),
    );
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create blocks
        let block_map: Vec<cranelift_codegen::ir::Block> =
            body.blocks.iter().map(|_| bcx.create_block()).collect();

        // Set up entry block
        let entry_block = block_map[0];
        bcx.switch_to_block(entry_block);
        bcx.append_block_params_for_function_params(entry_block);

        // Create locals
        let mut local_map = Vec::with_capacity(body.locals.len());
        for local in &body.locals {
            let layout = local_layout(local, layouts)?;
            let place = match layout.backend_repr {
                BackendRepr::Scalar(scalar) => {
                    let clif_ty = scalar_to_clif_type(dl, &scalar);
                    let var = bcx.declare_var(clif_ty);
                    CPlace::new_var_raw(var, layout)
                }
                BackendRepr::ScalarPair(a, b) => {
                    let a_clif = scalar_to_clif_type(dl, &a);
                    let b_clif = scalar_to_clif_type(dl, &b);
                    let var1 = bcx.declare_var(a_clif);
                    let var2 = bcx.declare_var(b_clif);
                    CPlace::new_var_pair_raw(var1, var2, layout)
                }
                _ if layout.is_zst() => {
                    CPlace::for_ptr(Pointer::dangling(layout.align.abi), layout)
                }
                _ => {
                    return Err("memory-repr locals not yet supported in mirdata codegen".into());
                }
            };
            local_map.push(place);
        }

        // Wire parameters to locals
        let block_params: Vec<Value> = bcx.block_params(entry_block).to_vec();
        let mut param_idx = 0;
        for i in 1..=body.arg_count as usize {
            let place = &local_map[i];
            if place.layout.is_zst() {
                continue;
            }
            match place.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    place.def_var(0, block_params[param_idx], &mut bcx);
                    param_idx += 1;
                }
                BackendRepr::ScalarPair(_, _) => {
                    place.def_var(0, block_params[param_idx], &mut bcx);
                    place.def_var(1, block_params[param_idx + 1], &mut bcx);
                    param_idx += 2;
                }
                _ => return Err("memory-repr params not yet supported".into()),
            }
        }

        let mut fx = FunctionCx {
            bcx,
            module,
            isa,
            pointer_type,
            dl,
            mir: MirSource::Mirdata { body, layouts },
            block_map,
            local_map,
            drop_flags: std::collections::HashMap::new(),
        };

        // Codegen blocks
        for (bb_idx, bb) in body.blocks.iter().enumerate() {
            let clif_block = fx.clif_block_idx(bb_idx);
            if bb_idx != 0 {
                fx.bcx.switch_to_block(clif_block);
            }

            for stmt in &bb.stmts {
                codegen_md_statement(&mut fx, stmt);
            }

            codegen_md_terminator(&mut fx, &bb.terminator);
        }

        fx.bcx.seal_all_blocks();
        fx.bcx.finalize();
    }

    let mut ctx = Context::for_function(func);
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define_function: {e}"))?;

    Ok(func_id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::convert_mirdata_layouts;
    use cranelift_jit::{JITBuilder, JITModule};
    use ra_mir_types::*;
    use rustc_abi::AddressSpace;

    fn make_jit_module() -> (JITModule, std::sync::Arc<dyn TargetIsa>, TargetDataLayout) {
        let isa = crate::build_host_isa(false);
        let dl = TargetDataLayout::parse_from_llvm_datalayout_string(
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
            AddressSpace(0),
        )
        .map_err(|_| "failed to parse data layout")
        .unwrap();
        let builder =
            JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());
        let module = JITModule::new(builder);
        (module, isa, dl)
    }

    /// JIT-compile a no-arg mirdata function, execute it, return the result.
    fn mirdata_jit_run<R: Copy>(
        fn_body: &FnBody,
        layout_entries: &[TypeLayoutEntry],
    ) -> R {
        let layouts = convert_mirdata_layouts(layout_entries);
        let (mut module, isa, dl) = make_jit_module();

        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            fn_body,
            &layouts,
            &fn_body.name,
            Linkage::Export,
        )
        .expect("compile_mirdata_fn failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn() -> R = std::mem::transmute(code);
            f()
        }
    }

    fn i32_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::Int(IntTy::I32),
            layout: LayoutInfo {
                size: 4,
                align: 4,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Int { size_bytes: 4, signed: true },
                    valid_range_start: 0,
                    valid_range_end: u32::MAX as u128,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    fn bool_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::Bool,
            layout: LayoutInfo {
                size: 1,
                align: 1,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Int { size_bytes: 1, signed: false },
                    valid_range_start: 0,
                    valid_range_end: 1,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: Some(ExportedNiche {
                    offset: 0,
                    scalar: ExportedScalar {
                        primitive: ExportedPrimitive::Int { size_bytes: 1, signed: false },
                        valid_range_start: 0,
                        valid_range_end: 1,
                    },
                }),
            },
        }
    }

    // -- fn foo() -> i32 { 42 } ------------------------------------------

    #[test]
    fn mirdata_return_42() {
        let layouts = vec![i32_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "foo".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::Use(Operand::Constant(ConstOperand {
                            ty: Ty::Int(IntTy::I32),
                            kind: ConstKind::Scalar(42, 4),
                        })),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layouts);
        assert_eq!(result, 42);
    }

    // -- fn foo() -> bool { true } ----------------------------------------

    #[test]
    fn mirdata_return_bool() {
        let layouts = vec![bool_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "foo_bool".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![Local { ty: Ty::Bool, layout: Some(0) }],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::Use(Operand::Constant(ConstOperand {
                            ty: Ty::Bool,
                            kind: ConstKind::Scalar(1, 1),
                        })),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: bool = mirdata_jit_run(&fn_body, &layouts);
        assert_eq!(result, true);
    }

    // -- fn foo() -> i32 { let x = 42; x } -------------------------------

    #[test]
    fn mirdata_copy_local() {
        let layouts = vec![i32_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "foo_copy".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Int(IntTy::I32),
                                kind: ConstKind::Scalar(42, 4),
                            })),
                        ),
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layouts);
        assert_eq!(result, 42);
    }

    // -- multi-block Goto -------------------------------------------------

    #[test]
    fn mirdata_goto() {
        let layouts = vec![i32_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "foo_goto".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Goto(1),
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Int(IntTy::I32),
                                kind: ConstKind::Scalar(99, 4),
                            })),
                        )],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layouts);
        assert_eq!(result, 99);
    }

    // -- fn identity(a: i32, b: i32) -> i32 { a } ------------------------

    #[test]
    fn mirdata_with_args() {
        let layout_entries = vec![i32_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "identity".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _0: return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _1: a
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _2: b
                ],
                arg_count: 2,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::Use(Operand::Copy(Place { local: 1, projections: vec![] })),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &fn_body,
            &layouts,
            "identity",
            Linkage::Export,
        )
        .expect("compile_mirdata_fn failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn(i32, i32) -> i32 = std::mem::transmute(code);
            assert_eq!(f(7, 13), 7);
            assert_eq!(f(100, 200), 100);
        }
    }
}
