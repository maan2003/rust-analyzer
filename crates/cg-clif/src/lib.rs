//! MIR → Cranelift IR codegen for rust-analyzer.
//!
//! Translates r-a's MIR representation to Cranelift IR and emits object files
//! via cranelift-object. Based on patterns from cg_clif (rustc's Cranelift backend).

use std::sync::Arc;

use cranelift_codegen::ir::{
    AbiParam, Block, InstBuilder, Signature, Type, Value, types,
};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use hir_ty::db::HirDatabase;
use hir_ty::mir::{
    BasicBlockId, LocalId, MirBody, OperandKind, Place, Rvalue, StatementKind, TerminatorKind,
};
use hir_ty::next_solver::{Const, ConstKind, IntoKind};
use hir_ty::traits::StoredParamEnvAndCrate;
use rustc_abi::{BackendRepr, Scalar, Size, TargetDataLayout};

mod pointer;

// ---------------------------------------------------------------------------
// Type mapping: r-a Ty → Cranelift Type
// ---------------------------------------------------------------------------

fn scalar_to_clif_type(dl: &TargetDataLayout, scalar: &Scalar) -> Type {
    use rustc_abi::Primitive;
    match scalar.primitive() {
        Primitive::Int(int, _signed) => match int.size().bits() {
            8 => types::I8,
            16 => types::I16,
            32 => types::I32,
            64 => types::I64,
            128 => types::I128,
            _ => unreachable!("unsupported int size: {}", int.size().bits()),
        },
        Primitive::Float(float) => match float {
            rustc_abi::Float::F16 => types::F16,
            rustc_abi::Float::F32 => types::F32,
            rustc_abi::Float::F64 => types::F64,
            rustc_abi::Float::F128 => types::F128,
        },
        Primitive::Pointer(_) => pointer_ty(dl),
    }
}

fn pointer_ty(dl: &TargetDataLayout) -> Type {
    match dl.pointer_size().bits() {
        16 => types::I16,
        32 => types::I32,
        64 => types::I64,
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// FunctionCx: per-function codegen state
// ---------------------------------------------------------------------------

struct FunctionCx<'a> {
    bcx: FunctionBuilder<'a>,
    pointer_type: Type,
    db: &'a dyn HirDatabase,
    dl: &'a TargetDataLayout,
    env: StoredParamEnvAndCrate,

    /// MIR basic block → Cranelift block
    block_map: Vec<(BasicBlockId, Block)>,
    /// MIR local → Cranelift variable (for scalar locals) or stack slot
    local_vars: Vec<(LocalId, LocalStorage)>,
}

#[derive(Clone)]
enum LocalStorage {
    Var(Variable, Type),
    // Stack(StackSlot),
}

impl FunctionCx<'_> {
    fn clif_block(&self, bb: BasicBlockId) -> Block {
        self.block_map
            .iter()
            .find(|(id, _)| *id == bb)
            .map(|(_, b)| *b)
            .expect("block not found in block_map")
    }

    fn local_storage(&self, local: LocalId) -> &LocalStorage {
        self.local_vars
            .iter()
            .find(|(id, _)| *id == local)
            .map(|(_, s)| s)
            .expect("local not found")
    }
}

// ---------------------------------------------------------------------------
// Constant extraction
// ---------------------------------------------------------------------------

fn const_to_i64(konst: Const<'_>, size: Size) -> i64 {
    match konst.kind() {
        ConstKind::Value(val) => {
            let bytes = &val.value.inner().memory;
            let n = size.bytes() as usize;
            let mut buf = [0u8; 8];
            let len = n.min(8);
            buf[..len].copy_from_slice(&bytes[..len]);
            i64::from_le_bytes(buf)
        }
        _ => panic!("const_to_i64: unsupported const kind: {:?}", konst),
    }
}

// ---------------------------------------------------------------------------
// Statement codegen
// ---------------------------------------------------------------------------

fn codegen_statement(fx: &mut FunctionCx<'_>, body: &MirBody, stmt: &StatementKind) {
    match stmt {
        StatementKind::Assign(place, rvalue) => {
            codegen_assign(fx, body, place, rvalue);
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {}
        StatementKind::Nop | StatementKind::FakeRead(_) => {}
        StatementKind::Deinit(_) => {}
        StatementKind::SetDiscriminant { .. } => todo!("SetDiscriminant"),
    }
}

fn codegen_assign(fx: &mut FunctionCx<'_>, body: &MirBody, place: &Place, rvalue: &Rvalue) {
    assert!(place.projection.is_empty(), "projections not yet supported");

    let val = codegen_rvalue(fx, body, rvalue);

    match fx.local_storage(place.local) {
        LocalStorage::Var(var, _) => {
            fx.bcx.def_var(*var, val);
        }
    }
}

fn codegen_rvalue(fx: &mut FunctionCx<'_>, body: &MirBody, rvalue: &Rvalue) -> Value {
    match rvalue {
        Rvalue::Use(operand) => codegen_operand(fx, body, &operand.kind),
        Rvalue::BinaryOp(op, lhs, rhs) => {
            let _ = (op, lhs, rhs);
            todo!("BinaryOp")
        }
        _ => todo!("rvalue: {:?}", rvalue),
    }
}

fn codegen_operand(fx: &mut FunctionCx<'_>, _body: &MirBody, kind: &OperandKind) -> Value {
    match kind {
        OperandKind::Constant { konst, ty } => {
            let ty = ty.as_ref();
            let layout = fx
                .db
                .layout_of_ty(ty.store(), fx.env.clone())
                .expect("layout error for constant type");
            if let BackendRepr::Scalar(scalar) = layout.backend_repr {
                let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
                let val = const_to_i64(konst.as_ref(), scalar.size(fx.dl));
                fx.bcx.ins().iconst(clif_ty, val)
            } else {
                todo!("non-scalar constant")
            }
        }
        OperandKind::Copy(place) | OperandKind::Move(place) => {
            assert!(place.projection.is_empty(), "projections not yet supported");
            match fx.local_storage(place.local) {
                LocalStorage::Var(var, _) => fx.bcx.use_var(*var),
            }
        }
        OperandKind::Static(_) => todo!("static operand"),
    }
}

// ---------------------------------------------------------------------------
// Terminator codegen
// ---------------------------------------------------------------------------

fn codegen_terminator(fx: &mut FunctionCx<'_>, body: &MirBody, term: &TerminatorKind) {
    match term {
        TerminatorKind::Return => {
            let ret_local = hir_ty::mir::return_slot();
            let ret_ty = &body.locals[ret_local].ty;
            let layout = fx
                .db
                .layout_of_ty(ret_ty.clone(), fx.env.clone())
                .expect("return type layout error");
            if layout.is_zst() {
                fx.bcx.ins().return_(&[]);
            } else {
                match fx.local_storage(ret_local) {
                    LocalStorage::Var(var, _) => {
                        let val = fx.bcx.use_var(*var);
                        fx.bcx.ins().return_(&[val]);
                    }
                }
            }
        }
        TerminatorKind::Goto { target } => {
            let block = fx.clif_block(*target);
            fx.bcx.ins().jump(block, &[]);
        }
        TerminatorKind::Unreachable => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());
        }
        _ => todo!("terminator: {:?}", term),
    }
}

// ---------------------------------------------------------------------------
// Top-level: compile a MIR body to an object file
// ---------------------------------------------------------------------------

/// Build a Cranelift ISA for the host machine.
pub fn build_host_isa() -> Arc<dyn TargetIsa> {
    let mut flags_builder = settings::builder();
    flags_builder.set("is_pic", "true").unwrap();
    flags_builder.set("opt_level", "none").unwrap();
    let flags = settings::Flags::new(flags_builder);

    let isa_builder = cranelift_native::builder().expect("host ISA not supported");
    isa_builder.finish(flags).expect("failed to build ISA")
}

/// Compile a single MIR body to a named function in an ObjectModule.
pub fn compile_fn(
    module: &mut ObjectModule,
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    fn_name: &str,
    linkage: Linkage,
) -> Result<FuncId, String> {
    let pointer_type = pointer_ty(dl);

    let ret_local = &body.locals[hir_ty::mir::return_slot()];
    let mut sig = Signature::new(isa.default_call_conv());

    // Return type
    let ret_layout = db
        .layout_of_ty(ret_local.ty.clone(), env.clone())
        .map_err(|e| format!("return type layout error: {:?}", e))?;
    if let BackendRepr::Scalar(scalar) = ret_layout.backend_repr {
        sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
    }

    // Parameter types
    for &param_local in &body.param_locals {
        let param = &body.locals[param_local];
        let param_layout = db
            .layout_of_ty(param.ty.clone(), env.clone())
            .map_err(|e| format!("param layout error: {:?}", e))?;
        if let BackendRepr::Scalar(scalar) = param_layout.backend_repr {
            sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
        } else if param_layout.is_zst() {
            // ZST params ignored
        } else {
            return Err("unsupported param type: non-scalar, non-ZST".into());
        }
    }

    // Declare function in module
    let func_id = module
        .declare_function(fn_name, linkage, &sig)
        .map_err(|e| format!("declare_function: {e}"))?;

    // Build the function body
    let mut func = cranelift_codegen::ir::Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
        sig.clone(),
    );
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create Cranelift blocks for MIR basic blocks
        let mut block_map = Vec::new();
        for (bb_id, _bb) in body.basic_blocks.iter() {
            let block = bcx.create_block();
            block_map.push((bb_id, block));
        }

        // Set up entry block
        let entry_block = block_map
            .iter()
            .find(|(id, _)| *id == body.start_block)
            .map(|(_, b)| *b)
            .unwrap();
        bcx.switch_to_block(entry_block);
        bcx.append_block_params_for_function_params(entry_block);

        // Set up locals as Cranelift variables
        let mut local_vars = Vec::new();

        for (local_id, local) in body.locals.iter() {
            let local_layout = db
                .layout_of_ty(local.ty.clone(), env.clone())
                .map_err(|e| format!("local layout error: {:?}", e))?;

            if let BackendRepr::Scalar(scalar) = local_layout.backend_repr {
                let clif_ty = scalar_to_clif_type(dl, &scalar);
                let var = bcx.declare_var(clif_ty);
                local_vars.push((local_id, LocalStorage::Var(var, clif_ty)));
            } else if local_layout.is_zst() {
                let var = bcx.declare_var(types::I8);
                local_vars.push((local_id, LocalStorage::Var(var, types::I8)));
            } else {
                // TODO: stack slot for non-scalar types
                return Err(format!("unsupported local type: non-scalar, non-ZST"));
            }
        }

        // Wire function parameters to their locals
        let block_params: Vec<Value> = bcx.block_params(entry_block).to_vec();
        let mut param_idx = 0;
        for &param_local in &body.param_locals {
            if let Some((_, LocalStorage::Var(var, _))) =
                local_vars.iter().find(|(id, _)| *id == param_local)
            {
                if param_idx < block_params.len() {
                    bcx.def_var(*var, block_params[param_idx]);
                    param_idx += 1;
                }
            }
        }

        let mut fx = FunctionCx {
            bcx,
            pointer_type,
            db,
            dl,
            env: env.clone(),
            block_map: block_map.clone(),
            local_vars,
        };

        // Codegen each basic block
        for (bb_id, bb) in body.basic_blocks.iter() {
            let clif_block = fx.clif_block(bb_id);
            if bb_id != body.start_block {
                fx.bcx.switch_to_block(clif_block);
            }

            for stmt in &bb.statements {
                codegen_statement(&mut fx, body, &stmt.kind);
            }

            if let Some(term) = &bb.terminator {
                codegen_terminator(&mut fx, body, &term.kind);
            }
        }

        fx.bcx.seal_all_blocks();
        fx.bcx.finalize();
    }

    // Compile and define the function
    let mut ctx = Context::for_function(func);
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define_function: {e}"))?;

    Ok(func_id)
}

/// Compile a MIR body and return the object file bytes.
pub fn compile_to_object(
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    fn_name: &str,
) -> Result<Vec<u8>, String> {
    let isa = build_host_isa();

    let builder =
        ObjectBuilder::new(isa.clone(), "rac_output", cranelift_module::default_libcall_names())
            .map_err(|e| format!("ObjectBuilder: {e}"))?;
    let mut module = ObjectModule::new(builder);

    compile_fn(&mut module, &*isa, db, dl, env, body, fn_name, Linkage::Export)?;

    let product = module.finish();
    let bytes = product.emit().map_err(|e| format!("emit: {e}"))?;
    Ok(bytes)
}

#[cfg(test)]
mod tests;
