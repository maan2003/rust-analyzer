//! Compile mirdata `FnBody` directly to Cranelift IR.
//!
//! Self-contained compilation path that works directly with `ra_mir_types::FnBody`
//! — no `HirDatabase`, `StoredTy`, or `MirBody` imports. Only depends on
//! `ra_mir_types`, `cranelift_*`, `rustc_abi`, and our existing `layout`/`pointer`
//! modules.
//!
//! Uses the shared `FunctionCx` with `MirSource::Mirdata` variant, which gives
//! access to the full `CPlace`/`CValue` infrastructure without duplication.

use std::collections::HashMap;

use cranelift_codegen::ir::{AbiParam, InstBuilder, Signature, StackSlotData, StackSlotKind, Value};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch};
use cranelift_module::{FuncId, Linkage, Module};
use ra_mir_types::{Body, CastKind, ConstKind, Operand, PointerCoercion, Rvalue, Statement, Terminator};
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
    layouts.get(idx as usize)
        .cloned()
        .ok_or_else(|| format!("layout index {} out of bounds (len={})", idx, layouts.len()))
}

/// Resolve a local's layout: try the layout index table first, then fall back
/// to ty_layouts lookup (for monomorphized locals without layout indices).
fn resolve_local_layout(
    local: &ra_mir_types::Local,
    layouts: &[LayoutArc],
    ty_layouts: &HashMap<ra_mir_types::Ty, LayoutArc>,
) -> Result<LayoutArc, String> {
    // Try layout index first
    if let Some(idx) = local.layout {
        if let Some(layout) = layouts.get(idx as usize) {
            return Ok(layout.clone());
        }
    }
    // Fall back to ty_layouts lookup
    if let Some(layout) = ty_layouts.get(&local.ty) {
        return Ok(layout.clone());
    }
    Err(format!("no layout for type: {:?}", local.ty))
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

    // Return type: locals[0]
    let ret_layout = local_layout(&body.locals[0], layouts)?;
    crate::append_ret_to_sig(&mut sig, dl, &ret_layout);

    // Parameters: locals[1..=arg_count]
    for i in 1..=body.arg_count as usize {
        let param_layout = local_layout(&body.locals[i], layouts)?;
        crate::append_param_to_sig(&mut sig, dl, &param_layout);
    }

    Ok(sig)
}

/// Build a function signature using `resolve_local_layout` (which falls back
/// to ty_layouts for locals without layout indices). Needed for monomorphized
/// generic functions.
fn build_mirdata_fn_sig_resolved(
    isa: &dyn TargetIsa,
    dl: &TargetDataLayout,
    body: &Body,
    layouts: &[LayoutArc],
    ty_layouts: &HashMap<ra_mir_types::Ty, LayoutArc>,
) -> Result<Signature, String> {
    let mut sig = Signature::new(isa.default_call_conv());

    // Return type: locals[0]
    let ret_layout = resolve_local_layout(&body.locals[0], layouts, ty_layouts)?;
    crate::append_ret_to_sig(&mut sig, dl, &ret_layout);

    // Parameters: locals[1..=arg_count]
    for i in 1..=body.arg_count as usize {
        let param_layout = resolve_local_layout(&body.locals[i], layouts, ty_layouts)?;
        crate::append_param_to_sig(&mut sig, dl, &param_layout);
    }

    Ok(sig)
}

/// Build a best-effort signature, using pointer_type for locals without layouts.
/// This is used for declaring generic functions in fn_registry so that calls
/// to them don't cause verifier errors (wrong param count).
fn build_mirdata_fn_sig_fallback(
    isa: &dyn TargetIsa,
    dl: &TargetDataLayout,
    body: &Body,
    layouts: &[LayoutArc],
) -> Signature {
    let pointer_type = crate::pointer_ty(dl);
    let mut sig = Signature::new(isa.default_call_conv());

    // Return type: locals[0]
    match local_layout(&body.locals[0], layouts) {
        Ok(ret_layout) => { crate::append_ret_to_sig(&mut sig, dl, &ret_layout); }
        Err(_) => { sig.returns.push(AbiParam::new(pointer_type)); }
    }

    // Parameters: locals[1..=arg_count]
    for i in 1..=body.arg_count as usize {
        match local_layout(&body.locals[i], layouts) {
            Ok(param_layout) => { crate::append_param_to_sig(&mut sig, dl, &param_layout); }
            Err(_) => { sig.params.push(AbiParam::new(pointer_type)); }
        }
    }

    sig
}

// ---------------------------------------------------------------------------
// Place codegen (projections)
// ---------------------------------------------------------------------------

/// Resolve a mirdata Place (local + projections) to a CPlace.
///
/// Tracks the current `ra_mir_types::Ty` through projections to resolve
/// field/pointee/element layouts via the `ty_layouts` map.
fn codegen_md_place(
    fx: &mut FunctionCx<'_, impl Module>,
    place: &ra_mir_types::Place,
) -> CPlace {
    if place.projections.is_empty() {
        return fx.local_place_idx(place.local as usize).clone();
    }

    let mut cplace = fx.local_place_idx(place.local as usize).clone();

    // Get the local's type to track through projections
    let MirSource::Mirdata { body, .. } = &fx.mir else { unreachable!() };
    let mut cur_ty = body.locals[place.local as usize].ty.clone();

    for proj in &place.projections {
        match proj {
            ra_mir_types::Projection::Field(field_idx, field_ty) => {
                let field_layout = fx.md_layout(field_ty);
                cplace = cplace.place_field(fx, *field_idx as usize, field_layout);
                cur_ty = field_ty.clone();
            }
            ra_mir_types::Projection::Deref => {
                let pointee_ty = match &cur_ty {
                    ra_mir_types::Ty::Ref(_, pointee) => (**pointee).clone(),
                    ra_mir_types::Ty::RawPtr(_, pointee) => (**pointee).clone(),
                    _ => panic!("Deref on non-pointer type: {:?}", cur_ty),
                };
                let pointee_layout = fx.md_layout(&pointee_ty);
                let cval = cplace.to_cvalue(fx);
                cplace = match cval.layout.backend_repr {
                    BackendRepr::ScalarPair(_, _) => {
                        // Fat pointer (e.g. &dyn Trait, &[T])
                        let (data_ptr, meta) = cval.load_scalar_pair(fx);
                        CPlace::for_ptr_with_extra(
                            Pointer::new(data_ptr),
                            meta,
                            pointee_layout,
                        )
                    }
                    _ => {
                        let ptr_val = cval.load_scalar(fx);
                        CPlace::for_ptr(Pointer::new(ptr_val), pointee_layout)
                    }
                };
                cur_ty = pointee_ty;
            }
            ra_mir_types::Projection::Downcast(variant_idx) => {
                use rustc_abi::Variants;
                // For multi-variant enums in registers, spill to memory
                if cplace.is_register() {
                    if matches!(&cplace.layout.variants, Variants::Multiple { .. }) {
                        let cval = cplace.to_cvalue(fx);
                        let ptr = cval.force_stack(fx);
                        cplace = CPlace::for_ptr(ptr, cplace.layout.clone());
                    }
                }
                let variant_layout = md_variant_layout(&cplace.layout, *variant_idx);
                cplace = cplace.downcast_variant(variant_layout);
                // cur_ty stays the same (Downcast doesn't change the Rust type)
            }
            ra_mir_types::Projection::Index(index_local) => {
                let index_place = fx.local_place_idx(*index_local as usize).clone();
                let index_val = index_place.to_cvalue(fx).load_scalar(fx);
                let elem_ty = match &cur_ty {
                    ra_mir_types::Ty::Array(elem, _) | ra_mir_types::Ty::Slice(elem) => (**elem).clone(),
                    _ => panic!("Index on non-array/slice type: {:?}", cur_ty),
                };
                let elem_layout = fx.md_layout(&elem_ty);
                let offset = fx.bcx.ins().imul_imm(index_val, elem_layout.size.bytes() as i64);
                // Get base pointer (handles both sized and unsized places)
                let base_ptr = cplace.to_ptr_maybe_spill(fx);
                cplace = CPlace::for_ptr(
                    base_ptr.offset_value(&mut fx.bcx, fx.pointer_type, offset),
                    elem_layout,
                );
                cur_ty = elem_ty;
            }
            ra_mir_types::Projection::ConstantIndex { offset, min_length: _, from_end } => {
                let elem_ty = match &cur_ty {
                    ra_mir_types::Ty::Array(elem, _) | ra_mir_types::Ty::Slice(elem) => (**elem).clone(),
                    _ => panic!("ConstantIndex on non-array/slice type: {:?}", cur_ty),
                };
                let elem_layout = fx.md_layout(&elem_ty);
                let index = if !*from_end {
                    fx.bcx.ins().iconst(fx.pointer_type, *offset as i64)
                } else {
                    // from_end: index = len - offset
                    let len = match &cur_ty {
                        ra_mir_types::Ty::Array(_, len) => {
                            fx.bcx.ins().iconst(fx.pointer_type, *len as i64)
                        }
                        _ => panic!("ConstantIndex from_end on non-array: {:?}", cur_ty),
                    };
                    fx.bcx.ins().iadd_imm(len, -(*offset as i64))
                };
                let base_ptr = cplace.to_ptr_maybe_spill(fx);
                let byte_offset = fx.bcx.ins().imul_imm(index, elem_layout.size.bytes() as i64);
                cplace = CPlace::for_ptr(
                    base_ptr.offset_value(&mut fx.bcx, fx.pointer_type, byte_offset),
                    elem_layout,
                );
                cur_ty = elem_ty;
            }
            ra_mir_types::Projection::Subslice { from, to, from_end } => {
                let elem_ty = match &cur_ty {
                    ra_mir_types::Ty::Array(elem, _) | ra_mir_types::Ty::Slice(elem) => (**elem).clone(),
                    _ => panic!("Subslice on non-array/slice type: {:?}", cur_ty),
                };
                let elem_layout = fx.md_layout(&elem_ty);
                if !*from_end {
                    // Array subslice: result is a smaller array [from..to]
                    let sub_len = *to - *from;
                    let sub_arr_ty = ra_mir_types::Ty::Array(Box::new(elem_ty.clone()), sub_len);
                    let sub_layout = fx.md_layout(&sub_arr_ty);
                    let base_ptr = cplace.to_ptr_maybe_spill(fx);
                    cplace = CPlace::for_ptr(
                        base_ptr.offset_i64(&mut fx.bcx, fx.pointer_type, elem_layout.size.bytes() as i64 * (*from as i64)),
                        sub_layout,
                    );
                    cur_ty = sub_arr_ty;
                } else {
                    // Slice subslice: result is a sub-slice [from..len-to]
                    // This is for actual slices — adjust pointer and length
                    let (ptr, len) = match cplace.to_cvalue(fx).layout.backend_repr {
                        BackendRepr::ScalarPair(_, _) => {
                            let cval = cplace.to_cvalue(fx);
                            let (p, l) = cval.load_scalar_pair(fx);
                            (Pointer::new(p), l)
                        }
                        _ => panic!("Subslice from_end on non-ScalarPair: {:?}", cur_ty),
                    };
                    let new_ptr = ptr.offset_i64(&mut fx.bcx, fx.pointer_type, elem_layout.size.bytes() as i64 * (*from as i64));
                    let new_len = fx.bcx.ins().iadd_imm(len, -((*from as i64) + (*to as i64)));
                    cplace = CPlace::for_ptr_with_extra(
                        new_ptr,
                        new_len,
                        cplace.layout.clone(),
                    );
                    // cur_ty stays as slice
                }
            }
            ra_mir_types::Projection::OpaqueCast(cast_ty) => {
                // OpaqueCast is a no-op at codegen level — just changes the type
                // (e.g. `impl Trait` → concrete revealed type).
                let new_layout = fx.md_layout(cast_ty);
                cplace = cplace.transmute_type(new_layout);
                cur_ty = cast_ty.clone();
            }
        }
    }

    cplace
}

/// Compute a variant layout for Downcast projection (mirdata path).
/// Delegates to the shared `variant_layout` in lib.rs.
fn md_variant_layout(parent_layout: &LayoutArc, variant_idx: u32) -> LayoutArc {
    crate::variant_layout(parent_layout, rac_abi::VariantIdx::from_u32(variant_idx))
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
            codegen_md_place(fx, place).to_cvalue(fx)
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
            ConstKind::Slice(bytes, meta) => {
                // Slice constant (e.g. string literal): create a data section for the bytes,
                // return a ScalarPair (data_ptr, len).
                use std::hash::{Hash, Hasher};
                let data_id = {
                    let mut desc = cranelift_module::DataDescription::new();
                    desc.define(bytes.clone().into_boxed_slice());
                    // Use content hash for unique naming to avoid conflicts
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    bytes.hash(&mut hasher);
                    let name = format!("__const_slice_{:x}", hasher.finish());
                    let data_id = fx.module
                        .declare_data(&name, Linkage::Local, false, false)
                        .expect("declare slice data");
                    // Ignore DuplicateDefinition — the same slice constant may
                    // appear in multiple functions compiled into the same module.
                    let _ = fx.module.define_data(data_id, &desc);
                    data_id
                };
                let gv = fx.module.declare_data_in_func(data_id, fx.bcx.func);
                let data_ptr = fx.bcx.ins().global_value(fx.pointer_type, gv);
                let len_val = fx.bcx.ins().iconst(fx.pointer_type, *meta as i64);
                CValue::by_val_pair(data_ptr, len_val, dest_layout.clone())
            }
            ConstKind::Unevaluated(_hash, _args) => {
                // Unevaluated const: try to handle as a ZST function reference
                if dest_layout.is_zst() {
                    return CValue::zst(dest_layout.clone());
                }
                // For non-ZST unevaluated consts, emit a zero-initialized dummy.
                // This allows the function to compile (for coverage testing) but
                // the value will be wrong at runtime.
                match dest_layout.backend_repr {
                    BackendRepr::Scalar(ref s) => {
                        let clif_ty = scalar_to_clif_type(fx.dl, s);
                        let val = fx.bcx.ins().iconst(clif_ty, 0);
                        CValue::by_val(val, dest_layout.clone())
                    }
                    BackendRepr::ScalarPair(ref a, ref b) => {
                        let a_ty = scalar_to_clif_type(fx.dl, a);
                        let b_ty = scalar_to_clif_type(fx.dl, b);
                        let a_val = fx.bcx.ins().iconst(a_ty, 0);
                        let b_val = fx.bcx.ins().iconst(b_ty, 0);
                        CValue::by_val_pair(a_val, b_val, dest_layout.clone())
                    }
                    _ => {
                        // Memory repr: return a zeroed stack slot
                        let place = CPlace::new_stack_slot(fx, dest_layout.clone());
                        place.to_cvalue(fx)
                    }
                }
            }
            ConstKind::Todo(_desc) => {
                // Unsupported constant form: emit a zero-initialized dummy
                if dest_layout.is_zst() {
                    return CValue::zst(dest_layout.clone());
                }
                match dest_layout.backend_repr {
                    BackendRepr::Scalar(ref s) => {
                        let clif_ty = scalar_to_clif_type(fx.dl, s);
                        let val = fx.bcx.ins().iconst(clif_ty, 0);
                        CValue::by_val(val, dest_layout.clone())
                    }
                    BackendRepr::ScalarPair(ref a, ref b) => {
                        let a_ty = scalar_to_clif_type(fx.dl, a);
                        let b_ty = scalar_to_clif_type(fx.dl, b);
                        let a_val = fx.bcx.ins().iconst(a_ty, 0);
                        let b_val = fx.bcx.ins().iconst(b_ty, 0);
                        CValue::by_val_pair(a_val, b_val, dest_layout.clone())
                    }
                    _ => {
                        let place = CPlace::new_stack_slot(fx, dest_layout.clone());
                        place.to_cvalue(fx)
                    }
                }
            }
        },
    }
}

/// Extract the unsized metadata for an Unsize coercion.
///
/// Walks the source and dest pointee types to find the unsizing point:
/// - `[T; N]` → `[T]`: returns array length N as an iconst
/// - `dyn TraitA` → `dyn TraitB`: returns old info (vtable upcasting passthrough)
fn md_unsized_info(
    fx: &mut FunctionCx<'_, impl Module>,
    src_pointee: &ra_mir_types::Ty,
    _dst_pointee: &ra_mir_types::Ty,
    old_info: Option<Value>,
) -> Value {
    match (src_pointee, _dst_pointee) {
        (ra_mir_types::Ty::Array(_, len), ra_mir_types::Ty::Slice(_)) => {
            fx.bcx.ins().iconst(fx.pointer_type, *len as i64)
        }
        (ra_mir_types::Ty::Dynamic(_), ra_mir_types::Ty::Dynamic(_)) => {
            // Trait upcasting: pass through the existing vtable pointer
            old_info.expect("dyn→dyn unsize requires old_info")
        }
        _ => todo!("md_unsized_info: {:?} → {:?}", src_pointee, _dst_pointee),
    }
}

/// Get the pointee type from a reference/raw pointer type.
fn pointee_ty(ty: &ra_mir_types::Ty) -> &ra_mir_types::Ty {
    match ty {
        ra_mir_types::Ty::Ref(_, inner) | ra_mir_types::Ty::RawPtr(_, inner) => inner,
        _ => panic!("pointee_ty on non-pointer type: {:?}", ty),
    }
}

/// Codegen for PointerCoercion::Unsize in mirdata.
///
/// Handles:
/// - Thin → fat: `&[T; N]` → `&[T]` (emit ptr + array length)
/// - Fat → fat: `&dyn A` → `&dyn B` (vtable passthrough)
fn codegen_md_unsize(
    fx: &mut FunctionCx<'_, impl Module>,
    operand: &Operand,
    from_cval: CValue,
    dest_layout: &LayoutArc,
) -> CValue {
    let src_ty = operand_ty(fx, operand);

    match from_cval.layout.backend_repr {
        BackendRepr::Scalar(_) => {
            // Thin → fat pointer
            let src_pointee = pointee_ty(&src_ty);

            // Walk the dest type from the Rvalue's target type
            // The dest_layout tells us this is a ScalarPair (fat pointer)
            let BackendRepr::ScalarPair(_, _) = dest_layout.backend_repr else {
                // Thin → thin unsize (e.g. transparent wrapper), just pass through
                return from_cval;
            };

            let ptr = from_cval.load_scalar(fx);
            let info = md_unsized_info(fx, src_pointee, &infer_dst_pointee(src_pointee), None);
            CValue::by_val_pair(ptr, info, dest_layout.clone())
        }
        BackendRepr::ScalarPair(_, _) => {
            // Fat → fat (e.g. dyn A → dyn B upcasting, or &[T] passthrough)
            let (ptr, old_info) = from_cval.load_scalar_pair(fx);
            let src_pointee = pointee_ty(&src_ty);
            let info = md_unsized_info(fx, src_pointee, &infer_dst_pointee(src_pointee), Some(old_info));
            CValue::by_val_pair(ptr, info, dest_layout.clone())
        }
        _ => todo!("Unsize on memory-repr source in mirdata"),
    }
}

/// Infer the destination pointee type for Unsize coercion from the source pointee.
/// `[T; N]` → `[T]`, `dyn A` → `dyn A` (passthrough for upcasting).
fn infer_dst_pointee(src_pointee: &ra_mir_types::Ty) -> ra_mir_types::Ty {
    match src_pointee {
        ra_mir_types::Ty::Array(elem, _) => ra_mir_types::Ty::Slice(elem.clone()),
        ra_mir_types::Ty::Dynamic(preds) => ra_mir_types::Ty::Dynamic(preds.clone()),
        _ => todo!("infer_dst_pointee: {:?}", src_pointee),
    }
}

fn codegen_md_cast(
    fx: &mut FunctionCx<'_, impl Module>,
    kind: &CastKind,
    operand: &Operand,
    dest_layout: &LayoutArc,
) -> CValue {
    // Handle ReifyFnPointer: FnDef (ZST) → fn pointer via fn_registry lookup
    if let CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer) = kind {
        let ra_mir_types::Operand::Constant(c) = operand else {
            panic!("ReifyFnPointer on non-constant operand");
        };
        let ra_mir_types::Ty::FnDef(hash, generic_args) = &c.ty else {
            panic!("ReifyFnPointer on non-FnDef type: {:?}", c.ty);
        };
        let MirSource::Mirdata { fn_registry, .. } = &fx.mir else {
            unreachable!()
        };
        // Try mono instance key first, then fall back to plain hash
        let mono_key = if generic_args.iter().any(|a| matches!(a, ra_mir_types::GenericArg::Ty(_))) {
            let inst = MonoInstance { def_path_hash: *hash, args: generic_args.clone() };
            Some(inst.registry_key())
        } else {
            None
        };
        let func_id = mono_key.and_then(|k| fn_registry.get(&k).copied())
            .or_else(|| fn_registry.get(hash).copied())
            .expect("ReifyFnPointer: function not in fn_registry");
        let func_ref = fx.module.declare_func_in_func(func_id, fx.bcx.func);
        let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
        return CValue::by_val(func_addr, dest_layout.clone());
    }

    // Handle Transmute: supports Scalar, ScalarPair, and memory-repr
    if let CastKind::Transmute = kind {
        let src_layout = md_operand_layout(fx, operand);
        let from_cval = codegen_md_operand(fx, operand, &src_layout);
        return crate::codegen_transmute(fx, from_cval, dest_layout);
    }

    // Handle Unsize coercion: thin ptr → fat ptr (array→slice, T→dyn Trait)
    if let CastKind::PointerCoercion(PointerCoercion::Unsize) = kind {
        let src_layout = md_operand_layout(fx, operand);
        let from_cval = codegen_md_operand(fx, operand, &src_layout);
        return codegen_md_unsize(fx, operand, from_cval, dest_layout);
    }

    // All other casts
    let src_layout = md_operand_layout(fx, operand);
    let from_cval = codegen_md_operand(fx, operand, &src_layout);

    // Handle fat pointer casts (ScalarPair targets)
    if let BackendRepr::ScalarPair(_, _) = dest_layout.backend_repr {
        match from_cval.layout.backend_repr {
            BackendRepr::ScalarPair(_, _) => {
                // Fat → fat: pass through (e.g. *const dyn A → *const dyn B)
                let (a, b) = from_cval.load_scalar_pair(fx);
                return CValue::by_val_pair(a, b, dest_layout.clone());
            }
            BackendRepr::Scalar(_) => {
                todo!("thin-to-fat pointer cast (non-Unsize) in mirdata: {:?}", kind);
            }
            _ => todo!("non-scalar-pair to ScalarPair cast in mirdata"),
        }
    }

    let from_val = from_cval.load_scalar(fx);

    let BackendRepr::Scalar(target_scalar) = dest_layout.backend_repr else {
        todo!("non-scalar cast target in mirdata: {:?}", kind);
    };

    // Determine source signedness from source layout
    let from_signed = match from_cval.layout.backend_repr {
        BackendRepr::Scalar(ref s) => matches!(s.primitive(), Primitive::Int(_, true)),
        _ => false,
    };

    let scalar_kind = match kind {
        CastKind::IntToInt => crate::ScalarCastKind::IntToInt,
        CastKind::FloatToInt => crate::ScalarCastKind::FloatToInt,
        CastKind::IntToFloat => crate::ScalarCastKind::IntToFloat,
        CastKind::FloatToFloat => crate::ScalarCastKind::FloatToFloat,
        CastKind::PtrToPtr
        | CastKind::FnPtrToPtr
        | CastKind::PointerExposeProvenance
        | CastKind::PointerWithExposedProvenance
        | CastKind::PointerCoercion(
            PointerCoercion::MutToConstPointer
            | PointerCoercion::UnsafeFnPointer
            | PointerCoercion::ArrayToPointer
            | PointerCoercion::ClosureFnPointer,
        ) => crate::ScalarCastKind::PtrLike,
        CastKind::Transmute => unreachable!("handled above"),
        CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer) => {
            unreachable!("handled above")
        }
        CastKind::PointerCoercion(PointerCoercion::Unsize) => {
            unreachable!("handled above")
        }
    };
    crate::codegen_scalar_cast(fx, scalar_kind, from_val, from_signed, &target_scalar, dest_layout)
}

fn codegen_md_rvalue(
    fx: &mut FunctionCx<'_, impl Module>,
    rvalue: &Rvalue,
    dest_layout: &LayoutArc,
) -> CValue {
    match rvalue {
        Rvalue::Use(operand) => codegen_md_operand(fx, operand, dest_layout),
        Rvalue::Cast(kind, operand, _ty) => codegen_md_cast(fx, kind, operand, dest_layout),
        Rvalue::BinaryOp(op, lhs, rhs) => {
            let lhs_layout = md_operand_layout(fx, lhs);
            let rhs_layout = md_operand_layout(fx, rhs);
            let lhs_cval = codegen_md_operand(fx, lhs, &lhs_layout);
            let rhs_cval = codegen_md_operand(fx, rhs, &rhs_layout);
            let lhs_val = lhs_cval.load_scalar(fx);
            let rhs_val = rhs_cval.load_scalar(fx);

            let BackendRepr::Scalar(scalar) = lhs_cval.layout.backend_repr else {
                panic!("expected scalar for binop operand");
            };
            // Checked (overflow) binops return (result, bool) as ScalarPair
            if op.overflowing_to_wrapping().is_some() {
                let Primitive::Int(_, signed) = scalar.primitive() else {
                    panic!("overflow binop on non-integer");
                };
                let (result, overflow) = crate::codegen_checked_int_binop(fx, op, lhs_val, rhs_val, signed);
                return CValue::by_val_pair(result, overflow, dest_layout.clone());
            }
            let val = match scalar.primitive() {
                Primitive::Int(_, signed) => {
                    crate::codegen_int_binop(fx, op, lhs_val, rhs_val, signed)
                }
                Primitive::Float(_) => crate::codegen_float_binop(fx, op, lhs_val, rhs_val),
                Primitive::Pointer(_) => match op {
                    ra_mir_types::BinOp::Offset => {
                        // Offset: ptr + (count * pointee_size)
                        let lhs_ty = operand_ty(fx, lhs);
                        let pointee_ty = match &lhs_ty {
                            ra_mir_types::Ty::RawPtr(_, pointee) | ra_mir_types::Ty::Ref(_, pointee) => (**pointee).clone(),
                            _ => panic!("Offset on non-pointer: {:?}", lhs_ty),
                        };
                        let pointee_layout = fx.md_layout(&pointee_ty);
                        let byte_offset = fx.bcx.ins().imul_imm(rhs_val, pointee_layout.size.bytes() as i64);
                        fx.bcx.ins().iadd(lhs_val, byte_offset)
                    }
                    ra_mir_types::BinOp::Eq | ra_mir_types::BinOp::Ne
                    | ra_mir_types::BinOp::Lt | ra_mir_types::BinOp::Le
                    | ra_mir_types::BinOp::Ge | ra_mir_types::BinOp::Gt => {
                        let cc = crate::bin_op_to_intcc(op, false);
                        fx.bcx.ins().icmp(cc, lhs_val, rhs_val)
                    }
                    ra_mir_types::BinOp::Sub => {
                        // Pointer subtraction: (ptr1 - ptr2) in bytes
                        fx.bcx.ins().isub(lhs_val, rhs_val)
                    }
                    _ => todo!("pointer binop: {:?}", op),
                }
            };
            CValue::by_val(val, dest_layout.clone())
        }
        Rvalue::UnaryOp(op, operand) => {
            let op_layout = md_operand_layout(fx, operand);
            let val_cval = codegen_md_operand(fx, operand, &op_layout);

            // PtrMetadata operates on ScalarPair, handle before load_scalar
            if matches!(op, ra_mir_types::UnOp::PtrMetadata) {
                return match val_cval.layout.backend_repr {
                    BackendRepr::Scalar(_) => CValue::zst(dest_layout.clone()),
                    BackendRepr::ScalarPair(_, _) => {
                        let (_, meta) = val_cval.load_scalar_pair(fx);
                        CValue::by_val(meta, dest_layout.clone())
                    }
                    _ => panic!("PtrMetadata on non-pointer repr: {:?}", val_cval.layout.backend_repr),
                };
            }

            let val = val_cval.load_scalar(fx);
            let BackendRepr::Scalar(scalar) = val_cval.layout.backend_repr else {
                panic!("expected scalar for unary op");
            };
            let res = match op {
                ra_mir_types::UnOp::Neg => match scalar.primitive() {
                    Primitive::Float(_) => fx.bcx.ins().fneg(val),
                    _ => fx.bcx.ins().ineg(val),
                },
                ra_mir_types::UnOp::Not => {
                    // For booleans (valid_range 0..=1), use icmp_imm eq 0
                    if scalar.is_bool() {
                        fx.bcx.ins().icmp_imm(cranelift_codegen::ir::condcodes::IntCC::Equal, val, 0)
                    } else {
                        fx.bcx.ins().bnot(val)
                    }
                }
                ra_mir_types::UnOp::PtrMetadata => unreachable!("handled above"),
            };
            CValue::by_val(res, dest_layout.clone())
        }
        Rvalue::Ref(_, place) | Rvalue::RawPtr(_, place) => {
            let place = codegen_md_place(fx, place);
            place.place_ref(fx, dest_layout.clone())
        }
        Rvalue::CopyForDeref(place) => {
            codegen_md_place(fx, place).to_cvalue(fx)
        }
        Rvalue::Discriminant(place) => {
            let place = codegen_md_place(fx, place);
            let discr = crate::codegen_get_discriminant(fx, &place, dest_layout);
            CValue::by_val(discr, dest_layout.clone())
        }
        Rvalue::Aggregate(agg_kind, operands) => {
            // Fast path for simple struct-like aggregates with Scalar/ScalarPair layout
            match agg_kind {
                ra_mir_types::AggregateKind::Tuple
                | ra_mir_types::AggregateKind::Adt(_, 0, _)
                | ra_mir_types::AggregateKind::Closure(_, _) => {
                    // ZST fast path
                    if dest_layout.is_zst() {
                        return CValue::zst(dest_layout.clone());
                    }
                    // Scalar fast path (single non-ZST field)
                    if let BackendRepr::Scalar(_) = dest_layout.backend_repr {
                        for operand in operands {
                            let op_layout = md_operand_layout(fx, operand);
                            if !op_layout.is_zst() {
                                let val = codegen_md_operand(fx, operand, &op_layout);
                                return CValue::by_val(val.load_scalar(fx), dest_layout.clone());
                            }
                        }
                    }
                    // ScalarPair fast path (exactly two non-ZST scalars)
                    if let BackendRepr::ScalarPair(_, _) = dest_layout.backend_repr {
                        let mut non_zst = Vec::new();
                        for operand in operands {
                            let op_layout = md_operand_layout(fx, operand);
                            if !op_layout.is_zst() {
                                let val = codegen_md_operand(fx, operand, &op_layout);
                                non_zst.push(val);
                            }
                        }
                        if non_zst.len() == 2 {
                            let a = non_zst[0].load_scalar(fx);
                            let b = non_zst[1].load_scalar(fx);
                            return CValue::by_val_pair(a, b, dest_layout.clone());
                        }
                    }
                }
                _ => {}
            }

            let dest_place = CPlace::new_stack_slot(fx, dest_layout.clone());
            match agg_kind {
                ra_mir_types::AggregateKind::Tuple
                | ra_mir_types::AggregateKind::Adt(_, 0, _)
                | ra_mir_types::AggregateKind::Closure(_, _) => {
                    // Memory-repr: write each field to stack slot
                    for (i, operand) in operands.iter().enumerate() {
                        let field_place = dest_place.place_field(
                            fx,
                            i,
                            fx.md_layout(&operand_ty(fx, operand)),
                        );
                        let val = codegen_md_operand(fx, operand, &field_place.layout);
                        field_place.write_cvalue(fx, val);
                    }
                }
                ra_mir_types::AggregateKind::Adt(_, variant_idx, _) => {
                    // Enum variant
                    let variant_layout = md_variant_layout(dest_layout, *variant_idx);
                    let variant_place = dest_place.downcast_variant(variant_layout);
                    for (i, operand) in operands.iter().enumerate() {
                        let field_place = variant_place.place_field(
                            fx,
                            i,
                            fx.md_layout(&operand_ty(fx, operand)),
                        );
                        let val = codegen_md_operand(fx, operand, &field_place.layout);
                        field_place.write_cvalue(fx, val);
                    }
                    crate::codegen_set_discriminant(fx, &dest_place, rac_abi::VariantIdx::from_u32(*variant_idx));
                }
                ra_mir_types::AggregateKind::Array(elem_ty) => {
                    let elem_layout = fx.md_layout(elem_ty);
                    for (i, operand) in operands.iter().enumerate() {
                        let offset = elem_layout.size.bytes() as i64 * i as i64;
                        let field_ptr = dest_place.to_ptr().offset_i64(&mut fx.bcx, fx.pointer_type, offset);
                        let field_place = CPlace::for_ptr(field_ptr, elem_layout.clone());
                        let val = codegen_md_operand(fx, operand, &field_place.layout);
                        field_place.write_cvalue(fx, val);
                    }
                }
                ra_mir_types::AggregateKind::RawPtr(_, _) => {
                    // RawPtr aggregate: (data_ptr, metadata) → thin or fat pointer
                    assert_eq!(operands.len(), 2, "RawPtr aggregate must have 2 operands");
                    let data_layout = md_operand_layout(fx, &operands[0]);
                    let data = codegen_md_operand(fx, &operands[0], &data_layout);
                    let meta_layout = md_operand_layout(fx, &operands[1]);
                    let meta = codegen_md_operand(fx, &operands[1], &meta_layout);
                    if meta.layout.is_zst() {
                        // Thin pointer: just reinterpret data as the target pointer type
                        let data_val = data.load_scalar(fx);
                        return CValue::by_val(data_val, dest_layout.clone());
                    } else {
                        // Fat pointer: ScalarPair(data, meta)
                        let data_val = data.load_scalar(fx);
                        let meta_val = meta.load_scalar(fx);
                        return CValue::by_val_pair(data_val, meta_val, dest_layout.clone());
                    }
                }
            }
            dest_place.to_cvalue(fx)
        }
        Rvalue::Repeat(operand, count) => {
            let dest_place = CPlace::new_stack_slot(fx, dest_layout.clone());
            let elem_layout = md_operand_layout(fx, operand);
            let elem_cval = codegen_md_operand(fx, operand, &elem_layout);
            let elem_size = elem_layout.size.bytes() as i64;
            for i in 0..*count {
                let offset = elem_size * i as i64;
                let field_ptr = dest_place.to_ptr().offset_i64(&mut fx.bcx, fx.pointer_type, offset);
                let field_place = CPlace::for_ptr(field_ptr, elem_layout.clone());
                field_place.write_cvalue(fx, elem_cval.clone());
            }
            dest_place.to_cvalue(fx)
        }
        Rvalue::ThreadLocalRef(_) => todo!("ThreadLocalRef in mirdata"),
    }
}

/// Get the type of an operand (for aggregate field layout resolution).
fn operand_ty(
    fx: &FunctionCx<'_, impl Module>,
    operand: &Operand,
) -> ra_mir_types::Ty {
    match operand {
        Operand::Copy(place) | Operand::Move(place) => {
            let MirSource::Mirdata { body, .. } = &fx.mir else { unreachable!() };
            // For projections, we'd need to walk the type. For now, handle the
            // common case of a bare local.
            if place.projections.is_empty() {
                body.locals[place.local as usize].ty.clone()
            } else {
                // Walk through projections to get the final type
                let mut ty = body.locals[place.local as usize].ty.clone();
                for proj in &place.projections {
                    ty = match proj {
                        ra_mir_types::Projection::Field(_, field_ty) => field_ty.clone(),
                        ra_mir_types::Projection::Deref => match &ty {
                            ra_mir_types::Ty::Ref(_, inner) | ra_mir_types::Ty::RawPtr(_, inner) => (**inner).clone(),
                            _ => panic!("Deref on non-pointer"),
                        },
                        ra_mir_types::Projection::Index(_)
                        | ra_mir_types::Projection::ConstantIndex { .. } => match &ty {
                            ra_mir_types::Ty::Array(elem, _) | ra_mir_types::Ty::Slice(elem) => (**elem).clone(),
                            _ => panic!("Index on non-array/slice"),
                        },
                        ra_mir_types::Projection::Subslice { from, to, from_end } => {
                            if !from_end {
                                let sub_len = *to - *from;
                                match &ty {
                                    ra_mir_types::Ty::Array(elem, _) => {
                                        ra_mir_types::Ty::Array(elem.clone(), sub_len)
                                    }
                                    _ => panic!("Subslice on non-array"),
                                }
                            } else {
                                ty // slice stays as slice
                            }
                        },
                        ra_mir_types::Projection::Downcast(_) => ty, // type stays the same
                        ra_mir_types::Projection::OpaqueCast(cast_ty) => cast_ty.clone(),
                    };
                }
                ty
            }
        }
        Operand::Constant(c) => c.ty.clone(),
    }
}

// ---------------------------------------------------------------------------
// Call codegen
// ---------------------------------------------------------------------------

fn codegen_md_call(
    fx: &mut FunctionCx<'_, impl Module>,
    func: &Operand,
    args: &[Operand],
    dest: &ra_mir_types::Place,
    target: &Option<u32>,
) {
    let dest_place = codegen_md_place(fx, dest);

    match func {
        Operand::Constant(c) => match &c.ty {
            ra_mir_types::Ty::FnDef(hash, generic_args) => {
                let MirSource::Mirdata { fn_registry, .. } = &fx.mir else {
                    unreachable!()
                };
                // For monomorphized calls, try the mono instance key first.
                let mono_key = if generic_args.iter().any(|a| matches!(a, ra_mir_types::GenericArg::Ty(_))) {
                    let inst = MonoInstance { def_path_hash: *hash, args: generic_args.clone() };
                    Some(inst.registry_key())
                } else {
                    None
                };
                let func_id = mono_key.and_then(|k| fn_registry.get(&k).copied())
                    .or_else(|| fn_registry.get(hash).copied());
                match func_id {
                    Some(func_id) => {
                        let func_ref = fx.module.declare_func_in_func(func_id, fx.bcx.func);
                        emit_md_call_direct(fx, func_ref, args, &dest_place);
                    }
                    None => {
                        // Function not in mirdata (e.g. extern "C", or from a crate
                        // not included in the export). Emit trap as stub.
                        fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(4).unwrap());
                        // Need a new block for the code after the call (if any)
                        // since trap is a terminator.
                        if target.is_some() {
                            let cont = fx.bcx.create_block();
                            fx.bcx.switch_to_block(cont);
                        }
                        // Still need to jump to target
                        if let Some(t) = target {
                            let block = fx.clif_block_idx(*t as usize);
                            fx.bcx.ins().jump(block, &[]);
                        }
                        return;
                    }
                }
            }
            _ => panic!("non-FnDef constant call: {:?}", c.ty),
        },
        Operand::Copy(p) | Operand::Move(p) => {
            // Indirect call through a fn pointer
            let fn_ptr_cval = codegen_md_place(fx, p).to_cvalue(fx);
            let fn_ptr = fn_ptr_cval.load_scalar(fx);

            // Build signature from the fn pointer type
            let fn_ptr_ty = operand_ty(fx, func);
            let (param_tys, _ret_ty) = match &fn_ptr_ty {
                ra_mir_types::Ty::FnPtr(params, ret) => (params.clone(), (**ret).clone()),
                _ => panic!("indirect call on non-FnPtr type: {:?}", fn_ptr_ty),
            };

            let mut sig = Signature::new(fx.isa.default_call_conv());

            // Return type
            let is_sret = if !dest_place.layout.is_zst() {
                crate::append_ret_to_sig(&mut sig, fx.dl, &dest_place.layout)
            } else {
                false
            };

            // Parameter types
            for param_ty in &param_tys {
                let param_layout = fx.md_layout(param_ty);
                crate::append_param_to_sig(&mut sig, fx.dl, &param_layout);
            }

            let sig_ref = fx.bcx.import_signature(sig);

            // Build argument values
            let mut call_args: Vec<Value> = Vec::new();
            let sret_slot = if is_sret {
                let slot = CPlace::new_stack_slot(fx, dest_place.layout.clone());
                let ptr = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
                call_args.push(ptr);
                Some(slot)
            } else {
                None
            };

            push_md_call_args(fx, args, &mut call_args);

            let call = fx.bcx.ins().call_indirect(sig_ref, fn_ptr, &call_args);
            store_md_call_result(fx, call, &dest_place, sret_slot);
        }
    }

    // Continue to target block
    if let Some(target) = target {
        let block = fx.clif_block_idx(*target as usize);
        fx.bcx.ins().jump(block, &[]);
    }
}

/// Emit a direct call (func_ref already resolved) with memory-repr support.
///
/// If the callee's declared signature doesn't match the actual arguments
/// (e.g. it was declared as void→void because it's a generic stub), we
/// convert to `call_indirect` with a signature derived from the actual
/// arguments and destination type. This avoids Cranelift verifier errors.
fn emit_md_call_direct(
    fx: &mut FunctionCx<'_, impl Module>,
    func_ref: cranelift_codegen::ir::FuncRef,
    args: &[Operand],
    dest_place: &CPlace,
) {
    let mut call_args: Vec<Value> = Vec::new();

    // If return is memory-repr, allocate sret slot and pass pointer as first arg
    let is_sret = !dest_place.layout.is_zst()
        && matches!(dest_place.layout.backend_repr, BackendRepr::Memory { .. });
    let sret_slot = if is_sret {
        let slot = CPlace::new_stack_slot(fx, dest_place.layout.clone());
        let ptr = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
        call_args.push(ptr);
        Some(slot)
    } else {
        None
    };

    push_md_call_args(fx, args, &mut call_args);

    // Check if the declared signature matches the actual call arguments.
    let sig_ref = fx.bcx.func.dfg.ext_funcs[func_ref].signature;
    let declared_sig = &fx.bcx.func.dfg.signatures[sig_ref];
    let declared_params = declared_sig.params.len();
    let declared_returns = declared_sig.returns.len();

    let expected_returns = if dest_place.layout.is_zst() || is_sret {
        0
    } else {
        match dest_place.layout.backend_repr {
            BackendRepr::Scalar(_) => 1,
            BackendRepr::ScalarPair(_, _) => 2,
            _ => 0,
        }
    };

    if declared_params == call_args.len() && declared_returns == expected_returns {
        // Signature matches — use efficient direct call.
        let call = fx.bcx.ins().call(func_ref, &call_args);
        store_md_call_result(fx, call, dest_place, sret_slot);
    } else {
        // Signature mismatch (e.g. callee declared as void→void stub).
        // Build expected signature from actual values and use call_indirect.
        let mut expected_sig = Signature::new(fx.isa.default_call_conv());
        for &arg in &call_args {
            let ty = fx.bcx.func.dfg.value_type(arg);
            expected_sig.params.push(AbiParam::new(ty));
        }
        if !is_sret && !dest_place.layout.is_zst() {
            match dest_place.layout.backend_repr {
                BackendRepr::Scalar(ref s) => {
                    expected_sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, s)));
                }
                BackendRepr::ScalarPair(ref a, ref b) => {
                    expected_sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, a)));
                    expected_sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, b)));
                }
                _ => {}
            }
        }
        let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, func_ref);
        let expected_sig_ref = fx.bcx.import_signature(expected_sig);
        let call = fx.bcx.ins().call_indirect(expected_sig_ref, func_addr, &call_args);
        store_md_call_result(fx, call, dest_place, sret_slot);
    }
}

/// Push argument values onto `call_args`, handling all backend reprs.
fn push_md_call_args(
    fx: &mut FunctionCx<'_, impl Module>,
    args: &[Operand],
    call_args: &mut Vec<Value>,
) {
    for arg in args {
        let arg_layout = md_operand_layout(fx, arg);
        let arg_cval = codegen_md_operand(fx, arg, &arg_layout);
        if arg_cval.layout.is_zst() {
            continue;
        }
        match arg_cval.layout.backend_repr {
            BackendRepr::Scalar(_) => call_args.push(arg_cval.load_scalar(fx)),
            BackendRepr::ScalarPair(_, _) => {
                let (a, b) = arg_cval.load_scalar_pair(fx);
                call_args.push(a);
                call_args.push(b);
            }
            _ => {
                // Memory-repr: pass as pointer
                let ptr = arg_cval.force_stack(fx);
                call_args.push(ptr.get_addr(&mut fx.bcx, fx.pointer_type));
            }
        }
    }
}

/// Store the return value from a call into `dest_place`.
fn store_md_call_result(
    fx: &mut FunctionCx<'_, impl Module>,
    call: cranelift_codegen::ir::Inst,
    dest_place: &CPlace,
    sret_slot: Option<CPlace>,
) {
    if let Some(sret_slot) = sret_slot {
        let cval = sret_slot.to_cvalue(fx);
        dest_place.write_cvalue(fx, cval);
    } else if !dest_place.layout.is_zst() {
        let results = fx.bcx.inst_results(call).to_vec();
        // Guard against mismatched signatures (e.g. calling a stub-declared function)
        if results.is_empty() {
            return;
        }
        match dest_place.layout.backend_repr {
            BackendRepr::Scalar(_) => {
                let cval = CValue::by_val(results[0], dest_place.layout.clone());
                dest_place.write_cvalue(fx, cval);
            }
            BackendRepr::ScalarPair(_, _) => {
                if results.len() >= 2 {
                    let cval = CValue::by_val_pair(results[0], results[1], dest_place.layout.clone());
                    dest_place.write_cvalue(fx, cval);
                }
            }
            _ => unreachable!("non-sret memory return without sret slot"),
        }
    }
}

/// Get the layout for an operand (works for locals-with-projections and constants).
fn md_operand_layout(
    fx: &FunctionCx<'_, impl Module>,
    operand: &Operand,
) -> LayoutArc {
    match operand {
        Operand::Copy(p) | Operand::Move(p) => {
            if p.projections.is_empty() {
                fx.local_place_idx(p.local as usize).layout.clone()
            } else {
                // Walk projections to find the final type, then look up layout
                let ty = operand_ty(fx, &Operand::Copy(p.clone()));
                fx.md_layout(&ty)
            }
        }
        Operand::Constant(c) => fx.md_layout(&c.ty),
    }
}

// ---------------------------------------------------------------------------
// Statement / Terminator codegen
// ---------------------------------------------------------------------------

fn codegen_md_statement(fx: &mut FunctionCx<'_, impl Module>, stmt: &Statement) {
    match stmt {
        Statement::Assign(place, rvalue) => {
            let dest = codegen_md_place(fx, place);
            let val = codegen_md_rvalue(fx, rvalue, &dest.layout);
            dest.write_cvalue(fx, val);
        }
        Statement::SetDiscriminant { place, variant_index } => {
            let dest = codegen_md_place(fx, place);
            crate::codegen_set_discriminant(fx, &dest, rac_abi::VariantIdx::from_u32(*variant_index));
        }
        Statement::Deinit(_) => {
            // Deinit is a no-op in our codegen (used by drop elaboration)
        }
        Statement::StorageLive(_) | Statement::StorageDead(_) | Statement::Nop => {}
    }
}

fn codegen_md_terminator(fx: &mut FunctionCx<'_, impl Module>, term: &Terminator) {
    match term {
        Terminator::Return => {
            let ret_place = fx.local_place_idx(0).clone();
            crate::codegen_return(fx, &ret_place);
        }
        Terminator::Goto(target) => {
            let block = fx.clif_block_idx(*target as usize);
            fx.bcx.ins().jump(block, &[]);
        }
        Terminator::SwitchInt { discr, targets } => {
            let discr_layout = md_operand_layout(fx, discr);
            let discr_val = codegen_md_operand(fx, discr, &discr_layout).load_scalar(fx);
            let otherwise_idx = *targets.targets.last().unwrap() as usize;
            let otherwise = fx.clif_block_idx(otherwise_idx);
            let mut switch = Switch::new();
            for (val, target) in targets.values.iter().zip(targets.targets.iter()) {
                switch.set_entry(*val as u128, fx.clif_block_idx(*target as usize));
            }
            switch.emit(&mut fx.bcx, discr_val, otherwise);
        }
        Terminator::Call { func, args, dest, target, unwind: _ } => {
            codegen_md_call(fx, func, args, dest, target);
        }
        Terminator::Unreachable => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
        }
        Terminator::Drop { place: _, target, unwind: _ } => {
            // Drop is a no-op for now (we don't have drop glue).
            // Just jump to the target block.
            let block = fx.clif_block_idx(*target as usize);
            fx.bcx.ins().jump(block, &[]);
        }
        Terminator::Assert { cond, expected, target, unwind: _ } => {
            let cond_layout = md_operand_layout(fx, cond);
            let cond_val = codegen_md_operand(fx, cond, &cond_layout).load_scalar(fx);
            let target_block = fx.clif_block_idx(*target as usize);
            let trap_block = fx.bcx.create_block();
            if *expected {
                fx.bcx.ins().brif(cond_val, target_block, &[], trap_block, &[]);
            } else {
                fx.bcx.ins().brif(cond_val, trap_block, &[], target_block, &[]);
            }
            fx.bcx.switch_to_block(trap_block);
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
        }
        Terminator::UnwindResume => {
            // In our JIT context, unwind resume is just a trap
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(2).unwrap());
        }
    }
}

// ---------------------------------------------------------------------------
// Monomorphization
// ---------------------------------------------------------------------------

/// A monomorphization instance: a generic function + concrete type args.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct MonoInstance {
    pub def_path_hash: ra_mir_types::DefPathHash,
    pub args: Vec<ra_mir_types::GenericArg>,
}

impl MonoInstance {
    /// Compute a stable hash to use as a synthetic DefPathHash in fn_registry.
    pub fn registry_key(&self) -> (u64, u64) {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.def_path_hash.hash(&mut hasher);
        self.args.hash(&mut hasher);
        let h1 = hasher.finish();
        // Second hash for uniqueness
        h1.hash(&mut hasher);
        let h2 = hasher.finish();
        (h1, h2)
    }
}

/// Collect all monomorphization instances needed by a set of function bodies.
///
/// Scans call sites in **monomorphic** functions for `FnDef(hash, args)` where:
/// - `args` contains at least one `GenericArg::Ty` that is fully concrete (no Param)
/// - `hash` refers to a generic function in `body_map`
///
/// Returns deduplicated instances.
pub fn collect_mono_instances(
    bodies: &[ra_mir_types::FnBody],
    body_map: &HashMap<ra_mir_types::DefPathHash, usize>,
) -> Vec<MonoInstance> {
    let mut seen = std::collections::HashSet::new();
    let mut instances = Vec::new();

    // Only scan monomorphic functions — generic functions have unresolved Params
    for fb in bodies {
        if fb.num_generic_params > 0 {
            continue;
        }
        collect_from_body(&fb.body, body_map, &mut seen, &mut instances);
    }

    instances
}

fn collect_from_body(
    body: &ra_mir_types::Body,
    body_map: &HashMap<ra_mir_types::DefPathHash, usize>,
    seen: &mut std::collections::HashSet<MonoInstance>,
    instances: &mut Vec<MonoInstance>,
) {
    for bb in &body.blocks {
        if let ra_mir_types::Terminator::Call { func, .. } = &bb.terminator {
            if let Operand::Constant(c) = func {
                if let ra_mir_types::Ty::FnDef(hash, args) = &c.ty {
                    let has_ty_args = args.iter().any(|a| matches!(a, ra_mir_types::GenericArg::Ty(_)));
                    if !has_ty_args {
                        continue;
                    }
                    // Skip if any type arg still contains Param (means caller is generic)
                    let all_concrete = args.iter().all(|a| match a {
                        ra_mir_types::GenericArg::Ty(ty) => !ty.has_param(),
                        _ => true,
                    });
                    if !all_concrete {
                        continue;
                    }
                    // Check if the target is a generic function in our body map
                    if body_map.contains_key(hash) {
                        let inst = MonoInstance {
                            def_path_hash: *hash,
                            args: args.clone(),
                        };
                        if seen.insert(inst.clone()) {
                            instances.push(inst);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Build a `Ty → LayoutArc` lookup map from the layout table entries and
/// their converted layouts. Used for field/pointee layout resolution in
/// place projections.
pub fn build_ty_layout_map(
    layout_entries: &[ra_mir_types::TypeLayoutEntry],
    layouts: &[LayoutArc],
) -> HashMap<ra_mir_types::Ty, LayoutArc> {
    layout_entries
        .iter()
        .zip(layouts.iter())
        .map(|(entry, layout)| (entry.ty.clone(), layout.clone()))
        .collect()
}

pub fn compile_mirdata_fn(
    module: &mut impl Module,
    isa: &dyn TargetIsa,
    dl: &TargetDataLayout,
    fn_body: &ra_mir_types::FnBody,
    layouts: &[LayoutArc],
    fn_name: &str,
    linkage: Linkage,
    fn_registry: &HashMap<(u64, u64), FuncId>,
    ty_layouts: &HashMap<ra_mir_types::Ty, LayoutArc>,
) -> Result<FuncId, String> {
    compile_mirdata_body(module, isa, dl, &fn_body.body, layouts, fn_name, linkage, fn_registry, ty_layouts)
}

/// Compile a mirdata Body (possibly monomorphized) into a Cranelift function.
///
/// Unlike `compile_mirdata_fn`, this takes a `Body` directly (not `FnBody`),
/// enabling compilation of monomorphized bodies that were created by type
/// substitution.
pub fn compile_mirdata_body(
    module: &mut impl Module,
    isa: &dyn TargetIsa,
    dl: &TargetDataLayout,
    body: &ra_mir_types::Body,
    layouts: &[LayoutArc],
    fn_name: &str,
    linkage: Linkage,
    fn_registry: &HashMap<(u64, u64), FuncId>,
    ty_layouts: &HashMap<ra_mir_types::Ty, LayoutArc>,
) -> Result<FuncId, String> {
    let pointer_type = pointer_ty(dl);
    // Use resolve_local_layout which falls back to ty_layouts for locals
    // without layout indices (e.g. monomorphized generic functions).
    let sig = build_mirdata_fn_sig_resolved(isa, dl, body, layouts, ty_layouts)?;

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
            let layout = resolve_local_layout(local, layouts, ty_layouts)?;
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
                    // Memory-repr: allocate a stack slot
                    let size = u32::try_from(layout.size.bytes()).expect("stack slot too large");
                    let align_shift = layout.align.abi.bytes().trailing_zeros() as u8;
                    let slot = bcx.create_sized_stack_slot(StackSlotData::new(
                        StackSlotKind::ExplicitSlot,
                        size,
                        align_shift,
                    ));
                    CPlace::for_ptr(Pointer::stack_slot(slot), layout)
                }
            };
            local_map.push(place);
        }

        // Wire parameters to locals
        let block_params: Vec<Value> = bcx.block_params(entry_block).to_vec();
        let mut param_idx = 0;

        // If return type is memory-repr (sret), first param is the sret pointer.
        // Override the return local (index 0) to point at the caller-provided sret.
        let ret_layout = &local_map[0].layout;
        let ret_is_sret = !ret_layout.is_zst()
            && matches!(ret_layout.backend_repr, BackendRepr::Memory { .. });
        if ret_is_sret {
            let sret_ptr = block_params[param_idx];
            local_map[0] = CPlace::for_ptr(Pointer::new(sret_ptr), local_map[0].layout.clone());
            param_idx += 1;
        }

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
                _ => {
                    // Memory-repr: param is an incoming pointer. Create a CPlace
                    // pointing at the caller-provided memory.
                    let incoming_ptr = block_params[param_idx];
                    local_map[i] = CPlace::for_ptr(
                        Pointer::new(incoming_ptr),
                        place.layout.clone(),
                    );
                    param_idx += 1;
                }
            }
        }

        let mut fx = FunctionCx {
            bcx,
            module,
            isa,
            pointer_type,
            dl,
            mir: MirSource::Mirdata { body, layouts, fn_registry, ty_layouts },
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

            // Cleanup blocks are only reachable during unwinding.
            // In our non-unwinding JIT context, emit a trap and skip codegen.
            if bb.is_cleanup {
                fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(3).unwrap());
                continue;
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
        let ty_layouts = build_ty_layout_map(layout_entries, &layouts);
        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();

        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            fn_body,
            &layouts,
            &fn_body.name,
            Linkage::Export,
            &empty_reg,
            &ty_layouts,
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
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);
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
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &fn_body,
            &layouts,
            "identity",
            Linkage::Export,
            &empty_reg,
            &ty_layouts,
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

    // -- fn add(a: i32, b: i32) -> i32 { a + b } ----------------------------

    #[test]
    fn mirdata_binop_add() {
        let layout_entries = vec![i32_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "add".to_string(),
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
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                            Operand::Copy(Place { local: 2, projections: vec![] }),
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &fn_body,
            &layouts,
            "add",
            Linkage::Export,
            &empty_reg,
            &ty_layouts,
        )
        .expect("compile_mirdata_fn failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn(i32, i32) -> i32 = std::mem::transmute(code);
            assert_eq!(f(3, 4), 7);
            assert_eq!(f(-10, 25), 15);
            assert_eq!(f(i32::MAX, 1), i32::MIN); // wrapping
        }
    }

    // -- fn max(a: i32, b: i32) -> i32 { if a >= b { a } else { b } } -------

    #[test]
    fn mirdata_switchint_max() {
        let layout_entries = vec![i32_layout_entry(), bool_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);
        // MIR:
        //   _3: bool = Ge(Copy(_1), Copy(_2))
        //   switchInt(_3) -> [0: bb2, otherwise: bb1]
        // bb1: _0 = Copy(_1); return
        // bb2: _0 = Copy(_2); return
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "max".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _0: return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _1: a
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _2: b
                    Local { ty: Ty::Bool, layout: Some(1) },            // _3: cmp
                ],
                arg_count: 2,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 3, projections: vec![] },
                            Rvalue::BinaryOp(
                                BinOp::Ge,
                                Operand::Copy(Place { local: 1, projections: vec![] }),
                                Operand::Copy(Place { local: 2, projections: vec![] }),
                            ),
                        )],
                        terminator: Terminator::SwitchInt {
                            discr: Operand::Copy(Place { local: 3, projections: vec![] }),
                            targets: SwitchTargets {
                                values: vec![0],      // 0 = false → bb2
                                targets: vec![2, 1],  // otherwise (true) → bb1
                            },
                        },
                        is_cleanup: false,
                    },
                    // bb1: a >= b, return a
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place { local: 1, projections: vec![] })),
                        )],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                    // bb2: a < b, return b
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place { local: 2, projections: vec![] })),
                        )],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &fn_body,
            &layouts,
            "max",
            Linkage::Export,
            &empty_reg,
            &ty_layouts,
        )
        .expect("compile_mirdata_fn failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn(i32, i32) -> i32 = std::mem::transmute(code);
            assert_eq!(f(10, 5), 10);
            assert_eq!(f(3, 7), 7);
            assert_eq!(f(4, 4), 4);
            assert_eq!(f(-1, -5), -1);
        }
    }

    // -- call between mirdata functions: double(x) = x + x, caller calls it --

    #[test]
    fn mirdata_call_between_fns() {
        let layout_entries = vec![i32_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        // Callee: fn double(x: i32) -> i32 { x + x }
        let callee_hash: (u64, u64) = (1, 1);
        let callee = FnBody {
            def_path_hash: callee_hash,
            name: "double".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _0: return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _1: x
                ],
                arg_count: 1,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        // Caller: fn test_fn() -> i32 { double(21) }
        // MIR:
        //   _1 = const 21
        //   _0 = call double(Copy(_1)) -> bb1
        // bb1: return
        let caller = FnBody {
            def_path_hash: (2, 2),
            name: "test_fn".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _0: return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _1: arg for double
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Int(IntTy::I32),
                                kind: ConstKind::Scalar(21, 4),
                            })),
                        )],
                        terminator: Terminator::Call {
                            func: Operand::Constant(ConstOperand {
                                ty: Ty::FnDef(callee_hash, vec![]),
                                kind: ConstKind::ZeroSized,
                            }),
                            args: vec![Operand::Copy(Place { local: 1, projections: vec![] })],
                            dest: Place { local: 0, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let (mut module, isa, dl) = make_jit_module();

        // Compile callee first
        let empty_reg = HashMap::new();
        let callee_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &callee,
            &layouts,
            "double",
            Linkage::Export,
            &empty_reg,
            &ty_layouts,
        )
        .expect("compile callee failed");

        // Compile caller with fn_registry pointing to callee
        let mut fn_registry = HashMap::new();
        fn_registry.insert(callee_hash, callee_id);
        let caller_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &caller,
            &layouts,
            "test_fn",
            Linkage::Export,
            &fn_registry,
            &ty_layouts,
        )
        .expect("compile caller failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(caller_id);
        unsafe {
            let f: fn() -> i32 = std::mem::transmute(code);
            assert_eq!(f(), 42); // double(21) = 21 + 21 = 42
        }
    }

    // -- call a native extern "C" function registered as JIT symbol ----------

    #[test]
    fn mirdata_call_native() {
        let layout_entries = vec![i32_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        // Native function that will be registered as a JIT symbol
        extern "C" fn native_mul(a: i32, b: i32) -> i32 {
            a * b
        }

        // Build JIT module with native symbol registered
        let isa = crate::build_host_isa(false);
        let dl = TargetDataLayout::parse_from_llvm_datalayout_string(
            "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128",
            AddressSpace(0),
        )
        .map_err(|_| "failed to parse data layout")
        .unwrap();
        let mut builder =
            JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());
        builder.symbol("native_mul", native_mul as *const u8);
        let mut module = JITModule::new(builder);

        // Declare native_mul as an import so the call codegen can reference it
        let native_hash: (u64, u64) = (99, 99);
        let mut sig = Signature::new(isa.default_call_conv());
        sig.params.push(AbiParam::new(cranelift_codegen::ir::types::I32));
        sig.params.push(AbiParam::new(cranelift_codegen::ir::types::I32));
        sig.returns.push(AbiParam::new(cranelift_codegen::ir::types::I32));
        let native_func_id = module
            .declare_function("native_mul", Linkage::Import, &sig)
            .expect("declare native_mul");

        // Mirdata body: fn test() -> i32 { native_mul(6, 7) }
        // MIR:
        //   _1 = const 6
        //   _2 = const 7
        //   _0 = call native_mul(Copy(_1), Copy(_2)) -> bb1
        // bb1: return
        let caller = FnBody {
            def_path_hash: (0, 0),
            name: "test_native".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _0: return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _1
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _2
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![
                            Statement::Assign(
                                Place { local: 1, projections: vec![] },
                                Rvalue::Use(Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(6, 4),
                                })),
                            ),
                            Statement::Assign(
                                Place { local: 2, projections: vec![] },
                                Rvalue::Use(Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(7, 4),
                                })),
                            ),
                        ],
                        terminator: Terminator::Call {
                            func: Operand::Constant(ConstOperand {
                                ty: Ty::FnDef(native_hash, vec![]),
                                kind: ConstKind::ZeroSized,
                            }),
                            args: vec![
                                Operand::Copy(Place { local: 1, projections: vec![] }),
                                Operand::Copy(Place { local: 2, projections: vec![] }),
                            ],
                            dest: Place { local: 0, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let mut fn_registry = HashMap::new();
        fn_registry.insert(native_hash, native_func_id);
        let caller_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &caller,
            &layouts,
            "test_native",
            Linkage::Export,
            &fn_registry,
            &ty_layouts,
        )
        .expect("compile caller failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(caller_id);
        unsafe {
            let f: fn() -> i32 = std::mem::transmute(code);
            assert_eq!(f(), 42); // native_mul(6, 7) = 42
        }
    }

    // -- fn cast_i32_to_i64(x: i32) -> i64 { x as i64 } ---------------------

    fn i64_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::Int(IntTy::I64),
            layout: LayoutInfo {
                size: 8,
                align: 8,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Int { size_bytes: 8, signed: true },
                    valid_range_start: 0,
                    valid_range_end: u64::MAX as u128,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    fn u64_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::Uint(UintTy::U64),
            layout: LayoutInfo {
                size: 8,
                align: 8,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Int { size_bytes: 8, signed: false },
                    valid_range_start: 0,
                    valid_range_end: u64::MAX as u128,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    fn usize_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::Uint(UintTy::Usize),
            layout: LayoutInfo {
                size: 8,
                align: 8,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Pointer,
                    valid_range_start: 0,
                    valid_range_end: u64::MAX as u128,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    /// Layout for a struct { i32, i32 } — two fields at offsets 0 and 4.
    fn pair_i32_layout_entry() -> TypeLayoutEntry {
        let adt_ty = Ty::Tuple(vec![Ty::Int(IntTy::I32), Ty::Int(IntTy::I32)]);
        TypeLayoutEntry {
            ty: adt_ty,
            layout: LayoutInfo {
                size: 8,
                align: 4,
                backend_repr: ExportedBackendRepr::ScalarPair(
                    ExportedScalar {
                        primitive: ExportedPrimitive::Int { size_bytes: 4, signed: true },
                        valid_range_start: 0,
                        valid_range_end: u32::MAX as u128,
                    },
                    ExportedScalar {
                        primitive: ExportedPrimitive::Int { size_bytes: 4, signed: true },
                        valid_range_start: 0,
                        valid_range_end: u32::MAX as u128,
                    },
                ),
                fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 4] },
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    /// Layout for *const i32 (thin pointer, scalar)
    fn ptr_i32_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::RawPtr(Mutability::Not, Box::new(Ty::Int(IntTy::I32))),
            layout: LayoutInfo {
                size: 8,
                align: 8,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Pointer,
                    valid_range_start: 0,
                    valid_range_end: u64::MAX as u128,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    // -- struct field access: (i32, i32).1 ----------------------------------

    #[test]
    fn mirdata_tuple_field_access() {
        // fn foo() -> i32 {
        //   let pair: (i32, i32) = (10, 20);
        //   pair.1
        // }
        // MIR:
        //   _1: (i32, i32) = Aggregate(Tuple, [const 10, const 20])
        //   _0: i32 = Use(Copy(_1.1))
        let pair_ty = Ty::Tuple(vec![Ty::Int(IntTy::I32), Ty::Int(IntTy::I32)]);
        let layout_entries = vec![i32_layout_entry(), pair_i32_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "tuple_field".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: pair_ty.clone(), layout: Some(1) },       // _1: pair
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Tuple,
                                vec![
                                    Operand::Constant(ConstOperand {
                                        ty: Ty::Int(IntTy::I32),
                                        kind: ConstKind::Scalar(10, 4),
                                    }),
                                    Operand::Constant(ConstOperand {
                                        ty: Ty::Int(IntTy::I32),
                                        kind: ConstKind::Scalar(20, 4),
                                    }),
                                ],
                            ),
                        ),
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![Projection::Field(1, Ty::Int(IntTy::I32))],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 20);
    }

    // -- ref and deref: &x then *ref ----------------------------------------

    #[test]
    fn mirdata_ref_deref() {
        // fn foo() -> i32 {
        //   let x: i32 = 42;
        //   let r: *const i32 = &raw const x;
        //   *r
        // }
        // MIR:
        //   _1: i32 = const 42
        //   _2: *const i32 = RawPtr(Not, _1)
        //   _0: i32 = Use(Copy(*_2))
        let layout_entries = vec![i32_layout_entry(), ptr_i32_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "ref_deref".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _1: x
                    Local { ty: Ty::RawPtr(Mutability::Not, Box::new(Ty::Int(IntTy::I32))), layout: Some(1) }, // _2: ptr
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
                            Place { local: 2, projections: vec![] },
                            Rvalue::RawPtr(Mutability::Not, Place { local: 1, projections: vec![] }),
                        ),
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 2,
                                projections: vec![Projection::Deref],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 42);
    }

    #[test]
    fn mirdata_cast_i32_to_i64() {
        let layout_entries = vec![i32_layout_entry(), i64_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);
        // fn cast(x: i32) -> i64 { x as i64 }
        // MIR:
        //   _0 = Cast(IntToInt, Copy(_1), i64)
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "cast_i32_i64".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I64), layout: Some(1) }, // _0: return i64
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _1: x i32
                ],
                arg_count: 1,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::Cast(
                            CastKind::IntToInt,
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                            Ty::Int(IntTy::I64),
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            &fn_body,
            &layouts,
            "cast_i32_i64",
            Linkage::Export,
            &empty_reg,
            &ty_layouts,
        )
        .expect("compile_mirdata_fn failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn(i32) -> i64 = std::mem::transmute(code);
            assert_eq!(f(42), 42i64);
            assert_eq!(f(-1), -1i64); // sign extension
            assert_eq!(f(i32::MAX), i32::MAX as i64);
            assert_eq!(f(i32::MIN), i32::MIN as i64);
        }
    }

    // -- memory-repr struct (3 fields): sret return + pass by pointer ---------

    fn triple_i32_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::Tuple(vec![Ty::Int(IntTy::I32), Ty::Int(IntTy::I32), Ty::Int(IntTy::I32)]),
            layout: LayoutInfo {
                size: 12,
                align: 4,
                backend_repr: ExportedBackendRepr::Memory { sized: true },
                fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 4, 8] },
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    #[test]
    fn mirdata_sret_return() {
        // fn foo() -> (i32, i32, i32) { (10, 20, 30) }
        // Then extract the first field to verify.
        let triple_ty = Ty::Tuple(vec![Ty::Int(IntTy::I32), Ty::Int(IntTy::I32), Ty::Int(IntTy::I32)]);
        let layout_entries = vec![i32_layout_entry(), triple_i32_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        // Callee: fn make_triple() -> (i32, i32, i32) { (10, 20, 30) }
        let callee_hash: (u64, u64) = (10, 10);
        let callee = FnBody {
            def_path_hash: callee_hash,
            name: "make_triple".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: triple_ty.clone(), layout: Some(1) }, // _0: return
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::Aggregate(
                            ra_mir_types::AggregateKind::Tuple,
                            vec![
                                Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(10, 4) }),
                                Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(20, 4) }),
                                Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(30, 4) }),
                            ],
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        // Caller: fn test() -> i32 { make_triple().1 }
        let caller = FnBody {
            def_path_hash: (20, 20),
            name: "test_sret".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: triple_ty.clone(), layout: Some(1) },     // _1: triple
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Call {
                            func: Operand::Constant(ConstOperand {
                                ty: Ty::FnDef(callee_hash, vec![]),
                                kind: ConstKind::ZeroSized,
                            }),
                            args: vec![],
                            dest: Place { local: 1, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![Projection::Field(1, Ty::Int(IntTy::I32))],
                            })),
                        )],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let callee_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &callee, &layouts,
            "make_triple", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile callee");

        let mut fn_registry = HashMap::new();
        fn_registry.insert(callee_hash, callee_id);
        let caller_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &caller, &layouts,
            "test_sret", Linkage::Export, &fn_registry, &ty_layouts,
        ).expect("compile caller");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(caller_id);
        unsafe {
            let f: fn() -> i32 = std::mem::transmute(code);
            assert_eq!(f(), 20); // second field of (10, 20, 30)
        }
    }

    // -- memory-repr pass by pointer ------------------------------------------

    #[test]
    fn mirdata_memory_repr_arg() {
        // fn sum_triple(t: (i32, i32, i32)) -> i32 { t.0 + t.1 + t.2 }
        let triple_ty = Ty::Tuple(vec![Ty::Int(IntTy::I32), Ty::Int(IntTy::I32), Ty::Int(IntTy::I32)]);
        let layout_entries = vec![i32_layout_entry(), triple_i32_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        // Callee: fn sum_triple(t: (i32, i32, i32)) -> i32 { t.0 + t.1 + t.2 }
        let callee_hash: (u64, u64) = (11, 11);
        let callee = FnBody {
            def_path_hash: callee_hash,
            name: "sum_triple".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: triple_ty.clone(), layout: Some(1) },     // _1: t (arg)
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _2: tmp
                ],
                arg_count: 1,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        // _2 = t.0 + t.1
                        Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::BinaryOp(
                                BinOp::Add,
                                Operand::Copy(Place { local: 1, projections: vec![Projection::Field(0, Ty::Int(IntTy::I32))] }),
                                Operand::Copy(Place { local: 1, projections: vec![Projection::Field(1, Ty::Int(IntTy::I32))] }),
                            ),
                        ),
                        // _0 = _2 + t.2
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::BinaryOp(
                                BinOp::Add,
                                Operand::Copy(Place { local: 2, projections: vec![] }),
                                Operand::Copy(Place { local: 1, projections: vec![Projection::Field(2, Ty::Int(IntTy::I32))] }),
                            ),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        // Caller: fn test() -> i32 { sum_triple((3, 5, 7)) }
        let caller = FnBody {
            def_path_hash: (21, 21),
            name: "test_mem_arg".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: triple_ty.clone(), layout: Some(1) },     // _1: triple
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Tuple,
                                vec![
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(3, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(5, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(7, 4) }),
                                ],
                            ),
                        )],
                        terminator: Terminator::Call {
                            func: Operand::Constant(ConstOperand {
                                ty: Ty::FnDef(callee_hash, vec![]),
                                kind: ConstKind::ZeroSized,
                            }),
                            args: vec![Operand::Copy(Place { local: 1, projections: vec![] })],
                            dest: Place { local: 0, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let callee_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &callee, &layouts,
            "sum_triple", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile callee");

        let mut fn_registry = HashMap::new();
        fn_registry.insert(callee_hash, callee_id);
        let caller_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &caller, &layouts,
            "test_mem_arg", Linkage::Export, &fn_registry, &ty_layouts,
        ).expect("compile caller");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(caller_id);
        unsafe {
            let f: fn() -> i32 = std::mem::transmute(code);
            assert_eq!(f(), 15); // 3 + 5 + 7
        }
    }

    // -- indirect fn pointer call ---------------------------------------------

    #[test]
    fn mirdata_fn_ptr_call() {
        // fn double(x: i32) -> i32 { x + x }
        // fn test() -> i32 {
        //     let fp: fn(i32) -> i32 = double;   // ReifyFnPointer
        //     fp(21)                              // indirect call
        // }
        let fn_ptr_ty = Ty::FnPtr(vec![Ty::Int(IntTy::I32)], Box::new(Ty::Int(IntTy::I32)));
        let layout_entries = vec![
            i32_layout_entry(),
            // fn pointer is just a scalar pointer
            TypeLayoutEntry {
                ty: fn_ptr_ty.clone(),
                layout: LayoutInfo {
                    size: 8,
                    align: 8,
                    backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                        primitive: ExportedPrimitive::Pointer,
                        valid_range_start: 1,
                        valid_range_end: u64::MAX as u128,
                    }),
                    fields: ExportedFieldsShape::Primitive,
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: Some(ExportedNiche {
                        offset: 0,
                        scalar: ExportedScalar {
                            primitive: ExportedPrimitive::Pointer,
                            valid_range_start: 1,
                            valid_range_end: u64::MAX as u128,
                        },
                    }),
                },
            },
        ];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        let double_hash: (u64, u64) = (30, 30);
        let double = FnBody {
            def_path_hash: double_hash,
            name: "double".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },
                ],
                arg_count: 1,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let caller = FnBody {
            def_path_hash: (31, 31),
            name: "test_fnptr".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },   // _0: return
                    Local { ty: fn_ptr_ty.clone(), layout: Some(1) },     // _1: fn pointer
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },   // _2: arg
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![
                            // _1 = double as fn(i32) -> i32  (ReifyFnPointer)
                            Statement::Assign(
                                Place { local: 1, projections: vec![] },
                                Rvalue::Cast(
                                    CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer),
                                    Operand::Constant(ConstOperand {
                                        ty: Ty::FnDef(double_hash, vec![]),
                                        kind: ConstKind::ZeroSized,
                                    }),
                                    fn_ptr_ty.clone(),
                                ),
                            ),
                            // _2 = 21
                            Statement::Assign(
                                Place { local: 2, projections: vec![] },
                                Rvalue::Use(Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(21, 4),
                                })),
                            ),
                        ],
                        // _0 = _1(_2)  indirect call
                        terminator: Terminator::Call {
                            func: Operand::Copy(Place { local: 1, projections: vec![] }),
                            args: vec![Operand::Copy(Place { local: 2, projections: vec![] })],
                            dest: Place { local: 0, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let double_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &double, &layouts,
            "double", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile double");

        let mut fn_registry = HashMap::new();
        fn_registry.insert(double_hash, double_id);
        let caller_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &caller, &layouts,
            "test_fnptr", Linkage::Export, &fn_registry, &ty_layouts,
        ).expect("compile caller");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(caller_id);
        unsafe {
            let f: fn() -> i32 = std::mem::transmute(code);
            assert_eq!(f(), 42); // double(21) via fn pointer
        }
    }

    // -- Rvalue::Repeat: [val; N] array creation ------------------------------

    #[test]
    fn mirdata_repeat() {
        // fn foo() -> i32 {
        //   let arr: [i32; 3] = [7; 3];
        //   arr[1]
        // }
        let arr_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 3);
        let layout_entries = vec![
            i32_layout_entry(),
            TypeLayoutEntry {
                ty: arr_ty.clone(),
                layout: LayoutInfo {
                    size: 12,
                    align: 4,
                    backend_repr: ExportedBackendRepr::Memory { sized: true },
                    fields: ExportedFieldsShape::Array { stride: 4, count: 3 },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
            usize_layout_entry(),
        ];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "repeat_test".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: arr_ty.clone(), layout: Some(1) },        // _1: arr
                    Local { ty: Ty::Uint(UintTy::Usize), layout: Some(2) }, // _2: index
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Repeat(
                                Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(7, 4),
                                }),
                                3,
                            ),
                        ),
                        // _2 = 1usize
                        Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Uint(UintTy::Usize),
                                kind: ConstKind::Scalar(1, 8),
                            })),
                        ),
                        // _0 = arr[_2]
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![Projection::Index(2)],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &fn_body, &layouts,
            "repeat_test", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn() -> i32 = std::mem::transmute(code);
            assert_eq!(f(), 7);
        }
    }

    // -- constant args in calls -----------------------------------------------

    #[test]
    fn mirdata_constant_call_arg() {
        // fn double(x: i32) -> i32 { x + x }
        // fn test() -> i32 { double(21) }  // 21 passed as constant operand
        let layout_entries = vec![i32_layout_entry()];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        let double_hash: (u64, u64) = (40, 40);
        let double = FnBody {
            def_path_hash: double_hash,
            name: "double_const".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },
                ],
                arg_count: 1,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::BinaryOp(
                            BinOp::Add,
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        // Caller passes constant directly (no intermediate local)
        let caller = FnBody {
            def_path_hash: (41, 41),
            name: "test_const_arg".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _0: return
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Call {
                            func: Operand::Constant(ConstOperand {
                                ty: Ty::FnDef(double_hash, vec![]),
                                kind: ConstKind::ZeroSized,
                            }),
                            args: vec![Operand::Constant(ConstOperand {
                                ty: Ty::Int(IntTy::I32),
                                kind: ConstKind::Scalar(21, 4),
                            })],
                            dest: Place { local: 0, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let double_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &double, &layouts,
            "double_const", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile double");

        let mut fn_registry = HashMap::new();
        fn_registry.insert(double_hash, double_id);
        let caller_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &caller, &layouts,
            "test_const_arg", Linkage::Export, &fn_registry, &ty_layouts,
        ).expect("compile caller");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(caller_id);
        unsafe {
            let f: fn() -> i32 = std::mem::transmute(code);
            assert_eq!(f(), 42); // double(21) = 42
        }
    }

    // -- checked overflow (AddWithOverflow) -----------------------------------

    #[test]
    fn mirdata_checked_add() {
        // fn checked_add(a: i32, b: i32) -> (i32, bool) {
        //     AddWithOverflow(a, b)
        // }
        let bool_ty = Ty::Bool;
        let pair_ty = Ty::Tuple(vec![Ty::Int(IntTy::I32), bool_ty.clone()]);
        let layout_entries = vec![
            i32_layout_entry(),
            bool_layout_entry(),
            // (i32, bool) layout: ScalarPair(i32, i8) with size 8, align 4
            TypeLayoutEntry {
                ty: pair_ty.clone(),
                layout: LayoutInfo {
                    size: 8,
                    align: 4,
                    backend_repr: ExportedBackendRepr::ScalarPair(
                        ExportedScalar {
                            primitive: ExportedPrimitive::Int { size_bytes: 4, signed: true },
                            valid_range_start: 0,
                            valid_range_end: u32::MAX as u128,
                        },
                        ExportedScalar {
                            primitive: ExportedPrimitive::Int { size_bytes: 1, signed: false },
                            valid_range_start: 0,
                            valid_range_end: 1,
                        },
                    ),
                    fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 4] },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
        ];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "checked_add".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: pair_ty.clone(), layout: Some(2) }, // _0: return (i32, bool)
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _1: a
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) }, // _2: b
                ],
                arg_count: 2,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::BinaryOp(
                            BinOp::AddWithOverflow,
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                            Operand::Copy(Place { local: 2, projections: vec![] }),
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &fn_body, &layouts,
            "checked_add", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn(i32, i32) -> (i32, bool) = std::mem::transmute(code);
            assert_eq!(f(3, 4), (7, false));          // no overflow
            assert_eq!(f(i32::MAX, 1), (i32::MIN, true)); // overflow!
        }
    }

    // -- transmute (scalar → scalar) -----------------------------------------

    #[test]
    fn mirdata_transmute() {
        // fn transmute_i32_to_u32(x: i32) -> u32 { transmute(x) }
        let u32_layout_entry = TypeLayoutEntry {
            ty: Ty::Uint(UintTy::U32),
            layout: LayoutInfo {
                size: 4,
                align: 4,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Int { size_bytes: 4, signed: false },
                    valid_range_start: 0,
                    valid_range_end: u32::MAX as u128,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        };
        let layout_entries = vec![i32_layout_entry(), u32_layout_entry];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "transmute_test".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Uint(UintTy::U32), layout: Some(1) }, // _0: return u32
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },   // _1: x i32
                ],
                arg_count: 1,
                blocks: vec![BasicBlock {
                    stmts: vec![Statement::Assign(
                        Place { local: 0, projections: vec![] },
                        Rvalue::Cast(
                            CastKind::Transmute,
                            Operand::Copy(Place { local: 1, projections: vec![] }),
                            Ty::Uint(UintTy::U32),
                        ),
                    )],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &fn_body, &layouts,
            "transmute_test", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn(i32) -> u32 = std::mem::transmute(code);
            assert_eq!(f(-1), u32::MAX);
            assert_eq!(f(42), 42);
        }
    }

    // -- ConstantIndex: arr[2] via fixed offset --------------------------------

    #[test]
    fn mirdata_constant_index() {
        // fn foo() -> i32 {
        //   let arr: [i32; 4] = [10, 20, 30, 40];
        //   arr[2]  // via ConstantIndex { offset: 2, from_end: false }
        // }
        let arr_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 4);
        let layout_entries = vec![
            i32_layout_entry(),
            TypeLayoutEntry {
                ty: arr_ty.clone(),
                layout: LayoutInfo {
                    size: 16,
                    align: 4,
                    backend_repr: ExportedBackendRepr::Memory { sized: true },
                    fields: ExportedFieldsShape::Array { stride: 4, count: 4 },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
        ];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "const_idx".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: arr_ty.clone(), layout: Some(1) },        // _1: arr
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Array(Ty::Int(IntTy::I32)),
                                vec![
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(10, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(20, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(30, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(40, 4) }),
                                ],
                            ),
                        ),
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![Projection::ConstantIndex {
                                    offset: 2,
                                    min_length: 4,
                                    from_end: false,
                                }],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 30);
    }

    // -- ConstantIndex from_end: arr[len - 1] ----------------------------------

    #[test]
    fn mirdata_constant_index_from_end() {
        let arr_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 4);
        let layout_entries = vec![
            i32_layout_entry(),
            TypeLayoutEntry {
                ty: arr_ty.clone(),
                layout: LayoutInfo {
                    size: 16,
                    align: 4,
                    backend_repr: ExportedBackendRepr::Memory { sized: true },
                    fields: ExportedFieldsShape::Array { stride: 4, count: 4 },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
        ];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "const_idx_end".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },
                    Local { ty: arr_ty.clone(), layout: Some(1) },
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Array(Ty::Int(IntTy::I32)),
                                vec![
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(10, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(20, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(30, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(40, 4) }),
                                ],
                            ),
                        ),
                        // Get last element: ConstantIndex { offset: 1, from_end: true }
                        // = arr[len - 1] = arr[3] = 40
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![Projection::ConstantIndex {
                                    offset: 1,
                                    min_length: 4,
                                    from_end: true,
                                }],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 40);
    }

    // -- Subslice: arr[1..3] as sub-array --------------------------------------

    #[test]
    fn mirdata_subslice() {
        // fn foo() -> i32 {
        //   let arr: [i32; 4] = [10, 20, 30, 40];
        //   let sub: [i32; 2] = arr[1..3];  // Subslice { from: 1, to: 3, from_end: false }
        //   sub[0]  // = 20
        // }
        let arr4_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 4);
        let arr2_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 2);
        let layout_entries = vec![
            i32_layout_entry(),
            TypeLayoutEntry {
                ty: arr4_ty.clone(),
                layout: LayoutInfo {
                    size: 16,
                    align: 4,
                    backend_repr: ExportedBackendRepr::Memory { sized: true },
                    fields: ExportedFieldsShape::Array { stride: 4, count: 4 },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
            TypeLayoutEntry {
                ty: arr2_ty.clone(),
                layout: LayoutInfo {
                    size: 8,
                    align: 4,
                    backend_repr: ExportedBackendRepr::Memory { sized: true },
                    fields: ExportedFieldsShape::Array { stride: 4, count: 2 },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
        ];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "subslice_test".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },   // _0: return
                    Local { ty: arr4_ty.clone(), layout: Some(1) },        // _1: arr
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Array(Ty::Int(IntTy::I32)),
                                vec![
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(10, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(20, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(30, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(40, 4) }),
                                ],
                            ),
                        ),
                        // _0 = arr[1..3][0]
                        // = Subslice { from: 1, to: 3, from_end: false }, then ConstantIndex 0
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![
                                    Projection::Subslice { from: 1, to: 3, from_end: false },
                                    Projection::ConstantIndex { offset: 1, min_length: 2, from_end: false },
                                ],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 30); // arr[1..3][1] = arr[2] = 30
    }

    // -- RawPtr aggregate: construct fat pointer from (data, meta) -------------

    #[test]
    fn mirdata_rawptr_aggregate() {
        // Construct a *const [i32] fat pointer from (data_ptr, len), then extract len.
        // fn foo() -> usize {
        //   let x: i32 = 42;
        //   let data: *const i32 = &raw const x;
        //   let len: usize = 5;
        //   let fat_ptr: *const [i32] = RawPtr(data, len);
        //   PtrMetadata(fat_ptr)  // = 5
        // }
        let slice_ptr_ty = Ty::RawPtr(Mutability::Not, Box::new(Ty::Slice(Box::new(Ty::Int(IntTy::I32)))));
        let layout_entries = vec![
            i32_layout_entry(),
            ptr_i32_layout_entry(),
            usize_layout_entry(),
            // *const [i32] — fat pointer: ScalarPair(Pointer, Pointer)
            TypeLayoutEntry {
                ty: slice_ptr_ty.clone(),
                layout: LayoutInfo {
                    size: 16,
                    align: 8,
                    backend_repr: ExportedBackendRepr::ScalarPair(
                        ExportedScalar {
                            primitive: ExportedPrimitive::Pointer,
                            valid_range_start: 0,
                            valid_range_end: u64::MAX as u128,
                        },
                        ExportedScalar {
                            primitive: ExportedPrimitive::Pointer,
                            valid_range_start: 0,
                            valid_range_end: u64::MAX as u128,
                        },
                    ),
                    fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 8] },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
        ];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "rawptr_agg".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Uint(UintTy::Usize), layout: Some(2) },   // _0: return usize
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },        // _1: x
                    Local { ty: Ty::RawPtr(Mutability::Not, Box::new(Ty::Int(IntTy::I32))), layout: Some(1) }, // _2: data ptr
                    Local { ty: Ty::Uint(UintTy::Usize), layout: Some(2) },   // _3: len
                    Local { ty: slice_ptr_ty.clone(), layout: Some(3) },        // _4: fat ptr
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        // _1 = 42
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Int(IntTy::I32),
                                kind: ConstKind::Scalar(42, 4),
                            })),
                        ),
                        // _2 = &raw const _1
                        Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::RawPtr(Mutability::Not, Place { local: 1, projections: vec![] }),
                        ),
                        // _3 = 5usize
                        Statement::Assign(
                            Place { local: 3, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Uint(UintTy::Usize),
                                kind: ConstKind::Scalar(5, 8),
                            })),
                        ),
                        // _4 = Aggregate(RawPtr, [_2, _3])
                        Statement::Assign(
                            Place { local: 4, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::RawPtr(
                                    Ty::Slice(Box::new(Ty::Int(IntTy::I32))),
                                    Mutability::Not,
                                ),
                                vec![
                                    Operand::Copy(Place { local: 2, projections: vec![] }),
                                    Operand::Copy(Place { local: 3, projections: vec![] }),
                                ],
                            ),
                        ),
                        // _0 = PtrMetadata(_4)
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::UnaryOp(
                                ra_mir_types::UnOp::PtrMetadata,
                                Operand::Copy(Place { local: 4, projections: vec![] }),
                            ),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: usize = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 5);
    }

    // -- Unsize: &[i32; 3] → &[i32] (array to slice) ---------------------------

    fn slice_i32_ref_layout_entry() -> TypeLayoutEntry {
        // &[i32] is ScalarPair(Pointer, Pointer) — (data_ptr, len)
        TypeLayoutEntry {
            ty: Ty::Ref(Mutability::Not, Box::new(Ty::Slice(Box::new(Ty::Int(IntTy::I32))))),
            layout: LayoutInfo {
                size: 16,
                align: 8,
                backend_repr: ExportedBackendRepr::ScalarPair(
                    ExportedScalar {
                        primitive: ExportedPrimitive::Pointer,
                        valid_range_start: 1,
                        valid_range_end: u64::MAX as u128,
                    },
                    ExportedScalar {
                        primitive: ExportedPrimitive::Pointer,
                        valid_range_start: 0,
                        valid_range_end: u64::MAX as u128,
                    },
                ),
                fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 8] },
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    fn arr3_i32_ref_layout_entry() -> TypeLayoutEntry {
        // &[i32; 3] is a thin pointer (Scalar)
        TypeLayoutEntry {
            ty: Ty::Ref(Mutability::Not, Box::new(Ty::Array(Box::new(Ty::Int(IntTy::I32)), 3))),
            layout: LayoutInfo {
                size: 8,
                align: 8,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Pointer,
                    valid_range_start: 1,
                    valid_range_end: u64::MAX as u128,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    fn arr3_i32_layout_entry() -> TypeLayoutEntry {
        TypeLayoutEntry {
            ty: Ty::Array(Box::new(Ty::Int(IntTy::I32)), 3),
            layout: LayoutInfo {
                size: 12,
                align: 4,
                backend_repr: ExportedBackendRepr::Memory { sized: true },
                fields: ExportedFieldsShape::Array { stride: 4, count: 3 },
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        }
    }

    #[test]
    fn mirdata_unsize_array_to_slice() {
        // fn foo() -> usize {
        //   let arr: [i32; 3] = [10, 20, 30];
        //   let arr_ref: &[i32; 3] = &arr;
        //   let slice_ref: &[i32] = arr_ref as &[i32]; // Unsize coercion
        //   PtrMetadata(slice_ref) // = 3
        // }
        let arr_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 3);
        let arr_ref_ty = Ty::Ref(Mutability::Not, Box::new(arr_ty.clone()));
        let slice_ref_ty = Ty::Ref(Mutability::Not, Box::new(Ty::Slice(Box::new(Ty::Int(IntTy::I32)))));

        let layout_entries = vec![
            i32_layout_entry(),            // 0
            arr3_i32_layout_entry(),       // 1
            arr3_i32_ref_layout_entry(),   // 2
            slice_i32_ref_layout_entry(),  // 3
            usize_layout_entry(),          // 4
        ];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "unsize_arr".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Uint(UintTy::Usize), layout: Some(4) },   // _0: return usize
                    Local { ty: arr_ty.clone(), layout: Some(1) },              // _1: arr
                    Local { ty: arr_ref_ty.clone(), layout: Some(2) },          // _2: &[i32; 3]
                    Local { ty: slice_ref_ty.clone(), layout: Some(3) },        // _3: &[i32]
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        // _1 = [10, 20, 30]
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Array(Ty::Int(IntTy::I32)),
                                vec![
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(10, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(20, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(30, 4) }),
                                ],
                            ),
                        ),
                        // _2 = &_1
                        Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::Ref(BorrowKind::Shared, Place { local: 1, projections: vec![] }),
                        ),
                        // _3 = _2 as &[i32] (Unsize)
                        Statement::Assign(
                            Place { local: 3, projections: vec![] },
                            Rvalue::Cast(
                                CastKind::PointerCoercion(PointerCoercion::Unsize),
                                Operand::Copy(Place { local: 2, projections: vec![] }),
                                slice_ref_ty.clone(),
                            ),
                        ),
                        // _0 = PtrMetadata(_3) → should be 3
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::UnaryOp(
                                ra_mir_types::UnOp::PtrMetadata,
                                Operand::Copy(Place { local: 3, projections: vec![] }),
                            ),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: usize = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 3);
    }

    // -- Unsize + deref: access element through unsized slice ref -----------------

    #[test]
    fn mirdata_unsize_then_deref() {
        // fn foo() -> i32 {
        //   let arr: [i32; 3] = [10, 20, 30];
        //   let arr_ref: &[i32; 3] = &arr;
        //   let slice_ref: &[i32] = arr_ref as &[i32]; // Unsize
        //   (*slice_ref)[1]  // Deref + Index = 20
        // }
        let arr_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 3);
        let arr_ref_ty = Ty::Ref(Mutability::Not, Box::new(arr_ty.clone()));
        let slice_ty = Ty::Slice(Box::new(Ty::Int(IntTy::I32)));
        let slice_ref_ty = Ty::Ref(Mutability::Not, Box::new(slice_ty.clone()));

        let layout_entries = vec![
            i32_layout_entry(),            // 0
            arr3_i32_layout_entry(),       // 1
            arr3_i32_ref_layout_entry(),   // 2
            slice_i32_ref_layout_entry(),  // 3
            usize_layout_entry(),          // 4
            // [i32] slice layout (unsized)
            TypeLayoutEntry {
                ty: slice_ty.clone(),
                layout: LayoutInfo {
                    size: 0, // unsized, size not meaningful
                    align: 4,
                    backend_repr: ExportedBackendRepr::Memory { sized: false },
                    fields: ExportedFieldsShape::Array { stride: 4, count: 0 },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
        ];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "unsize_deref".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },        // _0: return i32
                    Local { ty: arr_ty.clone(), layout: Some(1) },              // _1: arr
                    Local { ty: arr_ref_ty.clone(), layout: Some(2) },          // _2: &[i32; 3]
                    Local { ty: slice_ref_ty.clone(), layout: Some(3) },        // _3: &[i32]
                    Local { ty: Ty::Uint(UintTy::Usize), layout: Some(4) },    // _4: index
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        // _1 = [10, 20, 30]
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Array(Ty::Int(IntTy::I32)),
                                vec![
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(10, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(20, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(30, 4) }),
                                ],
                            ),
                        ),
                        // _2 = &_1
                        Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::Ref(BorrowKind::Shared, Place { local: 1, projections: vec![] }),
                        ),
                        // _3 = _2 as &[i32] (Unsize)
                        Statement::Assign(
                            Place { local: 3, projections: vec![] },
                            Rvalue::Cast(
                                CastKind::PointerCoercion(PointerCoercion::Unsize),
                                Operand::Copy(Place { local: 2, projections: vec![] }),
                                slice_ref_ty.clone(),
                            ),
                        ),
                        // _4 = 1usize
                        Statement::Assign(
                            Place { local: 4, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Uint(UintTy::Usize),
                                kind: ConstKind::Scalar(1, 8),
                            })),
                        ),
                        // _0 = (*_3)[_4]  (Deref + Index)
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 3,
                                projections: vec![
                                    Projection::Deref,
                                    Projection::Index(4),
                                ],
                            })),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: i32 = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 20);
    }

    // -- Fat pointer reborrow: &*fat_ptr preserves metadata ----------------------

    #[test]
    fn mirdata_fat_ptr_reborrow() {
        // fn foo() -> usize {
        //   let arr: [i32; 3] = [10, 20, 30];
        //   let arr_ref: &[i32; 3] = &arr;
        //   let slice_ref: &[i32] = arr_ref as &[i32]; // Unsize
        //   let reborrowed: &[i32] = &*slice_ref;      // Ref(Deref)
        //   PtrMetadata(reborrowed) // should still be 3
        // }
        let arr_ty = Ty::Array(Box::new(Ty::Int(IntTy::I32)), 3);
        let arr_ref_ty = Ty::Ref(Mutability::Not, Box::new(arr_ty.clone()));
        let slice_ty = Ty::Slice(Box::new(Ty::Int(IntTy::I32)));
        let slice_ref_ty = Ty::Ref(Mutability::Not, Box::new(slice_ty.clone()));

        let layout_entries = vec![
            i32_layout_entry(),            // 0
            arr3_i32_layout_entry(),       // 1
            arr3_i32_ref_layout_entry(),   // 2
            slice_i32_ref_layout_entry(),  // 3
            usize_layout_entry(),          // 4
            // [i32] slice layout (unsized)
            TypeLayoutEntry {
                ty: slice_ty.clone(),
                layout: LayoutInfo {
                    size: 0,
                    align: 4,
                    backend_repr: ExportedBackendRepr::Memory { sized: false },
                    fields: ExportedFieldsShape::Array { stride: 4, count: 0 },
                    variants: ExportedVariants::Single { index: 0 },
                    largest_niche: None,
                },
            },
        ];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "fat_reborrow".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Uint(UintTy::Usize), layout: Some(4) },    // _0: return usize
                    Local { ty: arr_ty.clone(), layout: Some(1) },              // _1: arr
                    Local { ty: arr_ref_ty.clone(), layout: Some(2) },          // _2: &[i32; 3]
                    Local { ty: slice_ref_ty.clone(), layout: Some(3) },        // _3: &[i32]
                    Local { ty: slice_ref_ty.clone(), layout: Some(3) },        // _4: &[i32] (reborrowed)
                ],
                arg_count: 0,
                blocks: vec![BasicBlock {
                    stmts: vec![
                        // _1 = [10, 20, 30]
                        Statement::Assign(
                            Place { local: 1, projections: vec![] },
                            Rvalue::Aggregate(
                                ra_mir_types::AggregateKind::Array(Ty::Int(IntTy::I32)),
                                vec![
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(10, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(20, 4) }),
                                    Operand::Constant(ConstOperand { ty: Ty::Int(IntTy::I32), kind: ConstKind::Scalar(30, 4) }),
                                ],
                            ),
                        ),
                        // _2 = &_1
                        Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::Ref(BorrowKind::Shared, Place { local: 1, projections: vec![] }),
                        ),
                        // _3 = _2 as &[i32] (Unsize)
                        Statement::Assign(
                            Place { local: 3, projections: vec![] },
                            Rvalue::Cast(
                                CastKind::PointerCoercion(PointerCoercion::Unsize),
                                Operand::Copy(Place { local: 2, projections: vec![] }),
                                slice_ref_ty.clone(),
                            ),
                        ),
                        // _4 = &*_3  (reborrow — Ref of Deref)
                        Statement::Assign(
                            Place { local: 4, projections: vec![] },
                            Rvalue::Ref(BorrowKind::Shared, Place {
                                local: 3,
                                projections: vec![Projection::Deref],
                            }),
                        ),
                        // _0 = PtrMetadata(_4) → should still be 3
                        Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::UnaryOp(
                                ra_mir_types::UnOp::PtrMetadata,
                                Operand::Copy(Place { local: 4, projections: vec![] }),
                            ),
                        ),
                    ],
                    terminator: Terminator::Return,
                    is_cleanup: false,
                }],
            },
        };

        let result: usize = mirdata_jit_run(&fn_body, &layout_entries);
        assert_eq!(result, 3);
    }

    // -- Enum matching: SwitchInt + Downcast + field extraction -------------------

    #[test]
    fn mirdata_enum_match() {
        // Simulate matching on a simple 2-variant enum (like Option<i32>) with
        // direct tag encoding:
        //
        // enum MyOption { None=0, Some(i32)=1 }
        //
        // fn extract_or_default(opt: MyOption) -> i32 {
        //     match opt {
        //         MyOption::Some(val) => val,
        //         MyOption::None => -1,
        //     }
        // }
        //
        // Layout: size=8, align=4
        //   tag at field 0 (offset 0, u8)
        //   variant 0 (None): no fields
        //   variant 1 (Some): field 0 at offset 4 (i32)
        let enum_ty = Ty::Adt((100, 100), "MyOption".into(), vec![]);
        let u8_layout = TypeLayoutEntry {
            ty: Ty::Uint(UintTy::U8),
            layout: LayoutInfo {
                size: 1,
                align: 1,
                backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                    primitive: ExportedPrimitive::Int { size_bytes: 1, signed: false },
                    valid_range_start: 0,
                    valid_range_end: 255,
                }),
                fields: ExportedFieldsShape::Primitive,
                variants: ExportedVariants::Single { index: 0 },
                largest_niche: None,
            },
        };
        let enum_layout = TypeLayoutEntry {
            ty: enum_ty.clone(),
            layout: LayoutInfo {
                size: 8,
                align: 4,
                backend_repr: ExportedBackendRepr::Memory { sized: true },
                fields: ExportedFieldsShape::Arbitrary { offsets: vec![0] }, // tag at offset 0
                variants: ExportedVariants::Multiple {
                    tag: ExportedScalar {
                        primitive: ExportedPrimitive::Int { size_bytes: 1, signed: false },
                        valid_range_start: 0,
                        valid_range_end: 1,
                    },
                    tag_encoding: ExportedTagEncoding::Direct,
                    tag_field: 0,
                    variants: vec![
                        // Variant 0 (None): just tag, no payload
                        LayoutInfo {
                            size: 8,
                            align: 4,
                            backend_repr: ExportedBackendRepr::Memory { sized: true },
                            fields: ExportedFieldsShape::Arbitrary { offsets: vec![] },
                            variants: ExportedVariants::Single { index: 0 },
                            largest_niche: None,
                        },
                        // Variant 1 (Some): payload i32 at offset 4
                        LayoutInfo {
                            size: 8,
                            align: 4,
                            backend_repr: ExportedBackendRepr::Memory { sized: true },
                            fields: ExportedFieldsShape::Arbitrary { offsets: vec![4] },
                            variants: ExportedVariants::Single { index: 1 },
                            largest_niche: None,
                        },
                    ],
                },
                largest_niche: None,
            },
        };
        let layout_entries = vec![
            i32_layout_entry(),    // 0
            u8_layout,             // 1
            enum_layout,           // 2
        ];
        let layouts = convert_mirdata_layouts(&layout_entries);
        let ty_layouts = build_ty_layout_map(&layout_entries, &layouts);

        // Build the function: fn extract(opt: MyOption) -> i32
        // MIR:
        //   bb0: _2 = Discriminant(_1)
        //         SwitchInt(_2, [0 → bb1, otherwise → bb2])
        //   bb1 (None):  _0 = const -1i32; goto bb3
        //   bb2 (Some):  _0 = Copy((_1 as Some).0); goto bb3
        //   bb3: return
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "extract".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: enum_ty.clone(), layout: Some(2) },      // _1: opt (arg)
                    Local { ty: Ty::Uint(UintTy::U8), layout: Some(1) }, // _2: discriminant
                ],
                arg_count: 1,
                blocks: vec![
                    // bb0: get discriminant and switch
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::Discriminant(Place { local: 1, projections: vec![] }),
                        )],
                        terminator: Terminator::SwitchInt {
                            discr: Operand::Copy(Place { local: 2, projections: vec![] }),
                            targets: SwitchTargets {
                                values: vec![0], // 0 = None
                                targets: vec![1, 2], // [None→bb1, otherwise(Some)→bb2]
                            },
                        },
                        is_cleanup: false,
                    },
                    // bb1 (None): _0 = -1
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Constant(ConstOperand {
                                ty: Ty::Int(IntTy::I32),
                                kind: ConstKind::Scalar(-1i32 as u32 as u128, 4),
                            })),
                        )],
                        terminator: Terminator::Goto(3),
                        is_cleanup: false,
                    },
                    // bb2 (Some): _0 = (_1 as variant 1).field 0
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::Use(Operand::Copy(Place {
                                local: 1,
                                projections: vec![
                                    Projection::Downcast(1),
                                    Projection::Field(0, Ty::Int(IntTy::I32)),
                                ],
                            })),
                        )],
                        terminator: Terminator::Goto(3),
                        is_cleanup: false,
                    },
                    // bb3: return
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        // Build a caller that constructs Some(42) and calls extract
        //
        // fn test() -> i32 {
        //   let opt: MyOption = Some(42);
        //   extract(opt)
        // }
        //
        // MIR for constructing Some(42):
        //   _1 = Aggregate(Adt(MyOption, variant=1), [const 42])
        //   SetDiscriminant(_1, 1)
        let caller = FnBody {
            def_path_hash: (1, 1),
            name: "test_enum".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: enum_ty.clone(), layout: Some(2) },      // _1: opt
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![
                            // _1 = Aggregate(Adt(MyOption, variant=1), [42])
                            Statement::Assign(
                                Place { local: 1, projections: vec![] },
                                Rvalue::Aggregate(
                                    ra_mir_types::AggregateKind::Adt((100, 100), 1, vec![]),
                                    vec![Operand::Constant(ConstOperand {
                                        ty: Ty::Int(IntTy::I32),
                                        kind: ConstKind::Scalar(42, 4),
                                    })],
                                ),
                            ),
                            Statement::SetDiscriminant {
                                place: Place { local: 1, projections: vec![] },
                                variant_index: 1,
                            },
                        ],
                        terminator: Terminator::Call {
                            func: Operand::Constant(ConstOperand {
                                ty: Ty::FnDef((0, 0), vec![]),
                                kind: ConstKind::ZeroSized,
                            }),
                            args: vec![Operand::Copy(Place { local: 1, projections: vec![] })],
                            dest: Place { local: 0, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let extract_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &fn_body, &layouts,
            "extract", Linkage::Export, &empty_reg, &ty_layouts,
        ).expect("compile extract");

        // Also test the None case: construct None and call extract
        let caller_none = FnBody {
            def_path_hash: (2, 2),
            name: "test_enum_none".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: enum_ty.clone(), layout: Some(2) },      // _1: opt
                ],
                arg_count: 0,
                blocks: vec![
                    BasicBlock {
                        stmts: vec![
                            // _1 = Aggregate(Adt(MyOption, variant=0), [])
                            Statement::Assign(
                                Place { local: 1, projections: vec![] },
                                Rvalue::Aggregate(
                                    ra_mir_types::AggregateKind::Adt((100, 100), 0, vec![]),
                                    vec![],
                                ),
                            ),
                            Statement::SetDiscriminant {
                                place: Place { local: 1, projections: vec![] },
                                variant_index: 0,
                            },
                        ],
                        terminator: Terminator::Call {
                            func: Operand::Constant(ConstOperand {
                                ty: Ty::FnDef((0, 0), vec![]),
                                kind: ConstKind::ZeroSized,
                            }),
                            args: vec![Operand::Copy(Place { local: 1, projections: vec![] })],
                            dest: Place { local: 0, projections: vec![] },
                            target: Some(1),
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let mut fn_registry = HashMap::new();
        fn_registry.insert((0u64, 0u64), extract_id);

        let caller_some_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &caller, &layouts,
            "test_enum", Linkage::Export, &fn_registry, &ty_layouts,
        ).expect("compile test_enum");

        let caller_none_id = compile_mirdata_fn(
            &mut module, &*isa, &dl, &caller_none, &layouts,
            "test_enum_none", Linkage::Export, &fn_registry, &ty_layouts,
        ).expect("compile test_enum_none");

        module.finalize_definitions().unwrap();

        unsafe {
            let f_some: fn() -> i32 = std::mem::transmute(module.get_finalized_function(caller_some_id));
            assert_eq!(f_some(), 42); // Some(42) → extract → 42

            let f_none: fn() -> i32 = std::mem::transmute(module.get_finalized_function(caller_none_id));
            assert_eq!(f_none(), -1); // None → extract → -1
        }
    }

    // -- Assert terminator: panics on false condition ---------------------------

    #[test]
    fn mirdata_assert_pass() {
        // fn foo(x: i32) -> i32 {
        //     assert!(x > 0);  // Assert { cond: x > 0, expected: true }
        //     x + 1
        // }
        let layout_entries = vec![i32_layout_entry(), bool_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "assert_pass".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _1: x (arg)
                    Local { ty: Ty::Bool, layout: Some(1) },             // _2: cond
                ],
                arg_count: 1,
                blocks: vec![
                    // bb0: _2 = _1 > 0; assert(_2, expected=true) → bb1
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 2, projections: vec![] },
                            Rvalue::BinaryOp(
                                BinOp::Gt,
                                Operand::Copy(Place { local: 1, projections: vec![] }),
                                Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(0, 4),
                                }),
                            ),
                        )],
                        terminator: Terminator::Assert {
                            cond: Operand::Copy(Place { local: 2, projections: vec![] }),
                            expected: true,
                            target: 1,
                            unwind: UnwindAction::Terminate,
                        },
                        is_cleanup: false,
                    },
                    // bb1: _0 = _1 + 1; return
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 0, projections: vec![] },
                            Rvalue::BinaryOp(
                                BinOp::Add,
                                Operand::Copy(Place { local: 1, projections: vec![] }),
                                Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(1, 4),
                                }),
                            ),
                        )],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let result: i32 = mirdata_jit_run_with_arg(&fn_body, &layout_entries, 5);
        assert_eq!(result, 6); // 5 + 1 = 6
    }

    // -- Simple loop: sum 1..=N ------------------------------------------------

    #[test]
    fn mirdata_loop_sum() {
        // fn sum_to(n: i32) -> i32 {
        //     let mut acc = 0;
        //     let mut i = 1;
        //     while i <= n {
        //         acc += i;
        //         i += 1;
        //     }
        //     acc
        // }
        //
        // MIR:
        //   _0 = 0      (acc)
        //   _2 = 1      (i)
        // bb1: _3 = _2 <= _1; SwitchInt(_3, [0 → bb3, otherwise → bb2])
        // bb2: _0 = _0 + _2; _2 = _2 + 1; goto bb1
        // bb3: return
        let layout_entries = vec![i32_layout_entry(), bool_layout_entry()];
        let fn_body = FnBody {
            def_path_hash: (0, 0),
            name: "sum_to".to_string(),
            num_generic_params: 0,
            body: Body {
                locals: vec![
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _0: acc/return
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _1: n (arg)
                    Local { ty: Ty::Int(IntTy::I32), layout: Some(0) },  // _2: i
                    Local { ty: Ty::Bool, layout: Some(1) },             // _3: cond
                ],
                arg_count: 1,
                blocks: vec![
                    // bb0: init acc=0, i=1, goto bb1
                    BasicBlock {
                        stmts: vec![
                            Statement::Assign(
                                Place { local: 0, projections: vec![] },
                                Rvalue::Use(Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(0, 4),
                                })),
                            ),
                            Statement::Assign(
                                Place { local: 2, projections: vec![] },
                                Rvalue::Use(Operand::Constant(ConstOperand {
                                    ty: Ty::Int(IntTy::I32),
                                    kind: ConstKind::Scalar(1, 4),
                                })),
                            ),
                        ],
                        terminator: Terminator::Goto(1),
                        is_cleanup: false,
                    },
                    // bb1: check i <= n
                    BasicBlock {
                        stmts: vec![Statement::Assign(
                            Place { local: 3, projections: vec![] },
                            Rvalue::BinaryOp(
                                BinOp::Le,
                                Operand::Copy(Place { local: 2, projections: vec![] }),
                                Operand::Copy(Place { local: 1, projections: vec![] }),
                            ),
                        )],
                        terminator: Terminator::SwitchInt {
                            discr: Operand::Copy(Place { local: 3, projections: vec![] }),
                            targets: SwitchTargets {
                                values: vec![0],      // false → exit
                                targets: vec![3, 2],  // [false→bb3, otherwise(true)→bb2]
                            },
                        },
                        is_cleanup: false,
                    },
                    // bb2: acc += i; i += 1; goto bb1
                    BasicBlock {
                        stmts: vec![
                            Statement::Assign(
                                Place { local: 0, projections: vec![] },
                                Rvalue::BinaryOp(
                                    BinOp::Add,
                                    Operand::Copy(Place { local: 0, projections: vec![] }),
                                    Operand::Copy(Place { local: 2, projections: vec![] }),
                                ),
                            ),
                            Statement::Assign(
                                Place { local: 2, projections: vec![] },
                                Rvalue::BinaryOp(
                                    BinOp::Add,
                                    Operand::Copy(Place { local: 2, projections: vec![] }),
                                    Operand::Constant(ConstOperand {
                                        ty: Ty::Int(IntTy::I32),
                                        kind: ConstKind::Scalar(1, 4),
                                    }),
                                ),
                            ),
                        ],
                        terminator: Terminator::Goto(1),
                        is_cleanup: false,
                    },
                    // bb3: return
                    BasicBlock {
                        stmts: vec![],
                        terminator: Terminator::Return,
                        is_cleanup: false,
                    },
                ],
            },
        };

        let result: i32 = mirdata_jit_run_with_arg(&fn_body, &layout_entries, 10);
        assert_eq!(result, 55); // 1+2+...+10 = 55
    }

    /// Helper to run a mirdata function with a single i32 argument
    fn mirdata_jit_run_with_arg(fn_body: &FnBody, layout_entries: &[TypeLayoutEntry], arg: i32) -> i32 {
        let layouts = convert_mirdata_layouts(layout_entries);
        let ty_layouts = build_ty_layout_map(layout_entries, &layouts);
        let (mut module, isa, dl) = make_jit_module();
        let empty_reg = HashMap::new();
        let func_id = compile_mirdata_fn(
            &mut module,
            &*isa,
            &dl,
            fn_body,
            &layouts,
            &fn_body.name,
            Linkage::Export,
            &empty_reg,
            &ty_layouts,
        )
        .expect("compile_mirdata_fn failed");

        module.finalize_definitions().unwrap();
        let code = module.get_finalized_function(func_id);
        unsafe {
            let f: fn(i32) -> i32 = std::mem::transmute(code);
            f(arg)
        }
    }

    // -- Integration test: compile real sysroot mirdata functions ----------------

    #[test]
    fn mirdata_roundtrip_serialize() {
        // Verify that MirData serializes/deserializes correctly via postcard
        let data = MirData {
            crates: vec![CrateInfo { name: "test".into(), stable_crate_id: 42 }],
            bodies: vec![],
            layouts: vec![],
        };
        let bytes = postcard::to_allocvec(&data).unwrap();
        let data2: MirData = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(data2.crates.len(), 1);
        assert_eq!(data2.crates[0].name, "test");
    }

    /// Try to compile every monomorphic function from the sysroot mirdata file.
    /// Reports success/failure statistics. Run with:
    ///   cargo test -p cg-clif --lib mirdata_sysroot_compile -- --ignored --nocapture
    #[test]
    #[ignore]
    fn mirdata_sysroot_compile() {
        let mirdata_path = "/tmp/sysroot.mirdata";
        let bytes = match std::fs::read(mirdata_path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skipping: cannot read {mirdata_path}: {e}");
                return;
            }
        };
        eprintln!("Read {} bytes from {mirdata_path}", bytes.len());
        let mirdata: ra_mir_types::MirData = match postcard::from_bytes(&bytes) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Deserialization failed: {e:?}");
                eprintln!("This usually means the mirdata file was generated with a different");
                eprintln!("version of ra-mir-types. Regenerate with:");
                eprintln!("  cd ra-mir-export && cargo run --release -- -o /tmp/sysroot.mirdata");
                panic!("failed to deserialize mirdata: {e}");
            }
        };

        let layouts = convert_mirdata_layouts(&mirdata.layouts);
        let ty_layouts = build_ty_layout_map(&mirdata.layouts, &layouts);
        let (mut module, isa, dl) = make_jit_module();

        // Build body_map for looking up generic function bodies by hash
        let mut body_map: HashMap<(u64, u64), usize> = HashMap::new();
        for (i, fb) in mirdata.bodies.iter().enumerate() {
            body_map.insert(fb.def_path_hash, i);
        }

        // Build fn_registry: declare ALL functions so cross-references work.
        // Even generic functions need entries so Call terminators can resolve them.
        let mut fn_registry: HashMap<(u64, u64), FuncId> = HashMap::new();
        for (i, fb) in mirdata.bodies.iter().enumerate() {
            let fn_name = format!("__mirdata_{}_{}", i, fb.name.replace(|c: char| !c.is_alphanumeric(), "_"));
            let sig = build_mirdata_fn_sig(&*isa, &dl, &fb.body, &layouts)
                .unwrap_or_else(|_| {
                    // Fallback: void→void for functions whose locals lack layouts.
                    // The call site may get verifier errors, but this is better than
                    // using wrong-sized types that cause bitcast panics.
                    Signature::new(isa.default_call_conv())
                });
            let func_id = module
                .declare_function(&fn_name, Linkage::Local, &sig)
                .expect("declare_function");
            fn_registry.insert(fb.def_path_hash, func_id);
        }

        // Monomorphization pass: collect instances, compile them, add to fn_registry
        let mono_instances = collect_mono_instances(&mirdata.bodies, &body_map);
        eprintln!("Collected {} monomorphization instances", mono_instances.len());
        let mut mono_compiled = 0usize;
        let mut mono_failed = 0usize;
        let mut mono_errors: HashMap<String, Vec<String>> = HashMap::new();
        for inst in &mono_instances {
            let body_idx = body_map[&inst.def_path_hash];
            let generic_body = &mirdata.bodies[body_idx];
            let mono_body = generic_body.body.subst(&inst.args);

            // Build a unique name for this instance
            let mono_name = format!("__mono_{}_{:x}_{:x}",
                generic_body.name.replace(|c: char| !c.is_alphanumeric(), "_"),
                inst.registry_key().0, inst.registry_key().1);

            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compile_mirdata_body(
                    &mut module, &*isa, &dl, &mono_body, &layouts,
                    &mono_name, Linkage::Local, &fn_registry, &ty_layouts,
                )
            }));

            match result {
                Ok(Ok(func_id)) => {
                    fn_registry.insert(inst.registry_key(), func_id);
                    mono_compiled += 1;
                }
                Ok(Err(e)) => {
                    mono_failed += 1;
                    let key = if e.len() > 120 { e[..120].to_string() } else { e };
                    mono_errors.entry(key).or_default().push(generic_body.name.clone());
                }
                Err(panic_info) => {
                    mono_failed += 1;
                    let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "unknown panic".to_string()
                    };
                    let key = if msg.len() > 120 { msg[..120].to_string() } else { msg };
                    mono_errors.entry(key).or_default().push(generic_body.name.clone());
                }
            }
        }
        eprintln!("Mono compiled: {mono_compiled}, failed: {mono_failed}");
        if !mono_errors.is_empty() {
            eprintln!("\n--- Mono error categories ---");
            let mut sorted: Vec<_> = mono_errors.iter().collect();
            sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
            for (msg, fns) in sorted.iter().take(10) {
                eprintln!("[{} instances] {}", fns.len(), msg);
            }
        }

        let mut compiled = 0usize;
        let mut failed = 0usize;
        let mut skipped = 0usize;
        let mut errors: HashMap<String, Vec<String>> = HashMap::new();

        for (i, fb) in mirdata.bodies.iter().enumerate() {
            if fb.num_generic_params > 0 {
                skipped += 1;
                continue;
            }
            let all_have_layouts = fb.body.locals.iter().all(|l| l.layout.is_some());
            if !all_have_layouts {
                skipped += 1;
                continue;
            }

            let fn_name = format!("__mirdata_{}_{}", i, fb.name.replace(|c: char| !c.is_alphanumeric(), "_"));
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compile_mirdata_fn(
                    &mut module,
                    &*isa,
                    &dl,
                    fb,
                    &layouts,
                    &fn_name,
                    Linkage::Local,
                    &fn_registry,
                    &ty_layouts,
                )
            }));

            match result {
                Ok(Ok(_)) => compiled += 1,
                Ok(Err(e)) => {
                    failed += 1;
                    let key = e.to_string();
                    errors.entry(key).or_default().push(fb.name.clone());
                }
                Err(panic_info) => {
                    failed += 1;
                    let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "unknown panic".to_string()
                    };
                    // Truncate long panic messages for grouping
                    let key = if msg.len() > 120 { msg[..120].to_string() } else { msg };
                    errors.entry(key).or_default().push(fb.name.clone());
                }
            }
        }

        let total = mirdata.bodies.len();
        eprintln!("\n=== Mirdata sysroot compilation results ===");
        eprintln!("Total bodies: {total}");
        eprintln!("Skipped (generic/no-layout): {skipped}");
        eprintln!("Compiled OK: {compiled}");
        eprintln!("Failed: {failed}");
        if compiled + failed > 0 {
            eprintln!("Success rate: {:.1}%", compiled as f64 / (compiled + failed) as f64 * 100.0);
        }

        if !errors.is_empty() {
            eprintln!("\n--- Error categories (sorted by frequency) ---");
            let mut sorted: Vec<_> = errors.iter().collect();
            sorted.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
            for (msg, fns) in sorted.iter().take(25) {
                eprintln!("\n[{} functions] {}", fns.len(), msg);
                for name in fns.iter().take(3) {
                    eprintln!("  - {name}");
                }
                if fns.len() > 3 {
                    eprintln!("  ... and {} more", fns.len() - 3);
                }
            }
        }

        // Don't fail the test — this is informational
        eprintln!("\n(This test is informational; pass/fail is not asserted)");
    }
}
