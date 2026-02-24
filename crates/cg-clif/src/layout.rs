//! Convert mirdata layout types (`ExportedXxx` from `ra-mir-types`) back to
//! `rustc_abi` layout types that cg-clif already matches on.
//!
//! This lets us use pre-computed layouts from `.mirdata` files without changing
//! the existing codegen code that pattern-matches on `BackendRepr`, `Scalar`, etc.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::ops::Bound;

use ra_mir_types::{
    AdtDefEntry, AdtKind, ExportedBackendRepr, ExportedFieldsShape, ExportedNiche,
    ExportedPrimitive, ExportedScalar, ExportedTagEncoding, ExportedVariants, LayoutInfo,
    TypeLayoutEntry,
};
use rac_abi::{FieldIdx, VariantIdx};
use rustc_abi::{
    AddressSpace, Align, BackendRepr, FieldsShape, Float, Integer, LayoutCalculator,
    LayoutData, Niche, Primitive, ReprOptions, Scalar, Size, StructKind, TagEncoding,
    TargetDataLayout, Variants, WrappingRange,
};
use rustc_hashes::Hash64;
use rustc_index::IndexVec;
use triomphe::Arc as TArc;

use hir_ty::layout::Layout;

/// Convert all mirdata layout entries into `TArc<Layout>` values that cg-clif
/// can use directly (same type as `db.layout_of_ty()` returns).
pub fn convert_mirdata_layouts(layouts: &[TypeLayoutEntry]) -> Vec<TArc<Layout>> {
    layouts.iter().map(|entry| TArc::new(convert_layout(&entry.layout))).collect()
}

fn convert_layout(info: &LayoutInfo) -> Layout {
    let size = Size::from_bytes(info.size);
    let align = rustc_abi::AbiAlign::new(
        Align::from_bytes(info.align).expect("invalid alignment"),
    );

    Layout {
        fields: convert_fields(&info.fields),
        variants: convert_variants(&info.variants),
        backend_repr: convert_backend_repr(&info.backend_repr),
        largest_niche: info.largest_niche.as_ref().map(convert_niche),
        uninhabited: false,
        align,
        size,
        max_repr_align: None,
        unadjusted_abi_align: align.abi,
        randomization_seed: Hash64::new(0),
    }
}

fn convert_backend_repr(r: &ExportedBackendRepr) -> BackendRepr {
    match r {
        ExportedBackendRepr::Scalar(s) => BackendRepr::Scalar(convert_scalar(s)),
        ExportedBackendRepr::ScalarPair(a, b) => {
            BackendRepr::ScalarPair(convert_scalar(a), convert_scalar(b))
        }
        ExportedBackendRepr::Memory { sized } => BackendRepr::Memory { sized: *sized },
    }
}

fn convert_scalar(s: &ExportedScalar) -> Scalar {
    let value = convert_primitive(&s.primitive);
    let valid_range = WrappingRange { start: s.valid_range_start, end: s.valid_range_end };
    Scalar::Initialized { value, valid_range }
}

fn convert_primitive(p: &ExportedPrimitive) -> Primitive {
    match p {
        ExportedPrimitive::Int { size_bytes, signed } => {
            let int = match size_bytes {
                1 => Integer::I8,
                2 => Integer::I16,
                4 => Integer::I32,
                8 => Integer::I64,
                16 => Integer::I128,
                _ => panic!("unsupported integer size: {size_bytes}"),
            };
            Primitive::Int(int, *signed)
        }
        ExportedPrimitive::Float { size_bytes } => {
            let float = match size_bytes {
                2 => Float::F16,
                4 => Float::F32,
                8 => Float::F64,
                16 => Float::F128,
                _ => panic!("unsupported float size: {size_bytes}"),
            };
            Primitive::Float(float)
        }
        ExportedPrimitive::Pointer => Primitive::Pointer(AddressSpace(0)),
    }
}

fn convert_fields(f: &ExportedFieldsShape) -> FieldsShape<FieldIdx> {
    match f {
        ExportedFieldsShape::Primitive => FieldsShape::Primitive,
        ExportedFieldsShape::Union(count) => {
            FieldsShape::Union(NonZeroUsize::new(*count).expect("union with 0 fields"))
        }
        ExportedFieldsShape::Array { stride, count } => FieldsShape::Array {
            stride: Size::from_bytes(*stride),
            count: *count,
        },
        ExportedFieldsShape::Arbitrary { offsets } => {
            let offsets: IndexVec<FieldIdx, Size> =
                offsets.iter().map(|&o| Size::from_bytes(o)).collect();
            // Memory index: identity mapping (source order = memory order).
            // The mirdata export doesn't preserve reordering info, but codegen
            // only uses `offsets`, not `memory_index`.
            let memory_index: IndexVec<FieldIdx, u32> =
                (0..offsets.len() as u32).map(|i| i).collect();
            FieldsShape::Arbitrary { offsets, memory_index }
        }
    }
}

fn convert_variants(v: &ExportedVariants) -> Variants<FieldIdx, VariantIdx> {
    match v {
        ExportedVariants::Empty => Variants::Empty,
        ExportedVariants::Single { index } => {
            Variants::Single { index: VariantIdx::from_u32(*index) }
        }
        ExportedVariants::Multiple { tag, tag_encoding, tag_field, variants } => {
            let variant_layouts: IndexVec<VariantIdx, Layout> =
                variants.iter().map(|v| convert_layout(v)).collect();
            Variants::Multiple {
                tag: convert_scalar(tag),
                tag_encoding: convert_tag_encoding(tag_encoding),
                tag_field: FieldIdx::from_u32(*tag_field),
                variants: variant_layouts,
            }
        }
    }
}

fn convert_tag_encoding(e: &ExportedTagEncoding) -> TagEncoding<VariantIdx> {
    match e {
        ExportedTagEncoding::Direct => TagEncoding::Direct,
        ExportedTagEncoding::Niche {
            untagged_variant,
            niche_variants_start,
            niche_variants_end,
            niche_start,
        } => TagEncoding::Niche {
            untagged_variant: VariantIdx::from_u32(*untagged_variant),
            niche_variants: VariantIdx::from_u32(*niche_variants_start)
                ..=VariantIdx::from_u32(*niche_variants_end),
            niche_start: *niche_start,
        },
    }
}

fn convert_niche(n: &ExportedNiche) -> Niche {
    let scalar = convert_scalar(&n.scalar);
    let Scalar::Initialized { value, valid_range } = scalar else {
        panic!("niche scalar must be Initialized");
    };
    Niche { offset: Size::from_bytes(n.offset), value, valid_range }
}

// ---------------------------------------------------------------------------
// Layout computation for monomorphized generic types
// ---------------------------------------------------------------------------

fn scalar_unit(dl: &TargetDataLayout, value: Primitive) -> Scalar {
    Scalar::Initialized { value, valid_range: WrappingRange::full(value.size(dl)) }
}

/// Compute a layout for a given `ra_mir_types::Ty`, caching results in `ty_layouts`.
/// Uses ADT definitions from `adt_defs` to compute layouts for struct/enum/union types
/// that may not be in the pre-exported layout table (e.g. due to monomorphization).
///
/// Returns `None` if the type cannot be computed (e.g. unsupported Opaque types).
pub fn try_compute_layout(
    ty: &ra_mir_types::Ty,
    adt_defs: &HashMap<ra_mir_types::DefPathHash, AdtDefEntry>,
    ty_layouts: &mut HashMap<ra_mir_types::Ty, TArc<Layout>>,
    dl: &TargetDataLayout,
) -> Option<TArc<Layout>> {
    // Check cache first
    if let Some(layout) = ty_layouts.get(ty) {
        return Some(layout.clone());
    }

    let calc = LayoutCalculator::new(dl);
    let result = compute_layout_inner(ty, adt_defs, ty_layouts, dl, &calc)?;
    ty_layouts.insert(ty.clone(), result.clone());
    Some(result)
}

/// Compute a layout, panicking if it can't be computed.
pub fn compute_layout(
    ty: &ra_mir_types::Ty,
    adt_defs: &HashMap<ra_mir_types::DefPathHash, AdtDefEntry>,
    ty_layouts: &mut HashMap<ra_mir_types::Ty, TArc<Layout>>,
    dl: &TargetDataLayout,
) -> TArc<Layout> {
    try_compute_layout(ty, adt_defs, ty_layouts, dl)
        .unwrap_or_else(|| panic!("compute_layout failed for type: {:?}", ty))
}

fn compute_layout_inner(
    ty: &ra_mir_types::Ty,
    adt_defs: &HashMap<ra_mir_types::DefPathHash, AdtDefEntry>,
    ty_layouts: &mut HashMap<ra_mir_types::Ty, TArc<Layout>>,
    dl: &TargetDataLayout,
    calc: &LayoutCalculator<&TargetDataLayout>,
) -> Option<TArc<Layout>> {
    use ra_mir_types::Ty;

    let layout = match ty {
        Ty::Bool => Layout::scalar(
            dl,
            Scalar::Initialized {
                value: Primitive::Int(Integer::I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            },
        ),
        Ty::Char => Layout::scalar(
            dl,
            Scalar::Initialized {
                value: Primitive::Int(Integer::I32, false),
                valid_range: WrappingRange { start: 0, end: 0x10FFFF },
            },
        ),
        Ty::Int(int_ty) => {
            let int = match int_ty {
                ra_mir_types::IntTy::Isize => dl.ptr_sized_integer(),
                ra_mir_types::IntTy::I8 => Integer::I8,
                ra_mir_types::IntTy::I16 => Integer::I16,
                ra_mir_types::IntTy::I32 => Integer::I32,
                ra_mir_types::IntTy::I64 => Integer::I64,
                ra_mir_types::IntTy::I128 => Integer::I128,
            };
            Layout::scalar(dl, scalar_unit(dl, Primitive::Int(int, true)))
        }
        Ty::Uint(uint_ty) => {
            let int = match uint_ty {
                ra_mir_types::UintTy::Usize => dl.ptr_sized_integer(),
                ra_mir_types::UintTy::U8 => Integer::I8,
                ra_mir_types::UintTy::U16 => Integer::I16,
                ra_mir_types::UintTy::U32 => Integer::I32,
                ra_mir_types::UintTy::U64 => Integer::I64,
                ra_mir_types::UintTy::U128 => Integer::I128,
            };
            Layout::scalar(dl, scalar_unit(dl, Primitive::Int(int, false)))
        }
        Ty::Float(float_ty) => {
            let float = match float_ty {
                ra_mir_types::FloatTy::F16 => Float::F16,
                ra_mir_types::FloatTy::F32 => Float::F32,
                ra_mir_types::FloatTy::F64 => Float::F64,
                ra_mir_types::FloatTy::F128 => Float::F128,
            };
            Layout::scalar(dl, scalar_unit(dl, Primitive::Float(float)))
        }
        Ty::Never => LayoutData::never_type(dl),
        Ty::Str => {
            let element = scalar_unit(dl, Primitive::Int(Integer::I8, false));
            calc.array_like::<_, _, ()>(&Layout::scalar(dl, element), None)
                .expect("str layout")
        }
        Ty::Tuple(tys) => {
            if tys.is_empty() {
                LayoutData::unit(dl, true)
            } else {
                let fields: Vec<_> = tys
                    .iter()
                    .map(|t| try_compute_layout(t, adt_defs, ty_layouts, dl))
                    .collect::<Option<Vec<_>>>()?;
                let field_refs: Vec<&Layout> = fields.iter().map(|f| &**f).collect();
                let field_iters: IndexVec<FieldIdx, &&Layout> = field_refs.iter().collect();
                calc.univariant(&field_iters, &ReprOptions::default(), StructKind::AlwaysSized)
                    .expect("tuple layout")
            }
        }
        Ty::Array(elem, count) => {
            let elem_layout = try_compute_layout(elem, adt_defs, ty_layouts, dl)?;
            calc.array_like::<_, _, ()>(&*elem_layout, Some(*count))
                .expect("array layout")
        }
        Ty::Slice(elem) => {
            let elem_layout = try_compute_layout(elem, adt_defs, ty_layouts, dl)?;
            calc.array_like::<_, _, ()>(&*elem_layout, None)
                .expect("slice layout")
        }
        Ty::Ref(_, pointee) | Ty::RawPtr(_, pointee) => {
            let is_ref = matches!(ty, Ty::Ref(..));
            let mut data_ptr = scalar_unit(dl, Primitive::Pointer(AddressSpace::ZERO));
            if is_ref {
                data_ptr.valid_range_mut().start = 1;
            }

            // Check if pointee is unsized (slice, str, dyn)
            match &**pointee {
                Ty::Slice(_) | Ty::Str => {
                    let metadata = scalar_unit(dl, Primitive::Int(dl.ptr_sized_integer(), false));
                    LayoutData::scalar_pair(dl, data_ptr, metadata)
                }
                Ty::Dynamic(_) => {
                    let mut vtable = scalar_unit(dl, Primitive::Pointer(AddressSpace::ZERO));
                    vtable.valid_range_mut().start = 1;
                    LayoutData::scalar_pair(dl, data_ptr, vtable)
                }
                _ => {
                    // Sized pointee — thin pointer
                    Layout::scalar(dl, data_ptr)
                }
            }
        }
        Ty::FnDef(_, _) => LayoutData::unit(dl, true),
        Ty::FnPtr(_, _) => {
            let mut ptr = scalar_unit(dl, Primitive::Pointer(dl.instruction_address_space));
            ptr.valid_range_mut().start = 1;
            Layout::scalar(dl, ptr)
        }
        Ty::Foreign(_) | Ty::Dynamic(_) => LayoutData::unit(dl, false),
        Ty::Closure(_, _) => {
            // Closures should have pre-computed layouts; fallback to ZST
            LayoutData::unit(dl, true)
        }
        Ty::Adt(hash, _name, args) => {
            return compute_adt_layout(hash, args, adt_defs, ty_layouts, dl, calc);
        }
        Ty::Param(_, _) | Ty::Opaque(_) => {
            // Can't compute layout for unresolved params or opaque types.
            // These should have pre-computed layouts in ty_layouts if needed.
            return None;
        }
    };

    Some(TArc::new(layout))
}

fn compute_adt_layout(
    hash: &ra_mir_types::DefPathHash,
    args: &[ra_mir_types::GenericArg],
    adt_defs: &HashMap<ra_mir_types::DefPathHash, AdtDefEntry>,
    ty_layouts: &mut HashMap<ra_mir_types::Ty, TArc<Layout>>,
    dl: &TargetDataLayout,
    calc: &LayoutCalculator<&TargetDataLayout>,
) -> Option<TArc<Layout>> {
    let adt = adt_defs.get(hash)?;

    // Compute layouts for each variant's fields after substitution
    let variants: Vec<Vec<TArc<Layout>>> = adt
        .variants
        .iter()
        .map(|variant| {
            variant
                .fields
                .iter()
                .map(|field_ty| {
                    let substituted = field_ty.subst(args);
                    try_compute_layout(&substituted, adt_defs, ty_layouts, dl)
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()?;

    let repr = convert_repr_options(&adt.repr, dl);

    let variant_refs: Vec<Vec<&Layout>> = variants
        .iter()
        .map(|v| v.iter().map(|l| &**l).collect())
        .collect();
    let variant_index_vecs: IndexVec<VariantIdx, IndexVec<FieldIdx, &&Layout>> = variant_refs
        .iter()
        .map(|v| v.iter().collect())
        .collect();

    let result = match adt.kind {
        AdtKind::Union => {
            calc.layout_of_union(&repr, &variant_index_vecs)
                .expect("union layout computation failed")
        }
        AdtKind::Struct | AdtKind::Enum => {
            let is_enum = adt.kind == AdtKind::Enum;
            calc.layout_of_struct_or_enum(
                &repr,
                &variant_index_vecs,
                is_enum,
                adt.is_special_no_niche,
                (Bound::Unbounded, Bound::Unbounded), // scalar_valid_range
                |min, max| {
                    repr_discr(dl, &repr, min, max)
                },
                // Discriminant values — for enums, use 0, 1, 2, ...
                if is_enum {
                    variant_index_vecs
                        .iter_enumerated()
                        .map(|(id, _)| (id, id.as_u32() as i128))
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                }
                .into_iter(),
                // always_sized: true for structs, check last field for enums
                !is_enum
                    && variant_index_vecs
                        .iter()
                        .next()
                        .and_then(|it| it.iter().last().map(|it| !it.is_unsized()))
                        .unwrap_or(true),
            )
            .expect("struct/enum layout computation failed")
        }
    };

    Some(TArc::new(result))
}

fn convert_repr_options(
    repr: &ra_mir_types::ExportedReprOptions,
    dl: &TargetDataLayout,
) -> ReprOptions {
    use rustc_abi::{IntegerType, ReprFlags};

    let mut flags = ReprFlags::empty();
    if repr.c {
        flags |= ReprFlags::IS_C;
    }
    if repr.transparent {
        flags |= ReprFlags::IS_TRANSPARENT;
    }

    let int = repr.int.as_ref().map(|ity| match ity {
        ra_mir_types::ExportedIntegerType::Fixed { size_bytes, signed } => {
            let int = match size_bytes {
                1 => Integer::I8,
                2 => Integer::I16,
                4 => Integer::I32,
                8 => Integer::I64,
                16 => Integer::I128,
                _ => panic!("unsupported integer size: {size_bytes}"),
            };
            IntegerType::Fixed(int, *signed)
        }
        ra_mir_types::ExportedIntegerType::Pointer(signed) => {
            IntegerType::Pointer(*signed)
        }
    });

    let _ = dl; // dl used for pack/align validation if needed

    ReprOptions {
        int,
        align: repr.align.map(|a| Align::from_bytes(a).expect("invalid repr align")),
        pack: repr.packed.map(|p| Align::from_bytes(p).expect("invalid repr pack")),
        flags,
        field_shuffle_seed: Hash64::new(0),
    }
}

/// Finds the appropriate Integer type and signedness for the given
/// signed discriminant range and `#[repr]` attribute.
fn repr_discr(
    dl: &TargetDataLayout,
    repr: &ReprOptions,
    min: i128,
    max: i128,
) -> (Integer, bool) {
    use std::cmp;

    let unsigned_fit = Integer::fit_unsigned(cmp::max(min as u128, max as u128));
    let signed_fit = cmp::max(Integer::fit_signed(min), Integer::fit_signed(max));

    if let Some(ity) = repr.int {
        let discr = Integer::from_attr(dl, ity);
        let fit = if ity.is_signed() { signed_fit } else { unsigned_fit };
        if discr < fit {
            return (discr, ity.is_signed());
        }
        return (discr, ity.is_signed());
    }

    let at_least = if repr.c() {
        dl.c_enum_min_size
    } else {
        Integer::I8
    };

    if min >= 0 {
        (cmp::max(unsigned_fit, at_least), false)
    } else {
        (cmp::max(signed_fit, at_least), true)
    }
}

// ---------------------------------------------------------------------------
// Body type walker: ensure all body types have layouts
// ---------------------------------------------------------------------------

/// Walk a monomorphized body and ensure all types referenced in it
/// have computed layouts in `ty_layouts`.
pub fn ensure_body_layouts(
    body: &ra_mir_types::Body,
    adt_defs: &HashMap<ra_mir_types::DefPathHash, AdtDefEntry>,
    ty_layouts: &mut HashMap<ra_mir_types::Ty, TArc<Layout>>,
    dl: &TargetDataLayout,
) {
    let mut types_to_process: Vec<ra_mir_types::Ty> = Vec::new();

    // Collect all types from the body
    for local in &body.locals {
        types_to_process.push(local.ty.clone());
    }

    for bb in &body.blocks {
        for stmt in &bb.stmts {
            collect_stmt_types(stmt, &mut types_to_process);
        }
        collect_terminator_types(&bb.terminator, &mut types_to_process);
    }

    // Ensure layout for each collected type and its components
    let mut visited = std::collections::HashSet::new();
    for ty in types_to_process {
        ensure_layout_recursive(&ty, adt_defs, ty_layouts, dl, &mut visited);
    }
}

fn collect_stmt_types(stmt: &ra_mir_types::Statement, out: &mut Vec<ra_mir_types::Ty>) {
    use ra_mir_types::Statement;
    match stmt {
        Statement::Assign(place, rvalue) => {
            collect_place_types(place, out);
            collect_rvalue_types(rvalue, out);
        }
        Statement::SetDiscriminant { place, .. } => {
            collect_place_types(place, out);
        }
        Statement::Deinit(place) => {
            collect_place_types(place, out);
        }
        Statement::StorageLive(_) | Statement::StorageDead(_) | Statement::Nop => {}
    }
}

fn collect_terminator_types(term: &ra_mir_types::Terminator, out: &mut Vec<ra_mir_types::Ty>) {
    use ra_mir_types::Terminator;
    match term {
        Terminator::Call { func, args, dest, .. } => {
            collect_operand_types(func, out);
            for arg in args {
                collect_operand_types(arg, out);
            }
            collect_place_types(dest, out);
        }
        Terminator::SwitchInt { discr, .. } => {
            collect_operand_types(discr, out);
        }
        Terminator::Drop { place, .. } => {
            collect_place_types(place, out);
        }
        Terminator::Assert { cond, .. } => {
            collect_operand_types(cond, out);
        }
        Terminator::Goto(_)
        | Terminator::Return
        | Terminator::Unreachable
        | Terminator::UnwindResume => {}
    }
}

fn collect_operand_types(op: &ra_mir_types::Operand, out: &mut Vec<ra_mir_types::Ty>) {
    match op {
        ra_mir_types::Operand::Copy(p) | ra_mir_types::Operand::Move(p) => {
            collect_place_types(p, out);
        }
        ra_mir_types::Operand::Constant(c) => {
            out.push(c.ty.clone());
        }
    }
}

fn collect_place_types(place: &ra_mir_types::Place, out: &mut Vec<ra_mir_types::Ty>) {
    for proj in &place.projections {
        match proj {
            ra_mir_types::Projection::Field(_, ty) => out.push(ty.clone()),
            ra_mir_types::Projection::OpaqueCast(ty) => out.push(ty.clone()),
            _ => {}
        }
    }
}

fn collect_rvalue_types(rv: &ra_mir_types::Rvalue, out: &mut Vec<ra_mir_types::Ty>) {
    use ra_mir_types::Rvalue;
    match rv {
        Rvalue::Cast(_, op, ty) => {
            collect_operand_types(op, out);
            out.push(ty.clone());
        }
        Rvalue::Aggregate(kind, ops) => {
            match kind {
                ra_mir_types::AggregateKind::Array(ty) => out.push(ty.clone()),
                _ => {}
            }
            for op in ops {
                collect_operand_types(op, out);
            }
        }
        Rvalue::Use(op) | Rvalue::Repeat(op, _) | Rvalue::UnaryOp(_, op) => {
            collect_operand_types(op, out);
        }
        Rvalue::BinaryOp(_, lhs, rhs) => {
            collect_operand_types(lhs, out);
            collect_operand_types(rhs, out);
        }
        Rvalue::Ref(_, place) | Rvalue::RawPtr(_, place) | Rvalue::Discriminant(place) | Rvalue::CopyForDeref(place) => {
            collect_place_types(place, out);
        }
        Rvalue::ThreadLocalRef(_) => {}
    }
}

/// Recursively ensure a type and all its component types have layouts.
fn ensure_layout_recursive(
    ty: &ra_mir_types::Ty,
    adt_defs: &HashMap<ra_mir_types::DefPathHash, AdtDefEntry>,
    ty_layouts: &mut HashMap<ra_mir_types::Ty, TArc<Layout>>,
    dl: &TargetDataLayout,
    visited: &mut std::collections::HashSet<ra_mir_types::Ty>,
) {
    if !visited.insert(ty.clone()) {
        return;
    }

    // Skip types with unresolved params
    if ty.has_param() {
        return;
    }

    // Try to compute layout for this type (may fail for unsupported types like Opaque)
    if try_compute_layout(ty, adt_defs, ty_layouts, dl).is_none() {
        return; // Can't compute — stop recursion
    }

    // Recurse into component types
    match ty {
        ra_mir_types::Ty::Ref(_, pointee) | ra_mir_types::Ty::RawPtr(_, pointee) => {
            ensure_layout_recursive(pointee, adt_defs, ty_layouts, dl, visited);
        }
        ra_mir_types::Ty::Array(elem, _) | ra_mir_types::Ty::Slice(elem) => {
            ensure_layout_recursive(elem, adt_defs, ty_layouts, dl, visited);
        }
        ra_mir_types::Ty::Tuple(tys) => {
            for t in tys {
                ensure_layout_recursive(t, adt_defs, ty_layouts, dl, visited);
            }
        }
        ra_mir_types::Ty::Adt(hash, _, args) => {
            if let Some(adt) = adt_defs.get(hash) {
                for variant in &adt.variants {
                    for field_ty in &variant.fields {
                        let substituted = field_ty.subst(args);
                        ensure_layout_recursive(&substituted, adt_defs, ty_layouts, dl, visited);
                    }
                }
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Export helpers (mirror ra-mir-export's translate.rs for roundtrip testing)
// ---------------------------------------------------------------------------

/// Export a `Layout` to `LayoutInfo` (same logic as ra-mir-export's `translate_layout_data`).
/// Used only for roundtrip tests.
#[cfg(test)]
pub fn export_layout(layout: &Layout) -> LayoutInfo {
    LayoutInfo {
        size: layout.size.bytes(),
        align: layout.align.abi.bytes(),
        backend_repr: export_backend_repr(&layout.backend_repr),
        fields: export_fields(&layout.fields),
        variants: export_variants(&layout.variants),
        largest_niche: layout.largest_niche.as_ref().map(export_niche),
    }
}

#[cfg(test)]
fn export_backend_repr(repr: &BackendRepr) -> ExportedBackendRepr {
    match repr {
        BackendRepr::Scalar(s) => ExportedBackendRepr::Scalar(export_scalar(s)),
        BackendRepr::ScalarPair(a, b) => {
            ExportedBackendRepr::ScalarPair(export_scalar(a), export_scalar(b))
        }
        BackendRepr::Memory { sized } => ExportedBackendRepr::Memory { sized: *sized },
        _ => ExportedBackendRepr::Memory { sized: true },
    }
}

#[cfg(test)]
fn export_scalar(s: &Scalar) -> ExportedScalar {
    match s {
        Scalar::Initialized { value, valid_range } => ExportedScalar {
            primitive: export_primitive(value),
            valid_range_start: valid_range.start,
            valid_range_end: valid_range.end,
        },
        Scalar::Union { value } => ExportedScalar {
            primitive: export_primitive(value),
            valid_range_start: 0,
            valid_range_end: u128::MAX,
        },
    }
}

#[cfg(test)]
fn export_primitive(p: &Primitive) -> ExportedPrimitive {
    match p {
        Primitive::Int(int, signed) => {
            let size_bytes = match int {
                Integer::I8 => 1,
                Integer::I16 => 2,
                Integer::I32 => 4,
                Integer::I64 => 8,
                Integer::I128 => 16,
            };
            ExportedPrimitive::Int { size_bytes, signed: *signed }
        }
        Primitive::Float(float) => {
            let size_bytes = match float {
                Float::F16 => 2,
                Float::F32 => 4,
                Float::F64 => 8,
                Float::F128 => 16,
            };
            ExportedPrimitive::Float { size_bytes }
        }
        Primitive::Pointer(_) => ExportedPrimitive::Pointer,
    }
}

#[cfg(test)]
fn export_fields(fields: &FieldsShape<FieldIdx>) -> ExportedFieldsShape {
    match fields {
        FieldsShape::Primitive => ExportedFieldsShape::Primitive,
        FieldsShape::Union(count) => ExportedFieldsShape::Union(count.get()),
        FieldsShape::Array { stride, count } => ExportedFieldsShape::Array {
            stride: stride.bytes(),
            count: *count,
        },
        FieldsShape::Arbitrary { offsets, .. } => ExportedFieldsShape::Arbitrary {
            offsets: offsets.iter().map(|o| o.bytes()).collect(),
        },
    }
}

#[cfg(test)]
fn export_variants(
    variants: &Variants<FieldIdx, VariantIdx>,
) -> ExportedVariants {
    match variants {
        Variants::Empty => ExportedVariants::Empty,
        Variants::Single { index } => ExportedVariants::Single { index: index.as_u32() },
        Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
            ExportedVariants::Multiple {
                tag: export_scalar(tag),
                tag_encoding: export_tag_encoding(tag_encoding),
                tag_field: tag_field.as_u32(),
                variants: variants.iter().map(export_layout).collect(),
            }
        }
    }
}

#[cfg(test)]
fn export_tag_encoding(enc: &TagEncoding<VariantIdx>) -> ExportedTagEncoding {
    match enc {
        TagEncoding::Direct => ExportedTagEncoding::Direct,
        TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            ExportedTagEncoding::Niche {
                untagged_variant: untagged_variant.as_u32(),
                niche_variants_start: niche_variants.start().as_u32(),
                niche_variants_end: niche_variants.end().as_u32(),
                niche_start: *niche_start,
            }
        }
    }
}

#[cfg(test)]
fn export_niche(niche: &Niche) -> ExportedNiche {
    ExportedNiche {
        offset: niche.offset.bytes(),
        scalar: ExportedScalar {
            primitive: export_primitive(&niche.value),
            valid_range_start: niche.valid_range.start,
            valid_range_end: niche.valid_range.end,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ra_mir_types::Ty;


    // -----------------------------------------------------------------------
    // Unit tests for individual converters
    // -----------------------------------------------------------------------

    #[test]
    fn convert_i32_layout() {
        let info = LayoutInfo {
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
        };

        let layout = convert_layout(&info);

        assert_eq!(layout.size, Size::from_bytes(4));
        assert_eq!(layout.align.abi.bytes(), 4);
        let BackendRepr::Scalar(scalar) = layout.backend_repr else {
            panic!("expected Scalar");
        };
        assert_eq!(scalar.primitive(), Primitive::Int(Integer::I32, true));
        assert!(matches!(layout.fields, FieldsShape::Primitive));
        assert!(matches!(layout.variants, Variants::Single { .. }));
    }

    #[test]
    fn convert_f64_layout() {
        let info = LayoutInfo {
            size: 8,
            align: 8,
            backend_repr: ExportedBackendRepr::Scalar(ExportedScalar {
                primitive: ExportedPrimitive::Float { size_bytes: 8 },
                valid_range_start: 0,
                valid_range_end: u64::MAX as u128,
            }),
            fields: ExportedFieldsShape::Primitive,
            variants: ExportedVariants::Single { index: 0 },
            largest_niche: None,
        };

        let layout = convert_layout(&info);

        assert_eq!(layout.size, Size::from_bytes(8));
        let BackendRepr::Scalar(scalar) = layout.backend_repr else {
            panic!("expected Scalar");
        };
        assert_eq!(scalar.primitive(), Primitive::Float(Float::F64));
    }

    #[test]
    fn convert_pointer_layout() {
        let info = LayoutInfo {
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
        };

        let layout = convert_layout(&info);

        let BackendRepr::Scalar(scalar) = layout.backend_repr else {
            panic!("expected Scalar");
        };
        assert_eq!(scalar.primitive(), Primitive::Pointer(AddressSpace(0)));
    }

    #[test]
    fn convert_scalar_pair_layout() {
        // Like (i32, i32) — ScalarPair with two 4-byte fields at offsets 0, 4
        let info = LayoutInfo {
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
        };

        let layout = convert_layout(&info);

        assert_eq!(layout.size, Size::from_bytes(8));
        let BackendRepr::ScalarPair(a, b) = layout.backend_repr else {
            panic!("expected ScalarPair");
        };
        assert_eq!(a.primitive(), Primitive::Int(Integer::I32, true));
        assert_eq!(b.primitive(), Primitive::Int(Integer::I32, true));

        let FieldsShape::Arbitrary { ref offsets, ref memory_index } = layout.fields else {
            panic!("expected Arbitrary");
        };
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[FieldIdx::from_u32(0)], Size::from_bytes(0));
        assert_eq!(offsets[FieldIdx::from_u32(1)], Size::from_bytes(4));
        assert_eq!(memory_index.len(), 2);
    }

    #[test]
    fn convert_memory_layout() {
        // 3-field struct: Memory repr with arbitrary offsets
        let info = LayoutInfo {
            size: 12,
            align: 4,
            backend_repr: ExportedBackendRepr::Memory { sized: true },
            fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 4, 8] },
            variants: ExportedVariants::Single { index: 0 },
            largest_niche: None,
        };

        let layout = convert_layout(&info);

        assert_eq!(layout.size, Size::from_bytes(12));
        assert!(matches!(layout.backend_repr, BackendRepr::Memory { sized: true }));
        let FieldsShape::Arbitrary { ref offsets, .. } = layout.fields else {
            panic!("expected Arbitrary");
        };
        assert_eq!(offsets.len(), 3);
    }

    #[test]
    fn convert_array_fields() {
        let info = LayoutInfo {
            size: 12,
            align: 4,
            backend_repr: ExportedBackendRepr::Memory { sized: true },
            fields: ExportedFieldsShape::Array { stride: 4, count: 3 },
            variants: ExportedVariants::Single { index: 0 },
            largest_niche: None,
        };

        let layout = convert_layout(&info);

        let FieldsShape::Array { stride, count } = layout.fields else {
            panic!("expected Array");
        };
        assert_eq!(stride, Size::from_bytes(4));
        assert_eq!(count, 3);
    }

    #[test]
    fn convert_union_fields() {
        let info = LayoutInfo {
            size: 8,
            align: 4,
            backend_repr: ExportedBackendRepr::Memory { sized: true },
            fields: ExportedFieldsShape::Union(2),
            variants: ExportedVariants::Single { index: 0 },
            largest_niche: None,
        };

        let layout = convert_layout(&info);

        let FieldsShape::Union(count) = layout.fields else {
            panic!("expected Union");
        };
        assert_eq!(count.get(), 2);
    }

    #[test]
    fn convert_enum_direct_tag() {
        // Enum with 2 variants, direct tag encoding
        let variant0 = LayoutInfo {
            size: 8,
            align: 4,
            backend_repr: ExportedBackendRepr::Memory { sized: true },
            fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 4] },
            variants: ExportedVariants::Single { index: 0 },
            largest_niche: None,
        };
        let variant1 = LayoutInfo {
            size: 8,
            align: 4,
            backend_repr: ExportedBackendRepr::Memory { sized: true },
            fields: ExportedFieldsShape::Arbitrary { offsets: vec![0, 4] },
            variants: ExportedVariants::Single { index: 1 },
            largest_niche: None,
        };
        let info = LayoutInfo {
            size: 12,
            align: 4,
            backend_repr: ExportedBackendRepr::Memory { sized: true },
            fields: ExportedFieldsShape::Arbitrary { offsets: vec![0] },
            variants: ExportedVariants::Multiple {
                tag: ExportedScalar {
                    primitive: ExportedPrimitive::Int { size_bytes: 1, signed: false },
                    valid_range_start: 0,
                    valid_range_end: 1,
                },
                tag_encoding: ExportedTagEncoding::Direct,
                tag_field: 0,
                variants: vec![variant0, variant1],
            },
            largest_niche: None,
        };

        let layout = convert_layout(&info);

        let Variants::Multiple { ref tag, ref tag_encoding, tag_field, ref variants } =
            layout.variants
        else {
            panic!("expected Multiple");
        };
        assert_eq!(tag.primitive(), Primitive::Int(Integer::I8, false));
        assert!(matches!(tag_encoding, TagEncoding::Direct));
        assert_eq!(tag_field, FieldIdx::from_u32(0));
        assert_eq!(variants.len(), 2);
    }

    #[test]
    fn convert_niche_tag_encoding() {
        let enc = ExportedTagEncoding::Niche {
            untagged_variant: 1,
            niche_variants_start: 0,
            niche_variants_end: 0,
            niche_start: 0,
        };

        let result = convert_tag_encoding(&enc);

        let TagEncoding::Niche { untagged_variant, ref niche_variants, niche_start } = result
        else {
            panic!("expected Niche");
        };
        assert_eq!(untagged_variant, VariantIdx::from_u32(1));
        assert_eq!(*niche_variants.start(), VariantIdx::from_u32(0));
        assert_eq!(*niche_variants.end(), VariantIdx::from_u32(0));
        assert_eq!(niche_start, 0);
    }

    #[test]
    fn convert_niche_value() {
        let exported = ExportedNiche {
            offset: 4,
            scalar: ExportedScalar {
                primitive: ExportedPrimitive::Int { size_bytes: 1, signed: false },
                valid_range_start: 0,
                valid_range_end: 0,
            },
        };

        let niche = convert_niche(&exported);

        assert_eq!(niche.offset, Size::from_bytes(4));
        assert_eq!(niche.value, Primitive::Int(Integer::I8, false));
        assert_eq!(niche.valid_range.start, 0);
        assert_eq!(niche.valid_range.end, 0);
    }

    #[test]
    fn convert_all_int_sizes() {
        for (size, expected) in [
            (1u8, Integer::I8),
            (2, Integer::I16),
            (4, Integer::I32),
            (8, Integer::I64),
            (16, Integer::I128),
        ] {
            let prim = convert_primitive(&ExportedPrimitive::Int { size_bytes: size, signed: false });
            assert_eq!(prim, Primitive::Int(expected, false));
            let prim = convert_primitive(&ExportedPrimitive::Int { size_bytes: size, signed: true });
            assert_eq!(prim, Primitive::Int(expected, true));
        }
    }

    #[test]
    fn convert_all_float_sizes() {
        for (size, expected) in [
            (2u8, Float::F16),
            (4, Float::F32),
            (8, Float::F64),
            (16, Float::F128),
        ] {
            let prim = convert_primitive(&ExportedPrimitive::Float { size_bytes: size });
            assert_eq!(prim, Primitive::Float(expected));
        }
    }

    #[test]
    fn convert_mirdata_layouts_batch() {
        let entries = vec![
            TypeLayoutEntry {
                ty: Ty::Int(ra_mir_types::IntTy::I32),
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
            },
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
            },
        ];

        let layouts = convert_mirdata_layouts(&entries);

        assert_eq!(layouts.len(), 2);
        // i32
        assert_eq!(layouts[0].size, Size::from_bytes(4));
        assert!(layouts[0].largest_niche.is_none());
        // bool
        assert_eq!(layouts[1].size, Size::from_bytes(1));
        assert!(layouts[1].largest_niche.is_some());
        let niche = layouts[1].largest_niche.unwrap();
        assert_eq!(niche.valid_range.end, 1);
    }

    // -----------------------------------------------------------------------
    // Roundtrip tests: Layout → export → convert back → compare
    // -----------------------------------------------------------------------

    /// Check that codegen-relevant fields match after roundtrip.
    fn assert_layout_roundtrip(original: &Layout) {
        let exported = export_layout(original);
        let converted = convert_layout(&exported);

        assert_eq!(original.size, converted.size, "size mismatch");
        assert_eq!(original.align, converted.align, "align mismatch");
        assert_eq!(original.backend_repr, converted.backend_repr, "backend_repr mismatch");
        assert_eq!(original.largest_niche, converted.largest_niche, "largest_niche mismatch");

        // Fields: compare count and offsets (memory_index may differ — we use identity)
        assert_eq!(original.fields.count(), converted.fields.count(), "field count mismatch");
        for i in 0..original.fields.count() {
            assert_eq!(
                original.fields.offset(i),
                converted.fields.offset(i),
                "field offset mismatch at index {i}"
            );
        }

        // Variants: compare discriminant structure
        match (&original.variants, &converted.variants) {
            (Variants::Empty, Variants::Empty) => {}
            (Variants::Single { index: a }, Variants::Single { index: b }) => {
                assert_eq!(a, b, "variant index mismatch");
            }
            (
                Variants::Multiple {
                    tag: tag_a,
                    tag_encoding: enc_a,
                    tag_field: tf_a,
                    variants: vars_a,
                },
                Variants::Multiple {
                    tag: tag_b,
                    tag_encoding: enc_b,
                    tag_field: tf_b,
                    variants: vars_b,
                },
            ) => {
                assert_eq!(tag_a, tag_b, "tag mismatch");
                assert_eq!(tf_a, tf_b, "tag_field mismatch");
                assert_eq!(vars_a.len(), vars_b.len(), "variant count mismatch");
                // tag_encoding
                match (enc_a, enc_b) {
                    (TagEncoding::Direct, TagEncoding::Direct) => {}
                    (
                        TagEncoding::Niche {
                            untagged_variant: uv_a,
                            niche_variants: nv_a,
                            niche_start: ns_a,
                        },
                        TagEncoding::Niche {
                            untagged_variant: uv_b,
                            niche_variants: nv_b,
                            niche_start: ns_b,
                        },
                    ) => {
                        assert_eq!(uv_a, uv_b);
                        assert_eq!(nv_a, nv_b);
                        assert_eq!(ns_a, ns_b);
                    }
                    _ => panic!("tag_encoding mismatch: {:?} vs {:?}", enc_a, enc_b),
                }
                // Recurse into variant layouts
                for (va, vb) in vars_a.iter().zip(vars_b.iter()) {
                    assert_layout_roundtrip(va);
                    let _ = vb; // compared via recursion on va
                }
            }
            _ => panic!("variant shape mismatch"),
        }
    }

    #[test]
    fn roundtrip_i32() {
        let layout = Layout {
            size: Size::from_bytes(4),
            align: rustc_abi::AbiAlign::new(Align::from_bytes(4).unwrap()),
            backend_repr: BackendRepr::Scalar(Scalar::Initialized {
                value: Primitive::Int(Integer::I32, true),
                valid_range: WrappingRange { start: 0, end: u32::MAX as u128 },
            }),
            fields: FieldsShape::Primitive,
            variants: Variants::Single { index: VariantIdx::from_u32(0) },
            largest_niche: None,
            uninhabited: false,
            max_repr_align: None,
            unadjusted_abi_align: Align::from_bytes(4).unwrap(),
            randomization_seed: Hash64::new(0),
        };
        assert_layout_roundtrip(&layout);
    }

    #[test]
    fn roundtrip_scalar_pair() {
        let i32_scalar = Scalar::Initialized {
            value: Primitive::Int(Integer::I32, true),
            valid_range: WrappingRange { start: 0, end: u32::MAX as u128 },
        };
        let offsets: IndexVec<FieldIdx, Size> =
            vec![Size::from_bytes(0), Size::from_bytes(4)].into_iter().collect();
        let memory_index: IndexVec<FieldIdx, u32> = vec![0, 1].into_iter().collect();
        let layout = Layout {
            size: Size::from_bytes(8),
            align: rustc_abi::AbiAlign::new(Align::from_bytes(4).unwrap()),
            backend_repr: BackendRepr::ScalarPair(i32_scalar, i32_scalar),
            fields: FieldsShape::Arbitrary { offsets, memory_index },
            variants: Variants::Single { index: VariantIdx::from_u32(0) },
            largest_niche: None,
            uninhabited: false,
            max_repr_align: None,
            unadjusted_abi_align: Align::from_bytes(4).unwrap(),
            randomization_seed: Hash64::new(0),
        };
        assert_layout_roundtrip(&layout);
    }

    #[test]
    fn roundtrip_enum_with_direct_tag() {
        let tag = Scalar::Initialized {
            value: Primitive::Int(Integer::I8, false),
            valid_range: WrappingRange { start: 0, end: 1 },
        };
        let make_variant = |idx: u32| Layout {
            size: Size::from_bytes(8),
            align: rustc_abi::AbiAlign::new(Align::from_bytes(4).unwrap()),
            backend_repr: BackendRepr::Memory { sized: true },
            fields: FieldsShape::Arbitrary {
                offsets: vec![Size::from_bytes(0), Size::from_bytes(4)].into_iter().collect(),
                memory_index: vec![0, 1].into_iter().collect(),
            },
            variants: Variants::Single { index: VariantIdx::from_u32(idx) },
            largest_niche: None,
            uninhabited: false,
            max_repr_align: None,
            unadjusted_abi_align: Align::from_bytes(4).unwrap(),
            randomization_seed: Hash64::new(0),
        };

        let layout = Layout {
            size: Size::from_bytes(12),
            align: rustc_abi::AbiAlign::new(Align::from_bytes(4).unwrap()),
            backend_repr: BackendRepr::Memory { sized: true },
            fields: FieldsShape::Arbitrary {
                offsets: vec![Size::from_bytes(0)].into_iter().collect(),
                memory_index: vec![0].into_iter().collect(),
            },
            variants: Variants::Multiple {
                tag,
                tag_encoding: TagEncoding::Direct,
                tag_field: FieldIdx::from_u32(0),
                variants: vec![make_variant(0), make_variant(1)].into_iter().collect(),
            },
            largest_niche: Some(Niche {
                offset: Size::from_bytes(0),
                value: Primitive::Int(Integer::I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            }),
            uninhabited: false,
            max_repr_align: None,
            unadjusted_abi_align: Align::from_bytes(4).unwrap(),
            randomization_seed: Hash64::new(0),
        };
        assert_layout_roundtrip(&layout);
    }

    #[test]
    fn roundtrip_bool_with_niche() {
        let layout = Layout {
            size: Size::from_bytes(1),
            align: rustc_abi::AbiAlign::new(Align::from_bytes(1).unwrap()),
            backend_repr: BackendRepr::Scalar(Scalar::Initialized {
                value: Primitive::Int(Integer::I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            }),
            fields: FieldsShape::Primitive,
            variants: Variants::Single { index: VariantIdx::from_u32(0) },
            largest_niche: Some(Niche {
                offset: Size::from_bytes(0),
                value: Primitive::Int(Integer::I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            }),
            uninhabited: false,
            max_repr_align: None,
            unadjusted_abi_align: Align::from_bytes(1).unwrap(),
            randomization_seed: Hash64::new(0),
        };
        assert_layout_roundtrip(&layout);
    }
}
