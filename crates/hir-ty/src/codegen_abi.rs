//! ABI computation for codegen.
//!
//! This module provides the glue between hir-ty's type/layout system and rac-abi's
//! real ABI computation (ported from rustc_target). The core types (`TyAndLayout`,
//! `TyAbiInterface`, `PassMode`, `ArgAbi`, `FnAbi`) come from rac-abi.

use std::fmt;

use rac_abi::{
    Interned, Layout, TyAbiInterface, TyAndLayout, VariantIdx,
    callconv::{ArgAbi, ArgAttributes, ArgExtension, FnAbi},
    spec::{HasTargetSpec, HasX86AbiOpt, X86Abi},
};
use rustc_abi::{
    CanonAbi, HasDataLayout, PointeeInfo, Primitive, Scalar, Size, TargetDataLayout, Variants,
};
use rustc_type_ir::inherent::IntoKind;

use crate::{
    db::HirDatabase,
    layout::Layout as HirLayout,
    next_solver::{DbInterner, Ty, TyKind},
    traits::StoredParamEnvAndCrate,
};

pub use rac_abi::callconv::{CastTarget, PassMode, Uniform};

// ---------------------------------------------------------------------------
// Display impl for Ty — required by TyAbiInterface
// ---------------------------------------------------------------------------

/// Newtype wrapper around Ty that implements Display (delegates to Debug).
/// rac-abi's TyAbiInterface requires Display, but hir-ty's Ty only has Debug.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct AbiTy<'db>(pub Ty<'db>);

impl<'db> fmt::Debug for AbiTy<'db> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

impl fmt::Display for AbiTy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, f)
    }
}

// ---------------------------------------------------------------------------
// CodegenCx — the context type that provides layout lookups
// ---------------------------------------------------------------------------

pub struct CodegenCx<'a> {
    pub db: &'a dyn HirDatabase,
    pub env: StoredParamEnvAndCrate,
    pub target_data: &'a TargetDataLayout,
    pub target_spec: rac_abi::spec::Target,
    pub arena: &'a typed_arena::Arena<HirLayout>,
}

impl HasDataLayout for CodegenCx<'_> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.target_data
    }
}

impl HasTargetSpec for CodegenCx<'_> {
    fn target_spec(&self) -> &rac_abi::spec::Target {
        &self.target_spec
    }
}

impl HasX86AbiOpt for CodegenCx<'_> {
    fn x86_abi_opt(&self) -> X86Abi {
        X86Abi { regparm: None, reg_struct_return: false }
    }
}

impl<'a> CodegenCx<'a> {
    fn layout_of<'db>(&self, ty: Ty<'db>) -> Option<TyAndLayout<'a, AbiTy<'db>>> {
        let layout_arc = self.db.layout_of_ty(ty.store(), self.env.clone()).ok()?;
        let layout_ref: &'a HirLayout = self.arena.alloc((*layout_arc).clone());
        Some(TyAndLayout { ty: AbiTy(ty), layout: Layout(Interned(layout_ref)) })
    }
}

// ---------------------------------------------------------------------------
// TyAbiInterface implementation for AbiTy
// ---------------------------------------------------------------------------

impl<'a: 'db, 'db> TyAbiInterface<'a, CodegenCx<'a>> for AbiTy<'db> {
    fn ty_and_layout_for_variant(
        this: TyAndLayout<'a, Self>,
        cx: &CodegenCx<'a>,
        variant_index: VariantIdx,
    ) -> TyAndLayout<'a, Self> {
        let layout = match this.layout.0.0.variants {
            Variants::Single { index } if index == variant_index => this.layout,
            Variants::Single { .. } => this.layout,
            Variants::Multiple { ref variants, .. } => {
                let variant_layout = &variants[variant_index];
                Layout(Interned(cx.arena.alloc(variant_layout.clone())))
            }
            Variants::Empty => this.layout,
        };
        TyAndLayout { ty: this.ty, layout }
    }

    fn ty_and_layout_field(
        this: TyAndLayout<'a, Self>,
        cx: &CodegenCx<'a>,
        i: usize,
    ) -> TyAndLayout<'a, Self> {
        match this.ty.0.kind() {
            TyKind::Adt(def, args) => {
                let variant_id = match &this.layout.0.0.variants {
                    Variants::Single { index } => match def.inner().id {
                        hir_def::AdtId::StructId(s) => hir_def::VariantId::StructId(s),
                        hir_def::AdtId::UnionId(u) => hir_def::VariantId::UnionId(u),
                        hir_def::AdtId::EnumId(e) => {
                            let variants = e.enum_variants(cx.db);
                            hir_def::VariantId::EnumVariantId(variants.variants[index.as_usize()].0)
                        }
                    },
                    _ => panic!("ty_and_layout_field on multi-variant without for_variant"),
                };
                let field_types = cx.db.field_types(variant_id);
                let fields: Vec<_> = field_types.iter().collect();
                let (_field_id, field_ty) = fields[i];
                let field_ty = (*field_ty).get().instantiate(DbInterner::new_no_crate(cx.db), args);
                cx.layout_of(field_ty).expect("field layout")
            }
            TyKind::Tuple(tys) => {
                let field_ty = tys.iter().nth(i).expect("tuple field index out of bounds");
                cx.layout_of(field_ty).expect("tuple field layout")
            }
            TyKind::Ref(_, _pointee, _) | TyKind::RawPtr(_pointee, _) => {
                let interner = DbInterner::new_with(cx.db, cx.env.krate);
                match i {
                    0 => {
                        let unit = Ty::new(
                            interner,
                            TyKind::Tuple(crate::next_solver::Tys::empty(interner)),
                        );
                        let ptr_ty =
                            Ty::new(interner, TyKind::RawPtr(unit, rustc_ast_ir::Mutability::Not));
                        cx.layout_of(ptr_ty).expect("pointer layout")
                    }
                    1 => {
                        let usize_ty = Ty::new(interner, TyKind::Uint(rustc_ast_ir::UintTy::Usize));
                        cx.layout_of(usize_ty).expect("metadata layout")
                    }
                    _ => panic!("pointer only has 2 fields"),
                }
            }
            _ => panic!("ty_and_layout_field called on non-aggregate type: {:?}", this.ty),
        }
    }

    fn ty_and_layout_pointee_info_at(
        _this: TyAndLayout<'a, Self>,
        _cx: &CodegenCx<'a>,
        _offset: Size,
    ) -> Option<PointeeInfo> {
        None // stub for MVP
    }

    fn is_adt(this: TyAndLayout<'a, Self>) -> bool {
        matches!(this.ty.0.kind(), TyKind::Adt(..))
    }

    fn is_never(this: TyAndLayout<'a, Self>) -> bool {
        matches!(this.ty.0.kind(), TyKind::Never)
    }

    fn is_tuple(this: TyAndLayout<'a, Self>) -> bool {
        matches!(this.ty.0.kind(), TyKind::Tuple(..))
    }

    fn is_unit(this: TyAndLayout<'a, Self>) -> bool {
        matches!(this.ty.0.kind(), TyKind::Tuple(tys) if tys.is_empty())
    }

    fn is_transparent(this: TyAndLayout<'a, Self>) -> bool {
        match this.ty.0.kind() {
            TyKind::Adt(def, _) => {
                let adt_id = def.inner().id;
                // We need a db reference but don't have one in a static method.
                // AttrFlags::repr requires a db. Since we can't access CodegenCx here
                // (it's the cx param), we use cx through the trait signature.
                // However, this is a static method without cx access for the db.
                // We'll return false as a safe default — transparent structs are rare
                // and this only affects ABI optimization (not correctness for most cases).
                //
                // TODO: Find a way to check repr(transparent) without db access,
                // or restructure to pass it differently.
                let _ = adt_id;
                false
            }
            _ => false,
        }
    }

    fn is_scalable_vector(_this: TyAndLayout<'a, Self>) -> bool {
        false
    }

    fn is_pass_indirectly_in_non_rustic_abis_flag_set(_this: TyAndLayout<'a, Self>) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// scalar_arg_attrs helper
// ---------------------------------------------------------------------------

fn scalar_arg_attrs(dl: &TargetDataLayout, scalar: Scalar, _offset: Size) -> ArgAttributes {
    let mut attrs = ArgAttributes::new();
    match scalar.primitive() {
        Primitive::Int(i, signed) => {
            if i.size().bits() < dl.pointer_size().bits() {
                if signed {
                    attrs.ext(ArgExtension::Sext);
                } else {
                    attrs.ext(ArgExtension::Zext);
                }
            }
        }
        Primitive::Float(_) => {}
        Primitive::Pointer(_) => {}
    }
    attrs
}

// ---------------------------------------------------------------------------
// compute_fn_abi entry point
// ---------------------------------------------------------------------------

/// Compute the `FnAbi` for a function signature.
///
/// Takes the argument types and return type, computes layouts, and determines
/// how each should be passed using rac-abi's real ABI adjustment logic.
pub fn compute_fn_abi<'a: 'db, 'db>(
    cx: &CodegenCx<'a>,
    arg_tys: &[Ty<'db>],
    ret_ty: Ty<'db>,
) -> Option<FnAbi<'a, AbiTy<'db>>> {
    let dl = cx.target_data;

    let ret_layout = cx.layout_of(ret_ty)?;
    let ret = ArgAbi::new(cx, ret_layout, |scalar, offset| scalar_arg_attrs(dl, scalar, offset));

    let args: Vec<_> = arg_tys
        .iter()
        .map(|&ty| {
            let layout = cx.layout_of(ty)?;
            Some(ArgAbi::new(cx, layout, |scalar, offset| scalar_arg_attrs(dl, scalar, offset)))
        })
        .collect::<Option<_>>()?;

    let mut fn_abi = FnAbi {
        args: args.into_boxed_slice(),
        ret,
        c_variadic: false,
        fixed_count: arg_tys.len() as u32,
        conv: CanonAbi::Rust,
        can_unwind: false,
    };

    // Apply Rust ABI adjustments — the real version from rac-abi with
    // full arch-specific handling.
    fn_abi.adjust_for_rust_abi(cx);

    Some(fn_abi)
}
