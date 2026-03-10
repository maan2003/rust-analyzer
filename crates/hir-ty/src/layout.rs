//! Compute the binary representation of a type

use std::fmt;

use hir_def::{
    AdtId, ItemContainerId, LocalFieldId, Lookup, StructId,
    attrs::AttrFlags,
    layout::{LayoutCalculatorError, LayoutData},
};
use rustc_abi::{
    AddressSpace, BackendRepr, Float, Integer, LayoutCalculator, Niche, Primitive, ReprOptions,
    Scalar, StructKind, TargetDataLayout, WrappingRange,
};
use rustc_index::IndexVec;
use rustc_type_ir::{
    ConstKind,
    FloatTy, IntTy, UintTy,
    inherent::{GenericArgs as _, IntoKind},
};
use triomphe::Arc;

use crate::{
    InferenceResult, ParamEnvAndCrate,
    consteval::try_const_usize,
    db::HirDatabase,
    mir::pad16,
    next_solver::{
        Const, DbInterner, GenericArgs, Pattern, StoredTy, Ty, TyKind, TypingMode,
        fulfill::FulfillmentCtxt,
        infer::{DbInternerInferExt, traits::ObligationCause},
    },
    traits::StoredParamEnvAndCrate,
};

pub(crate) use self::adt::layout_of_adt_cycle_result;
pub use self::{adt::layout_of_adt_query, target::target_data_layout_query};

pub(crate) mod adt;
pub(crate) mod target;

pub use rac_abi::{FieldIdx as RustcFieldIdx, VariantIdx as RustcEnumVariantIdx};

pub type Layout = LayoutData<RustcFieldIdx, RustcEnumVariantIdx>;
pub type TagEncoding = hir_def::layout::TagEncoding<RustcEnumVariantIdx>;
pub type Variants = hir_def::layout::Variants<RustcFieldIdx, RustcEnumVariantIdx>;

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum LayoutError {
    // FIXME: Remove more variants once they get added to LayoutCalculatorError
    BadCalc(LayoutCalculatorError<()>),
    HasErrorConst,
    HasErrorType,
    HasPlaceholder,
    InvalidSimdType,
    NotImplemented,
    RecursiveTypeWithoutIndirection,
    TargetLayoutNotAvailable,
    Unknown,
    UserReprTooSmall,
}

impl std::error::Error for LayoutError {}
impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayoutError::BadCalc(err) => err.fallback_fmt(f),
            LayoutError::HasErrorConst => write!(f, "type contains an unevaluatable const"),
            LayoutError::HasErrorType => write!(f, "type contains an error"),
            LayoutError::HasPlaceholder => write!(f, "type contains placeholders"),
            LayoutError::InvalidSimdType => write!(f, "invalid simd type definition"),
            LayoutError::NotImplemented => write!(f, "not implemented"),
            LayoutError::RecursiveTypeWithoutIndirection => {
                write!(f, "recursive type without indirection")
            }
            LayoutError::TargetLayoutNotAvailable => write!(f, "target layout not available"),
            LayoutError::Unknown => write!(f, "unknown"),
            LayoutError::UserReprTooSmall => {
                write!(f, "the `#[repr]` hint is too small to hold the discriminants of the enum")
            }
        }
    }
}

impl<F> From<LayoutCalculatorError<F>> for LayoutError {
    fn from(err: LayoutCalculatorError<F>) -> Self {
        LayoutError::BadCalc(err.without_payload())
    }
}

struct LayoutCx<'a> {
    calc: LayoutCalculator<&'a TargetDataLayout>,
}

impl<'a> LayoutCx<'a> {
    fn new(target: &'a TargetDataLayout) -> Self {
        Self { calc: LayoutCalculator::new(target) }
    }
}

// FIXME: move this to the `rustc_abi`.
fn layout_of_simd_ty<'db>(
    db: &'db dyn HirDatabase,
    id: StructId,
    repr_packed: bool,
    args: &GenericArgs<'db>,
    env: ParamEnvAndCrate<'db>,
    dl: &TargetDataLayout,
) -> Result<Arc<Layout>, LayoutError> {
    // Supported SIMD vectors are homogeneous ADTs with exactly one array field:
    //
    // * #[repr(simd)] struct S([T; 4])
    //
    // where T is a primitive scalar (integer/float/pointer).
    let fields = db.field_types(id.into());
    let mut fields = fields.iter();
    let Some(TyKind::Array(e_ty, e_len)) = fields
        .next()
        .filter(|_| fields.next().is_none())
        .map(|f| (*f.1).get().instantiate(DbInterner::new_no_crate(db), args).kind())
    else {
        return Err(LayoutError::InvalidSimdType);
    };

    let e_len = try_const_usize(db, e_len).ok_or(LayoutError::HasErrorConst)? as u64;
    let e_ly = db.layout_of_ty(e_ty.store(), env.store())?;

    let cx = LayoutCx::new(dl);
    Ok(Arc::new(cx.calc.simd_type(e_ly, e_len, repr_packed)?))
}

pub fn layout_of_ty_query(
    db: &dyn HirDatabase,
    ty: StoredTy,
    trait_env: StoredParamEnvAndCrate,
) -> Result<Arc<Layout>, LayoutError> {
    let krate = trait_env.krate;
    let interner = DbInterner::new_with(db, krate);
    let Ok(target) = db.target_data_layout(krate) else {
        return Err(LayoutError::TargetLayoutNotAvailable);
    };
    let dl = &*target;
    let cx = LayoutCx::new(dl);
    if let Some(layout) = layout_of_not_all_ones_ty(db, dl, ty.as_ref()) {
        return layout;
    }
    let infer_ctxt = interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let cause = ObligationCause::dummy();
    let normalized_env = trait_env.clone();
    let original_ty = ty.as_ref();
    let at = infer_ctxt.at(&cause, normalized_env.param_env());
    let ty = match at.deeply_normalize(original_ty) {
        Ok(normalized)
            if !(matches!(normalized.kind(), TyKind::Error(_))
                && matches!(original_ty.kind(), TyKind::Alias(..))) =>
        {
            normalized
        }
        _ if matches!(original_ty.kind(), TyKind::Alias(..)) => {
            let mut fulfill_cx = FulfillmentCtxt::new(&infer_ctxt);
            at.structurally_normalize_ty(original_ty, &mut fulfill_cx).unwrap_or(original_ty)
        }
        _ => original_ty,
    };
    let result = match ty.kind() {
        TyKind::Adt(def, args) => {
            match def.inner().id {
                hir_def::AdtId::StructId(s) => {
                    let repr = AttrFlags::repr(db, s.into()).unwrap_or_default();
                    if repr.simd() {
                        return layout_of_simd_ty(
                            db,
                            s,
                            repr.packed(),
                            &args,
                            trait_env.as_ref(),
                            &target,
                        );
                    }
                }
                _ => {}
            }
            return db.layout_of_adt(def.inner().id, args.store(), trait_env);
        }
        TyKind::Bool => Layout::scalar(
            dl,
            Scalar::Initialized {
                value: Primitive::Int(Integer::I8, false),
                valid_range: WrappingRange { start: 0, end: 1 },
            },
        ),
        TyKind::Char => Layout::scalar(
            dl,
            Scalar::Initialized {
                value: Primitive::Int(Integer::I32, false),
                valid_range: WrappingRange { start: 0, end: 0x10FFFF },
            },
        ),
        TyKind::Int(i) => Layout::scalar(
            dl,
            scalar_unit(
                dl,
                Primitive::Int(
                    match i {
                        IntTy::Isize => dl.ptr_sized_integer(),
                        IntTy::I8 => Integer::I8,
                        IntTy::I16 => Integer::I16,
                        IntTy::I32 => Integer::I32,
                        IntTy::I64 => Integer::I64,
                        IntTy::I128 => Integer::I128,
                    },
                    true,
                ),
            ),
        ),
        TyKind::Uint(i) => Layout::scalar(
            dl,
            scalar_unit(
                dl,
                Primitive::Int(
                    match i {
                        UintTy::Usize => dl.ptr_sized_integer(),
                        UintTy::U8 => Integer::I8,
                        UintTy::U16 => Integer::I16,
                        UintTy::U32 => Integer::I32,
                        UintTy::U64 => Integer::I64,
                        UintTy::U128 => Integer::I128,
                    },
                    false,
                ),
            ),
        ),
        TyKind::Float(f) => Layout::scalar(
            dl,
            scalar_unit(
                dl,
                Primitive::Float(match f {
                    FloatTy::F16 => Float::F16,
                    FloatTy::F32 => Float::F32,
                    FloatTy::F64 => Float::F64,
                    FloatTy::F128 => Float::F128,
                }),
            ),
        ),
        TyKind::Tuple(tys) => {
            let kind =
                if tys.is_empty() { StructKind::AlwaysSized } else { StructKind::MaybeUnsized };

            let fields = tys
                .iter()
                .map(|k| db.layout_of_ty(k.store(), trait_env.clone()))
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<IndexVec<_, _>>();
            cx.calc.univariant(&fields, &ReprOptions::default(), kind)?
        }
        TyKind::Array(element, count) => {
            let count = try_const_usize(db, count).ok_or(LayoutError::HasErrorConst)? as u64;
            let element = db.layout_of_ty(element.store(), trait_env)?;
            cx.calc.array_like::<_, _, ()>(&element, Some(count))?
        }
        TyKind::Slice(element) => {
            let element = db.layout_of_ty(element.store(), trait_env)?;
            cx.calc.array_like::<_, _, ()>(&element, None)?
        }
        TyKind::Str => {
            let element = scalar_unit(dl, Primitive::Int(Integer::I8, false));
            cx.calc.array_like::<_, _, ()>(&Layout::scalar(dl, element), None)?
        }
        // Potentially-wide pointers.
        TyKind::Ref(_, pointee, _) | TyKind::RawPtr(pointee, _) => {
            let mut data_ptr = scalar_unit(dl, Primitive::Pointer(AddressSpace::ZERO));
            if matches!(ty.kind(), TyKind::Ref(..)) {
                data_ptr.valid_range_mut().start = 1;
            }

            // FIXME(next-solver)
            // let pointee = tcx.normalize_erasing_regions(param_env, pointee);
            // if pointee.is_sized(tcx.at(DUMMY_SP), param_env) {
            //     return Ok(tcx.mk_layout(LayoutS::scalar(cx, data_ptr)));
            // }

            let unsized_part = struct_tail_erasing_lifetimes(db, pointee);
            // FIXME(next-solver)
            /*
            if let TyKind::AssociatedType(id, subst) = unsized_part.kind(Interner) {
                unsized_part = TyKind::Alias(chalk_ir::AliasTy::Projection(ProjectionTy {
                    associated_ty_id: *id,
                    substitution: subst.clone(),
                }))
                .intern(Interner);
            }
            unsized_part = normalize(db, trait_env, unsized_part);
            */
            let metadata = match unsized_part.kind() {
                TyKind::Slice(_) | TyKind::Str => {
                    scalar_unit(dl, Primitive::Int(dl.ptr_sized_integer(), false))
                }
                TyKind::Dynamic(..) => {
                    let mut vtable = scalar_unit(dl, Primitive::Pointer(AddressSpace::ZERO));
                    vtable.valid_range_mut().start = 1;
                    vtable
                }
                _ => {
                    // pointee is sized
                    return Ok(Arc::new(Layout::scalar(dl, data_ptr)));
                }
            };

            // Effectively a (ptr, meta) tuple.
            LayoutData::scalar_pair(dl, data_ptr, metadata)
        }
        TyKind::Never => LayoutData::never_type(dl),
        TyKind::FnDef(..) => LayoutData::unit(dl, true),
        TyKind::Dynamic(..) | TyKind::Foreign(_) => LayoutData::unit(dl, false),
        TyKind::FnPtr(..) => {
            let mut ptr = scalar_unit(dl, Primitive::Pointer(dl.instruction_address_space));
            ptr.valid_range_mut().start = 1;
            Layout::scalar(dl, ptr)
        }
        TyKind::Closure(id, args) => {
            let def = db.lookup_intern_closure(id.0);
            let infer = InferenceResult::for_body(db, def.0);
            let (captures, _) = infer.closure_info(id.0);
            let fields = captures
                .iter()
                .map(|it| {
                    let ty = it.ty.get().instantiate(interner, args.as_closure().parent_args());
                    db.layout_of_ty(ty.store(), trait_env.clone())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let fields = fields.iter().map(|it| &**it).collect::<Vec<_>>();
            let fields = fields.iter().collect::<IndexVec<_, _>>();
            cx.calc.univariant(&fields, &ReprOptions::default(), StructKind::AlwaysSized)?
        }

        TyKind::Coroutine(_, _)
        | TyKind::CoroutineWitness(_, _)
        | TyKind::CoroutineClosure(_, _) => {
            return Err(LayoutError::NotImplemented);
        }

        TyKind::Pat(ty, pat) => return layout_of_pattern_ty(db, dl, trait_env, ty, pat),
        TyKind::UnsafeBinder(_) => return Err(LayoutError::NotImplemented),

        TyKind::Error(_) => return Err(LayoutError::HasErrorType),
        TyKind::Placeholder(_)
        | TyKind::Bound(..)
        | TyKind::Infer(..)
        | TyKind::Param(..)
        | TyKind::Alias(..) => {
            return Err(LayoutError::HasPlaceholder);
        }
    };
    Ok(Arc::new(result))
}

fn layout_of_not_all_ones_ty<'db>(
    db: &'db dyn HirDatabase,
    dl: &TargetDataLayout,
    ty: Ty<'db>,
) -> Option<Result<Arc<Layout>, LayoutError>> {
    let (int, signed, bits) = match ty.kind() {
        TyKind::Alias(rustc_type_ir::AliasTyKind::Projection, alias) => {
            let crate::next_solver::SolverDefId::TypeAliasId(type_alias_id) = alias.def_id else {
                return None;
            };
            let ItemContainerId::TraitId(trait_id) = type_alias_id.lookup(db).container else {
                return None;
            };
            if db.type_alias_signature(type_alias_id).name.as_str() != "Type"
                || db.trait_signature(trait_id).name.as_str() != "NotAllOnesHelper"
            {
                return None;
            }
            let [arg] = alias.args.as_slice() else {
                return Some(Err(LayoutError::HasErrorType));
            };
            let Some(arg_ty) = arg.ty() else {
                return Some(Err(LayoutError::HasErrorType));
            };
            not_all_ones_scalar_info(arg_ty)?
        }
        TyKind::Adt(def, _) => match def.inner().id {
            AdtId::StructId(id) => {
                // Cross-crate attrs are currently collected from syntax only, so std/core
                // wrappers like `U32NotAllOnes` lose their valid-range metadata here.
                match db.struct_signature(id).name.as_str() {
                    "I32NotAllOnes" => (Integer::I32, true, 32),
                    "U32NotAllOnes" => (Integer::I32, false, 32),
                    "I64NotAllOnes" => (Integer::I64, true, 64),
                    "U64NotAllOnes" => (Integer::I64, false, 64),
                    _ => return None,
                }
            }
            _ => return None,
        },
        _ => return None,
    };
    let scalar = Scalar::Initialized {
        value: Primitive::Int(int, signed),
        valid_range: WrappingRange { start: 0, end: (1u128 << bits) - 2 },
    };
    Some(Ok(Arc::new(Layout::scalar(dl, scalar))))
}

fn not_all_ones_scalar_info<'db>(arg_ty: Ty<'db>) -> Option<(Integer, bool, u32)> {
    Some(match arg_ty.kind() {
        TyKind::Int(IntTy::I32) => (Integer::I32, true, 32),
        TyKind::Int(IntTy::I64) => (Integer::I64, true, 64),
        TyKind::Uint(UintTy::U32) => (Integer::I32, false, 32),
        TyKind::Uint(UintTy::U64) => (Integer::I64, false, 64),
        _ => return None,
    })
}

pub(crate) fn layout_of_ty_cycle_result(
    _: &dyn HirDatabase,
    _: salsa::Id,
    _: StoredTy,
    _: StoredParamEnvAndCrate,
) -> Result<Arc<Layout>, LayoutError> {
    Err(LayoutError::RecursiveTypeWithoutIndirection)
}

fn struct_tail_erasing_lifetimes<'a>(db: &'a dyn HirDatabase, pointee: Ty<'a>) -> Ty<'a> {
    match pointee.kind() {
        TyKind::Adt(def, args) => {
            let struct_id = match def.inner().id {
                AdtId::StructId(id) => id,
                _ => return pointee,
            };
            let data = struct_id.fields(db);
            let mut it = data.fields().iter().rev();
            match it.next() {
                Some((f, _)) => {
                    let last_field_ty = field_ty(db, struct_id.into(), f, args);
                    struct_tail_erasing_lifetimes(db, last_field_ty)
                }
                None => pointee,
            }
        }
        TyKind::Tuple(tys) => {
            if let Some(last_field_ty) = tys.iter().next_back() {
                struct_tail_erasing_lifetimes(db, last_field_ty)
            } else {
                pointee
            }
        }
        _ => pointee,
    }
}

fn layout_of_pattern_ty<'db>(
    db: &'db dyn HirDatabase,
    dl: &TargetDataLayout,
    trait_env: StoredParamEnvAndCrate,
    ty: Ty<'db>,
    pat: Pattern<'db>,
) -> Result<Arc<Layout>, LayoutError> {
    let mut layout = (*db.layout_of_ty(ty.store(), trait_env)?).clone();
    match pat.kind() {
        rustc_type_ir::PatternKind::NotNull => {
            let BackendRepr::Scalar(mut scalar) = layout.backend_repr else {
                return Err(LayoutError::NotImplemented);
            };
            let Scalar::Initialized { value: Primitive::Pointer(_), valid_range } = &mut scalar else {
                return Err(LayoutError::NotImplemented);
            };
            valid_range.start = 1;
            layout.backend_repr = BackendRepr::Scalar(scalar);
            layout.largest_niche = Niche::from_scalar(dl, rustc_abi::Size::ZERO, scalar);
        }
        rustc_type_ir::PatternKind::Range { start, end } => {
            let BackendRepr::Scalar(Scalar::Initialized { value, .. }) = layout.backend_repr else {
                return Err(LayoutError::NotImplemented);
            };
            let start = const_to_bits(db, start)?;
            let end = const_to_bits(db, end)?;
            let scalar = Scalar::Initialized {
                value,
                valid_range: WrappingRange { start, end },
            };
            layout.backend_repr = BackendRepr::Scalar(scalar);
            layout.largest_niche = Niche::from_scalar(dl, rustc_abi::Size::ZERO, scalar);
        }
        rustc_type_ir::PatternKind::Or(_) => return Err(LayoutError::NotImplemented),
    }
    Ok(Arc::new(layout))
}

fn const_to_bits<'db>(db: &'db dyn HirDatabase, c: Const<'db>) -> Result<u128, LayoutError> {
    match c.kind() {
        ConstKind::Unevaluated(unevaluated_const) => match unevaluated_const.def.0 {
            hir_def::GeneralConstId::ConstId(id) => {
                const_to_bits(db, db.const_eval(id, unevaluated_const.args, None).map_err(|_| LayoutError::HasErrorConst)?)
            }
            hir_def::GeneralConstId::StaticId(id) => {
                const_to_bits(db, db.const_eval_static(id).map_err(|_| LayoutError::HasErrorConst)?)
            }
        },
        ConstKind::Value(val) => Ok(u128::from_le_bytes(pad16(&val.value.inner().memory, false))),
        ConstKind::Param(_)
        | ConstKind::Infer(_)
        | ConstKind::Bound(_, _)
        | ConstKind::Placeholder(_)
        | ConstKind::Error(_)
        | ConstKind::Expr(_) => Err(LayoutError::HasErrorConst),
    }
}

fn field_ty<'a>(
    db: &'a dyn HirDatabase,
    def: hir_def::VariantId,
    fd: LocalFieldId,
    args: GenericArgs<'a>,
) -> Ty<'a> {
    db.field_types(def)[fd].get().instantiate(DbInterner::new_no_crate(db), args)
}

fn scalar_unit(dl: &TargetDataLayout, value: Primitive) -> Scalar {
    Scalar::Initialized { value, valid_range: WrappingRange::full(value.size(dl)) }
}

#[cfg(test)]
mod tests;
