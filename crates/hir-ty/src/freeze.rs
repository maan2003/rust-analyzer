//! Utilities for reasoning about `Freeze` on monomorphic types.

use hir_def::{AdtId, signatures::StructFlags};
use rustc_hash::FxHashSet;
use rustc_type_ir::inherent::{AdtDef, IntoKind};

use crate::{
    InferenceResult,
    consteval,
    next_solver::{DbInterner, Ty, TyKind},
};

/// Returns whether a fully monomorphized type implements `Freeze`.
///
/// This is conservative for unresolved forms (`Param`, `Alias`, `Dynamic`, etc.):
/// they return `false` rather than guessing. That keeps downstream callers safe
/// when this is used to decide whether immutable statics may reside in readonly
/// memory.
pub fn is_freeze_mono<'db>(interner: DbInterner<'db>, ty: Ty<'db>) -> bool {
    is_freeze_mono_impl(interner, ty, &mut FxHashSet::default())
}

fn is_freeze_mono_impl<'db>(
    interner: DbInterner<'db>,
    ty: Ty<'db>,
    visited: &mut FxHashSet<Ty<'db>>,
) -> bool {
    if !visited.insert(ty) {
        return true;
    }

    match ty.kind() {
        TyKind::Int(_)
        | TyKind::Uint(_)
        | TyKind::Float(_)
        | TyKind::Bool
        | TyKind::Char
        | TyKind::Str
        | TyKind::Never
        | TyKind::Ref(..)
        | TyKind::RawPtr(_, _)
        | TyKind::FnDef(..)
        | TyKind::FnPtr(..)
        | TyKind::Error(_) => true,
        TyKind::Tuple(tys) => tys.iter().all(|ty| is_freeze_mono_impl(interner, ty, visited)),
        TyKind::Pat(ty, _) | TyKind::Slice(ty) => is_freeze_mono_impl(interner, ty, visited),
        TyKind::Array(ty, len) => {
            consteval::try_const_usize(interner.db, len) == Some(0)
                || is_freeze_mono_impl(interner, ty, visited)
        }
        TyKind::Adt(adt_def, subst) => match adt_def.def_id().0 {
            AdtId::StructId(id) => {
                let flags = interner.db.struct_signature(id).flags;
                if flags.contains(StructFlags::IS_UNSAFE_CELL) {
                    return false;
                }
                if flags.contains(StructFlags::IS_PHANTOM_DATA) {
                    return true;
                }

                interner.db.field_types(id.into()).iter().all(|(_, field_ty)| {
                    is_freeze_mono_impl(interner, field_ty.get().instantiate(interner, subst), visited)
                })
            }
            AdtId::UnionId(id) => interner.db.field_types(id.into()).iter().all(|(_, field_ty)| {
                is_freeze_mono_impl(interner, field_ty.get().instantiate(interner, subst), visited)
            }),
            AdtId::EnumId(id) => id.enum_variants(interner.db).variants.iter().all(
                |&(variant, _, _)| {
                    interner.db.field_types(variant.into()).iter().all(|(_, field_ty)| {
                        is_freeze_mono_impl(
                            interner,
                            field_ty.get().instantiate(interner, subst),
                            visited,
                        )
                    })
                },
            ),
        },
        TyKind::Closure(closure_id, subst) => {
            let owner = interner.db.lookup_intern_closure(closure_id.0).0;
            let infer = InferenceResult::for_body(interner.db, owner);
            let (captures, _) = infer.closure_info(closure_id.0);
            captures
                .iter()
                .all(|capture| is_freeze_mono_impl(interner, capture.ty(interner.db, subst), visited))
        }
        TyKind::Param(_)
        | TyKind::Bound(..)
        | TyKind::Alias(..)
        | TyKind::Dynamic(..)
        | TyKind::Foreign(_)
        | TyKind::Placeholder(_)
        | TyKind::Coroutine(..)
        | TyKind::CoroutineWitness(..)
        | TyKind::CoroutineClosure(..)
        | TyKind::Infer(_)
        | TyKind::UnsafeBinder(_) => false,
    }
}
