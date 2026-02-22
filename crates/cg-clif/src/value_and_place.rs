//! CValue / CPlace abstractions for codegen.
//!
//! Adapted from cg_clif's `value_and_place.rs`. CValue is a read-only value
//! with its layout; CPlace is a mutable location with its layout.

use cranelift_codegen::ir::{InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};
use cranelift_frontend::Variable;
use cranelift_module::Module;
use rustc_abi::{BackendRepr, Scalar, Size, TargetDataLayout};
use triomphe::Arc;

/// Compute the align_shift (log2 of alignment) for a stack slot.
fn align_to_shift(align_bytes: u64) -> u8 {
    assert!(align_bytes.is_power_of_two(), "alignment must be a power of 2");
    align_bytes.trailing_zeros() as u8
}

use crate::pointer::Pointer;
use crate::{FunctionCx, scalar_to_clif_type};
use hir_ty::layout::Layout;

fn scalar_pair_b_offset(dl: &TargetDataLayout, a: Scalar, b: Scalar) -> i64 {
    let b_offset = a.size(dl).align_to(b.align(dl).abi);
    i64::try_from(b_offset.bytes()).unwrap()
}

// ---------------------------------------------------------------------------
// CValue: read-only value with layout
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct CValue {
    inner: CValueInner,
    pub(crate) layout: Arc<Layout>,
}

#[derive(Clone, Copy)]
enum CValueInner {
    ByRef(Pointer),
    ByVal(Value),
    ByValPair(Value, Value),
}

impl CValue {
    pub(crate) fn by_ref(ptr: Pointer, layout: Arc<Layout>) -> Self {
        CValue { inner: CValueInner::ByRef(ptr), layout }
    }

    pub(crate) fn by_val(value: Value, layout: Arc<Layout>) -> Self {
        CValue { inner: CValueInner::ByVal(value), layout }
    }

    pub(crate) fn by_val_pair(a: Value, b: Value, layout: Arc<Layout>) -> Self {
        CValue { inner: CValueInner::ByValPair(a, b), layout }
    }

    pub(crate) fn zst(layout: Arc<Layout>) -> Self {
        assert!(layout.is_zst());
        CValue::by_ref(Pointer::dangling(layout.align.abi), layout)
    }

    /// Load a single scalar value. Panics if the layout is not `Scalar`.
    pub(crate) fn load_scalar(&self, fx: &mut FunctionCx<'_, impl Module>) -> Value {
        match self.inner {
            CValueInner::ByVal(val) => val,
            CValueInner::ByRef(ptr) => {
                let BackendRepr::Scalar(scalar) = self.layout.backend_repr else {
                    panic!("load_scalar on non-Scalar layout: {:?}", self.layout.backend_repr);
                };
                let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
                let mut flags = MemFlags::new();
                flags.set_notrap();
                ptr.load(&mut fx.bcx, clif_ty, flags)
            }
            CValueInner::ByValPair(_, _) => {
                panic!("load_scalar on ByValPair; use load_scalar_pair instead")
            }
        }
    }

    /// Load a scalar pair. Panics if the layout is not `ScalarPair`.
    pub(crate) fn load_scalar_pair(
        &self,
        fx: &mut FunctionCx<'_, impl Module>,
    ) -> (Value, Value) {
        match self.inner {
            CValueInner::ByValPair(a, b) => (a, b),
            CValueInner::ByRef(ptr) => {
                let BackendRepr::ScalarPair(a_scalar, b_scalar) = self.layout.backend_repr else {
                    panic!(
                        "load_scalar_pair on non-ScalarPair layout: {:?}",
                        self.layout.backend_repr
                    );
                };
                let a_clif = scalar_to_clif_type(fx.dl, &a_scalar);
                let b_clif = scalar_to_clif_type(fx.dl, &b_scalar);
                let b_off = scalar_pair_b_offset(fx.dl, a_scalar, b_scalar);
                let mut flags = MemFlags::new();
                flags.set_notrap();
                let a_val = ptr.load(&mut fx.bcx, a_clif, flags);
                let b_val =
                    ptr.offset_i64(&mut fx.bcx, fx.pointer_type, b_off).load(&mut fx.bcx, b_clif, flags);
                (a_val, b_val)
            }
            CValueInner::ByVal(_) => {
                panic!("load_scalar_pair on ByVal; use load_scalar instead")
            }
        }
    }

    /// Force the value into memory and return the pointer.
    pub(crate) fn force_stack(self, fx: &mut FunctionCx<'_, impl Module>) -> Pointer {
        match self.inner {
            CValueInner::ByRef(ptr) => ptr,
            CValueInner::ByVal(_) | CValueInner::ByValPair(_, _) => {
                let place = CPlace::new_stack_slot(fx, self.layout.clone());
                place.write_cvalue(fx, self);
                place.to_ptr()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPlace: mutable location with layout
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct CPlace {
    inner: CPlaceInner,
    pub(crate) layout: Arc<Layout>,
}

#[derive(Clone, Copy)]
pub(crate) enum CPlaceInner {
    Var(Variable),
    VarPair(Variable, Variable),
    Addr(Pointer),
}

impl CPlace {
    pub(crate) fn new_stack_slot(
        fx: &mut FunctionCx<'_, impl Module>,
        layout: Arc<Layout>,
    ) -> Self {
        if layout.size.bytes() == 0 {
            return CPlace {
                inner: CPlaceInner::Addr(Pointer::dangling(layout.align.abi)),
                layout,
            };
        }
        let size = u32::try_from(layout.size.bytes()).expect("stack slot too large");
        let slot = fx.bcx.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size,
            align_to_shift(layout.align.abi.bytes()),
        ));
        CPlace { inner: CPlaceInner::Addr(Pointer::stack_slot(slot)), layout }
    }

    pub(crate) fn new_var(
        fx: &mut FunctionCx<'_, impl Module>,
        layout: Arc<Layout>,
    ) -> Self {
        let BackendRepr::Scalar(scalar) = layout.backend_repr else {
            panic!("new_var requires Scalar layout, got {:?}", layout.backend_repr);
        };
        let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
        let var = fx.bcx.declare_var(clif_ty);
        CPlace { inner: CPlaceInner::Var(var), layout }
    }

    pub(crate) fn new_var_pair(
        fx: &mut FunctionCx<'_, impl Module>,
        layout: Arc<Layout>,
    ) -> Self {
        let BackendRepr::ScalarPair(a, b) = layout.backend_repr else {
            panic!("new_var_pair requires ScalarPair layout, got {:?}", layout.backend_repr);
        };
        let a_clif = scalar_to_clif_type(fx.dl, &a);
        let b_clif = scalar_to_clif_type(fx.dl, &b);
        let var1 = fx.bcx.declare_var(a_clif);
        let var2 = fx.bcx.declare_var(b_clif);
        CPlace { inner: CPlaceInner::VarPair(var1, var2), layout }
    }

    /// Create a Var CPlace from an already-declared variable.
    /// Used during local setup before FunctionCx is built.
    pub(crate) fn new_var_raw(var: Variable, layout: Arc<Layout>) -> Self {
        CPlace { inner: CPlaceInner::Var(var), layout }
    }

    /// Create a VarPair CPlace from already-declared variables.
    /// Used during local setup before FunctionCx is built.
    pub(crate) fn new_var_pair_raw(
        var1: Variable,
        var2: Variable,
        layout: Arc<Layout>,
    ) -> Self {
        CPlace { inner: CPlaceInner::VarPair(var1, var2), layout }
    }

    pub(crate) fn for_ptr(ptr: Pointer, layout: Arc<Layout>) -> Self {
        CPlace { inner: CPlaceInner::Addr(ptr), layout }
    }

    /// Define the i-th SSA variable in this place. Used during parameter
    /// wiring before FunctionCx is available.
    ///   - For `Var`: i must be 0
    ///   - For `VarPair`: i must be 0 or 1
    pub(crate) fn def_var(
        &self,
        i: usize,
        val: Value,
        bcx: &mut cranelift_frontend::FunctionBuilder<'_>,
    ) {
        match self.inner {
            CPlaceInner::Var(var) => {
                assert_eq!(i, 0);
                bcx.def_var(var, val);
            }
            CPlaceInner::VarPair(var1, var2) => match i {
                0 => bcx.def_var(var1, val),
                1 => bcx.def_var(var2, val),
                _ => panic!("def_var index {i} out of range for VarPair"),
            },
            CPlaceInner::Addr(_) => panic!("def_var on Addr CPlace"),
        }
    }

    /// Returns true if this place is stored in registers (Var or VarPair),
    /// not in memory (Addr).
    pub(crate) fn is_register(&self) -> bool {
        matches!(self.inner, CPlaceInner::Var(_) | CPlaceInner::VarPair(_, _))
    }

    pub(crate) fn to_ptr(&self) -> Pointer {
        match self.inner {
            CPlaceInner::Addr(ptr) => ptr,
            _ => panic!("to_ptr on non-Addr CPlace"),
        }
    }

    /// Read the place as a CValue.
    pub(crate) fn to_cvalue(&self, fx: &mut FunctionCx<'_, impl Module>) -> CValue {
        match self.inner {
            CPlaceInner::Var(var) => {
                let val = fx.bcx.use_var(var);
                CValue::by_val(val, self.layout.clone())
            }
            CPlaceInner::VarPair(var1, var2) => {
                let val1 = fx.bcx.use_var(var1);
                let val2 = fx.bcx.use_var(var2);
                CValue::by_val_pair(val1, val2, self.layout.clone())
            }
            CPlaceInner::Addr(ptr) => {
                if self.layout.is_zst() {
                    CValue::zst(self.layout.clone())
                } else {
                    CValue::by_ref(ptr, self.layout.clone())
                }
            }
        }
    }

    /// Write a CValue into this place.
    pub(crate) fn write_cvalue(
        &self,
        fx: &mut FunctionCx<'_, impl Module>,
        from: CValue,
    ) {
        if self.layout.is_zst() {
            return;
        }

        match self.inner {
            CPlaceInner::Var(var) => {
                let val = from.load_scalar(fx);
                let dst_ty = match self.layout.backend_repr {
                    BackendRepr::Scalar(s) => scalar_to_clif_type(fx.dl, &s),
                    _ => panic!("Var place with non-Scalar layout"),
                };
                let src_ty = fx.bcx.func.dfg.value_type(val);
                let val = if src_ty == dst_ty {
                    val
                } else {
                    assert_eq!(src_ty.bytes(), dst_ty.bytes(), "bitcast size mismatch");
                    fx.bcx.ins().bitcast(dst_ty, MemFlags::new(), val)
                };
                fx.bcx.def_var(var, val);
            }
            CPlaceInner::VarPair(var1, var2) => {
                let (val1, val2) = from.load_scalar_pair(fx);
                fx.bcx.def_var(var1, val1);
                fx.bcx.def_var(var2, val2);
            }
            CPlaceInner::Addr(ptr) => {
                if self.layout.size == Size::ZERO {
                    return;
                }
                let mut flags = MemFlags::new();
                flags.set_notrap();
                match from.inner {
                    CValueInner::ByVal(val) => {
                        ptr.store(&mut fx.bcx, val, flags);
                    }
                    CValueInner::ByValPair(val1, val2) => {
                        let BackendRepr::ScalarPair(a_scalar, b_scalar) =
                            self.layout.backend_repr
                        else {
                            panic!("writing ByValPair to non-ScalarPair memory");
                        };
                        let b_off = scalar_pair_b_offset(fx.dl, a_scalar, b_scalar);
                        ptr.store(&mut fx.bcx, val1, flags);
                        ptr.offset_i64(&mut fx.bcx, fx.pointer_type, b_off)
                            .store(&mut fx.bcx, val2, flags);
                    }
                    CValueInner::ByRef(from_ptr) => {
                        // Delegate to scalar/pair load if possible, else memcpy
                        match self.layout.backend_repr {
                            BackendRepr::Scalar(scalar) => {
                                let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
                                let val = from_ptr.load(&mut fx.bcx, clif_ty, flags);
                                ptr.store(&mut fx.bcx, val, flags);
                            }
                            BackendRepr::ScalarPair(a_scalar, b_scalar) => {
                                let a_clif = scalar_to_clif_type(fx.dl, &a_scalar);
                                let b_clif = scalar_to_clif_type(fx.dl, &b_scalar);
                                let b_off = scalar_pair_b_offset(fx.dl, a_scalar, b_scalar);
                                let val1 = from_ptr.load(&mut fx.bcx, a_clif, flags);
                                let val2 = from_ptr
                                    .offset_i64(&mut fx.bcx, fx.pointer_type, b_off)
                                    .load(&mut fx.bcx, b_clif, flags);
                                ptr.store(&mut fx.bcx, val1, flags);
                                ptr.offset_i64(&mut fx.bcx, fx.pointer_type, b_off)
                                    .store(&mut fx.bcx, val2, flags);
                            }
                            _ => {
                                // Memory-to-memory copy
                                let from_addr =
                                    from_ptr.get_addr(&mut fx.bcx, fx.pointer_type);
                                let to_addr = ptr.get_addr(&mut fx.bcx, fx.pointer_type);
                                let size = self.layout.size.bytes();
                                let dst_align: u8 =
                                    self.layout.align.abi.bytes().try_into().unwrap_or(128);
                                let src_align: u8 =
                                    from.layout.align.abi.bytes().try_into().unwrap_or(128);
                                fx.bcx.emit_small_memory_copy(
                                    fx.isa.frontend_config(),
                                    to_addr,
                                    from_addr,
                                    size,
                                    dst_align,
                                    src_align,
                                    true,
                                    flags,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Project to a field of this place.
    ///
    /// `field_idx` is the field index in the layout's `FieldsShape`.
    /// `field_layout` is the layout of the field being projected to.
    pub(crate) fn place_field(
        &self,
        fx: &mut FunctionCx<'_, impl Module>,
        field_idx: usize,
        field_layout: Arc<Layout>,
    ) -> CPlace {
        match self.inner {
            CPlaceInner::VarPair(var1, var2) => {
                // ScalarPair: field 0 = first scalar, field 1 = second scalar
                match field_idx {
                    0 => CPlace { inner: CPlaceInner::Var(var1), layout: field_layout },
                    1 => CPlace { inner: CPlaceInner::Var(var2), layout: field_layout },
                    _ => panic!("field index {field_idx} out of range for VarPair"),
                }
            }
            CPlaceInner::Var(var) => {
                // Scalar wrapper struct: field 0 is the scalar itself
                assert_eq!(field_idx, 0, "field index {field_idx} out of range for single Var");
                CPlace { inner: CPlaceInner::Var(var), layout: field_layout }
            }
            CPlaceInner::Addr(ptr) => {
                let offset = self.layout.fields.offset(field_idx);
                let field_ptr = ptr.offset_i64(
                    &mut fx.bcx,
                    fx.pointer_type,
                    i64::try_from(offset.bytes()).unwrap(),
                );
                CPlace::for_ptr(field_ptr, field_layout)
            }
        }
    }

    /// Change the layout to represent a specific enum variant without moving
    /// the pointer. Used for `ProjectionElem::Downcast`.
    pub(crate) fn downcast_variant(
        &self,
        variant_layout: Arc<Layout>,
    ) -> CPlace {
        CPlace { inner: self.inner, layout: variant_layout }
    }
}
