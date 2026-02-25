//! Pointer abstraction for Cranelift IR generation.
//!
//! Adapted from cg_clif's pointer.rs.

use cranelift_codegen::ir::immediates::Offset32;
use cranelift_codegen::ir::{InstBuilder, MemFlags, StackSlot, Type, Value};
use cranelift_frontend::FunctionBuilder;
use rustc_abi::Align;

/// A pointer pointing either to a certain address, a certain stack slot or nothing.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Pointer {
    base: PointerBase,
    offset: Offset32,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum PointerBase {
    Addr(Value),
    Stack(StackSlot),
    Dangling(Align),
}

impl Pointer {
    pub(crate) fn new(addr: Value) -> Self {
        Pointer { base: PointerBase::Addr(addr), offset: Offset32::new(0) }
    }

    pub(crate) fn stack_slot(stack_slot: StackSlot) -> Self {
        Pointer { base: PointerBase::Stack(stack_slot), offset: Offset32::new(0) }
    }

    pub(crate) fn dangling(align: Align) -> Self {
        Pointer { base: PointerBase::Dangling(align), offset: Offset32::new(0) }
    }

    pub(crate) fn get_addr(self, bcx: &mut FunctionBuilder<'_>, pointer_type: Type) -> Value {
        match self.base {
            PointerBase::Addr(base_addr) => {
                let offset: i64 = self.offset.into();
                if offset == 0 { base_addr } else { bcx.ins().iadd_imm(base_addr, offset) }
            }
            PointerBase::Stack(stack_slot) => {
                bcx.ins().stack_addr(pointer_type, stack_slot, self.offset)
            }
            PointerBase::Dangling(align) => {
                bcx.ins().iconst(pointer_type, i64::try_from(align.bytes()).unwrap())
            }
        }
    }

    pub(crate) fn offset_i64(
        self,
        bcx: &mut FunctionBuilder<'_>,
        pointer_type: Type,
        extra_offset: i64,
    ) -> Self {
        if let Some(new_offset) = self.offset.try_add_i64(extra_offset) {
            Pointer { base: self.base, offset: new_offset }
        } else {
            let base_offset: i64 = self.offset.into();
            let new_offset = base_offset.checked_add(extra_offset).expect("offset overflow");
            let base_addr = match self.base {
                PointerBase::Addr(addr) => addr,
                PointerBase::Stack(stack_slot) => bcx.ins().stack_addr(pointer_type, stack_slot, 0),
                PointerBase::Dangling(align) => {
                    bcx.ins().iconst(pointer_type, i64::try_from(align.bytes()).unwrap())
                }
            };
            let addr = bcx.ins().iadd_imm(base_addr, new_offset);
            Pointer { base: PointerBase::Addr(addr), offset: Offset32::new(0) }
        }
    }

    pub(crate) fn load(self, bcx: &mut FunctionBuilder<'_>, ty: Type, flags: MemFlags) -> Value {
        match self.base {
            PointerBase::Addr(base_addr) => bcx.ins().load(ty, flags, base_addr, self.offset),
            PointerBase::Stack(stack_slot) => bcx.ins().stack_load(ty, stack_slot, self.offset),
            PointerBase::Dangling(_) => unreachable!("load from dangling pointer"),
        }
    }

    pub(crate) fn offset_value(
        self,
        bcx: &mut FunctionBuilder<'_>,
        pointer_type: Type,
        extra_offset: Value,
    ) -> Self {
        match self.base {
            PointerBase::Addr(addr) => Pointer {
                base: PointerBase::Addr(bcx.ins().iadd(addr, extra_offset)),
                offset: self.offset,
            },
            PointerBase::Stack(stack_slot) => {
                let base_addr = bcx.ins().stack_addr(pointer_type, stack_slot, self.offset);
                Pointer {
                    base: PointerBase::Addr(bcx.ins().iadd(base_addr, extra_offset)),
                    offset: Offset32::new(0),
                }
            }
            PointerBase::Dangling(align) => {
                let addr = bcx.ins().iconst(pointer_type, i64::try_from(align.bytes()).unwrap());
                Pointer {
                    base: PointerBase::Addr(bcx.ins().iadd(addr, extra_offset)),
                    offset: self.offset,
                }
            }
        }
    }

    pub(crate) fn store(self, bcx: &mut FunctionBuilder<'_>, value: Value, flags: MemFlags) {
        match self.base {
            PointerBase::Addr(base_addr) => {
                bcx.ins().store(flags, value, base_addr, self.offset);
            }
            PointerBase::Stack(stack_slot) => {
                bcx.ins().stack_store(value, stack_slot, self.offset);
            }
            PointerBase::Dangling(_) => unreachable!("store to dangling pointer"),
        }
    }
}
