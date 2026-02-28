//! PassMode-specific lowering helpers.

use cranelift_codegen::ir::{
    AbiParam, ArgumentPurpose, MemFlags, StackSlotData, StackSlotKind, Value, types,
};
use cranelift_codegen::ir::InstBuilder;
use cranelift_module::Module;
use rac_abi::callconv::{ArgAttributes, ArgExtension, CastTarget, PassMode, Reg, RegKind};
use rustc_abi::{BackendRepr, Size, TargetDataLayout};

use crate::abi::ArgAbi;
use crate::value_and_place::CValue;
use crate::{CPlace, FunctionCx, LayoutArc, scalar_to_clif_type};

fn reg_to_abi_param(reg: Reg) -> AbiParam {
    let clif_ty = match (reg.kind, reg.size.bytes()) {
        (RegKind::Integer, 1) => types::I8,
        (RegKind::Integer, 2) => types::I16,
        (RegKind::Integer, 3..=4) => types::I32,
        (RegKind::Integer, 5..=8) => types::I64,
        (RegKind::Integer, 9..=16) => types::I128,
        (RegKind::Float, 2) => types::F16,
        (RegKind::Float, 4) => types::F32,
        (RegKind::Float, 8) => types::F64,
        (RegKind::Float, 16) => types::F128,
        (RegKind::Vector, size) => types::I8.by(u32::try_from(size).unwrap()).unwrap(),
        _ => unreachable!("unsupported register layout in CastTarget: {reg:?}"),
    };
    AbiParam::new(clif_ty)
}

fn apply_attrs_to_abi_param(param: AbiParam, attrs: ArgAttributes) -> AbiParam {
    match attrs.arg_ext {
        ArgExtension::None => param,
        ArgExtension::Zext => param.uext(),
        ArgExtension::Sext => param.sext(),
    }
}

fn cast_target_to_abi_params(cast: &CastTarget) -> Vec<(Size, AbiParam)> {
    if let Some(offset_from_start) = cast.rest_offset {
        assert!(cast.prefix[1..].iter().all(|p| p.is_none()));
        assert_eq!(cast.rest.unit.size, cast.rest.total);
        let first = cast.prefix[0].unwrap();
        let second = cast.rest.unit;
        return vec![
            (Size::ZERO, reg_to_abi_param(first)),
            (offset_from_start, reg_to_abi_param(second)),
        ];
    }

    let (rest_count, rem_bytes) = if cast.rest.unit.size.bytes() == 0 {
        (0, 0)
    } else {
        (
            cast.rest.total.bytes() / cast.rest.unit.size.bytes(),
            cast.rest.total.bytes() % cast.rest.unit.size.bytes(),
        )
    };

    let args = cast
        .prefix
        .iter()
        .flatten()
        .copied()
        .map(reg_to_abi_param)
        .chain((0..rest_count).map(|_| reg_to_abi_param(cast.rest.unit)));

    let mut result = Vec::new();
    let mut offset = Size::ZERO;
    for arg in args {
        result.push((offset, arg));
        offset += Size::from_bytes(arg.value_type.bytes());
    }

    if rem_bytes != 0 {
        assert_eq!(cast.rest.unit.kind, RegKind::Integer);
        result.push((
            offset,
            reg_to_abi_param(Reg { kind: RegKind::Integer, size: Size::from_bytes(rem_bytes) }),
        ));
    }

    result
}

pub(crate) fn abi_params_for_arg(dl: &TargetDataLayout, arg_abi: &ArgAbi) -> Vec<AbiParam> {
    match arg_abi.mode {
        PassMode::Ignore => Vec::new(),
        PassMode::Direct(attrs) => {
            let layout = arg_abi.layout.as_ref().expect("Direct arg ABI without layout");
            let BackendRepr::Scalar(scalar) = layout.backend_repr else {
                panic!("Direct pass mode requires Scalar layout, got {:?}", layout.backend_repr);
            };
            vec![apply_attrs_to_abi_param(AbiParam::new(scalar_to_clif_type(dl, &scalar)), attrs)]
        }
        PassMode::Pair(attrs_a, attrs_b) => {
            let layout = arg_abi.layout.as_ref().expect("Pair arg ABI without layout");
            let BackendRepr::ScalarPair(a, b) = layout.backend_repr else {
                panic!("Pair pass mode requires ScalarPair layout, got {:?}", layout.backend_repr);
            };
            vec![
                apply_attrs_to_abi_param(AbiParam::new(scalar_to_clif_type(dl, &a)), attrs_a),
                apply_attrs_to_abi_param(AbiParam::new(scalar_to_clif_type(dl, &b)), attrs_b),
            ]
        }
        PassMode::Cast { ref cast, pad_i32 } => {
            let mut params = Vec::new();
            if pad_i32 {
                params.push(AbiParam::new(types::I32));
            }
            params.extend(cast_target_to_abi_params(cast).into_iter().map(|(_, param)| param));
            params
        }
        PassMode::Indirect { attrs, meta_attrs: None, on_stack } => {
            if on_stack {
                let layout = arg_abi.layout.as_ref().expect("Indirect on_stack ABI without layout");
                let size = layout.size.align_to(dl.pointer_align().abi);
                vec![apply_attrs_to_abi_param(
                    AbiParam::special(
                        crate::pointer_ty(dl),
                        ArgumentPurpose::StructArgument(size.bytes().try_into().unwrap()),
                    ),
                    attrs,
                )]
            } else {
                vec![apply_attrs_to_abi_param(AbiParam::new(crate::pointer_ty(dl)), attrs)]
            }
        }
        PassMode::Indirect { attrs, meta_attrs: Some(meta_attrs), on_stack } => {
            assert!(!on_stack, "unsized on_stack ABI is unsupported");
            vec![
                apply_attrs_to_abi_param(AbiParam::new(crate::pointer_ty(dl)), attrs),
                apply_attrs_to_abi_param(AbiParam::new(crate::pointer_ty(dl)), meta_attrs),
            ]
        }
    }
}

pub(crate) fn abi_return_for_arg(
    dl: &TargetDataLayout,
    ret_abi: &ArgAbi,
) -> (Option<AbiParam>, Vec<AbiParam>) {
    match ret_abi.mode {
        PassMode::Ignore => (None, Vec::new()),
        PassMode::Direct(attrs) => {
            let layout = ret_abi.layout.as_ref().expect("Direct return ABI without layout");
            let BackendRepr::Scalar(scalar) = layout.backend_repr else {
                panic!(
                    "Direct return pass mode requires Scalar layout, got {:?}",
                    layout.backend_repr
                );
            };
            (
                None,
                vec![apply_attrs_to_abi_param(
                    AbiParam::new(scalar_to_clif_type(dl, &scalar)),
                    attrs,
                )],
            )
        }
        PassMode::Pair(attrs_a, attrs_b) => {
            let layout = ret_abi.layout.as_ref().expect("Pair return ABI without layout");
            let BackendRepr::ScalarPair(a, b) = layout.backend_repr else {
                panic!(
                    "Pair return pass mode requires ScalarPair layout, got {:?}",
                    layout.backend_repr
                );
            };
            (
                None,
                vec![
                    apply_attrs_to_abi_param(AbiParam::new(scalar_to_clif_type(dl, &a)), attrs_a),
                    apply_attrs_to_abi_param(AbiParam::new(scalar_to_clif_type(dl, &b)), attrs_b),
                ],
            )
        }
        PassMode::Cast { ref cast, pad_i32: _ } => {
            (None, cast_target_to_abi_params(cast).into_iter().map(|(_, param)| param).collect())
        }
        PassMode::Indirect { attrs, meta_attrs: None, on_stack } => {
            assert!(!on_stack, "sret on_stack ABI is unsupported");
            (
                Some(apply_attrs_to_abi_param(
                    AbiParam::special(crate::pointer_ty(dl), ArgumentPurpose::StructReturn),
                    attrs,
                )),
                Vec::new(),
            )
        }
        PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized return ABI is unsupported")
        }
    }
}

pub(crate) fn to_casted_value(
    fx: &mut FunctionCx<'_, impl Module>,
    arg: CValue,
    cast: &CastTarget,
) -> Vec<Value> {
    let ptr = arg.force_stack(fx);
    cast_target_to_abi_params(cast)
        .into_iter()
        .map(|(offset, param)| {
            ptr.offset_i64(&mut fx.bcx, fx.pointer_type, offset.bytes() as i64).load(
                &mut fx.bcx,
                param.value_type,
                MemFlags::new(),
            )
        })
        .collect()
}

pub(crate) fn from_casted_value(
    fx: &mut FunctionCx<'_, impl Module>,
    values: &[Value],
    layout: LayoutArc,
    cast: &CastTarget,
) -> CValue {
    let abi_params = cast_target_to_abi_params(cast);
    let abi_param_size: u32 = abi_params.iter().map(|(_, param)| param.value_type.bytes()).sum();
    let layout_size = u32::try_from(layout.size.bytes()).expect("cast layout size does not fit in u32");
    let slot_size = abi_param_size.max(layout_size);
    let place = if slot_size == 0 {
        CPlace::for_ptr(crate::pointer::Pointer::dangling(layout.align.abi), layout.clone())
    } else {
        let align_bytes = layout.align.abi.bytes();
        assert!(align_bytes.is_power_of_two(), "alignment must be a power of 2");
        let slot = fx.bcx.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            slot_size,
            align_bytes.trailing_zeros() as u8,
        ));
        CPlace::for_ptr(crate::pointer::Pointer::stack_slot(slot), layout.clone())
    };
    let ptr = place.to_ptr();

    let mut values_iter = values.iter().copied();
    for (offset, _) in abi_params {
        let value = values_iter.next().expect("missing casted value component");
        ptr.offset_i64(&mut fx.bcx, fx.pointer_type, offset.bytes() as i64).store(
            &mut fx.bcx,
            value,
            MemFlags::new(),
        );
    }
    assert_eq!(values_iter.next(), None, "leftover casted values");

    CValue::by_ref(ptr, layout)
}

pub(crate) fn adjust_arg_for_abi(
    fx: &mut FunctionCx<'_, impl Module>,
    arg: CValue,
    arg_abi: &ArgAbi,
    is_owned: bool,
) -> Vec<Value> {
    match arg_abi.mode {
        PassMode::Ignore => Vec::new(),
        PassMode::Direct(_) => vec![arg.load_scalar(fx)],
        PassMode::Pair(_, _) => {
            let (a, b) = arg.load_scalar_pair(fx);
            vec![a, b]
        }
        PassMode::Cast { ref cast, pad_i32 } => {
            let mut values = to_casted_value(fx, arg, cast);
            if pad_i32 {
                values.insert(0, fx.bcx.ins().iconst(types::I32, 0));
            }
            values
        }
        PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ } => {
            if is_owned {
                vec![arg.force_stack(fx).get_addr(&mut fx.bcx, fx.pointer_type)]
            } else {
                let layout = arg_abi.layout.as_ref().expect("Indirect ABI without layout").clone();
                let tmp = CPlace::new_stack_slot(fx, layout);
                tmp.write_cvalue(fx, arg);
                vec![tmp.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type)]
            }
        }
        PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized by-value arguments are not supported")
        }
    }
}
