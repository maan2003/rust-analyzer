//! Return lowering helpers.

use cranelift_codegen::ir::InstBuilder;
use cranelift_module::Module;
use rac_abi::callconv::PassMode;

use crate::abi::{ArgAbi, pass_mode};
use crate::{CPlace, FunctionCx};

pub(crate) fn codegen_return(
    fx: &mut FunctionCx<'_, impl Module>,
    ret_place: &CPlace,
    ret_abi: &ArgAbi,
) {
    match ret_abi.mode {
        PassMode::Ignore | PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ } => {
            fx.bcx.ins().return_(&[]);
        }
        PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized return ABI is unsupported")
        }
        PassMode::Direct(_) => {
            let ret_val = ret_place.to_cvalue(fx).load_scalar(fx);
            fx.bcx.ins().return_(&[ret_val]);
        }
        PassMode::Pair(_, _) => {
            let (a, b) = ret_place.to_cvalue(fx).load_scalar_pair(fx);
            fx.bcx.ins().return_(&[a, b]);
        }
        PassMode::Cast { ref cast, .. } => {
            let ret_val = ret_place.to_cvalue(fx);
            let casted = pass_mode::to_casted_value(fx, ret_val, cast);
            fx.bcx.ins().return_(&casted);
        }
    }
}
