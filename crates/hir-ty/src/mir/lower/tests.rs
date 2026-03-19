use hir_def::{DefWithBodyId, HasModule, db::DefDatabase};
use span::Edition;
use test_fixture::WithFixture;

use crate::{InferenceResult, db::HirDatabase, setup_tracing, test_db::TestDB};

fn lower_mir(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let _tracing = setup_tracing();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    crate::attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let module_id = db.module_for_file(file_id.file_id(&db));
        let def_map = module_id.def_map(&db);
        let scope = &def_map[module_id].scope;
        let funcs = scope.declarations().filter_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(it) => Some(it),
            _ => None,
        });
        for func in funcs {
            _ = db.mir_body(func.into());
        }
    })
}

fn lower_mir_including_closures(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let _tracing = setup_tracing();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    crate::attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let module_id = db.module_for_file(file_id.file_id(&db));
        let def_map = module_id.def_map(&db);
        let scope = &def_map[module_id].scope;
        let funcs = scope.declarations().filter_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(it) => Some(it),
            _ => None,
        });
        for func in funcs {
            _ = db.mir_body(func.into());
            let infer = InferenceResult::for_body(&db, func.into());
            for closure in infer.closure_info.keys() {
                _ = db.mir_body_for_closure(*closure);
            }
        }
    })
}

fn lowered_mir(#[rust_analyzer::rust_fixture] ra_fixture: &str) -> String {
    let _tracing = setup_tracing();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    crate::attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let module_id = db.module_for_file(file_id.file_id(&db));
        let def_map = module_id.def_map(&db);
        let scope = &def_map[module_id].scope;
        let func = scope
            .declarations()
            .find_map(|x| match x {
                hir_def::ModuleDefId::FunctionId(it)
                    if db
                        .function_signature(it)
                        .name
                        .display(&db, Edition::CURRENT)
                        .to_string()
                        == "main" =>
                {
                    Some(it)
                }
                _ => None,
            })
            .unwrap();
        let body = db.mir_body(func.into()).unwrap();
        body.pretty_print(&db, crate::display::DisplayTarget::from_crate(&db, func.krate(&db)))
    })
}

#[test]
fn dyn_projection_with_auto_traits_regression_next_solver() {
    lower_mir(
        r#"
//- minicore: sized, send
pub trait Deserializer {}

pub trait Strictest {
    type Object: ?Sized;
}

impl Strictest for dyn CustomValue {
    type Object = dyn CustomValue + Send;
}

pub trait CustomValue: Send {}

impl CustomValue for () {}

struct Box<T: ?Sized>;

type DeserializeFn<T> = fn(&mut dyn Deserializer) -> Box<T>;

fn foo() {
    (|deserializer| Box::new(())) as DeserializeFn<<dyn CustomValue as Strictest>::Object>;
}
    "#,
    );
}

#[test]
fn try_in_generic_closure_monomorphizes() {
    lower_mir(
        r#"
//- minicore: try, result, from, fn
struct BuildError;
struct Trie;
impl Trie {
    fn iter<E, F: FnMut() -> Result<(), E>>(&self, mut f: F) -> Result<(), E> {
        f()
    }
}

struct Utf8Compiler;
impl Utf8Compiler {
    fn add(&mut self) -> Result<(), BuildError> {
        Result::Ok(())
    }
}

fn foo(trie: &Trie, utf8c: &mut Utf8Compiler) -> Result<(), BuildError> {
    trie.iter(|| {
        utf8c.add()?;
        Ok(())
    })?;
    Ok(())
}
    "#,
    );
}

#[test]
fn try_in_generic_closure_with_slice_arg_monomorphizes() {
    lower_mir_including_closures(
        r#"
//- minicore: try, result, from, slice, fn
struct BuildError;
struct Utf8Range;
struct Trie;
impl Trie {
    fn iter<E, F: FnMut(&[Utf8Range]) -> Result<(), E>>(&self, mut f: F) -> Result<(), E> {
        let ranges: &[Utf8Range] = loop {};
        f(ranges)
    }
}

struct Utf8Compiler;
impl Utf8Compiler {
    fn add(&mut self, ranges: &[Utf8Range]) -> Result<(), BuildError> {
        Result::Ok(())
    }
}

fn foo(trie: &Trie, utf8c: &mut Utf8Compiler) -> Result<(), BuildError> {
    trie.iter(|seq| {
        utf8c.add(&seq)?;
        Ok(())
    })?;
    Ok(())
}
        "#,
    );
}

fn check_borrowck(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let _tracing = setup_tracing();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    crate::attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let module_id = db.module_for_file(file_id.file_id(&db));
        let def_map = module_id.def_map(&db);
        let scope = &def_map[module_id].scope;

        let mut bodies: Vec<DefWithBodyId> = Vec::new();

        for decl in scope.declarations() {
            if let hir_def::ModuleDefId::FunctionId(f) = decl {
                bodies.push(f.into());
            }
        }

        for impl_id in scope.impls() {
            let impl_items = impl_id.impl_items(&db);
            for (_, item) in impl_items.items.iter() {
                if let hir_def::AssocItemId::FunctionId(f) = item {
                    bodies.push((*f).into());
                }
            }
        }

        for body in bodies {
            let _ = db.borrowck(body);
        }
    })
}

#[test]
fn regression_21173_const_generic_impl_with_assoc_type() {
    check_borrowck(
        r#"
pub trait Tr {
    type Assoc;
    fn f(&self, handle: Self::Assoc) -> i32;
}

pub struct ConstGeneric<const N: usize>;

impl<const N: usize> Tr for &ConstGeneric<N> {
    type Assoc = AssocTy;

    fn f(&self, a: Self::Assoc) -> i32 {
        a.x
    }
}

pub struct AssocTy {
    x: i32,
}
    "#,
    );
}

#[test]
fn super_let_extends_top_level_temporary_lifetime() {
    let mir = lowered_mir(
        r#"
fn id(_: &i32) -> i32 {
    0
}

fn main() -> i32 {
    super let x = {
        let y = 1;
        y
    };
    id(&x)
}
        "#,
    );
    let call = mir.find("Call {").unwrap();
    let drop_x = mir.find("StorageDead(x_1)").unwrap();
    let drop_temp = mir.find("StorageDead(_3)").unwrap();
    let drop_inner = mir.find("StorageDead(y_2)").unwrap();
    let bind_x = mir.find("StorageLive(x_1)").unwrap();

    assert!(call < drop_x, "{mir}");
    assert!(drop_x < drop_temp, "{mir}");
    assert!(drop_inner < bind_x, "{mir}");
}
