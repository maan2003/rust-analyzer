use std::{fmt, panic, sync::Mutex};

use cranelift_module::Module;

use base_db::{
    CrateGraphBuilder, CratesMap, FileSourceRootInput, FileText, Nonce, RootQueryDb,
    SourceDatabase, SourceRoot, SourceRootId, SourceRootInput,
};
use hir_def::{HasModule, ModuleId, db::DefDatabase, nameres::crate_def_map};
use hir_expand::EditionedFileId;
use rustc_abi::TargetDataLayout;
use salsa::Durability;
use span::{Edition, FileId};
use test_fixture::WithFixture;
use triomphe::Arc;

use hir_ty::{
    ParamEnvAndCrate,
    attach_db,
    db::HirDatabase,
    next_solver::{DbInterner, GenericArgs},
};

unsafe extern "C" {
    fn fmodf(x: f32, y: f32) -> f32;
    fn fmod(x: f64, y: f64) -> f64;
}

use crate::symbol_mangling;

// ---------------------------------------------------------------------------
// TestDB (same pattern as hir-ty's test_db)
// ---------------------------------------------------------------------------

#[salsa_macros::db]
struct TestDB {
    storage: salsa::Storage<Self>,
    files: Arc<base_db::Files>,
    crates_map: Arc<CratesMap>,
    events: Arc<Mutex<Option<Vec<salsa::Event>>>>,
    nonce: Nonce,
}

impl Default for TestDB {
    fn default() -> Self {
        let events = <Arc<Mutex<Option<Vec<salsa::Event>>>>>::default();
        let mut this = Self {
            storage: salsa::Storage::new(Some(Box::new({
                let events = events.clone();
                move |event| {
                    let mut events = events.lock().unwrap();
                    if let Some(events) = &mut *events {
                        events.push(event);
                    }
                }
            }))),
            events,
            files: Default::default(),
            crates_map: Default::default(),
            nonce: Nonce::new(),
        };
        this.set_expand_proc_attr_macros_with_durability(true, Durability::HIGH);
        this.set_all_crates(Arc::new(Box::new([])));
        _ = base_db::LibraryRoots::builder(Default::default())
            .durability(Durability::MEDIUM)
            .new(&this);
        _ = base_db::LocalRoots::builder(Default::default())
            .durability(Durability::MEDIUM)
            .new(&this);
        CrateGraphBuilder::default().set_in_db(&mut this);
        this
    }
}

impl Clone for TestDB {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            files: self.files.clone(),
            crates_map: self.crates_map.clone(),
            events: self.events.clone(),
            nonce: Nonce::new(),
        }
    }
}

impl fmt::Debug for TestDB {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TestDB").finish()
    }
}

#[salsa_macros::db]
impl SourceDatabase for TestDB {
    fn file_text(&self, file_id: base_db::FileId) -> FileText {
        self.files.file_text(file_id)
    }

    fn set_file_text(&mut self, file_id: base_db::FileId, text: &str) {
        let files = Arc::clone(&self.files);
        files.set_file_text(self, file_id, text);
    }

    fn set_file_text_with_durability(
        &mut self,
        file_id: base_db::FileId,
        text: &str,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_file_text_with_durability(self, file_id, text, durability);
    }

    fn source_root(&self, source_root_id: SourceRootId) -> SourceRootInput {
        self.files.source_root(source_root_id)
    }

    fn set_source_root_with_durability(
        &mut self,
        source_root_id: SourceRootId,
        source_root: Arc<SourceRoot>,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_source_root_with_durability(self, source_root_id, source_root, durability);
    }

    fn file_source_root(&self, id: base_db::FileId) -> FileSourceRootInput {
        self.files.file_source_root(id)
    }

    fn set_file_source_root_with_durability(
        &mut self,
        id: base_db::FileId,
        source_root_id: SourceRootId,
        durability: Durability,
    ) {
        let files = Arc::clone(&self.files);
        files.set_file_source_root_with_durability(self, id, source_root_id, durability);
    }

    fn crates_map(&self) -> Arc<CratesMap> {
        self.crates_map.clone()
    }

    fn nonce_and_revision(&self) -> (Nonce, salsa::Revision) {
        (self.nonce, salsa::plumbing::ZalsaDatabase::zalsa(self).current_revision())
    }
}

#[salsa_macros::db]
impl salsa::Database for TestDB {}

impl panic::RefUnwindSafe for TestDB {}

fn module_for_file(db: &TestDB, file_id: impl Into<FileId>) -> ModuleId {
    let file_id = file_id.into();
    for &krate in db.relevant_crates(file_id).iter() {
        let crate_def_map = crate_def_map(db, krate);
        for (module_id, data) in crate_def_map.modules() {
            if data.origin.file_id().map(|file_id| file_id.file_id(db)) == Some(file_id) {
                return module_id;
            }
        }
    }
    panic!("module not found for file");
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn find_fn(db: &TestDB, file_id: EditionedFileId, name: &str) -> hir_def::FunctionId {
    let module_id = module_for_file(db, file_id.file_id(db));
    let def_map = module_id.def_map(db);
    let scope = &def_map[module_id].scope;
    scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(x) => {
                if db.function_signature(x).name.display(db, Edition::CURRENT).to_string() == name {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("function `{name}` not found"))
}

fn get_mir_and_env(
    db: &TestDB,
    func_id: hir_def::FunctionId,
) -> (
    triomphe::Arc<hir_ty::mir::MirBody>,
    hir_ty::traits::StoredParamEnvAndCrate,
) {
    let interner = DbInterner::new_no_crate(db);
    let env = ParamEnvAndCrate {
        param_env: db.trait_environment(func_id.into()),
        krate: func_id.krate(db),
    }
    .store();
    let body = db
        .monomorphized_mir_body(func_id.into(), GenericArgs::empty(interner).store(), env.clone())
        .expect("failed to lower MIR");
    (body, env)
}

fn get_target_data_layout(db: &TestDB, func_id: hir_def::FunctionId) -> triomphe::Arc<TargetDataLayout> {
    db.target_data_layout(func_id.krate(db)).expect("no target data layout")
}

/// Get the v0-mangled name for a function (non-generic).
fn mangled_name(db: &TestDB, func_id: hir_def::FunctionId) -> String {
    let interner = DbInterner::new_no_crate(db);
    let generic_args = GenericArgs::empty(interner);
    symbol_mangling::mangle_function(db, func_id, generic_args)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn compile_return_i32() {
    let (db, file_ids) = TestDB::with_many_files(
        r#"
//- /main.rs
fn foo() -> i32 {
    42
}
"#,
    );
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let func_id = find_fn(&db, file_id, "foo");
        let (body, env) = get_mir_and_env(&db, func_id);
        let dl = get_target_data_layout(&db, func_id);
        let fn_name = mangled_name(&db, func_id);

        let obj_bytes =
            crate::compile_to_object(&db, &dl, &env, &body, &fn_name).expect("compilation failed");

        // Verify we got a non-empty object file
        assert!(!obj_bytes.is_empty(), "object file should not be empty");

        // Parse the object file and verify it contains the symbol
        let obj = object::read::File::parse(&*obj_bytes).expect("failed to parse object file");
        use object::{Object, ObjectSymbol};
        let symbols: Vec<_> = obj
            .symbols()
            .filter(|s| s.name() == Ok(fn_name.as_str()))
            .collect();
        assert!(!symbols.is_empty(), "symbol '{fn_name}' not found in object file");
    });
}

/// Helper: compile a function named "foo" from source and assert it produces a valid object file.
fn compile_fn_to_object(src: &str) -> Vec<u8> {
    let full_src = fixture_src(src);
    let (db, file_ids) = TestDB::with_many_files(&full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let func_id = find_fn(&db, file_id, "foo");
        let (body, env) = get_mir_and_env(&db, func_id);
        let dl = get_target_data_layout(&db, func_id);
        let fn_name = mangled_name(&db, func_id);

        let obj_bytes =
            crate::compile_to_object(&db, &dl, &env, &body, &fn_name).expect("compilation failed");
        assert!(!obj_bytes.is_empty());

        let obj = object::read::File::parse(&*obj_bytes).expect("failed to parse object file");
        use object::{Object, ObjectSymbol};
        assert!(obj.symbols().any(|s| s.name() == Ok(fn_name.as_str())), "symbol '{fn_name}' not found");

        obj_bytes
    })
}

/// Helper: JIT-compile functions from source, execute a no-arg entry function,
/// and return its result. Follows the cg_clif JIT pattern:
/// JITBuilder → JITModule → compile_fn → finalize → get_finalized_function → transmute → call.
///
/// The `entry` function must take no arguments and return `R`. Tests that need
/// to pass arguments should write a wrapper function in the source:
/// ```ignore
/// fn add(a: i32, b: i32) -> i32 { a + b }
/// fn test() -> i32 { add(3, 4) }
/// // jit_run::<i32>(src, &["add", "test"], "test") → 7
/// ```
///
/// `src` may also be a full test fixture (e.g. with `//- minicore: ...` and
/// explicit file entries).
fn jit_run<R: Copy>(src: &str, fn_names: &[&str], entry: &str) -> R {
    let full_src = fixture_src(src);
    let (db, file_ids) = TestDB::with_many_files(&full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let isa = crate::build_host_isa(false);

        let mut jit_builder =
            cranelift_jit::JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());
        jit_builder.symbol("fmodf", fmodf as *const u8);
        jit_builder.symbol("fmod", fmod as *const u8);
        let mut jit_module = cranelift_jit::JITModule::new(jit_builder);

        for &name in fn_names {
            let func_id = find_fn(&db, file_id, name);
            let (body, env) = get_mir_and_env(&db, func_id);
            let dl = get_target_data_layout(&db, func_id);
            let fn_name = mangled_name(&db, func_id);

            crate::compile_fn(
                &mut jit_module,
                &*isa,
                &db,
                &dl,
                &env,
                &body,
                &fn_name,
                cranelift_module::Linkage::Export,
            )
            .unwrap_or_else(|e| panic!("compiling `{name}` failed: {e}"));
        }

        // Finalize: make all compiled code executable
        jit_module.finalize_definitions().unwrap();

        // Look up the entry function pointer
        let entry_func_id = find_fn(&db, file_id, entry);
        let (entry_body, entry_env) = get_mir_and_env(&db, entry_func_id);
        let dl = get_target_data_layout(&db, entry_func_id);
        let sig = crate::build_fn_sig(&*isa, &db, &dl, &entry_env, &entry_body).expect("sig");
        let entry_mangled = mangled_name(&db, entry_func_id);

        let entry_id = jit_module
            .declare_function(&entry_mangled, cranelift_module::Linkage::Import, &sig)
            .expect("declare entry");
        let code_ptr = jit_module.get_finalized_function(entry_id);

        // SAFETY: The entry function is `extern "C" fn() -> R`. The JITModule
        // stays alive until after we call f(), so the code is still mapped.
        unsafe {
            let f: extern "C" fn() -> R = std::mem::transmute(code_ptr);
            f()
        }
    })
}

/// Helper: compile multiple functions from source into one object module.
/// `fn_names` lists function names to compile, in order.
fn compile_fns_to_object(src: &str, fn_names: &[&str]) -> Vec<u8> {
    let full_src = fixture_src(src);
    let (db, file_ids) = TestDB::with_many_files(&full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let isa = crate::build_host_isa(true);

        let builder = cranelift_object::ObjectBuilder::new(
            isa.clone(),
            "rac_output",
            cranelift_module::default_libcall_names(),
        )
        .expect("ObjectBuilder");
        let mut module = cranelift_object::ObjectModule::new(builder);

        let mut mangled_names = Vec::new();
        for &name in fn_names {
            let func_id = find_fn(&db, file_id, name);
            let (body, env) = get_mir_and_env(&db, func_id);
            let dl = get_target_data_layout(&db, func_id);
            let fn_name = mangled_name(&db, func_id);

            crate::compile_fn(
                &mut module,
                &*isa,
                &db,
                &dl,
                &env,
                &body,
                &fn_name,
                cranelift_module::Linkage::Export,
            )
            .unwrap_or_else(|e| panic!("compiling `{name}` failed: {e}"));
            mangled_names.push(fn_name);
        }

        let product = module.finish();
        let obj_bytes = product.emit().expect("emit");
        assert!(!obj_bytes.is_empty());

        let obj = object::read::File::parse(&*obj_bytes).expect("failed to parse object file");
        use object::{Object, ObjectSymbol};
        for mangled in &mangled_names {
            assert!(obj.symbols().any(|s| s.name() == Ok(mangled.as_str())), "symbol '{mangled}' not found");
        }

        obj_bytes
    })
}

fn fixture_src(src: &str) -> String {
    let trimmed = src.trim_start();
    if trimmed.starts_with("//-") { src.to_owned() } else { format!("//- /main.rs\n{src}") }
}

#[test]
fn compile_int_arithmetic() {
    compile_fn_to_object(
        r#"
fn foo(a: i32, b: i32) -> i32 {
    a + b
}
"#,
    );
}

#[test]
fn compile_int_sub_mul_div() {
    compile_fn_to_object(
        r#"
fn foo(a: i32, b: i32) -> i32 {
    (a - b) * b / (b + 1)
}
"#,
    );
}

#[test]
fn compile_bitwise_ops() {
    compile_fn_to_object(
        r#"
fn foo(a: u32, b: u32) -> u32 {
    (a & b) | (a ^ b)
}
"#,
    );
}

#[test]
fn compile_shift_ops() {
    compile_fn_to_object(
        r#"
fn foo(a: u32, b: u32) -> u32 {
    (a << b) >> b
}
"#,
    );
}

#[test]
fn compile_negation() {
    compile_fn_to_object(
        r#"
fn foo(a: i32) -> i32 {
    -a
}
"#,
    );
}

#[test]
fn compile_float_arithmetic() {
    compile_fn_to_object(
        r#"
fn foo(a: f64, b: f64) -> f64 {
    a + b * a - b / a
}
"#,
    );
}

#[test]
fn compile_float_negation() {
    compile_fn_to_object(
        r#"
fn foo(a: f64) -> f64 {
    -a
}
"#,
    );
}

#[test]
fn compile_comparison() {
    compile_fn_to_object(
        r#"
fn foo(a: i32, b: i32) -> bool {
    a < b
}
"#,
    );
}

#[test]
fn compile_if_else() {
    compile_fn_to_object(
        r#"
fn foo(a: i32) -> i32 {
    if a > 0 { a } else { 0 }
}
"#,
    );
}

#[test]
fn compile_multiple_branches() {
    compile_fn_to_object(
        r#"
fn foo(a: i32) -> i32 {
    if a > 10 {
        1
    } else if a > 0 {
        2
    } else {
        3
    }
}
"#,
    );
}

#[test]
fn compile_direct_call() {
    compile_fns_to_object(
        r#"
fn bar(x: i32) -> i32 {
    x + 1
}
fn foo(a: i32) -> i32 {
    bar(a)
}
"#,
        &["bar", "foo"],
    );
}

#[test]
fn compile_call_chain() {
    compile_fns_to_object(
        r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
fn double(x: i32) -> i32 {
    add(x, x)
}
fn foo(n: i32) -> i32 {
    double(add(n, 1))
}
"#,
        &["add", "double", "foo"],
    );
}

#[test]
fn compile_call_with_branch() {
    compile_fns_to_object(
        r#"
fn abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
fn foo(a: i32) -> i32 {
    abs(a) + abs(-a)
}
"#,
        &["abs", "foo"],
    );
}

// ---------------------------------------------------------------------------
// JIT execution tests — actually run the compiled code and verify results
// ---------------------------------------------------------------------------

#[test]
fn jit_return_constant() {
    let result = jit_run::<i32>(
        r#"
fn foo() -> i32 {
    42
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_arithmetic() {
    let result = jit_run::<i32>(
        r#"
fn foo() -> i32 {
    3 + 4 * 2 - 1
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 3 + 4 * 2 - 1);
}

#[test]
fn jit_if_else() {
    // Test the "then" branch
    let result = jit_run::<i32>(
        r#"
fn pick(x: i32) -> i32 {
    if x > 0 { 100 } else { 200 }
}
fn foo() -> i32 {
    pick(5)
}
"#,
        &["pick", "foo"],
        "foo",
    );
    assert_eq!(result, 100);
}

#[test]
fn jit_if_else_negative() {
    // Test the "else" branch
    let result = jit_run::<i32>(
        r#"
fn pick(x: i32) -> i32 {
    if x > 0 { 100 } else { 200 }
}
fn foo() -> i32 {
    pick(-3)
}
"#,
        &["pick", "foo"],
        "foo",
    );
    assert_eq!(result, 200);
}

#[test]
fn jit_function_call() {
    let result = jit_run::<i32>(
        r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
fn foo() -> i32 {
    add(10, 32)
}
"#,
        &["add", "foo"],
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_call_chain() {
    let result = jit_run::<i32>(
        r#"
fn square(x: i32) -> i32 {
    x * x
}
fn sum_of_squares(a: i32, b: i32) -> i32 {
    square(a) + square(b)
}
fn foo() -> i32 {
    sum_of_squares(3, 4)
}
"#,
        &["square", "sum_of_squares", "foo"],
        "foo",
    );
    assert_eq!(result, 25); // 9 + 16
}

#[test]
fn jit_recursive_like_call() {
    // Not actual recursion (that needs the same function to call itself),
    // but a chain that exercises multiple calls and branches.
    let result = jit_run::<i32>(
        r#"
fn abs(x: i32) -> i32 {
    if x < 0 { -x } else { x }
}
fn max(a: i32, b: i32) -> i32 {
    if a > b { a } else { b }
}
fn foo() -> i32 {
    max(abs(-10), abs(7))
}
"#,
        &["abs", "max", "foo"],
        "foo",
    );
    assert_eq!(result, 10);
}

#[test]
fn jit_float_arithmetic() {
    let result = jit_run::<f64>(
        r#"
fn foo() -> f64 {
    1.5 + 2.5
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 4.0);
}

#[test]
fn jit_float_rem() {
    let result = jit_run::<f64>(
        r#"
fn foo() -> f64 {
    7.25 % 2.0
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 1.25);
}

#[test]
fn jit_casts_int_float() {
    let result = jit_run::<i32>(
        r#"
fn foo() -> i32 {
    let a = 250u8 as i32;
    let b = 3.75f32 as i32;
    let c = 2i16 as f32;
    a + b + c as i32
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 255);
}

#[test]
fn jit_pointer_int_cast_roundtrip() {
    let result = jit_run::<usize>(
        r#"
fn foo() -> usize {
    let p = 0x1000usize as *const i32;
    p as usize
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 0x1000);
}

#[test]
fn jit_accepts_full_fixture_input() {
    let result = jit_run::<i32>(
        r#"
//- minicore: sized
//- /main.rs
fn foo() -> i32 {
    7
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 7);
}

#[test]
fn jit_pointer_offset_scales_by_pointee_size() {
    let result = jit_run::<usize>(
        r#"
extern "rust-intrinsic" {
    pub fn offset<Ptr, Delta>(dst: Ptr, offset: Delta) -> Ptr;
}

fn foo() -> usize {
    let p = 0x1000usize as *const i32;
    let q = unsafe { offset(p, 2isize) };
    q as usize - p as usize
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 8);
}

#[test]
fn compile_pointer_offset_method_via_minicore() {
    compile_fn_to_object(
        r#"
//- minicore: ptr_offset
//- /main.rs
fn foo() -> usize {
    let p = 0x1000usize as *const i32;
    let q = unsafe { p.offset(2) };
    q as usize - p as usize
}
"#,
    );
}

#[test]
fn jit_pointer_arith_offset_signed() {
    let result = jit_run::<usize>(
        r#"
extern "rust-intrinsic" {
    pub fn arith_offset<T>(dst: *const T, offset: isize) -> *const T;
}

fn foo() -> usize {
    let p = 0x1000usize as *const i32;
    let q = unsafe { arith_offset(arith_offset(p, 3), -1) };
    q as usize - p as usize
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 8);
}

#[test]
fn jit_pointer_offset_from_intrinsics() {
    let result = jit_run::<isize>(
        r#"
extern "rust-intrinsic" {
    pub fn ptr_offset_from<T>(ptr: *const T, base: *const T) -> isize;
    pub fn ptr_offset_from_unsigned<T>(ptr: *const T, base: *const T) -> usize;
}

fn foo() -> isize {
    let base = 0x1000usize as *const i32;
    let ptr = 0x1018usize as *const i32;
    let r1 = unsafe { ptr_offset_from(ptr, base) };
    let r2 = unsafe { ptr_offset_from(base, ptr) };
    let r3 = unsafe { ptr_offset_from_unsigned(ptr, base) } as isize;
    r3 * 100 + r1 * 10 + r2
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 654);
}

#[test]
fn compile_pointer_offset_from_methods_via_minicore() {
    compile_fn_to_object(
        r#"
//- minicore: ptr_offset
//- /main.rs
fn foo() -> isize {
    let base = 0x1000usize as *const i32;
    let ptr = 0x1018usize as *const i32;
    let r1 = unsafe { ptr.offset_from(base) };
    let r2 = unsafe { ptr.offset_from_unsigned(base) } as isize;
    r2 * 10 + r1
}
"#,
    );
}

#[test]
fn jit_bitwise_ops() {
    let result = jit_run::<u32>(
        r#"
fn foo() -> u32 {
    (0xFF00u32 & 0x0FF0u32) | 0x000Fu32
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, (0xFF00u32 & 0x0FF0u32) | 0x000Fu32);
}

#[test]
fn jit_bool_not() {
    let result = jit_run::<i32>(
        r#"
fn check(x: bool) -> i32 {
    if !x { 1 } else { 0 }
}
fn foo() -> i32 {
    check(false)
}
"#,
        &["check", "foo"],
        "foo",
    );
    assert_eq!(result, 1);
}

// ---------------------------------------------------------------------------
// Symbol mangling tests
// ---------------------------------------------------------------------------

/// Helper: get the mangled name for a function by name.
fn mangle_fn(src: &str, name: &str) -> String {
    let full_src = format!("//- /main.rs\n{src}");
    let (db, file_ids) = TestDB::with_many_files(&full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let func_id = find_fn(&db, file_id, name);
        let interner = DbInterner::new_no_crate(&db);
        let generic_args = GenericArgs::empty(interner);
        symbol_mangling::mangle_function(&db, func_id, generic_args)
    })
}

#[test]
fn mangle_crate_root_fn() {
    let mangled = mangle_fn("fn main() {}", "main");
    // Should start with _R (v0 prefix)
    assert!(mangled.starts_with("_R"), "expected v0 prefix, got: {mangled}");
    // Should contain the crate name "ra_test_fixture" (15 chars)
    assert!(mangled.contains("15ra_test_fixture"), "expected crate name, got: {mangled}");
    // Should contain the function name "main" (4 chars)
    assert!(mangled.contains("4main"), "expected fn name, got: {mangled}");
    // Should have the structure: _R N v C <dis> <crate> <dis> <fn>
    // i.e. _RNvC...15ra_test_fixture...4main
    assert!(mangled.contains("NvC"), "expected NvC path prefix, got: {mangled}");
}

#[test]
fn mangle_different_fn_name() {
    let mangled = mangle_fn("fn quux() -> i32 { 42 }", "quux");
    assert!(mangled.starts_with("_R"));
    assert!(mangled.contains("4quux"), "expected fn name 'quux', got: {mangled}");
    assert!(mangled.contains("NvC"), "expected NvC path structure, got: {mangled}");
}

#[test]
fn mangle_different_functions_differ() {
    let mangled_a = mangle_fn("fn alpha() {}\nfn beta() {}", "alpha");
    let mangled_b = mangle_fn("fn alpha() {}\nfn beta() {}", "beta");
    assert_ne!(mangled_a, mangled_b, "different functions should have different mangled names");
}

#[test]
fn jit_with_mangled_names() {
    // Test that functions can call each other using mangled symbol names.
    // The codegen_direct_call now uses mangled names, so this JIT test
    // verifies that caller and callee agree on the mangled symbol.
    let result = jit_run::<i32>(
        r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
fn foo() -> i32 {
    add(10, 32)
}
"#,
        &["add", "foo"],
        "foo",
    );
    assert_eq!(result, 42);
}

// ---------------------------------------------------------------------------
// End-to-end: compile → link → run
// ---------------------------------------------------------------------------

#[test]
fn compile_and_run_empty_main() {
    let full_src = "//- /main.rs\nfn main() {}\n";
    let (db, file_ids) = TestDB::with_many_files(full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let func_id = find_fn(&db, file_id, "main");
        let (body, env) = get_mir_and_env(&db, func_id);
        let dl = get_target_data_layout(&db, func_id);
        let interner = DbInterner::new_no_crate(&db);
        let generic_args = GenericArgs::empty(interner);

        let tmp_dir = std::env::temp_dir().join(format!("rac_test_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let output_path = tmp_dir.join("empty_main");

        let result = crate::compile_executable(
            &db, &dl, &env, &body, func_id, generic_args, &output_path,
        );

        // Clean up on both success and failure
        let cleanup = || {
            let _ = std::fs::remove_dir_all(&tmp_dir);
        };

        match result {
            Ok(()) => {
                // Run the executable
                let output = std::process::Command::new(&output_path)
                    .output()
                    .expect("failed to run compiled binary");

                cleanup();

                assert!(
                    output.status.success(),
                    "binary exited with {}: stderr={}",
                    output.status,
                    String::from_utf8_lossy(&output.stderr),
                );
            }
            Err(e) => {
                cleanup();
                panic!("compile_executable failed: {e}");
            }
        }
    });
}

#[test]
fn compile_and_run_exit_code() {
    let full_src = r#"
//- /main.rs
extern "C" {
    fn exit(code: i32) -> !;
}
fn main() -> ! {
    unsafe { exit(42) }
}
"#;
    let (db, file_ids) = TestDB::with_many_files(full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let func_id = find_fn(&db, file_id, "main");
        let (body, env) = get_mir_and_env(&db, func_id);
        let dl = get_target_data_layout(&db, func_id);
        let interner = DbInterner::new_no_crate(&db);
        let generic_args = GenericArgs::empty(interner);

        let tmp_dir = std::env::temp_dir().join(format!("rac_test_exit_{}", std::process::id()));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let output_path = tmp_dir.join("exit_code");

        let result = crate::compile_executable(
            &db, &dl, &env, &body, func_id, generic_args, &output_path,
        );

        let cleanup = || {
            let _ = std::fs::remove_dir_all(&tmp_dir);
        };

        match result {
            Ok(()) => {
                let output = std::process::Command::new(&output_path)
                    .output()
                    .expect("failed to run compiled binary");

                cleanup();

                assert_eq!(
                    output.status.code(),
                    Some(42),
                    "expected exit code 42, got {:?}: stderr={}",
                    output.status.code(),
                    String::from_utf8_lossy(&output.stderr),
                );
            }
            Err(e) => {
                cleanup();
                panic!("compile_executable failed: {e}");
            }
        }
    });
}
