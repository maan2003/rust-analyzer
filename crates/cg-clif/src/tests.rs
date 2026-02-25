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
    ParamEnvAndCrate, attach_db,
    db::HirDatabase,
    next_solver::{DbInterner, GenericArgs},
};

unsafe extern "C" {
    fn fmodf(x: f32, y: f32) -> f32;
    fn fmod(x: f64, y: f64) -> f64;
    fn dlopen(filename: *const std::ffi::c_char, flags: i32) -> *mut std::ffi::c_void;
    fn dlsym(
        handle: *mut std::ffi::c_void,
        symbol: *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;
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

    // First try module-level declarations
    if let Some(func) = scope.declarations().find_map(|x| match x {
        hir_def::ModuleDefId::FunctionId(x) => {
            if db.function_signature(x).name.display(db, Edition::CURRENT).to_string() == name {
                Some(x)
            } else {
                None
            }
        }
        _ => None,
    }) {
        return func;
    }

    // Then try functions inside impl blocks
    for impl_id in scope.impls() {
        let impl_items = impl_id.impl_items(db);
        for (item_name, item) in impl_items.items.iter() {
            if let hir_def::AssocItemId::FunctionId(fid) = item {
                if item_name.display(db, Edition::CURRENT).to_string() == name {
                    return *fid;
                }
            }
        }
    }

    panic!("function `{name}` not found")
}

fn get_mir_and_env(
    db: &TestDB,
    func_id: hir_def::FunctionId,
) -> (triomphe::Arc<hir_ty::mir::MirBody>, hir_ty::traits::StoredParamEnvAndCrate) {
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

fn get_target_data_layout(
    db: &TestDB,
    func_id: hir_def::FunctionId,
) -> triomphe::Arc<TargetDataLayout> {
    db.target_data_layout(func_id.krate(db)).expect("no target data layout")
}

/// Get the v0-mangled name for a function (non-generic).
fn mangled_name(db: &TestDB, func_id: hir_def::FunctionId) -> String {
    let interner = DbInterner::new_no_crate(db);
    let generic_args = GenericArgs::empty(interner);
    let empty_map = std::collections::HashMap::new();
    symbol_mangling::mangle_function(db, func_id, generic_args, &empty_map)
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
            crate::compile_to_object(&db, &dl, &env, &body, &fn_name, func_id.krate(&db))
                .expect("compilation failed");

        // Verify we got a non-empty object file
        assert!(!obj_bytes.is_empty(), "object file should not be empty");

        // Parse the object file and verify it contains the symbol
        let obj = object::read::File::parse(&*obj_bytes).expect("failed to parse object file");
        use object::{Object, ObjectSymbol};
        let symbols: Vec<_> = obj.symbols().filter(|s| s.name() == Ok(fn_name.as_str())).collect();
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
            crate::compile_to_object(&db, &dl, &env, &body, &fn_name, func_id.krate(&db))
                .expect("compilation failed");
        assert!(!obj_bytes.is_empty());

        let obj = object::read::File::parse(&*obj_bytes).expect("failed to parse object file");
        use object::{Object, ObjectSymbol};
        assert!(
            obj.symbols().any(|s| s.name() == Ok(fn_name.as_str())),
            "symbol '{fn_name}' not found"
        );

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
        let empty_map = std::collections::HashMap::new();

        let mut jit_builder = cranelift_jit::JITBuilder::with_isa(
            isa.clone(),
            cranelift_module::default_libcall_names(),
        );
        jit_builder.symbol("fmodf", fmodf as *const u8);
        jit_builder.symbol("fmod", fmod as *const u8);
        let mut jit_module = cranelift_jit::JITModule::new(jit_builder);

        // Use the first function's crate as local_crate (all test fns are in same crate)
        let first_func = find_fn(&db, file_id, fn_names[0]);
        let local_crate = first_func.krate(&db);

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
                local_crate,
                &empty_map,
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
        let empty_map = std::collections::HashMap::new();

        let builder = cranelift_object::ObjectBuilder::new(
            isa.clone(),
            "rac_output",
            cranelift_module::default_libcall_names(),
        )
        .expect("ObjectBuilder");
        let mut module = cranelift_object::ObjectModule::new(builder);

        let first_func = find_fn(&db, file_id, fn_names[0]);
        let local_crate = first_func.krate(&db);

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
                local_crate,
                &empty_map,
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
            assert!(
                obj.symbols().any(|s| s.name() == Ok(mangled.as_str())),
                "symbol '{mangled}' not found"
            );
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
// Tuple / aggregate / field projection tests
// ---------------------------------------------------------------------------

#[test]
fn jit_tuple_field_access() {
    let result = jit_run::<i32>(
        r#"
fn make_pair(a: i32, b: i32) -> (i32, i32) {
    (a, b)
}
fn foo() -> i32 {
    let pair = make_pair(10, 20);
    pair.0 + pair.1
}
"#,
        &["make_pair", "foo"],
        "foo",
    );
    assert_eq!(result, 30);
}

#[test]
fn jit_tuple_swap() {
    let result = jit_run::<i32>(
        r#"
fn swap(pair: (i32, i32)) -> (i32, i32) {
    (pair.1, pair.0)
}
fn foo() -> i32 {
    let a = (3, 7);
    let b = swap(a);
    b.0 * 10 + b.1
}
"#,
        &["swap", "foo"],
        "foo",
    );
    assert_eq!(result, 73);
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
        let empty_map = std::collections::HashMap::new();
        symbol_mangling::mangle_function(&db, func_id, generic_args, &empty_map)
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

/// Helper: compile source to an executable, run it, return the exit code.
///
/// Uses crate disambiguators from `link::extract_crate_disambiguators()`.
fn compile_and_run_legacy(src: &str, test_name: &str) -> i32 {
    let full_src = fixture_src(src);
    let (db, file_ids) = TestDB::with_many_files(&full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let func_id = find_fn(&db, file_id, "main");
        let (_, env) = get_mir_and_env(&db, func_id);
        let dl = get_target_data_layout(&db, func_id);

        let tmp_dir =
            std::env::temp_dir().join(format!("rac_test_{}_{}", test_name, std::process::id()));
        std::fs::create_dir_all(&tmp_dir).expect("create temp dir");
        let output_path = tmp_dir.join(test_name);

        // Crate disambiguators come from RA_MIRDATA metadata (crate name + StableCrateId).
        let disambiguators = crate::link::extract_crate_disambiguators().unwrap_or_default();
        let result =
            crate::compile_executable(&db, &dl, &env, func_id, &output_path, &disambiguators);

        let cleanup = || {
            let _ = std::fs::remove_dir_all(&tmp_dir);
        };

        match result {
            Ok(()) => {
                let output = std::process::Command::new(&output_path)
                    .output()
                    .expect("failed to run compiled binary");

                cleanup();

                output.status.code().unwrap_or_else(|| {
                    panic!(
                        "binary killed by signal: stderr={}",
                        String::from_utf8_lossy(&output.stderr),
                    )
                })
            }
            Err(e) => {
                cleanup();
                panic!("compile_executable failed: {e}");
            }
        }
    })
}

#[test]
fn compile_and_run_empty_main() {
    let code = compile_and_run_legacy("fn main() {}", "empty_main");
    assert_eq!(code, 0);
}

#[test]
fn compile_and_run_exit_code() {
    let code = compile_and_run_legacy(
        r#"
extern "C" {
    fn exit(code: i32) -> !;
}
fn main() -> ! {
    unsafe { exit(42) }
}
"#,
        "exit_code",
    );
    assert_eq!(code, 42);
}

#[test]
fn compile_and_run_multi_fn() {
    let code = compile_and_run_legacy(
        r#"
extern "C" {
    fn exit(code: i32) -> !;
}
fn add(a: i32, b: i32) -> i32 {
    if a > 0 { a + b } else { b }
}
fn main() -> ! {
    unsafe { exit(add(19, 23)) }
}
"#,
        "multi_fn",
    );
    assert_eq!(code, 42);
}

// ---------------------------------------------------------------------------
// Struct / enum / ADT tests
// ---------------------------------------------------------------------------

#[test]
fn jit_struct_field_access() {
    let result = jit_run::<i32>(
        r#"
struct Point {
    x: i32,
    y: i32,
}
fn foo() -> i32 {
    let p = Point { x: 10, y: 20 };
    p.x + p.y
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 30);
}

#[test]
fn jit_struct_single_field() {
    let result = jit_run::<i32>(
        r#"
struct Wrapper {
    val: i32,
}
fn foo() -> i32 {
    let w = Wrapper { val: 99 };
    w.val
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 99);
}

#[test]
fn jit_struct_pass_and_return() {
    let result = jit_run::<i32>(
        r#"
struct Pair {
    a: i32,
    b: i32,
}
fn swap(p: Pair) -> Pair {
    Pair { a: p.b, b: p.a }
}
fn foo() -> i32 {
    let p = Pair { a: 3, b: 7 };
    let q = swap(p);
    q.a * 10 + q.b
}
"#,
        &["swap", "foo"],
        "foo",
    );
    assert_eq!(result, 73);
}

#[test]
fn jit_enum_match() {
    let result = jit_run::<i32>(
        r#"
enum AB {
    A(i32),
    B(i32),
}
fn extract(x: AB) -> i32 {
    match x {
        AB::A(v) => v,
        AB::B(v) => v + 100,
    }
}
fn foo() -> i32 {
    let a = AB::A(10);
    let b = AB::B(20);
    extract(a) + extract(b)
}
"#,
        &["extract", "foo"],
        "foo",
    );
    assert_eq!(result, 10 + 120);
}

#[test]
fn jit_enum_unit_variants() {
    let result = jit_run::<i32>(
        r#"
enum Color {
    Red,
    Green,
    Blue,
}
fn to_int(c: Color) -> i32 {
    match c {
        Color::Red => 1,
        Color::Green => 2,
        Color::Blue => 3,
    }
}
fn foo() -> i32 {
    to_int(Color::Red) * 100 + to_int(Color::Green) * 10 + to_int(Color::Blue)
}
"#,
        &["to_int", "foo"],
        "foo",
    );
    assert_eq!(result, 123);
}

#[test]
fn jit_generic_struct() {
    let result = jit_run::<i32>(
        r#"
struct Pair<T> {
    first: T,
    second: T,
}
fn sum(p: Pair<i32>) -> i32 {
    p.first + p.second
}
fn foo() -> i32 {
    let p = Pair { first: 17, second: 25 };
    sum(p)
}
"#,
        &["sum", "foo"],
        "foo",
    );
    assert_eq!(result, 42);
}

// Cross-crate compile_and_run tests with fake fixture std were removed.
// The same functionality is covered by JIT tests above.

// ---------------------------------------------------------------------------
// Trait objects / dynamic dispatch tests (M10)
// ---------------------------------------------------------------------------

#[test]
fn jit_dyn_dispatch() {
    let result = jit_run::<i32>(
        r#"
//- minicore: sized, unsize, coerce_unsized, dispatch_from_dyn
//- /main.rs
trait Animal {
    fn legs(&self) -> i32;
}

struct Dog;

impl Animal for Dog {
    fn legs(&self) -> i32 { 4 }
}

fn count(a: &dyn Animal) -> i32 {
    a.legs()
}

fn foo() -> i32 {
    count(&Dog)
}
"#,
        &["legs", "count", "foo"],
        "foo",
    );
    assert_eq!(result, 4);
}

#[test]
fn jit_dyn_dispatch_multiple_methods() {
    let result = jit_run::<i32>(
        r#"
//- minicore: sized, unsize, coerce_unsized, dispatch_from_dyn
//- /main.rs
trait Shape {
    fn width(&self) -> i32;
    fn height(&self) -> i32;
}

struct Rect;

impl Shape for Rect {
    fn width(&self) -> i32 { 10 }
    fn height(&self) -> i32 { 20 }
}

fn area(s: &dyn Shape) -> i32 {
    s.width() * s.height()
}

fn foo() -> i32 {
    area(&Rect)
}
"#,
        &["width", "height", "area", "foo"],
        "foo",
    );
    assert_eq!(result, 200);
}

#[test]
fn compile_and_run_dyn_dispatch() {
    let code = compile_and_run_legacy(
        r#"
//- minicore: sized, unsize, coerce_unsized, dispatch_from_dyn
//- /main.rs
extern "C" {
    fn exit(code: i32) -> !;
}
trait Animal {
    fn legs(&self) -> i32;
}
struct Dog;
impl Animal for Dog {
    fn legs(&self) -> i32 { 4 }
}
fn count(a: &dyn Animal) -> i32 {
    a.legs()
}
fn main() -> ! {
    unsafe { exit(count(&Dog)) }
}
"#,
        "dyn_dispatch",
    );
    assert_eq!(code, 4);
}

#[test]
fn compile_and_run_generics() {
    let code = compile_and_run_legacy(
        r#"
extern "C" {
    fn exit(code: i32) -> !;
}
fn pick<T>(a: T, b: T, first: bool) -> T {
    if first { a } else { b }
}
fn main() -> ! {
    unsafe { exit(pick(7, 3, true)) }
}
"#,
        "generics",
    );
    assert_eq!(code, 7);
}

#[test]
fn compile_and_run_structs_and_enums() {
    let code = compile_and_run_legacy(
        r#"
extern "C" {
    fn exit(code: i32) -> !;
}
struct Point { x: i32, y: i32 }
enum Dir { Left, Right }

fn main() -> ! {
    let p = Point { x: 3, y: 4 };
    let code = match Dir::Right {
        Dir::Left => p.x,
        Dir::Right => p.y,
    };
    unsafe { exit(code) }
}
"#,
        "structs_and_enums",
    );
    assert_eq!(code, 4);
}

#[test]
fn compile_and_run_heap_alloc() {
    // Manual heap allocation using the real allocator shims (__rust_alloc/dealloc).
    // This exercises the same allocator path that Vec uses internally.
    let code = compile_and_run_legacy(
        r#"
//- minicore: drop, sized, copy, size_of
//- /main.rs
unsafe extern "C" {
    fn exit(code: i32) -> !;
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

struct Vec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> Vec<T> {
    fn new() -> Vec<T> {
        Vec { ptr: 0usize as *mut T, len: 0, cap: 0 }
    }

    fn push(&mut self, val: T) {
        if self.len == self.cap {
            self.grow();
        }
        let dest = (self.ptr as usize + self.len * core::mem::size_of::<T>()) as *mut T;
        unsafe { *dest = val; }
        self.len = self.len + 1;
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_size = new_cap * core::mem::size_of::<T>();
        self.ptr = unsafe { malloc(new_size) } as *mut T;
        self.cap = new_cap;
    }

    fn get(&self, idx: usize) -> &T {
        let src = (self.ptr as usize + idx * core::mem::size_of::<T>()) as *const T;
        unsafe { &*src }
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            unsafe { free(self.ptr as *mut u8); }
        }
    }
}

fn main() -> ! {
    let mut v = Vec::<i32>::new();
    v.push(10);
    v.push(20);
    v.push(30);
    let val = *v.get(1);
    unsafe { exit(val) }
}
"#,
        "heap_alloc",
    );
    assert_eq!(code, 20);
}

// ---------------------------------------------------------------------------
// PassMode::Indirect tests (M11a) — memory-repr structs passed/returned by pointer
// ---------------------------------------------------------------------------

#[test]
fn jit_pass_and_return_big_struct() {
    let result = jit_run::<i32>(
        r#"
struct Big {
    a: i32,
    b: i32,
    c: i32,
}
fn make_big(x: i32) -> Big {
    Big { a: x, b: x + 1, c: x + 2 }
}
fn sum_big(s: Big) -> i32 {
    s.a + s.b + s.c
}
fn foo() -> i32 {
    let b = make_big(10);
    sum_big(b)
}
"#,
        &["make_big", "sum_big", "foo"],
        "foo",
    );
    assert_eq!(result, 33); // 10 + 11 + 12
}

#[test]
fn jit_pass_big_struct_and_modify() {
    let result = jit_run::<i32>(
        r#"
struct Big {
    a: i32,
    b: i32,
    c: i32,
}
fn double_fields(s: Big) -> Big {
    Big { a: s.a * 2, b: s.b * 2, c: s.c * 2 }
}
fn foo() -> i32 {
    let b = Big { a: 1, b: 2, c: 3 };
    let d = double_fields(b);
    d.a + d.b + d.c
}
"#,
        &["double_fields", "foo"],
        "foo",
    );
    assert_eq!(result, 12); // 2 + 4 + 6
}

#[test]
fn jit_big_struct_through_call_chain() {
    let result = jit_run::<i32>(
        r#"
struct Triple {
    x: i32,
    y: i32,
    z: i32,
}
fn make(a: i32, b: i32, c: i32) -> Triple {
    Triple { x: a, y: b, z: c }
}
fn add_triples(a: Triple, b: Triple) -> Triple {
    make(a.x + b.x, a.y + b.y, a.z + b.z)
}
fn foo() -> i32 {
    let t1 = make(1, 2, 3);
    let t2 = make(10, 20, 30);
    let t3 = add_triples(t1, t2);
    t3.x + t3.y + t3.z
}
"#,
        &["make", "add_triples", "foo"],
        "foo",
    );
    assert_eq!(result, 66); // 11 + 22 + 33
}

// ---------------------------------------------------------------------------
// Array indexing tests (M11b)
// ---------------------------------------------------------------------------

#[test]
fn jit_array_index() {
    let result: i32 = jit_run(
        r#"
fn foo() -> i32 {
    let arr = [10, 20, 30];
    arr[1usize]
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 20);
}

// ---------------------------------------------------------------------------
// Fn pointer / indirect call tests (M11e)
// ---------------------------------------------------------------------------

#[test]
fn jit_fn_pointer_call() {
    let result: i32 = jit_run(
        r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
fn foo() -> i32 {
    let f: fn(i32, i32) -> i32 = add;
    f(3, 4)
}
"#,
        &["add", "foo"],
        "foo",
    );
    assert_eq!(result, 7);
}

#[test]
fn jit_fn_pointer_higher_order() {
    let result: i32 = jit_run(
        r#"
fn double(x: i32) -> i32 {
    x * 2
}
fn apply(f: fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}
fn foo() -> i32 {
    apply(double, 21)
}
"#,
        &["double", "apply", "foo"],
        "foo",
    );
    assert_eq!(result, 42);
}

/// Helper: JIT-compile using automatic reachable function discovery (including closures).
///
/// Unlike `jit_run`, this doesn't require listing function names explicitly.
/// It uses `collect_reachable_fns` to discover all reachable functions and
/// closures from the entry function.
fn jit_run_reachable<R: Copy>(src: &str, entry: &str) -> R {
    let full_src = fixture_src(src);
    let (db, file_ids) = TestDB::with_many_files(&full_src);
    attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let isa = crate::build_host_isa(false);
        let empty_map = std::collections::HashMap::new();

        let mut jit_builder = cranelift_jit::JITBuilder::with_isa(
            isa.clone(),
            cranelift_module::default_libcall_names(),
        );
        jit_builder.symbol("fmodf", fmodf as *const u8);
        jit_builder.symbol("fmod", fmod as *const u8);
        let mut jit_module = cranelift_jit::JITModule::new(jit_builder);

        let entry_func_id = find_fn(&db, file_id, entry);
        let local_crate = entry_func_id.krate(&db);
        let env = hir_ty::ParamEnvAndCrate {
            param_env: db.trait_environment(entry_func_id.into()),
            krate: local_crate,
        }
        .store();

        // Discover all reachable functions, closures, and drop types
        let (reachable_fns, reachable_closures, drop_types) =
            crate::collect_reachable_fns(&db, &env, entry_func_id, local_crate);

        // Compile all reachable functions
        let dl = get_target_data_layout(&db, entry_func_id);
        for (func_id, generic_args) in &reachable_fns {
            let fn_name = crate::symbol_mangling::mangle_function(
                &db,
                *func_id,
                generic_args.as_ref(),
                &empty_map,
            );
            let body = db
                .monomorphized_mir_body((*func_id).into(), generic_args.clone(), env.clone())
                .unwrap_or_else(|e| panic!("MIR error for {fn_name}: {:?}", e));
            crate::compile_fn(
                &mut jit_module,
                &*isa,
                &db,
                &dl,
                &env,
                &body,
                &fn_name,
                cranelift_module::Linkage::Export,
                local_crate,
                &empty_map,
            )
            .unwrap_or_else(|e| panic!("compiling fn failed: {e}"));
        }

        // Compile all reachable closures
        for (closure_id, closure_subst) in &reachable_closures {
            let body = db
                .monomorphized_mir_body_for_closure(*closure_id, closure_subst.clone(), env.clone())
                .unwrap_or_else(|e| panic!("closure MIR error: {:?}", e));
            let closure_name = crate::symbol_mangling::mangle_closure(&db, *closure_id, &empty_map);
            crate::compile_fn(
                &mut jit_module,
                &*isa,
                &db,
                &dl,
                &env,
                &body,
                &closure_name,
                cranelift_module::Linkage::Export,
                local_crate,
                &empty_map,
            )
            .unwrap_or_else(|e| panic!("compiling closure failed: {e}"));
        }

        // Compile drop_in_place glue functions
        for ty in &drop_types {
            crate::compile_drop_in_place(
                &mut jit_module,
                &*isa,
                &db,
                &dl,
                &env,
                ty,
                local_crate,
                &empty_map,
            )
            .unwrap_or_else(|e| panic!("compiling drop_in_place failed: {e}"));
        }

        // Finalize: make all compiled code executable
        jit_module.finalize_definitions().unwrap();

        // Look up the entry function pointer
        let (entry_body, entry_env) = get_mir_and_env(&db, entry_func_id);
        let sig = crate::build_fn_sig(&*isa, &db, &dl, &entry_env, &entry_body).expect("sig");
        let entry_mangled = mangled_name(&db, entry_func_id);

        let entry_id = jit_module
            .declare_function(&entry_mangled, cranelift_module::Linkage::Import, &sig)
            .expect("declare entry");
        let code_ptr = jit_module.get_finalized_function(entry_id);

        unsafe {
            let f: extern "C" fn() -> R = std::mem::transmute(code_ptr);
            f()
        }
    })
}

const RTLD_LAZY: i32 = 0x1;
const RTLD_GLOBAL: i32 = 0x100;

fn load_libstd_global() -> *mut std::ffi::c_void {
    static LIBSTD_HANDLE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

    let handle = *LIBSTD_HANDLE.get_or_init(|| {
        let libdir = crate::link::find_target_libdir().expect("rustc target-libdir unavailable");
        let libstd = crate::link::find_libstd_so(&libdir).expect("libstd-*.so not found");
        let c_path = std::ffi::CString::new(libstd.to_string_lossy().as_bytes())
            .expect("libstd path contains interior NUL");
        let handle = unsafe { dlopen(c_path.as_ptr(), RTLD_LAZY | RTLD_GLOBAL) };
        assert!(!handle.is_null(), "dlopen({}) failed", libstd.display());
        handle as usize
    });

    handle as *mut std::ffi::c_void
}

fn libstd_exports_symbol(libstd_handle: *mut std::ffi::c_void, symbol: &str) -> bool {
    let c_symbol =
        std::ffi::CString::new(symbol).expect("mangled symbol contains interior NUL byte");
    let ptr = unsafe { dlsym(libstd_handle, c_symbol.as_ptr()) };
    !ptr.is_null()
}

fn find_sysroot_src_dir() -> std::path::PathBuf {
    let output = std::process::Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .expect("failed to run `rustc --print sysroot`");
    assert!(output.status.success(), "rustc --print sysroot failed");

    let sysroot = String::from_utf8(output.stdout).expect("non-UTF8 sysroot path");
    let src_dir = std::path::PathBuf::from(sysroot.trim()).join("lib/rustlib/src/rust/library");
    assert!(src_dir.exists(), "sysroot sources not found at {}", src_dir.display());
    src_dir
}

fn walk_rs_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

fn rustc_cfg_options() -> cfg::CfgOptions {
    let output = std::process::Command::new("rustc")
        .args(["--print", "cfg"])
        .output()
        .expect("failed to run `rustc --print cfg`");
    assert!(output.status.success(), "rustc --print cfg failed");

    let stdout = String::from_utf8(output.stdout).expect("non-UTF8 rustc cfg output");
    let mut cfg = cfg::CfgOptions::default();
    for line in stdout.lines().map(str::trim).filter(|line| !line.is_empty()) {
        if let Some((key, raw_value)) = line.split_once('=') {
            let value = raw_value.trim_matches('"');
            cfg.insert_key_value(intern::Symbol::intern(key), intern::Symbol::intern(value));
        } else {
            cfg.insert_atom(intern::Symbol::intern(line));
        }
    }
    cfg
}

/// Load real sysroot crates (core/alloc/std) and user source into TestDB.
fn load_sysroot_and_user_code(user_src: &str) -> (TestDB, EditionedFileId, base_db::Crate) {
    use base_db::{
        CrateDisplayName, CrateOrigin, CrateWorkspaceData, DependencyBuilder, Env, FileChange,
        LangCrateOrigin,
    };
    use vfs::{AbsPathBuf, file_set::FileSet};

    let sysroot_src = find_sysroot_src_dir();
    let mut source_change = FileChange::default();
    let mut crate_graph = CrateGraphBuilder::default();
    let mut roots = Vec::new();
    let mut next_file_raw = 0u32;

    let proc_macro_cwd = Arc::new(AbsPathBuf::assert_utf8(std::env::current_dir().unwrap()));
    let crate_ws_data = Arc::new(CrateWorkspaceData {
        target: Ok(base_db::target::TargetData {
            arch: base_db::target::Arch::Other,
            data_layout:
                "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
                    .into(),
        }),
        toolchain: None,
    });

    let cfg = rustc_cfg_options();

    let mut load_sysroot_crate =
        |crate_name: &str, sub_path: &str, origin: LangCrateOrigin| -> base_db::CrateBuilderId {
            let crate_dir = sysroot_src.join(sub_path);
            let src_dir = crate_dir.join("src");
            let lib_rs = src_dir.join("lib.rs");
            assert!(lib_rs.exists(), "{} not found", lib_rs.display());

            let mut rs_files = Vec::new();
            walk_rs_files(&src_dir, &mut rs_files);

            let root_file_id = FileId::from_raw(next_file_raw);
            next_file_raw += 1;

            let lib_rs_content = std::fs::read_to_string(&lib_rs)
                .unwrap_or_else(|e| panic!("read {}: {e}", lib_rs.display()));
            source_change.change_file(root_file_id, Some(lib_rs_content));

            let mut file_set = FileSet::default();
            file_set.insert(
                root_file_id,
                base_db::VfsPath::new_virtual_path(format!("/sysroot/{crate_name}/src/lib.rs")),
            );

            for rs_path in rs_files {
                if rs_path == lib_rs {
                    continue;
                }

                let rel = rs_path
                    .strip_prefix(&src_dir)
                    .unwrap_or_else(|e| panic!("strip_prefix {}: {e}", rs_path.display()));
                let file_id = FileId::from_raw(next_file_raw);
                next_file_raw += 1;

                let content = std::fs::read_to_string(&rs_path)
                    .unwrap_or_else(|e| panic!("read {}: {e}", rs_path.display()));
                source_change.change_file(file_id, Some(content));
                file_set.insert(
                    file_id,
                    base_db::VfsPath::new_virtual_path(format!(
                        "/sysroot/{crate_name}/src/{}",
                        rel.display()
                    )),
                );
            }

            roots.push(base_db::SourceRoot::new_library(file_set));

            crate_graph.add_crate_root(
                root_file_id,
                Edition::Edition2021,
                Some(CrateDisplayName::from_canonical_name(crate_name)),
                None,
                cfg.clone(),
                Some(cfg.clone()),
                Env::default(),
                CrateOrigin::Lang(origin),
                Vec::new(),
                false,
                proc_macro_cwd.clone(),
                crate_ws_data.clone(),
            )
        };

    let core_id = load_sysroot_crate("core", "core", LangCrateOrigin::Core);
    let alloc_id = load_sysroot_crate("alloc", "alloc", LangCrateOrigin::Alloc);
    let std_id = load_sysroot_crate("std", "std", LangCrateOrigin::Std);

    crate_graph
        .add_dep(
            alloc_id,
            DependencyBuilder::with_prelude(
                base_db::CrateName::new("core").unwrap(),
                core_id,
                true,
                true,
            ),
        )
        .unwrap();
    crate_graph
        .add_dep(
            std_id,
            DependencyBuilder::with_prelude(
                base_db::CrateName::new("core").unwrap(),
                core_id,
                true,
                true,
            ),
        )
        .unwrap();
    crate_graph
        .add_dep(
            std_id,
            DependencyBuilder::with_prelude(
                base_db::CrateName::new("alloc").unwrap(),
                alloc_id,
                true,
                true,
            ),
        )
        .unwrap();

    let user_file_id = FileId::from_raw(next_file_raw);
    let mut user_file_set = FileSet::default();
    source_change.change_file(user_file_id, Some(user_src.to_owned()));
    user_file_set.insert(user_file_id, base_db::VfsPath::new_virtual_path("/main.rs".to_owned()));
    roots.push(base_db::SourceRoot::new_local(user_file_set));

    let user_crate_id = crate_graph.add_crate_root(
        user_file_id,
        Edition::Edition2021,
        Some(CrateDisplayName::from_canonical_name("test")),
        None,
        cfg.clone(),
        Some(cfg),
        Env::default(),
        CrateOrigin::Local { repo: None, name: None },
        Vec::new(),
        false,
        proc_macro_cwd,
        crate_ws_data,
    );

    crate_graph
        .add_dep(
            user_crate_id,
            DependencyBuilder::with_prelude(
                base_db::CrateName::new("std").unwrap(),
                std_id,
                true,
                true,
            ),
        )
        .unwrap();
    crate_graph
        .add_dep(
            user_crate_id,
            DependencyBuilder::with_prelude(
                base_db::CrateName::new("core").unwrap(),
                core_id,
                true,
                true,
            ),
        )
        .unwrap();
    crate_graph
        .add_dep(
            user_crate_id,
            DependencyBuilder::with_prelude(
                base_db::CrateName::new("alloc").unwrap(),
                alloc_id,
                true,
                true,
            ),
        )
        .unwrap();

    source_change.set_roots(roots);
    source_change.set_crate_graph(crate_graph);

    let mut db = TestDB::default();
    source_change.apply(&mut db);

    let user_crate = module_for_file(&db, user_file_id).krate(&db);
    let user_file = EditionedFileId::new(&db, user_file_id, Edition::Edition2021, user_crate);
    (db, user_file, user_crate)
}

/// JIT harness for std calls in the mirdataless architecture.
///
/// - Real sysroot sources are loaded into the test DB for type/MIR queries.
/// - Local reachable functions, cross-crate generic instantiations, and
///   cross-crate monomorphic functions that are not exported from `libstd.so`
///   (commonly `#[inline]`) are compiled.
/// - Cross-crate monomorphic calls are left as imports and resolved from libstd/libc
///   through symbol mangling + disambiguators (`RA_MIRDATA` metadata only).
fn jit_run_with_std<R: Copy>(src: &str, entry: &str) -> R {
    let libstd_handle = load_libstd_global();
    let disambiguators = crate::link::extract_crate_disambiguators()
        .expect("failed to load crate disambiguators; run via `just test-clif`");
    let (db, file_id, local_crate) = load_sysroot_and_user_code(src);

    attach_db(&db, || {
        let isa = crate::build_host_isa(false);
        let mut jit_builder = cranelift_jit::JITBuilder::with_isa(
            isa.clone(),
            cranelift_module::default_libcall_names(),
        );
        jit_builder.symbol("fmodf", fmodf as *const u8);
        jit_builder.symbol("fmod", fmod as *const u8);
        let mut jit_module = cranelift_jit::JITModule::new(jit_builder);

        let entry_func_id = find_fn(&db, file_id, entry);
        let env = hir_ty::ParamEnvAndCrate {
            param_env: db.trait_environment(entry_func_id.into()),
            krate: local_crate,
        }
        .store();

        let (reachable_fns, reachable_closures, drop_types) =
            crate::collect_reachable_fns(&db, &env, entry_func_id, local_crate);

        let dl = get_target_data_layout(&db, entry_func_id);

        for (func_id, generic_args) in &reachable_fns {
            let is_cross_crate = func_id.krate(&db) != local_crate;
            let is_generic_instance = !generic_args.as_ref().is_empty();
            let fn_name = crate::symbol_mangling::mangle_function(
                &db,
                *func_id,
                generic_args.as_ref(),
                &disambiguators,
            );
            let exported_in_libstd =
                is_cross_crate && libstd_exports_symbol(libstd_handle, &fn_name);
            let should_compile_cross_crate = is_generic_instance || !exported_in_libstd;

            if std::env::var_os("CG_CLIF_STD_JIT_TRACE").is_some() {
                eprintln!(
                    "std-jit: fn={fn_name} cross_crate={is_cross_crate} generic={is_generic_instance} exported_in_libstd={exported_in_libstd} compile={}",
                    !is_cross_crate || should_compile_cross_crate
                );
            }

            // For external calls, keep monomorphic functions as dylib imports by default,
            // but compile monomorphic callees from MIR when their symbols are not
            // exported from libstd (e.g. many `#[inline]` methods such as core::str::len).
            if is_cross_crate && !should_compile_cross_crate {
                continue;
            }
            let body = match db.monomorphized_mir_body(
                (*func_id).into(),
                generic_args.clone(),
                env.clone(),
            ) {
                Ok(body) => body,
                Err(e) if is_cross_crate => {
                    if std::env::var_os("CG_CLIF_STD_JIT_TRACE").is_some() {
                        eprintln!(
                            "std-jit: fn={fn_name} fallback=import reason=mir_error error={e:?}"
                        );
                    }
                    continue;
                }
                Err(e) => panic!("MIR error for compiled fn {fn_name}: {:?}", e),
            };
            crate::compile_fn(
                &mut jit_module,
                &*isa,
                &db,
                &dl,
                &env,
                &body,
                &fn_name,
                cranelift_module::Linkage::Export,
                local_crate,
                &disambiguators,
            )
            .unwrap_or_else(|e| panic!("compiling fn {fn_name} failed: {e}"));
        }

        for (closure_id, closure_subst) in &reachable_closures {
            let body = db
                .monomorphized_mir_body_for_closure(*closure_id, closure_subst.clone(), env.clone())
                .unwrap_or_else(|e| panic!("closure MIR error: {:?}", e));
            let closure_name =
                crate::symbol_mangling::mangle_closure(&db, *closure_id, &disambiguators);
            crate::compile_fn(
                &mut jit_module,
                &*isa,
                &db,
                &dl,
                &env,
                &body,
                &closure_name,
                cranelift_module::Linkage::Export,
                local_crate,
                &disambiguators,
            )
            .unwrap_or_else(|e| panic!("compiling closure failed: {e}"));
        }

        for ty in &drop_types {
            crate::compile_drop_in_place(
                &mut jit_module,
                &*isa,
                &db,
                &dl,
                &env,
                ty,
                local_crate,
                &disambiguators,
            )
            .unwrap_or_else(|e| panic!("compiling drop_in_place failed: {e}"));
        }

        jit_module.finalize_definitions().unwrap();

        let (entry_body, entry_env) = get_mir_and_env(&db, entry_func_id);
        let entry_sig = crate::build_fn_sig(&*isa, &db, &dl, &entry_env, &entry_body).expect("sig");
        let entry_name = crate::symbol_mangling::mangle_function(
            &db,
            entry_func_id,
            GenericArgs::empty(DbInterner::new_no_crate(&db)),
            &disambiguators,
        );

        let entry_id = jit_module
            .declare_function(&entry_name, cranelift_module::Linkage::Import, &entry_sig)
            .expect("declare entry");
        let code_ptr = jit_module.get_finalized_function(entry_id);

        unsafe {
            let f: extern "C" fn() -> R = std::mem::transmute(code_ptr);
            f()
        }
    })
}

#[test]
fn std_jit_process_id_nonzero() {
    let result: i32 = jit_run_with_std(
        r#"
fn foo() -> i32 {
    (std::process::id() != 0) as i32
}
"#,
        "foo",
    );
    assert_eq!(result, 1);
}

#[test]
fn std_jit_generic_identity_i32() {
    let result: i32 = jit_run_with_std(
        r#"
fn foo() -> i32 {
    core::convert::identity(42)
}
"#,
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn std_jit_str_len_smoke() {
    let result: i32 = jit_run_with_std(
        r#"
fn foo() -> i32 {
    ("hello".len() == 5) as i32
}
"#,
        "foo",
    );
    assert_eq!(result, 1);
}

#[test]
fn std_jit_array_index_smoke() {
    let result: i32 = jit_run_with_std(
        r#"
fn foo() -> i32 {
    let x = [1, 2, 3];
    x[0]
}
"#,
        "foo",
    );
    assert_eq!(result, 1);
}

#[test]
#[ignore = "currently fails during codegen cast: load_scalar on ByValPair"]
fn std_jit_vec_new_smoke() {
    let result: i32 = jit_run_with_std(
        r#"
fn foo() -> i32 {
    let v: Vec<i32> = Vec::new();
    (v.len() == 0) as i32
}
"#,
        "foo",
    );
    assert_eq!(result, 1);
}

#[test]
#[ignore = "currently fails during codegen: non-value const in ScalarPair constant"]
fn std_jit_env_var_roundtrip() {
    let result: i32 = jit_run_with_std(
        r#"
fn foo() -> i32 {
    let key = "CG_CLIF_STD_JIT_ENV_VAR";
    unsafe { std::env::set_var(key, "hello") };
    match std::env::var(key) {
        Ok(value) => (value == "hello") as i32,
        Err(_) => 0,
    }
}
"#,
        "foo",
    );
    assert_eq!(result, 1);
}

#[test]
fn std_jit_process_id_is_stable_across_calls() {
    let result: i32 = jit_run_with_std(
        r#"
fn foo() -> i32 {
    let a = std::process::id();
    let b = std::process::id();
    (a == b) as i32
}
"#,
        "foo",
    );
    assert_eq!(result, 1);
}

#[test]
fn jit_closure_basic() {
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: fn
//- /main.rs
fn apply(f: impl Fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}
fn foo() -> i32 {
    let offset = 10;
    apply(|x| x + offset, 32)
}
"#,
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_drop_basic() {
    // Verify that Drop::drop is called on scope exit.
    // The drop impl modifies a field; we verify the function compiles and runs correctly.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy
//- /main.rs
struct Guard {
    val: i32,
}
impl Drop for Guard {
    fn drop(&mut self) {
        self.val = 99;
    }
}
fn foo() -> i32 {
    let g = Guard { val: 42 };
    g.val
}
"#,
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_drop_side_effect() {
    // Verify Drop::drop is actually called by observing its side effect.
    // The drop writes through a raw pointer to modify a caller's local.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy
//- /main.rs
struct Guard {
    target: *mut i32,
    val: i32,
}
impl Drop for Guard {
    fn drop(&mut self) {
        unsafe { *(self.target) = self.val; }
    }
}
fn write_via_drop(target: *mut i32) {
    let _g = Guard { target, val: 42 };
}
fn foo() -> i32 {
    let mut result: i32 = 0;
    write_via_drop(&mut result as *mut i32);
    result
}
"#,
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_drop_no_drop_impl() {
    // Verify that types without Drop impls work fine (no-op drop).
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy
//- /main.rs
struct Pair { a: i32, b: i32 }
fn foo() -> i32 {
    let p = Pair { a: 10, b: 32 };
    p.a + p.b
}
"#,
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_needs_drop_transitive() {
    // needs_drop should return true for a struct that doesn't impl Drop
    // itself but contains a field that does.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy
//- /main.rs
#[rustc_intrinsic]
const fn needs_drop<T>() -> bool { true }

struct Inner;
impl Drop for Inner {
    fn drop(&mut self) {}
}
struct Outer { _inner: Inner }

fn foo() -> i32 {
    if needs_drop::<Outer>() { 1 } else { 0 }
}
"#,
        "foo",
    );
    assert_eq!(result, 1);
}

#[test]
fn jit_str_constant() {
    // Verify that &str constants work: the data pointer points to a valid
    // data section and the length is correct. We use transmute to extract
    // the (pointer, length) pair from the fat pointer.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: sized, copy
//- /main.rs
#[rustc_intrinsic]
unsafe fn transmute<Src, Dst>(src: Src) -> Dst { loop {} }

fn foo() -> i32 {
    let s: &str = "hello";
    let pair: (usize, usize) = unsafe { transmute(s) };
    pair.1 as i32
}
"#,
        "foo",
    );
    assert_eq!(result, 5);
}

#[test]
fn jit_str_constant_read_byte() {
    // Verify that the data pointer in a &str constant points to the actual
    // string bytes by reading the first byte through a raw pointer.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: sized, copy
//- /main.rs
#[rustc_intrinsic]
unsafe fn transmute<Src, Dst>(src: Src) -> Dst { loop {} }

fn foo() -> i32 {
    let s: &str = "hello";
    let (ptr, _len): (*const u8, usize) = unsafe { transmute(s) };
    unsafe { *ptr as i32 }
}
"#,
        "foo",
    );
    assert_eq!(result, 104); // 'h' = 104
}

#[test]
fn jit_drop_field_recursive() {
    // Verify that dropping a struct without a Drop impl still drops its fields.
    // Outer has no Drop impl, but contains Inner which does.
    // Inner's drop writes through a raw pointer, proving it was called.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy
//- /main.rs
struct Inner {
    target: *mut i32,
}
impl Drop for Inner {
    fn drop(&mut self) {
        unsafe { *(self.target) = 42; }
    }
}
struct Outer {
    inner: Inner,
}
fn make_and_drop(target: *mut i32) {
    let _o = Outer { inner: Inner { target } };
}
fn foo() -> i32 {
    let mut result: i32 = 0;
    make_and_drop(&mut result as *mut i32);
    result
}
"#,
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_drop_generic() {
    // Verify that Drop::drop works for generic types.
    // The drop impl writes through a raw pointer — proving the correct
    // monomorphized drop function is called with the right generic args.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy
//- /main.rs
struct Guard<T> {
    target: *mut T,
    val: T,
}
impl<T> Drop for Guard<T> {
    fn drop(&mut self) {
        unsafe { *(self.target) = self.val; }
    }
}
fn write_via_drop(target: *mut i32) {
    let _g = Guard { target, val: 42 };
}
fn foo() -> i32 {
    let mut result: i32 = 0;
    write_via_drop(&mut result as *mut i32);
    result
}
"#,
        "foo",
    );
    assert_eq!(result, 42);
}

#[test]
fn jit_heap_alloc_basic() {
    // Simulate Vec-like heap allocation: malloc, write through pointer,
    // read back, free. Exercises extern calls + raw pointer ops.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: sized, copy
//- /main.rs
unsafe extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

fn at(base: *mut i32, idx: usize) -> *mut i32 {
    (base as usize + idx * 4) as *mut i32
}

fn foo() -> i32 {
    unsafe {
        let ptr = malloc(12) as *mut i32; // room for 3 i32s
        *at(ptr, 0) = 10;
        *at(ptr, 1) = 20;
        *at(ptr, 2) = 30;
        let val = *at(ptr, 1); // read second element
        free(ptr as *mut u8);
        val
    }
}
"#,
        "foo",
    );
    assert_eq!(result, 20);
}

#[test]
fn jit_heap_vec_like() {
    // A manual Vec-like struct: heap-allocate, write elements, read back,
    // auto-drop deallocates. Exercises: extern calls, raw ptr ops, struct
    // with Drop, field access, and drop_in_place glue.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy
//- /main.rs
unsafe extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
}

struct IntBuf {
    ptr: *mut i32,
    len: usize,
}

impl IntBuf {
    fn new(cap: usize) -> IntBuf {
        IntBuf {
            ptr: unsafe { malloc(cap * 4) } as *mut i32,
            len: 0,
        }
    }
    fn push(&mut self, val: i32) {
        let dest = (self.ptr as usize + self.len * 4) as *mut i32;
        unsafe { *dest = val; }
        self.len = self.len + 1;
    }
    fn get(&self, idx: usize) -> i32 {
        let src = (self.ptr as usize + idx * 4) as *const i32;
        unsafe { *src }
    }
}

impl Drop for IntBuf {
    fn drop(&mut self) {
        unsafe { free(self.ptr as *mut u8); }
    }
}

fn foo() -> i32 {
    let mut buf = IntBuf::new(4);
    buf.push(10);
    buf.push(20);
    buf.push(30);
    let val = buf.get(1);
    // buf is dropped here, calling free()
    val
}
"#,
        "foo",
    );
    assert_eq!(result, 20);
}

#[test]
fn jit_generic_vec() {
    // A generic Vec<T> with heap allocation, push, index, and Drop.
    // Exercises: generics + size_of intrinsic + raw pointer arithmetic +
    // copy_nonoverlapping + generic drop glue + realloc.
    let result: i32 = jit_run_reachable(
        r#"
//- minicore: drop, sized, copy, size_of
//- /main.rs
unsafe extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn realloc(ptr: *mut u8, new_size: usize) -> *mut u8;
}

struct Vec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> Vec<T> {
    fn new() -> Vec<T> {
        Vec { ptr: 0usize as *mut T, len: 0, cap: 0 }
    }

    fn push(&mut self, val: T) {
        if self.len == self.cap {
            self.grow();
        }
        let dest = (self.ptr as usize + self.len * core::mem::size_of::<T>()) as *mut T;
        unsafe { *dest = val; }
        self.len = self.len + 1;
    }

    fn grow(&mut self) {
        let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
        let new_size = new_cap * core::mem::size_of::<T>();
        let new_ptr = if self.cap == 0 {
            unsafe { malloc(new_size) }
        } else {
            unsafe { realloc(self.ptr as *mut u8, new_size) }
        };
        self.ptr = new_ptr as *mut T;
        self.cap = new_cap;
    }

    fn get(&self, idx: usize) -> &T {
        let src = (self.ptr as usize + idx * core::mem::size_of::<T>()) as *const T;
        unsafe { &*src }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            unsafe { free(self.ptr as *mut u8); }
        }
    }
}

fn foo() -> i32 {
    let mut v = Vec::<i32>::new();
    v.push(10);
    v.push(20);
    v.push(30);
    let val = *v.get(1);
    let len = v.len() as i32;
    // v is dropped here, calling free()
    val + len  // 20 + 3 = 23
}
"#,
        "foo",
    );
    assert_eq!(result, 23);
}

// ---------------------------------------------------------------------------
// Essential intrinsics tests (M11c)
// ---------------------------------------------------------------------------

#[test]
fn jit_ptr_metadata_str() {
    let result: usize = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn ptr_metadata<P: ?Sized>(ptr: *const P) -> usize;
}

fn foo() -> usize {
    let s: &str = "hello";
    let p: *const str = s as *const str;
    unsafe { ptr_metadata::<str>(p) }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 5);
}

#[test]
fn jit_size_of() {
    let result: usize = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn size_of<T>() -> usize;
}
fn foo() -> usize {
    unsafe { size_of::<i32>() }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 4);
}

#[test]
fn jit_min_align_of() {
    let result: usize = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn min_align_of<T>() -> usize;
}
fn foo() -> usize {
    unsafe { min_align_of::<i32>() }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 4);
}

#[test]
fn jit_bswap() {
    let result: u32 = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn bswap<T>(x: T) -> T;
}
fn foo() -> u32 {
    unsafe { bswap(0x12345678u32) }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 0x78563412);
}

#[test]
fn jit_wrapping_add() {
    let result: i32 = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn wrapping_add<T>(a: T, b: T) -> T;
}
fn foo() -> i32 {
    unsafe { wrapping_add(100i32, 200i32) }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 300);
}

#[test]
fn jit_transmute() {
    let result: u32 = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn transmute<T, U>(x: T) -> U;
}
fn foo() -> u32 {
    unsafe { transmute::<f32, u32>(1.0f32) }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 0x3f800000); // IEEE 754 representation of 1.0f32
}

#[test]
fn jit_ctlz() {
    let result: u32 = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn ctlz<T>(x: T) -> u32;
}
fn foo() -> u32 {
    unsafe { ctlz(0x00FF0000u32) }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 8);
}

#[test]
fn jit_rotate_left() {
    let result: u32 = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn rotate_left<T>(x: T, y: u32) -> T;
}
fn foo() -> u32 {
    unsafe { rotate_left(0x12345678u32, 8u32) }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 0x34567812);
}

#[test]
fn jit_exact_div() {
    let result: i32 = jit_run(
        r#"
extern "rust-intrinsic" {
    pub fn exact_div<T>(x: T, y: T) -> T;
}
fn foo() -> i32 {
    unsafe { exact_div(42i32, 6i32) }
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 7);
}

// ---------------------------------------------------------------------------
// Non-scalar constants tests (M11d)
// ---------------------------------------------------------------------------

#[test]
fn jit_const_array_index() {
    // Tests memory-repr constant (array stored in data section) + indexing
    let result: i32 = jit_run(
        r#"
fn foo() -> i32 {
    let arr = [10i32, 20, 30];
    arr[0usize] + arr[2usize]
}
"#,
        &["foo"],
        "foo",
    );
    assert_eq!(result, 40); // 10 + 30
}

// ---------------------------------------------------------------------------
