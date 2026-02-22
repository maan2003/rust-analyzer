use std::{fmt, panic, sync::Mutex};

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

        let obj_bytes =
            crate::compile_to_object(&db, &dl, &env, &body, "foo").expect("compilation failed");

        // Verify we got a non-empty object file
        assert!(!obj_bytes.is_empty(), "object file should not be empty");

        // Parse the object file and verify it contains the symbol
        let obj = object::read::File::parse(&*obj_bytes).expect("failed to parse object file");
        use object::{Object, ObjectSymbol};
        let symbols: Vec<_> = obj
            .symbols()
            .filter(|s| s.name() == Ok("foo"))
            .collect();
        assert!(!symbols.is_empty(), "symbol 'foo' not found in object file");
    });
}
