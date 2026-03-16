//! Rustc v0 symbol mangling for r-a's codegen backend.
//!
//! Encoding primitives (`push_integer_62`, `push_ident`, `base_62_encode`, type
//! char mapping) are ported from `rustc_symbol_mangling/src/v0.rs` (RFC 2603).
//! We keep them close to the originals for easier future syncing.
//!
//! Deferred: full rustc-compatible const encoding (we currently hash const
//! args for uniqueness), `dyn Trait` / fn-pointer encoding, punycode,
//! closure/coroutine encoding.

use std::collections::HashMap;
use std::fmt::Write;
use std::ops::Range;

use base_db::Crate;
use hir_def::{
    AdtId, FunctionId, HasModule, ImplId, ItemContainerId, ModuleDefId, ModuleId, StaticId, TraitId,
};
use hir_ty::db::HirDatabase;
use hir_ty::next_solver::Mutability;
use hir_ty::next_solver::{
    Binder, BoundExistentialPredicates, BoundVarKind, Const, ExistentialPredicate, GenericArg,
    GenericArgKind, GenericArgs, IntoKind, Region, RegionKind, TermKind, Ty, TyKind,
};
use hir_ty::primitive::{FloatTy, IntTy, UintTy};
use rustc_type_ir::BoundVarIndexKind;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Produce a v0-mangled symbol name for a monomorphic function instance.
///
/// `ext_crate_disambiguators` maps crate names to their real disambiguator
/// values extracted from sysroot rlibs. For local crates (not in the map),
/// the FileId index is used as the disambiguator.
pub fn mangle_function(
    db: &dyn HirDatabase,
    func_id: FunctionId,
    generic_args: GenericArgs<'_>,
    ext_crate_disambiguators: &HashMap<String, u64>,
) -> String {
    let out = String::from("_R");
    let mut m = SymbolMangler {
        db,
        start_offset: out.len(),
        out,
        ext_crate_disambiguators,
        module_paths: HashMap::new(),
        binders: vec![],
    };

    let container = func_id.loc(db).container;
    let fn_name = db.function_signature(func_id).name.as_str().to_owned();
    let fn_disambiguator = m.function_disambiguator(func_id, container, &fn_name);

    let has_non_lifetime_args =
        generic_args.iter().any(|arg| !matches!(arg.kind(), GenericArgKind::Lifetime(_)));

    if !has_non_lifetime_args {
        // No generic args — simple path.
        m.out.push_str("N");
        m.out.push('v'); // ValueNs
        m.print_container_path(container);
        m.push_disambiguator(fn_disambiguator);
        m.push_ident(&fn_name);
    } else {
        // Wrap with I...E for generic instantiation.
        m.out.push_str("I");
        m.out.push_str("N");
        m.out.push('v'); // ValueNs
        m.print_container_path(container);
        m.push_disambiguator(fn_disambiguator);
        m.push_ident(&fn_name);
        for arg in generic_args.iter() {
            match arg.kind() {
                GenericArgKind::Type(ty) => m.print_type(ty),
                GenericArgKind::Const(konst) => m.print_const(konst),
                GenericArgKind::Lifetime(_) => {}
            }
        }
        m.out.push_str("E");
    }

    m.out
}

/// Produce a v0-mangled symbol name for a static item.
///
/// For extern statics declared in an extern block, this returns the declared
/// symbol name directly. Otherwise this encodes a value namespace path.
pub fn mangle_static(
    db: &dyn HirDatabase,
    static_id: StaticId,
    ext_crate_disambiguators: &HashMap<String, u64>,
) -> String {
    let static_loc = static_id.loc(db);
    let static_name = db.static_signature(static_id).name.as_str().to_owned();
    if matches!(static_loc.container, ItemContainerId::ExternBlockId(_)) {
        return static_name;
    }

    let out = String::from("_R");
    let mut m = SymbolMangler {
        db,
        start_offset: out.len(),
        out,
        ext_crate_disambiguators,
        module_paths: HashMap::new(),
        binders: vec![],
    };

    m.out.push_str("N");
    m.out.push('v'); // ValueNs
    m.print_container_path(static_loc.container);
    m.push_ident(&static_name);
    m.out
}

/// Produce a symbol name for a closure.
///
/// Uses a simple scheme based on `_Rclosure_{owner_crate}_{closure_id}` with
/// optional type-argument encoding for monomorphized closure instances.
pub fn mangle_closure(
    db: &dyn HirDatabase,
    closure_id: hir_ty::db::InternedClosureId,
    generic_args: GenericArgs<'_>,
    ext_crate_disambiguators: &HashMap<String, u64>,
) -> String {
    let def = db.lookup_intern_closure(closure_id);
    let owner = def.0;
    let krate = owner.module(db).krate(db);
    let file_dis = krate.data(db).root_file_id.index() as u64;
    let extra = krate.extra_data(db);
    let crate_name = extra
        .display_name
        .as_ref()
        .map(|dn| dn.crate_name().to_string())
        .unwrap_or_else(|| format!("crate{}", file_dis));
    let disamb = ext_crate_disambiguators.get(&crate_name).copied().unwrap_or(file_dis);

    let out = format!("_Rclosure_{}_{:x}_{:?}", crate_name, disamb, closure_id);

    // Mirror function symbol behavior: encode non-lifetime args so multiple
    // monomorphized instances of the same closure id don't collide.
    let has_non_lifetime_args =
        generic_args.iter().any(|arg| !matches!(arg.kind(), GenericArgKind::Lifetime(_)));

    if !has_non_lifetime_args {
        return out;
    }

    let mut m = SymbolMangler {
        db,
        start_offset: out.len(),
        out,
        ext_crate_disambiguators,
        module_paths: HashMap::new(),
        binders: vec![],
    };
    m.out.push('I');
    for arg in generic_args.iter() {
        match arg.kind() {
            GenericArgKind::Type(ty) => m.print_type(ty),
            GenericArgKind::Const(konst) => m.print_const(konst),
            GenericArgKind::Lifetime(_) => {}
        }
    }
    m.out.push('E');
    m.out
}

/// Produce a symbol name for a `drop_in_place::<T>` glue function.
///
/// Uses a `_Rdrop_` prefix followed by the v0-encoded type to ensure
/// uniqueness per monomorphized type.
pub fn mangle_drop_in_place(
    db: &dyn HirDatabase,
    ty: Ty<'_>,
    ext_crate_disambiguators: &HashMap<String, u64>,
) -> String {
    let out = String::from("_Rdrop_");
    let mut m = SymbolMangler {
        db,
        start_offset: out.len(),
        out,
        ext_crate_disambiguators,
        module_paths: HashMap::new(),
        binders: vec![],
    };
    m.print_type(ty);
    m.out
}

/// Produce a symbol name for a vtable for `dyn Trait` over a concrete `T`.
///
/// This must use the full trait+type identity, not fast-reject summaries, so
/// distinct closure/object types don't collapse onto the same vtable symbol.
pub fn mangle_vtable(
    db: &dyn HirDatabase,
    concrete_ty: Ty<'_>,
    trait_id: TraitId,
    ext_crate_disambiguators: &HashMap<String, u64>,
) -> String {
    let out = String::from("_Rvtable_");
    let mut m = SymbolMangler {
        db,
        start_offset: out.len(),
        out,
        ext_crate_disambiguators,
        module_paths: HashMap::new(),
        binders: vec![],
    };
    m.print_trait_path(trait_id, None);
    m.print_type(concrete_ty);
    m.out
}

// ---------------------------------------------------------------------------
// SymbolMangler
// ---------------------------------------------------------------------------

struct SymbolMangler<'a> {
    db: &'a dyn HirDatabase,
    /// Start offset for backref indexes.
    start_offset: usize,
    out: String,
    ext_crate_disambiguators: &'a HashMap<String, u64>,
    /// Cache of printed module paths to emit rustc-style `B..._` backrefs.
    module_paths: HashMap<ModuleId, usize>,
    binders: Vec<BinderLevel>,
}

struct BinderLevel {
    /// Flatten bound lifetime vars across nested binders into true de Bruijn
    /// indices, following rustc's v0 mangler scheme.
    lifetime_depths: Range<u32>,
}

impl<'a> SymbolMangler<'a> {
    // -- Encoding primitives (ported from rustc v0.rs) ----------------------

    /// Push a `_`-terminated base 62 integer (RFC spec `<base-62-number>`):
    /// * `x = 0` → `"_"`
    /// * `x > 0` → base62(x - 1) + `"_"`
    fn push_integer_62(&mut self, x: u64) {
        push_integer_62(x, &mut self.out)
    }

    /// Push a `tag`-prefixed base 62 integer, elided when `x == 0`:
    /// * `x = 0` → nothing
    /// * `x > 0` → tag + integer_62(x - 1)
    fn push_opt_integer_62(&mut self, tag: &str, x: u64) {
        if let Some(x) = x.checked_sub(1) {
            self.out.push_str(tag);
            self.push_integer_62(x);
        }
    }

    fn push_disambiguator(&mut self, dis: u64) {
        self.push_opt_integer_62("s", dis);
    }

    fn push_ident(&mut self, ident: &str) {
        push_ident(ident, &mut self.out)
    }

    fn print_backref(&mut self, i: usize) {
        debug_assert!(i >= self.start_offset);
        self.out.push('B');
        self.push_integer_62((i - self.start_offset) as u64);
    }

    fn wrap_binder<T>(&mut self, value: &Binder<'_, T>, print_value: impl FnOnce(&mut Self, &T))
    where
        T: rustc_type_ir::TypeVisitable<hir_ty::next_solver::DbInterner<'a>>,
    {
        let mut lifetime_depths =
            self.binders.last().map(|binder| binder.lifetime_depths.end).map_or(0..0, |i| i..i);
        let lifetimes = value
            .bound_vars()
            .iter()
            .filter(|var| matches!(var, BoundVarKind::Region(_)))
            .count() as u32;

        self.push_opt_integer_62("G", lifetimes as u64);
        lifetime_depths.end += lifetimes;
        self.binders.push(BinderLevel { lifetime_depths });
        print_value(self, value.as_ref().skip_binder());
        self.binders.pop();
    }

    fn print_region(&mut self, region: Region<'a>) {
        let idx = match region.kind() {
            RegionKind::ReErased
            | RegionKind::ReStatic
            | RegionKind::ReEarlyParam(_)
            | RegionKind::ReLateParam(_)
            | RegionKind::RePlaceholder(_)
            | RegionKind::ReVar(_)
            | RegionKind::ReError(_) => 0,
            RegionKind::ReBound(BoundVarIndexKind::Bound(debruijn), bound_region) => {
                let binder = &self.binders[self.binders.len() - 1 - debruijn.index()];
                let depth = binder.lifetime_depths.start + bound_region.var.as_u32();
                1 + (self.binders.last().expect("bound region without binder").lifetime_depths.end
                    - 1
                    - depth)
            }
            RegionKind::ReBound(BoundVarIndexKind::Canonical, _) => 0,
        };
        self.out.push('L');
        self.push_integer_62(idx as u64);
    }

    fn region_has_nontrivial_mangling(&self, region: Region<'a>) -> bool {
        matches!(region.kind(), RegionKind::ReBound(BoundVarIndexKind::Bound(..), _))
    }

    fn print_generic_arg(&mut self, arg: GenericArg<'a>) {
        match arg.kind() {
            GenericArgKind::Lifetime(region) => self.print_region(region),
            GenericArgKind::Type(ty) => self.print_type(ty),
            GenericArgKind::Const(konst) => {
                self.out.push('K');
                self.print_const(konst);
            }
        }
    }

    fn print_path_with_generic_args(
        &mut self,
        print_prefix: impl FnOnce(&mut Self),
        args: GenericArgs<'a>,
        skip_self_arg: bool,
    ) {
        let print_regions = args.iter().enumerate().any(|(idx, arg)| {
            (!skip_self_arg || idx != 0)
                && matches!(arg.kind(), GenericArgKind::Lifetime(region) if self.region_has_nontrivial_mangling(region))
        });
        let has_any_args = args.iter().enumerate().any(|(idx, arg)| {
            (!skip_self_arg || idx != 0)
                && !matches!(arg.kind(), GenericArgKind::Lifetime(_) if !print_regions)
        });

        if !has_any_args {
            print_prefix(self);
            return;
        }

        self.out.push('I');
        print_prefix(self);
        for (idx, arg) in args.iter().enumerate() {
            if skip_self_arg && idx == 0 {
                continue;
            }
            if matches!(arg.kind(), GenericArgKind::Lifetime(_)) && !print_regions {
                continue;
            }
            self.print_generic_arg(arg);
        }
        self.out.push('E');
    }

    fn function_disambiguator(
        &self,
        func_id: FunctionId,
        container: ItemContainerId,
        fn_name: &str,
    ) -> u64 {
        let mut same_named = Vec::new();
        match container {
            ItemContainerId::ModuleId(module) => {
                let def_map = module.def_map(self.db);
                let scope = &def_map[module].scope;
                for decl in scope.declarations() {
                    if let ModuleDefId::FunctionId(fid) = decl
                        && self.db.function_signature(fid).name.as_str() == fn_name
                    {
                        same_named.push(fid);
                    }
                }
            }
            ItemContainerId::ImplId(impl_id) => {
                let impl_items = impl_id.impl_items(self.db);
                for (_name, item) in impl_items.items.iter() {
                    if let hir_def::AssocItemId::FunctionId(fid) = item
                        && self.db.function_signature(*fid).name.as_str() == fn_name
                    {
                        same_named.push(*fid);
                    }
                }
            }
            ItemContainerId::TraitId(trait_id) => {
                let trait_items = trait_id.trait_items(self.db);
                for (_name, item) in trait_items.items.iter() {
                    if let hir_def::AssocItemId::FunctionId(fid) = item
                        && self.db.function_signature(*fid).name.as_str() == fn_name
                    {
                        same_named.push(*fid);
                    }
                }
            }
            ItemContainerId::ExternBlockId(_) => {
                // Extern symbols use function names directly and don't rely on rust-path disambiguators.
                return 0;
            }
        }

        if same_named.len() <= 1 {
            return 0;
        }

        same_named.sort_by_key(|fid| format!("{fid:?}"));
        same_named.iter().position(|fid| *fid == func_id).map(|idx| idx as u64).unwrap_or(0)
    }

    /// Match rustc's impl-path disambiguator behavior closely by using the
    /// deterministic order of impl blocks within the containing module scope.
    fn impl_disambiguator(&self, impl_id: ImplId) -> u64 {
        let module = impl_id.module(self.db);
        let def_map = module.def_map(self.db);
        let scope = &def_map[module].scope;
        scope.impls().position(|id| id == impl_id).map(|idx| idx as u64).unwrap_or(0)
    }

    // -- Path encoding ------------------------------------------------------

    fn print_crate(&mut self, krate: Crate) {
        self.out.push('C');
        // Crate name
        let extra = krate.extra_data(self.db);
        let root_file = krate.data(self.db).root_file_id;
        let file_dis = root_file.index() as u64;
        let name = extra
            .display_name
            .as_ref()
            .map(|dn| dn.crate_name().to_string())
            .unwrap_or_else(|| format!("crate{}", file_dis));

        // Use real disambiguator from rlib symbols for external crates,
        // fall back to FileId index for local crates.
        let dis = if let Some(&ext_dis) = self.ext_crate_disambiguators.get(&name) {
            ext_dis
        } else {
            file_dis
        };
        self.push_disambiguator(dis);
        self.push_ident(&name);
    }

    fn print_module_path(&mut self, module: hir_def::ModuleId) {
        if let Some(&start) = self.module_paths.get(&module) {
            self.print_backref(start);
            return;
        }
        let start = self.out.len();
        match module.containing_module(self.db) {
            None => {
                // Crate root module.
                self.print_crate(module.krate(self.db));
            }
            Some(parent) => {
                let (name, disambiguator) = match module.name(self.db) {
                    Some(name) => (name.as_str().to_owned(), 0),
                    None => (String::new(), self.anon_module_disambiguator(module)),
                };
                self.out.push('N');
                self.out.push('t'); // TypeNs
                self.print_module_path(parent);
                self.push_disambiguator(disambiguator);
                self.push_ident(&name);
            }
        }
        self.module_paths.insert(module, start);
    }

    /// Anonymous/block modules need a disambiguator; otherwise all local-item
    /// scopes collapse to the same mangled path (`...0`), causing collisions.
    fn anon_module_disambiguator(&self, module: hir_def::ModuleId) -> u64 {
        debug_assert!(module.name(self.db).is_none());

        // FNV-1a over the module debug identity keeps this deterministic per
        // database while avoiding dependence on HashMap's randomized hasher.
        let hash = stable_hash_bytes(format!("{module:?}").as_bytes());
        hash.max(1)
    }

    fn print_container_path(&mut self, container: ItemContainerId) {
        match container {
            ItemContainerId::ModuleId(module) => {
                self.print_module_path(module);
            }
            ItemContainerId::ImplId(impl_id) => {
                self.print_impl_path(impl_id);
            }
            ItemContainerId::TraitId(trait_id) => {
                self.print_trait_path(trait_id, None);
            }
            ItemContainerId::ExternBlockId(extern_id) => {
                // Skip to parent module (matches rustc's ForeignMod handling).
                let module = extern_id.module(self.db);
                self.print_module_path(module);
            }
        }
    }

    fn print_impl_path(&mut self, impl_id: ImplId) {
        let disambiguator = self.impl_disambiguator(impl_id);
        let module = impl_id.module(self.db);
        let self_ty = self.db.impl_self_ty(impl_id).skip_binder();

        if let Some(trait_ref) = self.db.impl_trait(impl_id).map(|it| it.skip_binder()) {
            // Trait impl: X + disambiguator + parent_path + self_ty + trait_path.
            self.out.push('X');
            self.push_disambiguator(disambiguator);
            self.print_module_path(module);
            self.print_type(self_ty);
            self.print_trait_path(trait_ref.def_id.0, Some(trait_ref.args));
        } else {
            // Inherent impl: M + disambiguator + parent_path + self_ty.
            self.out.push('M');
            self.push_disambiguator(disambiguator);
            self.print_module_path(module);
            self.print_type(self_ty);
        }
    }

    fn print_trait_path(&mut self, trait_id: TraitId, args: Option<GenericArgs<'a>>) {
        self.print_trait_path_with_args(trait_id, args, true);
    }

    fn print_existential_trait_path(&mut self, trait_id: TraitId, args: GenericArgs<'a>) {
        self.print_trait_path_with_args(trait_id, Some(args), false);
    }

    fn print_trait_path_with_args(
        &mut self,
        trait_id: TraitId,
        args: Option<GenericArgs<'a>>,
        skip_self_arg: bool,
    ) {
        let print_simple = |this: &mut Self| {
            let module = trait_id.module(this.db);
            let name = this.db.trait_signature(trait_id).name.as_str().to_owned();
            this.out.push('N');
            this.out.push('t'); // TypeNs
            this.print_module_path(module);
            this.push_disambiguator(0);
            this.push_ident(&name);
        };

        let Some(args) = args else {
            print_simple(self);
            return;
        };
        self.print_path_with_generic_args(print_simple, args, skip_self_arg);
    }

    fn print_dyn_existential(&mut self, predicates: BoundExistentialPredicates<'a>) {
        let Some(first) = predicates.iter().next() else {
            self.out.push('E');
            return;
        };
        self.wrap_binder(&first, |this, _| {
            for predicate in predicates.iter() {
                match predicate.skip_binder() {
                    ExistentialPredicate::Trait(trait_ref) => {
                        this.print_existential_trait_path(trait_ref.def_id.0, trait_ref.args);
                    }
                    ExistentialPredicate::Projection(projection) => {
                        this.out.push('p');
                        let assoc_name = this
                            .db
                            .type_alias_signature(projection.def_id.expect_type_alias())
                            .name
                            .as_str()
                            .to_owned();
                        this.push_ident(&assoc_name);
                        match projection.term.kind() {
                            TermKind::Ty(ty) => this.print_type(ty),
                            TermKind::Const(konst) => {
                                this.out.push('K');
                                this.print_const(konst);
                            }
                        }
                    }
                    ExistentialPredicate::AutoTrait(trait_id) => {
                        this.print_trait_path(trait_id.0, None);
                    }
                }
            }
        });
        self.out.push('E');
    }

    // -- Type encoding ------------------------------------------------------

    fn print_type(&mut self, ty: Ty<'a>) {
        let basic = match ty.kind() {
            TyKind::Bool => "b",
            TyKind::Char => "c",
            TyKind::Str => "e",
            TyKind::Never => "z",
            TyKind::Int(IntTy::I8) => "a",
            TyKind::Int(IntTy::I16) => "s",
            TyKind::Int(IntTy::I32) => "l",
            TyKind::Int(IntTy::I64) => "x",
            TyKind::Int(IntTy::I128) => "n",
            TyKind::Int(IntTy::Isize) => "i",
            TyKind::Uint(UintTy::U8) => "h",
            TyKind::Uint(UintTy::U16) => "t",
            TyKind::Uint(UintTy::U32) => "m",
            TyKind::Uint(UintTy::U64) => "y",
            TyKind::Uint(UintTy::U128) => "o",
            TyKind::Uint(UintTy::Usize) => "j",
            TyKind::Float(FloatTy::F16) => "C3f16",
            TyKind::Float(FloatTy::F32) => "f",
            TyKind::Float(FloatTy::F64) => "d",
            TyKind::Float(FloatTy::F128) => "C4f128",
            TyKind::Tuple(tys) if tys.is_empty() => "u", // unit
            _ => "",
        };
        if !basic.is_empty() {
            self.out.push_str(basic);
            return;
        }

        match ty.kind() {
            TyKind::Ref(region, inner_ty, mutbl) => {
                self.out.push_str(match mutbl {
                    Mutability::Not => "R",
                    Mutability::Mut => "Q",
                });
                if self.region_has_nontrivial_mangling(region) {
                    self.print_region(region);
                }
                self.print_type(inner_ty);
            }
            TyKind::RawPtr(inner_ty, mutbl) => {
                self.out.push_str(match mutbl {
                    Mutability::Not => "P",
                    Mutability::Mut => "O",
                });
                self.print_type(inner_ty);
            }
            TyKind::Slice(inner_ty) => {
                self.out.push('S');
                self.print_type(inner_ty);
            }
            TyKind::Array(inner_ty, len) => {
                self.out.push('A');
                self.print_type(inner_ty);
                self.print_const(len);
            }
            TyKind::Dynamic(predicates, region) => {
                self.out.push('D');
                self.print_dyn_existential(predicates);
                self.print_region(region);
            }
            TyKind::Tuple(tys) => {
                self.out.push('T');
                for ty in tys.iter() {
                    self.print_type(ty);
                }
                self.out.push('E');
            }
            TyKind::Adt(adt_def, substs) => {
                self.print_adt_path(adt_def.inner().id, substs);
            }
            TyKind::Param(param) => {
                // `p` alone collides when unrelated generic parameters share the
                // same index. Include both index and defining param identity.
                self.out.push('p');
                self.push_integer_62(param.index as u64);
                self.push_integer_62(stable_hash_bytes(format!("{:?}", param.id).as_bytes()));
            }
            // Fallback for anything else.
            _ => {
                self.out.push('u');
                self.push_integer_62(stable_hash_bytes(format!("{ty:?}").as_bytes()));
            }
        }
    }

    fn print_const(&mut self, konst: Const<'a>) {
        // Minimal const encoding for symbol uniqueness between distinct const
        // generic instantiations.
        self.out.push('p');
        self.push_integer_62(stable_hash_bytes(format!("{konst:?}").as_bytes()));
    }

    fn print_adt_path(&mut self, adt_id: AdtId, substs: GenericArgs<'a>) {
        let name = match adt_id {
            AdtId::StructId(id) => self.db.struct_signature(id).name.as_str().to_owned(),
            AdtId::EnumId(id) => self.db.enum_signature(id).name.as_str().to_owned(),
            AdtId::UnionId(id) => self.db.union_signature(id).name.as_str().to_owned(),
        };
        let module = match adt_id {
            AdtId::StructId(id) => id.module(self.db),
            AdtId::EnumId(id) => id.module(self.db),
            AdtId::UnionId(id) => id.module(self.db),
        };

        self.print_path_with_generic_args(
            |this| {
                this.out.push('N');
                this.out.push('t'); // TypeNs
                this.print_module_path(module);
                this.push_disambiguator(0);
                this.push_ident(&name);
            },
            substs,
            false,
        );
    }
}

// ---------------------------------------------------------------------------
// Free functions (ported from rustc v0.rs)
// ---------------------------------------------------------------------------

/// Base-62 encode into output: 0-9 a-z A-Z
fn base_62_encode(mut x: u64, output: &mut String) {
    if x == 0 {
        output.push('0');
        return;
    }
    let mut buf = [0u8; 16];
    let mut i = buf.len();
    while x > 0 {
        i -= 1;
        let digit = (x % 62) as u8;
        buf[i] = match digit {
            0..=9 => b'0' + digit,
            10..=35 => b'a' + (digit - 10),
            36..=61 => b'A' + (digit - 36),
            _ => unreachable!(),
        };
        x /= 62;
    }
    output.push_str(std::str::from_utf8(&buf[i..]).unwrap());
}

fn push_integer_62(x: u64, output: &mut String) {
    if let Some(x) = x.checked_sub(1) {
        base_62_encode(x, output);
    }
    output.push('_');
}

fn stable_hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn push_ident(ident: &str, output: &mut String) {
    let _ = write!(output, "{}", ident.len());
    // Separator `_` needed when ident starts with a digit or `_`.
    if let Some(first) = ident.bytes().next() {
        if first == b'_' || first.is_ascii_digit() {
            output.push('_');
        }
    }
    output.push_str(ident);
}
