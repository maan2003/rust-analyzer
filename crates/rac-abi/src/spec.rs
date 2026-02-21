//! Minimal target specification types needed by callconv code.

use std::borrow::Cow;

/// Target architecture.
///
/// Subset of rustc_target::spec::Arch — add variants as needed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arch {
    AArch64,
    AmdGpu,
    Arm,
    Arm64EC,
    Avr,
    Bpf,
    CSky,
    Hexagon,
    LoongArch32,
    LoongArch64,
    M68k,
    Mips,
    Mips32r6,
    Mips64,
    Mips64r6,
    Msp430,
    Nvptx64,
    PowerPC,
    PowerPC64,
    RiscV32,
    RiscV64,
    S390x,
    Sparc,
    Sparc64,
    SpirV,
    Wasm32,
    Wasm64,
    X86,
    X86_64,
    Xtensa,
    Other(Cow<'static, str>),
}

impl std::fmt::Display for Arch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Arch::AArch64 => write!(f, "aarch64"),
            Arch::AmdGpu => write!(f, "amdgpu"),
            Arch::Arm => write!(f, "arm"),
            Arch::Arm64EC => write!(f, "arm64ec"),
            Arch::Avr => write!(f, "avr"),
            Arch::Bpf => write!(f, "bpf"),
            Arch::CSky => write!(f, "csky"),
            Arch::Hexagon => write!(f, "hexagon"),
            Arch::LoongArch32 => write!(f, "loongarch32"),
            Arch::LoongArch64 => write!(f, "loongarch64"),
            Arch::M68k => write!(f, "m68k"),
            Arch::Mips => write!(f, "mips"),
            Arch::Mips32r6 => write!(f, "mips32r6"),
            Arch::Mips64 => write!(f, "mips64"),
            Arch::Mips64r6 => write!(f, "mips64r6"),
            Arch::Msp430 => write!(f, "msp430"),
            Arch::Nvptx64 => write!(f, "nvptx64"),
            Arch::PowerPC => write!(f, "powerpc"),
            Arch::PowerPC64 => write!(f, "powerpc64"),
            Arch::RiscV32 => write!(f, "riscv32"),
            Arch::RiscV64 => write!(f, "riscv64"),
            Arch::S390x => write!(f, "s390x"),
            Arch::Sparc => write!(f, "sparc"),
            Arch::Sparc64 => write!(f, "sparc64"),
            Arch::SpirV => write!(f, "spirv"),
            Arch::Wasm32 => write!(f, "wasm32"),
            Arch::Wasm64 => write!(f, "wasm64"),
            Arch::X86 => write!(f, "x86"),
            Arch::X86_64 => write!(f, "x86_64"),
            Arch::Xtensa => write!(f, "xtensa"),
            Arch::Other(s) => write!(f, "{s}"),
        }
    }
}

/// Target ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Abi {
    SoftFloat,
    Other(Cow<'static, str>),
}

/// Target environment (libc).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Env {
    Gnu,
    Musl,
    Uclibc,
    Other(Cow<'static, str>),
}

/// Target operating system.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Os {
    Linux,
    FreeBsd,
    Aix,
    Other(Cow<'static, str>),
}

/// Rustc-specific ABI overrides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RustcAbi {
    X86Sse2,
    Softfloat,
}

/// Minimal target specification.
#[derive(Clone, Debug)]
pub struct Target {
    pub arch: Arch,
    pub os: Os,
    pub env: Env,
    pub abi: Abi,
    pub rustc_abi: Option<RustcAbi>,
    pub llvm_target: String,
    pub llvm_abiname: String,
    pub pointer_width: u32,
    pub abi_return_struct_as_int: bool,
    pub is_like_darwin: bool,
    pub is_like_windows: bool,
    pub is_like_msvc: bool,
    pub simd_types_indirect: bool,
}

pub trait HasTargetSpec {
    fn target_spec(&self) -> &Target;
}

impl HasTargetSpec for Target {
    #[inline]
    fn target_spec(&self) -> &Target {
        self
    }
}

/// x86 (32-bit) ABI options.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct X86Abi {
    pub regparm: Option<u32>,
    pub reg_struct_return: bool,
}

pub trait HasX86AbiOpt {
    fn x86_abi_opt(&self) -> X86Abi;
}
