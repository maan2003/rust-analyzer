{ ... }:

{
    cachix.enable = false;
    languages.rust.enable = true;
    languages.rust.toolchainFile = ./cg_clif/rust-toolchain.toml;
}
