{ pkgs ? import <nixpkgs> {} }:
(pkgs.buildFHSUserEnv {
  name = "rma-kinetics-nix-env";
  targetPkgs = pkgs: (with pkgs; [
    python313
    uv
    mpich
    #cudaPackages.cudatoolkit
  ]);
  runscript = "zsh";
}).env
