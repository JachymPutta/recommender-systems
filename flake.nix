
{
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs";
  };

  outputs = {
    self,
    flake-utils,
    nixpkgs,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = (import nixpkgs) {
          inherit system;
        };
      in rec {
        devShell = pkgs.mkShell {
          nativeBuildInputs = with pkgs; [
            (pkgs.python312.withPackages (
              python-pkgs: with python-pkgs; [
                python-dotenv
                flake8
                pandas
                numpy
                scipy
                # implicit TODO: doesn't exist
                # surprise
                scikit-learn
                torch
                transformers
              ])
            )
            just
          ];
        };
      }
    );
}

