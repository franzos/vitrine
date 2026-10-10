use clap::Parser;
use vitrine::cli::{self, Cli};

fn main() -> anyhow::Result<()> {
    cli::run(Cli::parse())
}
