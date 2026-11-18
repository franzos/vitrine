//! The desktop portal's OpenURI interface, which also reaches the desktop from inside a sandbox.

use std::fs::File;
use std::io;
use std::path::Path;

use ashpd::desktop::open_uri::{OpenDirectoryRequest, OpenFileRequest};

/// Opens `path` in the desktop's default app for its type; returns once the portal answers.
pub fn open_file(path: &Path) -> io::Result<()> {
    let file = File::open(path)?;
    block_on(async {
        OpenFileRequest::default()
            .send_file(&file)
            .await?
            .response()
    })
}

/// Shows `path` selected in the file manager; returns once the portal answers.
pub fn open_directory(path: &Path) -> io::Result<()> {
    let file = File::open(path)?;
    block_on(async {
        OpenDirectoryRequest::default()
            .send(&file)
            .await?
            .response()
    })
}

fn block_on(request: impl Future<Output = ashpd::Result<()>>) -> io::Result<()> {
    ashpd::zbus::block_on(request).map_err(io::Error::other)
}
