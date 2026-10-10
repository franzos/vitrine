use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

use eframe::egui::{ColorImage, Context, TextureHandle, TextureOptions};

use super::cache::Lru;
use super::loader::decode;
use crate::catalog::{self, faces::FaceId};

const TEXTURES: usize = 512;

type Loaded = (FaceId, Result<ColorImage, String>);

pub struct Crops {
    catalog_db: PathBuf,
    worker: Option<(Sender<FaceId>, Receiver<Loaded>)>,
    textures: Lru<FaceId, TextureHandle>,
    pending: HashSet<FaceId>,
    failed: HashSet<FaceId>,
}

impl Crops {
    #[must_use]
    pub fn new(catalog_db: PathBuf) -> Self {
        Self {
            catalog_db,
            worker: None,
            textures: Lru::new(TEXTURES),
            pending: HashSet::new(),
            failed: HashSet::new(),
        }
    }

    /// Failed crops aren't retried.
    pub fn get(&mut self, ctx: &Context, face: FaceId) -> Option<TextureHandle> {
        self.receive(ctx);
        if let Some(texture) = self.textures.get(&face) {
            return Some(texture.clone());
        }
        if !self.pending.contains(&face)
            && !self.failed.contains(&face)
            && let Some((tx, _)) = self.worker(ctx)
            && tx.send(face).is_ok()
        {
            self.pending.insert(face);
        }
        None
    }

    fn receive(&mut self, ctx: &Context) {
        let Some((_, rx)) = &self.worker else {
            return;
        };
        for (face, image) in rx.try_iter() {
            self.pending.remove(&face);
            match image {
                Ok(image) => {
                    let texture = ctx.load_texture("face", image, TextureOptions::LINEAR);
                    self.textures.insert(face, texture);
                }
                Err(_) => {
                    self.failed.insert(face);
                }
            }
        }
    }

    /// Starts the worker on first use; it ends when `self` is dropped.
    fn worker(&mut self, ctx: &Context) -> Option<&(Sender<FaceId>, Receiver<Loaded>)> {
        if self.worker.is_none() {
            let (requests, rx) = mpsc::channel();
            let (tx, results) = mpsc::channel();
            let db = self.catalog_db.clone();
            let ctx = ctx.clone();
            thread::Builder::new()
                .name("face-crops".to_owned())
                .spawn(move || work(&db, &rx, &tx, &ctx))
                .ok()?;
            self.worker = Some((requests, results));
        }
        self.worker.as_ref()
    }
}

fn work(catalog_db: &std::path::Path, rx: &Receiver<FaceId>, tx: &Sender<Loaded>, ctx: &Context) {
    let conn = catalog::open_read_only(catalog_db).map_err(|e| e.to_string());
    for face in rx {
        let image = conn.as_ref().map_err(Clone::clone).and_then(|conn| {
            let jpeg = catalog::faces::crop(conn, face)
                .map_err(|e| e.to_string())?
                .ok_or_else(|| "face is gone".to_owned())?;
            decode(&jpeg).map_err(|e| e.to_string())
        });
        if tx.send((face, image)).is_err() {
            return;
        }
        ctx.request_repaint();
    }
}
