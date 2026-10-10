use std::path::{Path, PathBuf};

use eframe::egui::{self, collapsing_header::CollapsingState};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Folder {
    pub path: PathBuf,
    pub name: String,
    /// Entries in this folder and all folders below it.
    pub count: u64,
    /// Sorted by name.
    pub children: Vec<Folder>,
}

impl Folder {
    fn new(path: PathBuf, name: String) -> Self {
        Self {
            path,
            name,
            count: 0,
            children: Vec::new(),
        }
    }

    /// Folders outside `root` are skipped.
    #[must_use]
    pub fn build(root: &Path, counts: &[(PathBuf, u64)]) -> Self {
        let mut tree = Self::new(root.to_owned(), root.display().to_string());
        for (dir, n) in counts {
            let Ok(rel) = dir.strip_prefix(root) else {
                continue;
            };
            tree.count += n;
            let mut node = &mut tree;
            for part in rel {
                let name = part.to_string_lossy();
                let at = match node
                    .children
                    .binary_search_by(|c| c.name.as_str().cmp(&name))
                {
                    Ok(at) => at,
                    Err(at) => {
                        let path = node.path.join(part);
                        node.children.insert(at, Self::new(path, name.into_owned()));
                        at
                    }
                };
                node = &mut node.children[at];
                node.count += n;
            }
        }
        tree
    }

    pub fn label(&self) -> String {
        format!("{} ({})", self.name, self.count)
    }
}

pub fn show(ui: &mut egui::Ui, folders: &[Folder], selected: Option<&Path>) -> Option<PathBuf> {
    let mut clicked = None;
    for folder in folders {
        node(ui, folder, selected, true, &mut clicked);
    }
    clicked
}

fn node(
    ui: &mut egui::Ui,
    folder: &Folder,
    selected: Option<&Path>,
    open: bool,
    clicked: &mut Option<PathBuf>,
) {
    let is_selected = selected == Some(folder.path.as_path());
    if folder.children.is_empty() {
        ui.horizontal(|ui| {
            ui.add_space(ui.spacing().icon_width + ui.spacing().item_spacing.x);
            if ui.selectable_label(is_selected, folder.label()).clicked() {
                *clicked = Some(folder.path.clone());
            }
        });
        return;
    }
    let id = ui.make_persistent_id(&folder.path);
    CollapsingState::load_with_default_open(ui.ctx(), id, open)
        .show_header(ui, |ui| {
            if ui.selectable_label(is_selected, folder.label()).clicked() {
                *clicked = Some(folder.path.clone());
            }
        })
        .body(|ui| {
            for child in &folder.children {
                node(ui, child, selected, false, clicked);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_nests_folders_with_recursive_counts() {
        let counts = [
            (PathBuf::from("/p"), 1),
            (PathBuf::from("/p/2024"), 2),
            (PathBuf::from("/p/2024/01"), 3),
            (PathBuf::from("/p/2023/12"), 4),
            (PathBuf::from("/q"), 9),
        ];
        let tree = Folder::build(Path::new("/p"), &counts);
        assert_eq!((tree.name.as_str(), tree.count), ("/p", 10));
        let names: Vec<_> = tree
            .children
            .iter()
            .map(|c| (c.name.as_str(), c.count))
            .collect();
        assert_eq!(names, [("2023", 4), ("2024", 5)]);
        let jan = &tree.children[1].children[0];
        assert_eq!(jan.path, Path::new("/p/2024/01"));
        assert_eq!(jan.label(), "01 (3)");
        assert_eq!(tree.children[0].children[0].name, "12");
    }
}
