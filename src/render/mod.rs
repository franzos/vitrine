// Copied from oculante, https://github.com/woelper/oculante (src/glow_renderer.rs).
// Copyright (c) 2020 Johann Woelper. MIT license, full text in NOTICE.

//! Images are drawn as a grid of GL texture tiles, so ones larger than the GPU's max texture size display.

use std::borrow::Cow;

use anyhow::{Result, anyhow};
use eframe::egui;
use eframe::glow::{self, HasContext};
use image::{ColorType, DynamicImage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexFormat {
    Rgba8,
    Rgb8,
    /// Gray in red, alpha in green
    Rg8,
    R8,
}

pub struct GlowTexture {
    pub texture: glow::Texture,
    pub width: u32,
    pub height: u32,
}

/// `Copy`, for egui paint callbacks that can't hold `&GlowRenderer`.
#[derive(Debug, Clone, Copy)]
pub struct ImageShader {
    program: glow::Program,
    vao: glow::VertexArray,
}

/// Target tile edge; effective size is `min(TILE_TARGET, max_texture_size)`.
/// 4096: an RGBA8 tile is 64 MiB (safe on any modern GPU), and images up to 4096x4096 fit one tile.
pub const TILE_TARGET: u32 = 4096;

/// `x`, `y` in image pixels.
pub struct GlowTile {
    pub texture: GlowTexture,
    pub x: u32,
    pub y: u32,
}

/// In image pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub struct GlowRenderer {
    image_program: glow::Program,
    /// Empty VAO; positions come from `gl_VertexID`
    quad_vao: glow::VertexArray,
    pub max_texture_size: u32,
}

// `compile_program` prepends the `#version` line for the GL flavour.
const VERTEX_SHADER: &str = r"
uniform vec2 u_offset;
uniform vec2 u_viewport;
// Quad size in pixels
uniform vec2 u_size;

out vec2 v_uv;

void main() {
    // Triangle strip: 0,1,2,3 -> quad corners
    vec2 pos = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1);
    v_uv = pos;

    // Position in pixels
    vec2 pixel_pos = u_offset + pos * u_size;
    // Convert to clip space: [-1, 1]
    vec2 clip = (pixel_pos / u_viewport) * 2.0 - 1.0;
    clip.y = -clip.y; // flip Y (screen coords -> GL)
    gl_Position = vec4(clip, 0.0, 1.0);
}
";

const IMAGE_FRAGMENT_SHADER: &str = r"
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform mat4 u_swizzle_mat;
uniform vec4 u_offset_vec;
out vec4 color;
void main() {
    color = (u_swizzle_mat * texture(u_texture, v_uv)) + u_offset_vec;
}
";

impl GlowRenderer {
    /// Call once per GL context.
    pub fn new(gl: &glow::Context) -> Result<Self> {
        let image_program = compile_program(gl, VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)?;
        // SAFETY: called on the thread that owns the current GL context.
        let (quad_vao, max_texture_size) = unsafe {
            let vao = gl.create_vertex_array().map_err(|e| anyhow!(e))?;
            (vao, gl.get_parameter_i32(glow::MAX_TEXTURE_SIZE))
        };
        Ok(Self {
            image_program,
            quad_vao,
            max_texture_size: u32::try_from(max_texture_size).unwrap_or(0),
        })
    }

    #[must_use]
    pub fn image_shader(&self) -> ImageShader {
        ImageShader {
            program: self.image_program,
            vao: self.quad_vao,
        }
    }

    /// Uploads a region of a larger image, read in place (no copy).
    fn create_texture_from(
        gl: &glow::Context,
        bytes: &[u8],
        region: Region,
        width: u32,
        height: u32,
        format: TexFormat,
    ) -> Result<GlowTexture> {
        region.assert_inside(bytes, width, height, format);
        // SAFETY: region checked to lie inside `bytes`; unpack state reset before returning (egui expects defaults).
        unsafe {
            let texture = gl.create_texture().map_err(|e| anyhow!(e))?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));

            let (internal, fmt, typ) = gl_format(format);
            set_unpack_region(gl, Some(region));
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                internal as i32,
                width as i32,
                height as i32,
                0,
                fmt,
                typ,
                glow::PixelUnpackData::Slice(Some(bytes)),
            );
            set_unpack_region(gl, None);

            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_S,
                glow::CLAMP_TO_EDGE as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_WRAP_T,
                glow::CLAMP_TO_EDGE as i32,
            );
            apply_filter(gl);

            gl.bind_texture(glow::TEXTURE_2D, None);

            Ok(GlowTexture {
                texture,
                width,
                height,
            })
        }
    }

    pub fn delete_texture(gl: &glow::Context, tex: GlowTexture) {
        // SAFETY: called on the thread that owns the current GL context.
        unsafe {
            gl.delete_texture(tex.texture);
        }
    }

    pub fn destroy(&self, gl: &glow::Context) {
        // SAFETY: called on the thread that owns the current GL context.
        unsafe {
            gl.delete_program(self.image_program);
            gl.delete_vertex_array(self.quad_vao);
        }
    }

    #[must_use]
    pub fn tile_size(&self) -> u32 {
        TILE_TARGET.min(self.max_texture_size).max(1)
    }

    /// The tiles an image of `width` x `height` is split into, row by row.
    #[must_use]
    pub fn tile_regions(&self, width: u32, height: u32) -> Vec<TileRegion> {
        tile_regions(self.tile_size(), width, height)
    }

    /// Read in place from the full-image buffer (no copy).
    pub fn create_tile(
        gl: &glow::Context,
        bytes: &[u8],
        image_width: u32,
        tile: TileRegion,
        format: TexFormat,
    ) -> Result<GlowTile> {
        let region = Region {
            row_length: image_width,
            x: tile.x,
            y: tile.y,
        };
        let texture =
            Self::create_texture_from(gl, bytes, region, tile.width, tile.height, format)?;
        Ok(GlowTile {
            texture,
            x: tile.x,
            y: tile.y,
        })
    }

    pub fn delete_tiles(gl: &glow::Context, tiles: Vec<GlowTile>) {
        for tile in tiles {
            Self::delete_texture(gl, tile.texture);
        }
    }
}

fn tile_regions(cap: u32, width: u32, height: u32) -> Vec<TileRegion> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let cols = width.div_ceil(cap);
    let rows = height.div_ceil(cap);
    let mut tiles = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        let y = row * cap;
        let height = cap.min(height - y);
        for col in 0..cols {
            let x = col * cap;
            let width = cap.min(width - x);
            tiles.push(TileRegion {
                x,
                y,
                width,
                height,
            });
        }
    }
    tiles
}

impl GlowTile {
    #[must_use]
    pub fn quad(&self, offset: [f32; 2], scale: f32) -> Quad {
        Quad {
            texture: self.texture.texture,
            pos: [
                offset[0] + self.x as f32 * scale,
                offset[1] + self.y as f32 * scale,
            ],
            size: [
                self.texture.width as f32 * scale,
                self.texture.height as f32 * scale,
            ],
        }
    }
}

/// A textured rectangle on screen, in egui points.
#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub texture: glow::Texture,
    pub pos: [f32; 2],
    pub size: [f32; 2],
}

/// Draws textured quads from an egui paint callback: one state setup, one draw call per quad.
pub fn paint_quads(
    gl: &glow::Context,
    shader: ImageShader,
    info: &egui::PaintCallbackInfo,
    format: TexFormat,
    quads: &[Quad],
) {
    if quads.is_empty() {
        return;
    }
    let ImageShader { program, vao } = shader;
    let (swizzle_mat, offset_vec) = swizzle(format);
    let screen_size_px = info.screen_size_px;
    // Quads are positioned in egui points, relative to the whole window
    let viewport = [
        screen_size_px[0] as f32 / info.pixels_per_point,
        screen_size_px[1] as f32 / info.pixels_per_point,
    ];
    // SAFETY: egui_glow runs paint callbacks on the GL thread with its context current.
    unsafe {
        gl.viewport(0, 0, screen_size_px[0] as i32, screen_size_px[1] as i32);
        gl.enable(glow::BLEND);
        // Keep the window's alpha; translucent pixels would make the window see-through on Wayland
        gl.blend_func_separate(
            glow::SRC_ALPHA,
            glow::ONE_MINUS_SRC_ALPHA,
            glow::ZERO,
            glow::ONE,
        );

        gl.use_program(Some(program));
        gl.bind_vertex_array(Some(vao));

        set_uniform_2f(gl, program, "u_viewport", viewport);

        let mat_loc = gl.get_uniform_location(program, "u_swizzle_mat");
        gl.uniform_matrix_4_f32_slice(mat_loc.as_ref(), false, &swizzle_mat);
        let off_loc = gl.get_uniform_location(program, "u_offset_vec");
        gl.uniform_4_f32_slice(off_loc.as_ref(), &offset_vec);

        gl.active_texture(glow::TEXTURE0);
        let tex_loc = gl.get_uniform_location(program, "u_texture");
        gl.uniform_1_i32(tex_loc.as_ref(), 0);

        for quad in quads {
            set_uniform_2f(gl, program, "u_offset", quad.pos);
            set_uniform_2f(gl, program, "u_size", quad.size);
            gl.bind_texture(glow::TEXTURE_2D, Some(quad.texture));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        }

        gl.bind_texture(glow::TEXTURE_2D, None);
        gl.use_program(None);
        gl.bind_vertex_array(None);
    }
}

unsafe fn apply_filter(gl: &glow::Context) {
    unsafe {
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MIN_FILTER,
            glow::LINEAR_MIPMAP_LINEAR as i32,
        );
        gl.tex_parameter_i32(
            glow::TEXTURE_2D,
            glow::TEXTURE_MAG_FILTER,
            glow::LINEAR as i32,
        );
        gl.generate_mipmap(glow::TEXTURE_2D);
    }
}

#[derive(Debug, Clone, Copy)]
struct Region {
    /// Width of the whole image in pixels
    row_length: u32,
    x: u32,
    y: u32,
}

impl Region {
    /// Panics unless the whole region lies in `bytes`; GL reads it through a raw pointer.
    fn assert_inside(&self, bytes: &[u8], width: u32, height: u32, format: TexFormat) {
        if width == 0 || height == 0 {
            return;
        }
        let last_row = (self.y + height - 1) as usize * self.row_length as usize;
        let needed = (last_row + (self.x + width) as usize) * bytes_per_pixel(format);
        assert!(
            self.x + width <= self.row_length && bytes.len() >= needed,
            "texture region {self:?} of {width}x{height} {format:?} needs {needed} bytes, the image has {}",
            bytes.len()
        );
    }
}

/// Points the next upload at `region` of its buffer, or resets to the defaults egui's own uploads expect.
unsafe fn set_unpack_region(gl: &glow::Context, region: Option<Region>) {
    let region = region.unwrap_or(Region {
        row_length: 0,
        x: 0,
        y: 0,
    });
    unsafe {
        // Rows of 1-3 bytes per pixel needn't end on 4-byte boundaries
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.pixel_store_i32(glow::UNPACK_ROW_LENGTH, region.row_length as i32);
        gl.pixel_store_i32(glow::UNPACK_SKIP_PIXELS, region.x as i32);
        gl.pixel_store_i32(glow::UNPACK_SKIP_ROWS, region.y as i32);
    }
}

fn bytes_per_pixel(format: TexFormat) -> usize {
    match format {
        TexFormat::Rgba8 => 4,
        TexFormat::Rgb8 => 3,
        TexFormat::Rg8 => 2,
        TexFormat::R8 => 1,
    }
}

fn gl_format(format: TexFormat) -> (u32, u32, u32) {
    match format {
        TexFormat::Rgba8 => (glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE),
        TexFormat::Rgb8 => (glow::RGB8, glow::RGB, glow::UNSIGNED_BYTE),
        TexFormat::Rg8 => (glow::RG8, glow::RG, glow::UNSIGNED_BYTE),
        TexFormat::R8 => (glow::R8, glow::RED, glow::UNSIGNED_BYTE),
    }
}

unsafe fn set_uniform_2f(gl: &glow::Context, program: glow::Program, name: &str, v: [f32; 2]) {
    unsafe {
        let loc = gl.get_uniform_location(program, name);
        gl.uniform_2_f32(loc.as_ref(), v[0], v[1]);
    }
}

fn compile_program(
    gl: &glow::Context,
    vertex_src: &str,
    fragment_src: &str,
) -> Result<glow::Program> {
    // Desktop GL and GLES need different headers, the shader bodies work for both
    let header = if gl.version().is_embedded {
        "#version 300 es"
    } else {
        "#version 330"
    };
    let vertex_src = &format!("{header}{vertex_src}");
    let fragment_src = &format!("{header}{fragment_src}");
    // SAFETY: called on the thread that owns the current GL context.
    unsafe {
        let program = gl.create_program().map_err(|e| anyhow!(e))?;

        let vs = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(|e| anyhow!(e))?;
        gl.shader_source(vs, vertex_src);
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            return Err(anyhow!(
                "vertex shader error: {}",
                gl.get_shader_info_log(vs)
            ));
        }

        let fs = gl
            .create_shader(glow::FRAGMENT_SHADER)
            .map_err(|e| anyhow!(e))?;
        gl.shader_source(fs, fragment_src);
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            return Err(anyhow!(
                "fragment shader error: {}",
                gl.get_shader_info_log(fs)
            ));
        }

        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            return Err(anyhow!(
                "program link error: {}",
                gl.get_program_info_log(program)
            ));
        }

        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        Ok(program)
    }
}

/// Channels are kept, at 8 bits each: gray takes a quarter of RGBA's memory, and more than 8 bits isn't visible on screen.
#[must_use]
pub fn texture_layout(img: &DynamicImage) -> (TexFormat, Cow<'_, DynamicImage>) {
    let img = match img.color() {
        ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => Cow::Borrowed(img),
        ColorType::L16 => Cow::Owned(DynamicImage::ImageLuma8(img.to_luma8())),
        ColorType::La16 => Cow::Owned(DynamicImage::ImageLumaA8(img.to_luma_alpha8())),
        ColorType::Rgb16 | ColorType::Rgb32F => Cow::Owned(DynamicImage::ImageRgb8(img.to_rgb8())),
        _ => Cow::Owned(DynamicImage::ImageRgba8(img.to_rgba8())),
    };
    let format = match img.color() {
        ColorType::L8 => TexFormat::R8,
        ColorType::La8 => TexFormat::Rg8,
        ColorType::Rgb8 => TexFormat::Rgb8,
        _ => TexFormat::Rgba8,
    };
    (format, img)
}

#[must_use]
pub fn into_texture_layout(img: DynamicImage) -> DynamicImage {
    let converted = match texture_layout(&img).1 {
        Cow::Owned(converted) => Some(converted),
        Cow::Borrowed(_) => None,
    };
    converted.unwrap_or(img)
}

/// The column-major swizzle matrix and offset vector that show a texture of `format` as RGBA.
fn swizzle(format: TexFormat) -> ([f32; 16], [f32; 4]) {
    const GRAY: [f32; 4] = [1.0, 1.0, 1.0, 0.0];
    const ALPHA: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
    const ZERO: [f32; 4] = [0.0; 4];
    let cols = |c: [[f32; 4]; 4]| c.concat().try_into().unwrap_or([0.0; 16]);
    match format {
        // gray is in the red channel, alpha comes from the offset
        TexFormat::R8 => (cols([GRAY, ZERO, ZERO, ZERO]), ALPHA),
        TexFormat::Rg8 => (cols([GRAY, ALPHA, ZERO, ZERO]), ZERO),
        // a texture without alpha is sampled with an alpha of one
        TexFormat::Rgb8 | TexFormat::Rgba8 => (
            cols([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                ALPHA,
            ]),
            ZERO,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shown(format: TexFormat, texel: [f32; 4]) -> [f32; 4] {
        let (mat, vec) = swizzle(format);
        std::array::from_fn(|row| {
            (0..4)
                .map(|col| mat[col * 4 + row] * texel[col])
                .sum::<f32>()
                + vec[row]
        })
    }

    #[test]
    fn images_keep_their_channels() {
        let layout = |img: DynamicImage| texture_layout(&img).0;
        assert_eq!(layout(DynamicImage::new_luma8(3, 3)), TexFormat::R8);
        assert_eq!(layout(DynamicImage::new_luma16(3, 3)), TexFormat::R8);
        assert_eq!(layout(DynamicImage::new_luma_a8(3, 3)), TexFormat::Rg8);
        assert_eq!(layout(DynamicImage::new_luma_a16(3, 3)), TexFormat::Rg8);
        assert_eq!(layout(DynamicImage::new_rgb8(3, 3)), TexFormat::Rgb8);
        assert_eq!(layout(DynamicImage::new_rgb16(3, 3)), TexFormat::Rgb8);
        assert_eq!(layout(DynamicImage::new_rgb32f(3, 3)), TexFormat::Rgb8);
        assert_eq!(layout(DynamicImage::new_rgba8(3, 3)), TexFormat::Rgba8);
        assert_eq!(layout(DynamicImage::new_rgba16(3, 3)), TexFormat::Rgba8);
        assert_eq!(layout(DynamicImage::new_rgba32f(3, 3)), TexFormat::Rgba8);
    }

    #[test]
    fn layout_has_one_byte_per_channel() {
        for img in [
            DynamicImage::new_luma16(5, 3),
            DynamicImage::new_luma_a16(5, 3),
            DynamicImage::new_rgb32f(5, 3),
            DynamicImage::new_rgba16(5, 3),
        ] {
            let (format, layout) = texture_layout(&img);
            assert_eq!(layout.as_bytes().len(), 5 * 3 * bytes_per_pixel(format));
            let converted = into_texture_layout(img);
            assert_eq!(converted.as_bytes().len(), 5 * 3 * bytes_per_pixel(format));
        }
    }

    #[test]
    fn gray_is_shown_on_all_color_channels() {
        // a gray texture is sampled as (gray, 0, 0, 1)
        assert_eq!(
            shown(TexFormat::R8, [0.5, 0.0, 0.0, 1.0]),
            [0.5, 0.5, 0.5, 1.0]
        );
    }

    #[test]
    fn gray_with_alpha_keeps_its_alpha() {
        // sampled as (gray, alpha, 0, 1)
        assert_eq!(
            shown(TexFormat::Rg8, [0.5, 0.25, 0.0, 1.0]),
            [0.5, 0.5, 0.5, 0.25]
        );
    }

    #[test]
    fn color_is_shown_as_is() {
        let texel = [0.1, 0.2, 0.3, 0.4];
        assert_eq!(shown(TexFormat::Rgba8, texel), texel);
        assert_eq!(
            shown(TexFormat::Rgb8, [0.1, 0.2, 0.3, 1.0]),
            [0.1, 0.2, 0.3, 1.0]
        );
    }

    #[test]
    fn regions_must_be_inside_the_image() {
        // a 10x4 gray image, split into a tile of 8 and one of 2 pixels width
        let bytes = vec![0u8; 10 * 4];
        let region = |x| Region {
            row_length: 10,
            x,
            y: 0,
        };
        region(0).assert_inside(&bytes, 8, 4, TexFormat::R8);
        region(8).assert_inside(&bytes, 2, 4, TexFormat::R8);
        // wider than the image, or more channels than the buffer has
        assert!(
            std::panic::catch_unwind(|| region(8).assert_inside(&bytes, 3, 4, TexFormat::R8))
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| region(0).assert_inside(&bytes, 8, 4, TexFormat::Rgb8))
                .is_err()
        );
    }

    #[test]
    fn large_images_are_split_into_tiles() {
        let tiles = tile_regions(4096, 12000, 9000);
        assert_eq!(tiles.len(), 9);
        assert_eq!(
            tiles[0],
            TileRegion {
                x: 0,
                y: 0,
                width: 4096,
                height: 4096
            }
        );
        assert_eq!(
            tiles[8],
            TileRegion {
                x: 8192,
                y: 8192,
                width: 3808,
                height: 808
            }
        );
        let area: u64 = tiles
            .iter()
            .map(|t| u64::from(t.width) * u64::from(t.height))
            .sum();
        assert_eq!(area, 12000 * 9000);
        assert_eq!(tile_regions(4096, 640, 480).len(), 1);
        assert!(tile_regions(4096, 0, 480).is_empty());
    }
}
