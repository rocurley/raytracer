// For reading and opening files
use anyhow::Result;
use ordered_float::NotNan;
use png;
use std::convert::TryInto;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, Sub, SubAssign};
use std::path::Path;

fn main() {
    let scene = Scene {
        objects: vec![
            Sphere {
                pt: V3::new(0., 0., 20.),
                r2: 100.0.try_into().unwrap(),
                color: [255, 0, 0],
            },
            Sphere {
                pt: V3::new(0., 0., 10.),
                r2: 9.0.try_into().unwrap(),
                color: [0, 255, 0],
            },
            Sphere {
                pt: V3::new(1., 1., 3.),
                r2: 1.0.try_into().unwrap(),
                color: [0, 0, 255],
            },
        ],
        camera: Camera {},
    };
    let image = scene.render();
    image.write("out.png").unwrap();
}

struct Image([u8; 4_000000]);

impl Image {
    fn new() -> Self {
        let mut image = Image([0; 4_000000]);
        for pixel in image.0.chunks_exact_mut(4) {
            pixel[3] = 255; // Alpha
        }
        image
    }
    fn write(&self, path: &str) -> Result<()> {
        let path = Path::new(path);
        let file = File::create(path)?;
        let ref mut w = BufWriter::new(file);

        let mut encoder = png::Encoder::new(w, 1000, 1000);
        encoder.set_color(png::ColorType::RGBA);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&self.0)?;
        Ok(())
    }
}

impl Index<(usize, usize)> for Image {
    type Output = [u8; 3];

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let i = (y * 1000 + x) * 4;
        self.0[i..i + 3].try_into().unwrap()
    }
}

impl IndexMut<(usize, usize)> for Image {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        let i = (y * 1000 + x) * 4;
        (&mut self.0[i..i + 3]).try_into().unwrap()
    }
}

struct Scene {
    objects: Vec<Sphere>,
    camera: Camera,
}

impl Scene {
    fn render(&self) -> Image {
        let mut out = Image::new();
        for (x, y, ray) in self.camera.rays() {
            let color = self
                .objects
                .iter()
                .filter_map(|obj| {
                    let (t, _) = obj.intersect(&ray)?;
                    Some((t, obj.color))
                })
                .min()
                .map_or([0; 3], |(_, color)| color);
            out[(x, y)] = color;
        }
        out
    }
}

struct Sphere {
    r2: NotNan<f64>, // squared
    pt: V3,
    color: [u8; 3],
}

struct Camera {}

impl Camera {
    fn rays(&self) -> impl Iterator<Item = (usize, usize, Ray)> {
        (0..1000).flat_map(|i| {
            let x = (i as f64 - 500.0) * PI / 2.0 / 1000.0;
            (0..1000).map(move |j| {
                let y = (j as f64 - 500.0) * PI / 2.0 / 1000.0;
                let theta = (x * x + y * y).sqrt();
                let phi = y.atan2(x);
                let z = theta.cos();
                let x = theta.sin() * phi.cos();
                let y = theta.sin() * phi.sin();
                let orientation = V3::new(x, y, z);
                let ray = Ray {
                    orientation,
                    origin: V3::new(0., 0., 0.),
                };
                (i, j, ray)
            })
        })
    }
}

struct Ray {
    origin: V3,
    orientation: V3,
}

impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<(NotNan<f64>, V3)> {
        let center = self.pt - ray.origin;
        let t_closest = center.dot(ray.orientation);
        if t_closest <= 0.0.try_into().unwrap() {
            return None;
        }
        let closest = t_closest * ray.orientation;
        let d2_closest = (center - closest).norm2();
        if d2_closest > self.r2 {
            return None;
        }
        let closest_depth = (self.r2 - d2_closest).sqrt();
        let t = t_closest - closest_depth;
        let pt = ray.origin + ray.orientation * t;
        Some((t, pt))
    }
}

#[derive(Clone, Copy, Debug)]
struct V3([NotNan<f64>; 3]);

impl V3 {
    fn new(x: f64, y: f64, z: f64) -> Self {
        V3([
            x.try_into().unwrap(),
            y.try_into().unwrap(),
            z.try_into().unwrap(),
        ])
    }
    fn dot(self, other: Self) -> NotNan<f64> {
        self.0[0] * other.0[0] + self.0[1] * other.0[1] + self.0[2] * other.0[2]
    }
    fn norm2(self) -> NotNan<f64> {
        self.dot(self)
    }
    fn norm(self) -> NotNan<f64> {
        self.norm2().sqrt().try_into().unwrap()
    }
}

impl Add for V3 {
    type Output = V3;
    fn add(self, other: Self) -> Self::Output {
        V3([
            self.0[0] + other.0[0],
            self.0[1] + other.0[1],
            self.0[2] + other.0[2],
        ])
    }
}

impl Sub for V3 {
    type Output = V3;
    fn sub(self, other: Self) -> Self::Output {
        V3([
            self.0[0] - other.0[0],
            self.0[1] - other.0[1],
            self.0[2] - other.0[2],
        ])
    }
}

impl Mul<NotNan<f64>> for V3 {
    type Output = V3;
    fn mul(self, other: NotNan<f64>) -> Self::Output {
        V3([self.0[0] * other, self.0[1] * other, self.0[2] * other])
    }
}

impl Mul<V3> for NotNan<f64> {
    type Output = V3;
    fn mul(self, other: V3) -> Self::Output {
        V3([other.0[0] * self, other.0[1] * self, other.0[2] * self])
    }
}
