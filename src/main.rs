// For reading and opening files
use anyhow::Result;
use cpuprofiler::PROFILER;
use ordered_float::NotNan;
use png;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, UnitSphere};
use serde::Deserialize;
use std::convert::TryInto;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufWriter;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};
use std::path::Path;

// TODO:
// * Investigate "sparkles"
// * Show rendering live
// * Profiling!

fn main() {
    let scene_str = std::fs::read_to_string("scene.toml").unwrap();
    let mut scene: Scene = toml::de::from_str(&scene_str).unwrap();
    println!("Read scene");
    PROFILER.lock().unwrap().start("./profile.pprof").unwrap();
    let image = scene.render();
    image.write("out.png").unwrap();
    PROFILER.lock().unwrap().stop().unwrap();
}

const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;

struct Image(Box<[Color]>);

const BLACK: Color = unsafe { unchecked_new_color([0.0; 3]) };
const WHITE: Color = unsafe { unchecked_new_color([1.0; 3]) };

impl Image {
    fn new() -> Self {
        let v = vec![BLACK; WIDTH * HEIGHT];
        Image(v.into_boxed_slice())
    }
    fn write(&self, path: &str) -> Result<()> {
        let mut buffer = [0; 4 * WIDTH * HEIGHT];
        let mut saturations: Vec<NotNan<f64>> =
            self.0.iter().flat_map(|px| px.iter()).copied().collect();
        saturations.sort();
        let max_exposure = saturations[saturations.len() * 19 / 20].into_inner();
        for (b_px, px) in buffer.chunks_exact_mut(4).zip(self.0.iter()) {
            for i in 0..3 {
                b_px[i] = (px[i].into_inner() * 255.0 / max_exposure) as u8;
            }
            b_px[3] = 255;
        }
        let path = Path::new(path);
        let file = File::create(path)?;
        let ref mut w = BufWriter::new(file);

        let mut encoder = png::Encoder::new(w, 1000, 1000);
        encoder.set_color(png::ColorType::RGBA);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(&buffer)?;
        Ok(())
    }
}

impl Index<(usize, usize)> for Image {
    type Output = [NotNan<f64>; 3];

    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        let i = y * WIDTH + x;
        &self.0[i]
    }
}

impl IndexMut<(usize, usize)> for Image {
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        let i = y * WIDTH + x;
        &mut self.0[i]
    }
}

#[derive(Deserialize)]
struct Scene {
    objects: Vec<Sphere>,
    #[serde(default)]
    camera: Camera,
    ambient: Color,
    #[serde(skip)]
    rng: rand::rngs::ThreadRng,
}

impl Scene {
    fn render(&mut self) -> Image {
        let mut out = Image::new();
        for (x, y, ray) in self.camera.rays() {
            for _ in 0..20 {
                out[(x, y)] = add_colors(out[(x, y)], self.sample_color(ray.clone()));
            }
        }
        out
    }
    fn sample_color(&mut self, mut ray: Ray) -> Color {
        let mut filter_color = WHITE;
        loop {
            let collision = self
                .objects
                .iter()
                .filter_map(|obj| {
                    let (t, x, n) = obj.intersect(&ray)?;
                    Some((t, x, n, obj))
                })
                .min_by_key(|(t, _, _, _)| *t);

            let (_t, x, n, obj) = match collision {
                None => return mul_colors(filter_color, self.ambient),
                Some(c) => c,
            };
            let diffuse_ray = Ray {
                origin: x,
                orientation: random_unit_vector_above_plane(&mut self.rng, n),
            };
            let options = [
                (obj.material.diffuse, Some(diffuse_ray)),
                (obj.material.luminosity, None),
            ];
            let (new_color, new_ray) = random_color_weighted(&mut self.rng, &options);
            filter_color = mul_colors(filter_color, new_color);
            match new_ray {
                Some(r) => ray = r.clone(),
                None => return filter_color,
            }
        }
    }
}

fn random_unit_vector<R: Rng>(rng: &mut R) -> V3 {
    let [x, y, z] = UnitSphere.sample(rng);
    V3::new(x, y, z)
}

fn random_unit_vector_above_plane<R: Rng>(rng: &mut R, normal: V3) -> V3 {
    let v = random_unit_vector(rng);
    let dot = v.dot(normal);
    if dot.into_inner() < 0.0 {
        v - NotNan::new(2.0).unwrap() * dot * normal
    } else {
        v
    }
}

#[derive(Deserialize)]
struct Sphere {
    r2: NotNan<f64>, // squared
    pt: V3,
    material: Material,
}

impl Sphere {
    fn intersect(&self, ray: &Ray) -> Option<(NotNan<f64>, V3, V3)> {
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
        let mut normal = pt - center;
        normal = normal / normal.norm();
        Some((t, pt, normal))
    }
}

#[derive(Deserialize)]
struct Material {
    diffuse: Color,
    luminosity: Color,
}

type Color = [NotNan<f64>; 3];

const unsafe fn unchecked_new_color([r, g, b]: [f64; 3]) -> Color {
    [
        NotNan::unchecked_new(r),
        NotNan::unchecked_new(g),
        NotNan::unchecked_new(b),
    ]
}

fn add_colors(x: Color, y: Color) -> Color {
    [x[0] + y[0], x[1] + y[1], x[2] + y[2]]
}

fn mul_colors(x: Color, y: Color) -> Color {
    [x[0] * y[0], x[1] * y[1], x[2] * y[2]]
}

/// We use this in cases where one ray would "reverse-scatter" into many rays. We want to keep
/// it to 1 (or 0) at every step. This could be changed in the future.
///
/// We can do this by assigning every possible ray a probability (such that the total
/// probability is 1), and scaling the color by the inverse of that probability. In expectation,
/// it doesn't matter what probabilities we pick: it'll be correct regardless. However, because
/// we're not going to trace an infinite number of rays, we care about expected error. The
/// obvious optimization criterion is to minimize the expected squared error. We get:
/// $$ E = \sum_{i,j} p_i \left(c_{ij}/p_i - \bar c_j\right)^2 $$
/// $$ E = \sum_{i,j} c_{ij}^2/p_i -2 c_{ij} \bar c_j +  \bar c_j^2 p_i $$
/// $$ E = \sum_j \bar c_j^2 + \sum_{i} c_{ij}^2/p_i -2 c_{ij} \bar c_j$$
/// $$ \frac{\partial E}{\partial p_i} = -\frac{\sum_j c_{ij}^2}{p_i^2} $$
/// Now since we're optimizing with respect to the constraint of the probabilities adding up to 1:
/// $$ \frac{\partial E}{\partial p_i} = -\frac{\sum_j c_{ij}^2}{p_i^2} =
/// \lambda \frac{\partial C}{\partial p_i} = \lambda $$
/// $$  -\sum_j c_{ij}^2 = \lambda p_i^2 $$
/// So the probability assigned to each ray is proportional to the l2 norm of the colors, which is
/// a pleasing result.
fn random_color_weighted<'a, 'b, T, R: Rng>(
    rng: &'a mut R,
    samples: &'b [(Color, T)],
) -> (Color, &'b T) {
    let total_norm: NotNan<f64> = samples.iter().map(|(c, _)| V3(*c).norm()).sum();
    let (c, x) = samples
        .choose_weighted(rng, |(c, _)| V3(*c).norm().into_inner())
        .unwrap();
    let cv = V3(*c);
    let p = cv.norm() / total_norm;
    ((cv / p).0, x)
}

#[derive(Deserialize, Default)]
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

#[derive(Clone, Debug)]
struct Ray {
    origin: V3,
    orientation: V3,
}

#[derive(Clone, Copy, Debug, Deserialize)]
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

impl Div<NotNan<f64>> for V3 {
    type Output = V3;
    fn div(self, other: NotNan<f64>) -> Self::Output {
        V3([self.0[0] / other, self.0[1] / other, self.0[2] / other])
    }
}

impl Div<V3> for NotNan<f64> {
    type Output = V3;
    fn div(self, other: V3) -> Self::Output {
        V3([other.0[0] / self, other.0[1] / self, other.0[2] / self])
    }
}
