#![warn(clippy::pedantic)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

extern crate itertools;
extern crate ode_solvers;
extern crate plotters;
extern crate transpose;

use ode_solvers::dop853::Dop853;
use ode_solvers::Vector3;
use plotters::prelude::*;

// Konstanten
const G: f64 = 9.81; // Gravitationskonstante [m/s²]
const RADIUS_CELESTIAL_BODY: f64 = 6_371_000.0; // Radius des Himmelskörpers [m]
const EPSILON: f64 = 0.8; // Emissivität
const SIGMA: f64 = 5.67e-8; // Stefan-Boltzmann-Konstante [W/(m²*K⁴)]
const TEMP_AMBIENT: f64 = 300.0; // Umgebungstemperatur [K]
const MARGIN: f64 = 0.0;
const FONT: &str = "Input Sans Condensed"; // Schriftart
const IMAGE_SIZE: (u32, u32) = (3840, 2160); // Bildgröße

// Hilfsfunktionen
fn rho(h: f64) -> f64 {
    1.2 * f64::exp(-1.244_268e-4 * h) // Atmosphärisches Höhenmodell, Luftdichte [kg/m³]
}

fn q(v: f64, h: f64) -> f64 {
    rho(h) * v.powi(2) / 2.0 // Dynamischer Druck [Pa]
}

// Typdefinitionen
type State = Vector3<f64>; // Zustandsvektor
type Time = f64; // Zeit

// Struktur für die Rakete
#[derive(Copy, Clone)]
struct Rocket {
    beta: f64,
    l_over_d: f64,
    c_d: f64,
}

// Implementierung der ODE
impl ode_solvers::System<State> for Rocket {
    fn system(&self, _t: Time, y: &State, dy: &mut State) {
        // Differentialgleichung für die Geschwindigkeit
        dy[0] = -q(y[0], y[2]) / self.beta
            + f64::sin(y[1])
                * (G - (y[0] * f64::cos(y[1])).powi(2) / (RADIUS_CELESTIAL_BODY + y[2]));

        // Differentialgleichung für den Flugbahnwinkel
        dy[1] = 1.0 / y[0]
            * (-q(y[0], y[2]) / self.beta * self.l_over_d
                + f64::cos(y[1])
                    * (G - (y[0] * f64::cos(y[1])).powi(2) / (RADIUS_CELESTIAL_BODY + y[2])));

        // Differentialgleichung für die Höhe
        dy[2] = -y[0] * f64::sin(y[1]);
    }
}

// Plotfunktion für eine einzelne Datenreihe
fn plot(
    path: &String,
    data: (Vec<f64>, Vec<f64>),
    size: (u32, u32),
    caption: &String,
    margin: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Erstellt Dateipfade für die Ausgabedateien
    let image_path: String = format!("{path}.png");
    let data_path: String = format!("{path}.csv");

    let (x_data, y_data): (Vec<f64>, Vec<f64>) = data;

    // Findet die Datenlimits für x und y
    let x_min: f64 = *x_data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or("No data points")?;
    let x_max: f64 = *x_data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or("No data points")?;
    let y_min: f64 = *y_data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or("No data points")?;
    let y_max: f64 = *y_data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or("No data points")?;

    // Berechnet die Achsenbegrenzungen mit einem 10 %-Puffer
    let x_range: f64 = x_max - x_min;
    let y_range: f64 = y_max - y_min;

    let x_min_bound: f64 = x_min - margin * x_range;
    let y_min_bound: f64 = y_min - margin * y_range;
    let x_max_bound: f64;
    let y_max_bound: f64;

    if margin < 0.05 {
        x_max_bound = x_max + (margin + 0.05) * x_range;
        y_max_bound = y_max + (margin + 0.05) * y_range;
    } else {
        x_max_bound = x_max + margin * x_range;
        y_max_bound = y_max + margin * y_range;
    }
    // Erschafft das Plotobjekt
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new(&image_path, (size.0, size.1)).into_drawing_area();

    // Füllt den Hintergrund mit weißer Farbe
    root.fill(&WHITE)?;

    // Erstellt das Koordinatensystem und passt die Größe der Achsenbeschriftung an
    let mut chart: ChartContext<
        '_,
        BitMapBackend<'_>,
        Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>,
    > = ChartBuilder::on(&root)
        .caption(caption, (FONT, size.1 / 20).into_font())
        .margin(size.1 / 30)
        .x_label_area_size(size.1 / 20)
        .y_label_area_size(size.1 / 20)
        .build_cartesian_2d(x_min_bound..x_max_bound, y_min_bound..y_max_bound)?;

    // Zeichnet das Koordinatensystem
    chart
        .configure_mesh()
        .x_label_style((FONT, size.1 / 75).into_font())
        .y_label_style((FONT, size.1 / 75).into_font())
        .x_desc(
            caption
                .split(", ")
                .nth(1)
                .unwrap()
                .split(" und ")
                .nth(1)
                .unwrap(),
        )
        .y_desc(
            caption
                .split(", ")
                .nth(1)
                .unwrap()
                .split(" und ")
                .next()
                .unwrap(),
        )
        .axis_desc_style((FONT, size.1 / 40).into_font())
        .draw()?;

    // Zeichnet die Datenreihe
    chart
        .draw_series(LineSeries::new(
            x_data.clone().into_iter().zip(y_data.clone().into_iter()),
            ShapeStyle::from(&RED).stroke_width(size.1 / 500),
        ))?
        .label(caption.split(", ").next().unwrap())
        .legend(|(x, y)| {
            PathElement::new(
                vec![(x, y), (x + 20, y)],
                ShapeStyle::from(&RED).stroke_width(size.1 / 300),
            )
        });

    // Zeichnet die Legende
    chart
        .configure_series_labels()
        .label_font((FONT, size.1 / 50).into_font())
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    // Speichert das Bild
    root.present()?;

    // Generiere die CSV-Header aus der Caption
    let headers: &[&str] = &[
        caption
            .split(", ")
            .nth(1)
            .unwrap()
            .split(" und ")
            .nth(1)
            .unwrap(),
        caption
            .split(", ")
            .nth(1)
            .unwrap()
            .split(" und ")
            .next()
            .unwrap(),
    ];

    // Schreibt die Daten in eine CSV-Datei
    write_to_csv(&data_path, headers, &x_data, &y_data)?;

    Ok(())
}

// Plotfunktion für mehrere Datenreihen

fn plot_multiple(
    path: &String,
    data: (Vec<f64>, Vec<Vec<f64>>), // Change the data type for y to Vec<Vec<f64>>
    size: (u32, u32),
    caption: &String,
    labels: &Vec<String>,
    margin: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    // Erstellt Dateipfade für die Ausgabedateien
    let image_path: String = format!("{path}.png");
    let data_path: String = format!("{path}.csv");

    let (x_data, y_data_vec): (Vec<f64>, Vec<Vec<f64>>) = data;

    // Findet die Datenlimits für x
    let x_min: f64 = *x_data
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or("No data points")?;
    let x_max: f64 = *x_data
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .ok_or("No data points")?;

    // Findet die Datenlimits für y: y_min ist zunächst auf den größten möglichen Wert gesetzt, y_max auf den kleinsten möglichen Wert
    let mut y_min: f64 = f64::INFINITY;
    let mut y_max: f64 = f64::NEG_INFINITY;

    // Iteriert über alle Datenreihen und findet das Minimum und Maximum; die Werte werden
    // sukzessive verringert bzw. erhöht, wenn ein neues Minimum bzw. Maximum gefunden wird
    for y_data in &y_data_vec {
        let y_min_temp: f64 = *y_data
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .ok_or("No data points")?;
        let y_max_temp: f64 = *y_data
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .ok_or("No data points")?;
        if y_min_temp < y_min {
            y_min = y_min_temp;
        }
        if y_max_temp > y_max {
            y_max = y_max_temp;
        }
    }

    // Berechnet die Achsenbegrenzungen mit einem 10 %-Puffer
    let x_range: f64 = x_max - x_min;
    let y_range: f64 = y_max - y_min;

    let x_min_bound: f64 = x_min - margin * x_range;
    let x_max_bound: f64 = x_max + (margin + 0.01) * x_range;
    let y_min_bound: f64 = y_min - margin * y_range;
    let y_max_bound: f64 = y_max + (margin + 0.01) * y_range;

    // Erschafft das Plotobjekt
    let root: DrawingArea<BitMapBackend<'_>, plotters::coord::Shift> =
        BitMapBackend::new(&image_path, (size.0, size.1)).into_drawing_area();

    // Füllt den Hintergrund mit weißer Farbe
    root.fill(&WHITE)?;

    // Erstellt das Koordinatensystem
    let mut chart: ChartContext<
        '_,
        BitMapBackend<'_>,
        Cartesian2d<plotters::coord::types::RangedCoordf64, plotters::coord::types::RangedCoordf64>,
    > = ChartBuilder::on(&root)
        .caption(caption, (FONT, size.1 / 20).into_font())
        .margin(size.1 / 50)
        .x_label_area_size(size.1 / 20)
        .y_label_area_size(size.1 / 20)
        .build_cartesian_2d(x_min_bound..x_max_bound, y_min_bound..y_max_bound)?;

    // Zeichnet das Koordinatensystem
    chart
        .configure_mesh()
        .x_label_style((FONT, size.1 / 75).into_font())
        .y_label_style((FONT, size.1 / 75).into_font())
        .x_desc(
            caption
                .split(", ")
                .nth(1)
                .unwrap()
                .split(" und ")
                .nth(1)
                .unwrap(),
        )
        .y_desc(
            caption
                .split(", ")
                .nth(1)
                .unwrap()
                .split(" und ")
                .next()
                .unwrap(),
        )
        .axis_desc_style((FONT, size.1 / 40).into_font())
        .draw()?;

    // Zeichnet die Datenreihen und färbt sie mit einer Farbe aus dem Palette99-Farbset ein
    for (idx, (y_data, label)) in y_data_vec.iter().zip(labels).enumerate() {
        chart
            .draw_series(LineSeries::new(
                x_data.clone().into_iter().zip(y_data.clone().into_iter()),
                ShapeStyle::from(&Palette99::pick(idx)).stroke_width(size.1 / 500),
            ))?
            .label(label)
            .legend(move |(x, y)| {
                PathElement::new(
                    vec![(x, y), (x + 20, y)],
                    ShapeStyle::from(&Palette99::pick(idx)).stroke_width(size.1 / 300),
                )
            });
    }

    // Zeichnet die Legende und färbt sie mit einer Farbe aus dem Palette99-Farbset ein
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .label_font((FONT, size.1 / 50).into_font())
        .border_style(BLACK)
        .draw()?;

    // Speichert das Bild
    root.present()?;

    // Generiere die CSV-Header aus der Caption
    let headers: &[&str] = &[
        caption
            .split(", ")
            .nth(1)
            .unwrap()
            .split(" und ")
            .nth(1)
            .unwrap(),
        caption
            .split(", ")
            .nth(1)
            .unwrap()
            .split(" und ")
            .next()
            .unwrap(),
    ];

    // Schreibt die Daten in eine CSV-Datei
    write_to_csv(&data_path, headers, &x_data, &y_data_vec[0])?;

    Ok(())
}

// Schreibt Daten in eine CSV-Datei
fn write_to_csv(
    path: &str,
    headers: &[&str],
    x_data: &[f64],
    y_data: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    // Erstellt einen Writer für die CSV-Datei
    let mut writer: csv::Writer<std::fs::File> = csv::Writer::from_path(path)?;

    // Schreibt die Header in die CSV-Datei
    writer.write_record([headers[0], headers[1]])?;

    // Schreibt die Daten Zeile für Zeile in die CSV-Datei
    for (&x, &y) in x_data.iter().zip(y_data.iter()) {
        writer.write_record(&[x.to_string(), y.to_string()])?;
    }

    // Schließt die CSV-Datei
    writer.flush()?;

    Ok(())
}

// Hauptfunktion

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Aufgabe 1 und 2

    // Erstellt eine Rakete mit den gegebenen Parametern
    //let rocket: Rocket = Rocket {
    //    beta: 400.0,
    //    l_over_d: 0.0,
    //    c_d: 1.0,
    //};

    //// Erstellt die Anfangszeitbedingungen
    //let t0: f64 = 0.0;
    //let tmax: f64 = 300.0;
    //let tstep: f64 = 0.1;

    //// Erstellt den Anfangszustand
    //let y0: Vector3<f64> = Vector3::new(7500.0, 10.0_f64.to_radians(), 100_000.0);

    //// Erstellt den Integrator
    //let mut stepper = Dop853::new(rocket, t0, tmax, tstep, y0, 1.0e-14, 1.0e-14);

    //// Integriert die ODE
    //let _ = stepper.integrate();

    //// Plottet die Flugbahn
    //plot(
    //    &String::from("aufgabe_2"),
    //    (
    //        stepper
    //            .y_out()
    //            .clone()
    //            .into_iter()
    //            .map(|x| x[0] / 1000.0)
    //            .collect(),
    //        stepper
    //            .y_out()
    //            .clone()
    //            .into_iter()
    //            .map(|x| x[2] / 1000.0)
    //            .collect(),
    //    ),
    //    IMAGE_SIZE,
    //    &String::from("Flugbahn, h [km] und v [km/s]"),
    //    0.1,
    //)?;

    //// Aufgabe 3

    //// Erschafft Vektoren für den Wärmestrom und die strahlungsadiabate Wandtemperatur
    //let mut q_sg: Vec<f64> = Vec::new();
    //let mut temp_wall: Vec<f64> = Vec::new();

    //// Iteriert über alle Datenpunkte und berechnet den Wärmestrom und die strahlungsadiabate Wandtemperatur
    //for data in stepper.x_out().iter().zip(stepper.y_out().iter()) {
    //    let v: f64 = data.1[0];
    //    let h: f64 = data.1[2];

    //    // Berechnet den Wärmestrom
    //    let q_sg_temp: f64 = 1.74e-4 * rho(h).sqrt() * v.powi(3);

    //    q_sg.push(q_sg_temp / 1_000_000.0);

    //    // Berechnet die strahlungsadiabate Wandtemperatur
    //    temp_wall.push((q_sg_temp / (EPSILON * SIGMA) + TEMP_AMBIENT.powi(4)).powf(1.0 / 4.0));
    //}

    //// Berechnet die Beschleunigung
    //let acceleration: Vec<f64> = stepper
    //    .y_out()
    //    .windows(2)
    //    .map(|window| {
    //        let dt: f64 = tstep;
    //        let dv: f64 = window[1][0] - window[0][0];
    //        dv / dt
    //    })
    //    .collect();

    //// Plot Wärmestrom
    //plot(
    //    &String::from("aufgabe_3_1_q"),
    //    (stepper.x_out().clone(), q_sg.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Wärmestrom im Nasenbereich, q [MW/m²] und t [s]"),
    //    MARGIN,
    //)?;

    //// Plot strahlungsadiabate Wandtemperatur
    //plot(
    //    &String::from("aufgabe_3_2_temp"),
    //    (stepper.x_out().clone(), temp_wall.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Strahlungsadiabate Wandtemperatur, T [K] und t [s]"),
    //    MARGIN,
    //)?;

    //// Plot Beschleunigung
    //plot(
    //    &String::from("aufgabe_3_3_acceleration"),
    //    (stepper.x_out().clone(), acceleration.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Beschleunigung, a [m/s²] und t [s]"),
    //    MARGIN,
    //)?;

    //// Plot Geschwindigkeit
    //plot(
    //    &String::from("aufgabe_3_4_v_over_t"),
    //    (
    //        stepper.x_out().clone(),
    //        stepper
    //            .y_out()
    //            .clone()
    //            .into_iter()
    //            .map(|x| x[0] / 1000.0)
    //            .collect(),
    //    ),
    //    IMAGE_SIZE,
    //    &String::from("Geschwindigkeit, v [km/s] und t [s]"),
    //    MARGIN,
    //)?;

    //// Plot Höhe
    //plot(
    //    &String::from("aufgabe_3_5_h_over_t"),
    //    (
    //        stepper.x_out().clone(),
    //        stepper
    //            .y_out()
    //            .clone()
    //            .into_iter()
    //            .map(|x| x[2] / 1000.0)
    //            .collect(),
    //    ),
    //    IMAGE_SIZE,
    //    &String::from("Höhe, h [km] und t [s]"),
    //    MARGIN,
    //)?;

    //// Aufgabe 4

    //// Berechnet die Gesamtwärmelast
    //let mut total_heat_load: Vec<f64> = Vec::new();

    //// Iteriert über alle Datenpunkte und berechnet die Gesamtwärmelast
    //for data in q_sg {
    //    total_heat_load.push(total_heat_load.last().unwrap_or(&0.0) + data * tstep);
    //}

    //// Plot Gesamtwärmelast
    //plot(
    //    &String::from("aufgabe_4_heat_load"),
    //    (stepper.x_out().clone(), total_heat_load.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Gesamtwärmelast, Q [MJ/m²] und t [s]"),
    //    MARGIN,
    //)?;

    //// Aufgabe 5

    //// Berechnet die Masse der Rakete anhand der Formel im Skript
    //let mass: f64 = rocket.beta * std::f64::consts::PI * rocket.c_d;

    //// Berechnet die kinetische Energie
    //let kinetic_energy: Vec<f64> = stepper
    //    .y_out()
    //    .iter()
    //    .map(|y| 0.5 * mass * y[0].powi(2) / 1_000_000.0)
    //    .collect();

    //// Plot kinetische Energie
    //plot(
    //    &String::from("aufgabe_5_1_kinetic_energy"),
    //    (stepper.x_out().clone(), kinetic_energy.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Kinetische Energie, E [MJ] und t [s]"),
    //    MARGIN,
    //)?;

    //// Berechnet die maximale kinetische Energie
    //let max_kinetic_energy: &f64 = kinetic_energy
    //    .iter()
    //    .max_by(|a, b| a.partial_cmp(b).unwrap())
    //    .ok_or("No data points")?;

    //// Berechnet den Anteil der kinetischen Energie, der in die Wand geht
    //let percentage_of_kinetic_energy_into_wall: f64 =
    //    total_heat_load.last().unwrap() * std::f64::consts::PI / max_kinetic_energy;

    //// Gibt die Gesamtwärmelast, die maximale kinetische Energie und den Anteil der kinetischen Energie, der in die Wand geht, aus
    //println!(
    //    "Gesamtwärmelast: {} MJ/m², maximale kinetische Energie: {} MJ, Anteil der kinetischen Energie, der in die Wand geht: {} %",
    //    total_heat_load.last().unwrap_or(&0.0),
    //    max_kinetic_energy,
    //    percentage_of_kinetic_energy_into_wall * 100.0,
    //);

    //// Aufgabe 6

    //// Berechnet die Kenngrößen für verschiedene β und γ
    //for beta in &[50.0, 100.0, 200.0, 400.0, 800.0] {
    //    // Erstellt Vektoren für die Daten
    //    let mut data = Vec::new();
    //    let mut accel: Vec<Vec<f64>> = Vec::new();
    //    let mut q: Vec<Vec<f64>> = Vec::new();
    //    let mut q_total: Vec<Vec<f64>> = Vec::new();

    //    let gammas: Vec<f64> = vec![1.0, 2.0, 5.0, 10.0, 20.0, 45.0];

    //    let labels: Vec<String> = gammas.iter().map(|gamma| format!("γ = {gamma}°")).collect();

    //    // Iteriert über alle γ und berechnet die Kenngrößen
    //    for gamma in &gammas {
    //        // Erstellt Rakete und löst die ODE
    //        let rocket: Rocket = Rocket {
    //            beta: *beta,
    //            l_over_d: 0.0,
    //            c_d: 1.0,
    //        };

    //        let gamma: f64 = *gamma;

    //        let t0: f64 = 0.0;
    //        let tmax: f64 = 300.0;
    //        let tstep: f64 = 0.1;

    //        let y0: Vector3<f64> = Vector3::new(7500.0, gamma.to_radians(), 100_000.0);

    //        let mut stepper = Dop853::new(rocket, t0, tmax, tstep, y0, 1.0e-14, 1.0e-14);

    //        let _ = stepper.integrate();

    //        // Berechnet die Beschleunigung
    //        let acceleration: Vec<f64> = stepper
    //            .y_out()
    //            .windows(2)
    //            .map(|window| {
    //                let dt: f64 = tstep;
    //                let dv: f64 = window[1][0] - window[0][0];
    //                dv / dt
    //            })
    //            .collect();

    //        // Berechnet den Wärmestrom
    //        let mut q_sg: Vec<f64> = Vec::new();

    //        for data in stepper.x_out().iter().zip(stepper.y_out().iter()) {
    //            let v: f64 = data.1[0];
    //            let h: f64 = data.1[2];

    //            let q_sg_temp: f64 = 1.74e-4 * rho(h).sqrt() * v.powi(3);

    //            q_sg.push(q_sg_temp / 1_000_000.0);
    //        }

    //        // Berechnet die Gesamtwärmelast
    //        let mut total_heat_load: Vec<f64> = Vec::new();

    //        for data in &q_sg {
    //            total_heat_load.push(total_heat_load.last().unwrap_or(&0.0) + data * tstep);
    //        }

    //        // Fügt die Daten den Vektoren hinzu
    //        q.push(q_sg);
    //        accel.push(acceleration);
    //        q_total.push(total_heat_load);
    //        data.push(stepper);
    //    }

    //    // Plot Beschleunigung für verschiedene γ zu je einem β
    //    plot_multiple(
    //        &format!("aufgabe_6_1_{beta}"),
    //        (data[0].x_out().clone(), accel),
    //        IMAGE_SIZE,
    //        &format!("Beschleunigung, a [m/s²] und t [s], β = {beta}"),
    //        &labels,
    //        MARGIN,
    //    )?;

    //    // Plot Wärmestrom für verschiedene γ zu je einem β
    //    plot_multiple(
    //        &format!("aufgabe_6_2_{beta}"),
    //        (data[0].x_out().clone(), q),
    //        IMAGE_SIZE,
    //        &format!("Wärmestrom, q [MW/m²] und t [s], β = {beta}"),
    //        &labels,
    //        MARGIN,
    //    )?;

    //    // Plot Gesamtwärmelast für verschiedene γ zu je einem β
    //    plot_multiple(
    //        &format!("aufgabe_6_3_{beta}"),
    //        (data[0].x_out().clone(), q_total),
    //        IMAGE_SIZE,
    //        &format!("Gesamtwärmelast, Q [MJ/m²] und t [s], β = {beta}"),
    //        &labels,
    //        MARGIN,
    //    )?;
    //}

    //// Aufgabe 7

    //// Erstellt Rakete, löst die ODE für Gleitkörper und plottet die berechneten Kenngrößen
    //let rocket: Rocket = Rocket {
    //    beta: 50.0,
    //    l_over_d: 1.0,
    //    c_d: 1.0,
    //};

    //let t0: f64 = 0.0;
    //let tmax: f64 = 2000.0;
    //let tstep: f64 = 0.1;

    //let y0: Vector3<f64> = Vector3::new(7500.0, 5.0_f64.to_radians(), 100_000.0);

    //let mut stepper = Dop853::new(rocket, t0, tmax, tstep, y0, 1.0e-14, 1.0e-14);

    //let _ = stepper.integrate();

    //let mut q_sg: Vec<f64> = Vec::new();

    //for data in stepper.x_out().iter().zip(stepper.y_out().iter()) {
    //    let v: f64 = data.1[0];
    //    let h: f64 = data.1[2];
    //    let q_sg_temp: f64 = 1.74e-4 * rho(h).sqrt() * v.powi(3);
    //    q_sg.push(q_sg_temp / 1_000_000.0);
    //}

    //let mut total_heat_load: Vec<f64> = Vec::new();

    //for data in &q_sg {
    //    total_heat_load.push(total_heat_load.last().unwrap_or(&0.0) + data * tstep);
    //}

    //let acceleration: Vec<f64> = stepper
    //    .y_out()
    //    .windows(2)
    //    .map(|window| {
    //        let dt: f64 = tstep;
    //        let dv: f64 = window[1][0] - window[0][0];
    //        dv / dt
    //    })
    //    .collect();

    //plot(
    //    &String::from("aufgabe_7_1_q"),
    //    (stepper.x_out().clone(), q_sg.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Wärmestrom im Nasenbereich, q [MW/m²] und t [s]"),
    //    MARGIN,
    //)?;

    //plot(
    //    &String::from("aufgabe_7_2_heat_load"),
    //    (stepper.x_out().clone(), total_heat_load.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Gesamtwärmelast, Q [MJ/m²] und t [s]"),
    //    MARGIN,
    //)?;

    //plot(
    //    &String::from("aufgabe_7_3_acceleration"),
    //    (stepper.x_out().clone(), acceleration.clone()),
    //    IMAGE_SIZE,
    //    &String::from("Beschleunigung, a [m/s²] und t [s]"),
    //    MARGIN,
    //)?;

    //let gamma_dot: Vec<f64> = stepper
    //    .y_out()
    //    .windows(2)
    //    .map(|window| {
    //        let dt: f64 = tstep;
    //        let dv: f64 = window[1][1] - window[0][1];
    //        dv / dt
    //    })
    //    .collect();

    //plot(
    //    &String::from("aufgabe_7_4_gamma_dot"),
    //    (stepper.x_out().clone(), gamma_dot.clone()),
    //    IMAGE_SIZE,
    //    &String::from("γ_dot, γ_dot [°/s] und t [s]"),
    //    MARGIN,
    //)?;

    //// Plot Höhe über Zeit
    //plot(
    //    &String::from("aufgabe_7_5_h_over_t"),
    //    (
    //        stepper.x_out().clone(),
    //        stepper.y_out().clone().into_iter().map(|x| x[2]).collect(),
    //    ),
    //    IMAGE_SIZE,
    //    &String::from("Höhe, h [m] und t [s]"),
    //    MARGIN,
    //)?;

    // Aufgabe 8 - Implementieren Sie einen einfachen aerodynamischen Kontrollmechanismus um die "Skip-Trajectory", also das Überschwingen im Flug zu vermeiden: Tipp: Reduzieren sie die Gleitzahl (L/D) in Abhängigkeit des gegenwärtigen Flugbahnwinkels, gültige Werte liegen zwischen 0 und 1.

    let mut rocket: Rocket = Rocket {
        beta: 50.0,
        l_over_d: 1.0,
        c_d: 1.0,
    };

    let t0: f64 = 0.0;
    let tmax: f64 = 2000.0;
    let tstep: f64 = 0.001;
    let mut tcurrent: f64 = t0;

    let mut y: Vector3<f64> = Vector3::new(7500.0, 5.0_f64.to_radians(), 100_000.0);

    let mut vector: Vec<(f64, Vector3<f64>)> = Vec::new();

    let mut l_over_d_vec: Vec<f64> = Vec::new();

    let mut i: usize = 0;

    while tcurrent < tmax {
        let mut stepper = Dop853::new(
            rocket,
            tcurrent,
            tcurrent + tstep,
            tstep,
            y,
            1.0e-14,
            1.0e-14,
        );

        let _ = stepper.integrate();

        y = *stepper.y_out().last().unwrap();

        if (tcurrent - t0).abs() < f64::EPSILON {
            vector.push((
                *stepper.x_out().first().unwrap(),
                *stepper.y_out().first().unwrap(),
            ));
        }

        vector.push((
            *stepper.x_out().last().unwrap(),
            *stepper.y_out().last().unwrap(),
        ));

        //rocket.l_over_d = f64::sin(vector[i].1[1]).powf(1.0 / 2.0);

        rocket.l_over_d = f64::atan(50_000.0 * vector[i].1[1]) / (std::f64::consts::PI / 2.0);

        l_over_d_vec.push(rocket.l_over_d);

        i += 1;
        tcurrent += tstep;

        if i >= (tmax / tstep) as usize {
            break;
        }

        if i % 1000 == 0 {
            println!("{}", i as f64 / (tmax / tstep) * 100.0);
        }
    }

    // Plot Höhe gegen Zeit
    plot(
        &String::from("aufgabe_8_1_h_over_t"),
        (
            vector.clone().into_iter().map(|x| x.0).collect(),
            vector.clone().into_iter().map(|x| x.1[2]).collect(),
        ),
        IMAGE_SIZE,
        &String::from("Höhe, h [m] und t [s]"),
        0.0,
    )?;

    plot(
        &String::from("aufgabe_8_2_v_over_t"),
        (
            vector.clone().into_iter().map(|x| x.0).collect(),
            vector.clone().into_iter().map(|x| x.1[0]).collect(),
        ),
        IMAGE_SIZE,
        &String::from("Geschwindigkeit, v [m/s] und t [s]"),
        MARGIN,
    )?;

    // Plot l_over_d_vec gegen Zeit

    plot(
        &String::from("aufgabe_8_3_l_over_d_over_t"),
        (
            vector.clone().into_iter().map(|x| x.0).collect(),
            l_over_d_vec,
        ),
        IMAGE_SIZE,
        &String::from("L/D, L/D und t [s]"),
        MARGIN,
    )?;

    let acceleration: Vec<f64> = vector
        .windows(2)
        .map(|window| {
            let dt = tstep;
            let dv = window[1].1[0] - window[0].1[0];
            dv / dt
        })
        .collect();

    // Plot Beschleunigung

    plot(
        &String::from("aufgabe_8_4_acceleration"),
        (
            vector.clone().into_iter().map(|x| x.0).collect(),
            acceleration,
        ),
        IMAGE_SIZE,
        &String::from("Beschleunigung, a [m/s²] und t [s]"),
        MARGIN,
    )?;

    let mut q_sg_aero: Vec<f64> = Vec::new();

    for data in &vector {
        let v: f64 = data.1[0];
        let h: f64 = data.1[2];
        let q_sg_temp: f64 = 1.74e-4 * rho(h).sqrt() * v.powi(3);
        q_sg_aero.push(q_sg_temp / 1_000_000.0);
    }

    let mut total_heat_load_aero: Vec<f64> = Vec::new();

    for data in &q_sg_aero {
        total_heat_load_aero.push(total_heat_load_aero.last().unwrap_or(&0.0) + data * tstep);
    }

    // Plot Wärmelast

    plot(
        &String::from("aufgabe_8_5_q"),
        (
            vector.clone().into_iter().map(|x| x.0).collect(),
            q_sg_aero.clone(),
        ),
        IMAGE_SIZE,
        &String::from("Wärmestrom im Nasenbereich, q [MW/m²] und t [s]"),
        MARGIN,
    )?;

    // Plot integrale Wärmelast

    plot(
        &String::from("aufgabe_8_6_heat_load"),
        (
            vector.clone().into_iter().map(|x| x.0).collect(),
            total_heat_load_aero.clone(),
        ),
        IMAGE_SIZE,
        &String::from("Gesamtwärmelast, Q [MJ/m²] und t [s]"),
        MARGIN,
    )?;

    // Aufgabe 9 - Vergleiche die Wärmelasten eines ballistischen Eintritts gegenüber der Wärmelast
    // von Aufgabe 8. Plotte dazu beide Kurven im selben Diagramm.

    let rocket: Rocket = Rocket {
        beta: 50.0,
        l_over_d: 0.0,
        c_d: 1.0,
    };

    let t0: f64 = 0.0;
    let tmax: f64 = 2000.0;
    let tstep: f64 = 0.001;

    let y0: Vector3<f64> = Vector3::new(7500.0, 5.0_f64.to_radians(), 100_000.0);

    let mut stepper = Dop853::new(rocket, t0, tmax, tstep, y0, 1.0e-14, 1.0e-14);

    let _ = stepper.integrate();

    let mut q_sg_ball: Vec<f64> = Vec::new();

    for data in stepper.x_out().iter().zip(stepper.y_out().iter()) {
        let v: f64 = data.1[0];
        let h: f64 = data.1[2];
        let q_sg_temp: f64 = 1.74e-4 * rho(h).sqrt() * v.powi(3);
        q_sg_ball.push(q_sg_temp / 1_000_000.0);
    }

    // Berechnet die Gesamtwärmelast
    let mut total_heat_load_ball: Vec<f64> = Vec::new();

    // Iteriert über alle Datenpunkte und berechnet die Gesamtwärmelast
    for data in &q_sg_ball {
        total_heat_load_ball.push(total_heat_load_ball.last().unwrap_or(&0.0) + data * tstep);
    }

    // Plot q_sg für ballistischen Eintritt vs aerodynamischen Eintritt

    plot_multiple(
        &String::from("aufgabe_9_1_q"),
        (
            stepper.x_out().clone(),
            vec![q_sg_ball.clone(), q_sg_aero.clone()],
        ),
        IMAGE_SIZE,
        &String::from("Wärmestrom im Nasenbereich, q [MW/m²] und t [s]"),
        &vec![
            String::from("Ballistischer Eintritt"),
            String::from("Aerodynamischer Eintritt"),
        ],
        MARGIN,
    )?;

    // Plot Gesamtwärmelast für ballistischen Eintritt vs aerodynamischen Eintritt

    plot_multiple(
        &String::from("aufgabe_9_2_heat_load"),
        (
            stepper.x_out().clone(),
            vec![total_heat_load_ball.clone(), total_heat_load_aero.clone()],
        ),
        IMAGE_SIZE,
        &String::from("Gesamtwärmelast, Q [MJ/m²] und t [s]"),
        &vec![
            String::from("Ballistischer Eintritt"),
            String::from("Aerodynamischer Eintritt"),
        ],
        MARGIN,
    )?;

    Ok(())
}
