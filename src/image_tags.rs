use anyhow::{Context, Result};
use chrono::{Datelike, FixedOffset, NaiveDate, NaiveDateTime, TimeZone, Timelike, Utc};
use exif::{In, Tag, Value};
use once_cell::sync::Lazy;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

const TAG_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize)]
pub struct ImageTags {
    pub version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub captured_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latitude: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub longitude: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub altitude_m: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nearest_town: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub nearby_towns: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nearest_town_distance_km: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub season: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_of_day: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solar_phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solar_elevation_deg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera_make: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lens_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_text: Option<String>,
}

impl Default for ImageTags {
    fn default() -> Self {
        Self {
            version: TAG_VERSION,
            captured_at: None,
            timestamp_source: None,
            latitude: None,
            longitude: None,
            altitude_m: None,
            country_code: None,
            country: None,
            region: None,
            nearest_town: None,
            nearby_towns: Vec::new(),
            nearest_town_distance_km: None,
            season: None,
            time_of_day: None,
            solar_phase: None,
            solar_elevation_deg: None,
            camera_make: None,
            camera_model: None,
            lens_model: None,
            search_text: None,
        }
    }
}

#[derive(Clone, Debug)]
struct City {
    name: String,
    country_code: String,
    admin1_code: String,
    timezone: Option<String>,
    feature_code: String,
    population: u64,
    latitude: f64,
    longitude: f64,
    unit: [f64; 3],
}

#[derive(Debug)]
struct GeoIndex {
    cities: Vec<City>,
    countries: HashMap<String, String>,
    regions: HashMap<String, String>,
}

static GEO_CACHE: Lazy<RwLock<HashMap<PathBuf, Arc<GeoIndex>>>> = Lazy::new(|| RwLock::new(HashMap::new()));

fn unit_position(latitude: f64, longitude: f64) -> [f64; 3] {
    let lat = latitude.to_radians();
    let lon = longitude.to_radians();
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

impl GeoIndex {
    fn load(assets: &Path) -> Result<Self> {
        let root = assets.join("geonames");
        let cities_path = root.join("cities5000.txt");
        let countries_path = root.join("countryInfo.txt");
        let regions_path = root.join("admin1CodesASCII.txt");
        let mut countries = HashMap::new();
        for line in
            BufReader::new(File::open(&countries_path).with_context(|| format!("open {}", countries_path.display()))?)
                .lines()
        {
            let line = line?;
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() > 4 {
                countries.insert(fields[0].to_string(), fields[4].to_string());
            }
        }

        let mut regions = HashMap::new();
        for line in
            BufReader::new(File::open(&regions_path).with_context(|| format!("open {}", regions_path.display()))?)
                .lines()
        {
            let line = line?;
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() >= 2 {
                regions.insert(fields[0].to_string(), fields[1].to_string());
            }
        }

        let mut cities = Vec::new();
        for line in
            BufReader::new(File::open(&cities_path).with_context(|| format!("open {}", cities_path.display()))?).lines()
        {
            let line = line?;
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 15 {
                continue;
            }
            if !matches!(
                fields[7],
                "PPL" | "PPLA" | "PPLA2" | "PPLA3" | "PPLA4" | "PPLA5" | "PPLC" | "PPLG"
            ) {
                continue;
            }
            let Ok(latitude) = fields[4].parse::<f64>() else {
                continue;
            };
            let Ok(longitude) = fields[5].parse::<f64>() else {
                continue;
            };
            let population = fields[14].parse::<u64>().unwrap_or(0);
            cities.push(City {
                name: fields[1].to_string(),
                country_code: fields[8].to_string(),
                admin1_code: fields[10].to_string(),
                timezone: fields
                    .get(17)
                    .filter(|value| !value.is_empty())
                    .map(|value| value.to_string()),
                feature_code: fields[7].to_string(),
                population,
                latitude,
                longitude,
                unit: unit_position(latitude, longitude),
            });
        }
        anyhow::ensure!(!cities.is_empty(), "GeoNames city database is empty");
        Ok(Self {
            cities,
            countries,
            regions,
        })
    }

    fn nearby_cities(&self, latitude: f64, longitude: f64) -> Option<Vec<(&City, f64)>> {
        let query = unit_position(latitude, longitude);
        let city = self
            .cities
            .iter()
            .filter(|city| city.population >= 15_000)
            .max_by(|left, right| {
                let left_dot = left.unit[0] * query[0] + left.unit[1] * query[1] + left.unit[2] * query[2];
                let right_dot = right.unit[0] * query[0] + right.unit[1] * query[1] + right.unit[2] * query[2];
                left_dot.total_cmp(&right_dot)
            })?;
        let distance = haversine_km(latitude, longitude, city.latitude, city.longitude);
        let mut result = vec![(city, distance)];
        if distance < 10.0 {
            return Some(result);
        }

        let radius_km = (distance * 2.0).max(50.0);
        let mut alternatives: Vec<(&City, f64)> = self
            .cities
            .iter()
            .filter(|candidate| {
                candidate.name != city.name
                    && candidate.country_code == city.country_code
                    && candidate.feature_code != "PPL"
                    && candidate.population >= 10_000
            })
            .filter_map(|candidate| {
                let distance = haversine_km(latitude, longitude, candidate.latitude, candidate.longitude);
                (distance <= radius_km).then_some((candidate, distance))
            })
            .collect();
        alternatives.sort_by(|left, right| {
            right
                .0
                .population
                .cmp(&left.0.population)
                .then_with(|| left.1.total_cmp(&right.1))
        });
        result.extend(alternatives.into_iter().take(2));
        Some(result)
    }
}

fn geo_index(assets: &Path) -> Option<Arc<GeoIndex>> {
    let key = assets.canonicalize().unwrap_or_else(|_| assets.to_path_buf());
    if let Some(index) = GEO_CACHE.read().ok()?.get(&key) {
        return Some(index.clone());
    }
    let index = match GeoIndex::load(&key) {
        Ok(index) => Arc::new(index),
        Err(error) => {
            log::debug!("offline GeoNames enrichment unavailable: {error:#}");
            return None;
        },
    };
    GEO_CACHE.write().ok()?.insert(key, index.clone());
    Some(index)
}

fn ascii_field(exif: &exif::Exif, tag: Tag) -> Option<String> {
    let field = exif.get_field(tag, In::PRIMARY)?;
    let Value::Ascii(values) = &field.value else {
        return None;
    };
    let value = String::from_utf8_lossy(values.first()?).trim().to_string();
    (!value.is_empty()).then_some(value)
}

fn rational(value: &exif::Rational) -> Option<f64> {
    (value.denom != 0).then_some(value.num as f64 / value.denom as f64)
}

fn gps_coordinate(exif: &exif::Exif, value_tag: Tag, reference_tag: Tag) -> Option<f64> {
    let field = exif.get_field(value_tag, In::PRIMARY)?;
    let Value::Rational(parts) = &field.value else {
        return None;
    };
    if parts.len() < 3 {
        return None;
    }
    let mut coordinate = rational(&parts[0])? + rational(&parts[1])? / 60.0 + rational(&parts[2])? / 3600.0;
    let reference = ascii_field(exif, reference_tag)?;
    if matches!(reference.as_bytes().first(), Some(b'S' | b'W' | b's' | b'w')) {
        coordinate = -coordinate;
    }
    Some(coordinate)
}

fn gps_altitude(exif: &exif::Exif) -> Option<f64> {
    let field = exif.get_field(Tag::GPSAltitude, In::PRIMARY)?;
    let Value::Rational(values) = &field.value else {
        return None;
    };
    let mut altitude = rational(values.first()?)?;
    if exif
        .get_field(Tag::GPSAltitudeRef, In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        == Some(1)
    {
        altitude = -altitude;
    }
    Some(altitude)
}

fn parse_exif_datetime(value: &str) -> Option<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value.trim_matches(char::from(0)), "%Y:%m:%d %H:%M:%S").ok()
}

fn parse_offset(value: &str) -> Option<FixedOffset> {
    let value = value.trim_matches(char::from(0));
    let sign = match value.as_bytes().first()? {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    let hours = value.get(1..3)?.parse::<i32>().ok()?;
    let minutes = value.get(4..6)?.parse::<i32>().ok()?;
    FixedOffset::east_opt(sign * (hours * 3600 + minutes * 60))
}

fn gps_datetime(exif: &exif::Exif) -> Option<chrono::DateTime<Utc>> {
    let date = ascii_field(exif, Tag::GPSDateStamp)?;
    let date = NaiveDate::parse_from_str(date.trim_matches(char::from(0)), "%Y:%m:%d").ok()?;
    let field = exif.get_field(Tag::GPSTimeStamp, In::PRIMARY)?;
    let Value::Rational(parts) = &field.value else {
        return None;
    };
    let hour = rational(parts.first()?)?.floor() as u32;
    let minute = rational(parts.get(1)?)?.floor() as u32;
    let seconds = rational(parts.get(2)?)?;
    let second = seconds.floor() as u32;
    let nanos = ((seconds - f64::from(second)) * 1_000_000_000.0).round() as u32;
    Some(Utc.from_utc_datetime(&date.and_hms_nano_opt(hour, minute, second, nanos)?))
}

fn capture_times(
    exif: &exif::Exif,
    longitude: Option<f64>,
    timezone: Option<&str>,
) -> (Option<NaiveDateTime>, Option<chrono::DateTime<Utc>>, Option<String>) {
    let local = ascii_field(exif, Tag::DateTimeOriginal).and_then(|value| parse_exif_datetime(&value));
    if let Some(utc) = gps_datetime(exif) {
        return (local, Some(utc), Some("exif_gps_utc".to_string()));
    }
    if let (Some(local), Some(offset)) = (
        local,
        ascii_field(exif, Tag::OffsetTimeOriginal).and_then(|value| parse_offset(&value)),
    ) {
        if let Some(timestamp) = offset.from_local_datetime(&local).single() {
            return (
                Some(local),
                Some(timestamp.with_timezone(&Utc)),
                Some("exif_offset".to_string()),
            );
        }
    }
    if let (Some(local), Some(timezone)) = (local, timezone.and_then(|value| value.parse::<chrono_tz::Tz>().ok())) {
        let mapped = timezone.from_local_datetime(&local);
        if let Some(timestamp) = mapped.single().or_else(|| mapped.earliest()) {
            return (
                Some(local),
                Some(timestamp.with_timezone(&Utc)),
                Some("geonames_timezone".to_string()),
            );
        }
    }
    if let (Some(local), Some(longitude)) = (local, longitude) {
        let approximate_hours = (longitude / 15.0).round().clamp(-12.0, 14.0) as i32;
        if let Some(offset) = FixedOffset::east_opt(approximate_hours * 3600) {
            if let Some(timestamp) = offset.from_local_datetime(&local).single() {
                return (
                    Some(local),
                    Some(timestamp.with_timezone(&Utc)),
                    Some("longitude_estimate".to_string()),
                );
            }
        }
    }
    (local, None, local.map(|_| "local_time_only".to_string()))
}

fn season(month: u32, latitude: Option<f64>) -> &'static str {
    let northern = latitude.unwrap_or(45.0) >= 0.0;
    match (northern, month) {
        (true, 12 | 1 | 2) | (false, 6 | 7 | 8) => "winter",
        (true, 3 | 4 | 5) | (false, 9 | 10 | 11) => "spring",
        (true, 6 | 7 | 8) | (false, 12 | 1 | 2) => "summer",
        _ => "autumn",
    }
}

fn solar_position(timestamp: chrono::DateTime<Utc>, latitude: f64, longitude: f64) -> (f64, f64) {
    let julian_day = timestamp.timestamp_millis() as f64 / 86_400_000.0 + 2_440_587.5;
    let days = julian_day - 2_451_545.0;
    let mean_longitude = (280.460 + 0.9856474 * days).rem_euclid(360.0).to_radians();
    let mean_anomaly = (357.528 + 0.9856003 * days).rem_euclid(360.0).to_radians();
    let ecliptic_longitude =
        mean_longitude + (1.915 * mean_anomaly.sin() + 0.020 * (2.0 * mean_anomaly).sin()).to_radians();
    let obliquity = (23.439 - 0.0000004 * days).to_radians();
    let right_ascension = (obliquity.cos() * ecliptic_longitude.sin()).atan2(ecliptic_longitude.cos());
    let declination = (obliquity.sin() * ecliptic_longitude.sin()).asin();
    let utc_hours = timestamp.hour() as f64 + timestamp.minute() as f64 / 60.0 + timestamp.second() as f64 / 3600.0;
    let gmst = (6.697375 + 0.0657098242 * days.floor() + 1.00273791 * utc_hours).rem_euclid(24.0);
    let right_ascension_hours = right_ascension.to_degrees().rem_euclid(360.0) / 15.0;
    let hour_angle_hours = (gmst + longitude / 15.0 - right_ascension_hours + 12.0).rem_euclid(24.0) - 12.0;
    let hour_angle = (hour_angle_hours * 15.0).to_radians();
    let latitude = latitude.to_radians();
    let elevation = (latitude.sin() * declination.sin() + latitude.cos() * declination.cos() * hour_angle.cos())
        .asin()
        .to_degrees();
    let solar_hour = (12.0 + hour_angle_hours).rem_euclid(24.0);
    (elevation, solar_hour)
}

fn solar_tags(elevation: f64, solar_hour: f64) -> (&'static str, &'static str) {
    let phase = if elevation < -6.0 {
        "night"
    } else if elevation < 0.0 {
        "civil twilight"
    } else if elevation < 10.0 {
        "low sun"
    } else {
        "daylight"
    };
    let time_of_day = if elevation < -6.0 {
        "nighttime"
    } else if elevation < 0.0 {
        if solar_hour < 12.0 {
            "dawn"
        } else {
            "dusk"
        }
    } else if elevation < 10.0 {
        if solar_hour < 12.0 {
            "dawn"
        } else {
            "evening"
        }
    } else {
        "daytime"
    };
    (time_of_day, phase)
}

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let lat1 = lat1.to_radians();
    let lat2 = lat2.to_radians();
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    6_371.0 * 2.0 * a.sqrt().atan2((1.0 - a).sqrt())
}

fn build_search_text(tags: &ImageTags) -> Option<String> {
    let mut parts = vec!["Photographed".to_string()];
    let area = match (&tags.region, &tags.country) {
        (Some(region), Some(country)) if region != country => Some(format!("{region}, {country}")),
        (_, Some(country)) => Some(country.clone()),
        (Some(region), None) => Some(region.clone()),
        (None, None) => None,
    };
    let in_nearest_town = tags
        .nearest_town_distance_km
        .is_some_and(|distance| distance.is_finite() && distance <= 10.0);
    if in_nearest_town {
        match (&tags.nearest_town, area) {
            (Some(town), Some(area)) => {
                let town_and_area = if area == *town || area.starts_with(&format!("{town},")) {
                    area
                } else {
                    format!("{town}, {area}")
                };
                parts.push(format!("in {town_and_area}"));
            },
            (Some(town), None) => parts.push(format!("in {town}")),
            (None, Some(area)) => parts.push(format!("in {area}")),
            (None, None) => {},
        }
    } else {
        match (tags.nearby_towns.as_slice(), &tags.nearest_town, area) {
            (towns, _, Some(area)) if towns.len() > 1 => parts.push(format!("near {} in {area}", human_list(towns))),
            (towns, _, None) if towns.len() > 1 => parts.push(format!("near {}", human_list(towns))),
            (_, Some(town), Some(area)) => parts.push(format!("near {town}, {area}")),
            (_, Some(town), None) => parts.push(format!("near {town}")),
            (_, None, Some(area)) => parts.push(format!("in {area}")),
            (_, None, None) => {},
        }
    }
    if let Some(season) = &tags.season {
        parts.push(format!("during {season}"));
    }
    if let Some(time_of_day) = &tags.time_of_day {
        match time_of_day.as_str() {
            "dawn" | "dusk" | "nighttime" => parts.push(format!("at {time_of_day}")),
            _ => parts.push(format!("in the {time_of_day}")),
        }
    }
    if let Some(captured_at) = &tags.captured_at {
        if let Some(timestamp) = parse_exif_datetime(captured_at) {
            parts.push(timestamp.format("on %B %-d, %Y").to_string());
        }
    }
    if let Some(altitude) = tags.altitude_m {
        let band = if altitude < 200.0 {
            "low elevation"
        } else if altitude < 1_000.0 {
            "moderate elevation"
        } else if altitude < 2_500.0 {
            "high elevation"
        } else {
            "very high elevation"
        };
        parts.push(format!("at {:.0} metres altitude, {band}", altitude));
    }
    (parts.len() > 1).then(|| format!("{}.", parts.join(" ")))
}

fn human_list(values: &[String]) -> String {
    match values {
        [] => String::new(),
        [one] => one.clone(),
        [first, second] => format!("{first} and {second}"),
        [first, middle @ .., last] => format!(
            "{}, and {last}",
            std::iter::once(first)
                .chain(middle)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

pub fn extract(path: &Path, assets: &Path) -> Result<ImageTags> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let exif = match exif::Reader::new().read_from_container(&mut reader) {
        Ok(exif) => exif,
        Err(exif::Error::NotFound(_)) => return Ok(ImageTags::default()),
        Err(error) => return Err(error.into()),
    };
    let latitude = gps_coordinate(&exif, Tag::GPSLatitude, Tag::GPSLatitudeRef);
    let longitude = gps_coordinate(&exif, Tag::GPSLongitude, Tag::GPSLongitudeRef);
    let nearest = match (latitude, longitude, geo_index(assets)) {
        (Some(latitude), Some(longitude), Some(index)) => index.nearby_cities(latitude, longitude).map(|cities| {
            let (city, distance) = cities[0];
            (
                city.name.clone(),
                city.country_code.clone(),
                city.timezone.clone(),
                index.countries.get(&city.country_code).cloned(),
                index
                    .regions
                    .get(&format!("{}.{}", city.country_code, city.admin1_code))
                    .cloned(),
                distance,
                cities
                    .into_iter()
                    .map(|(city, _)| city.name.clone())
                    .collect::<Vec<_>>(),
            )
        }),
        _ => None,
    };
    let timezone = nearest.as_ref().and_then(|nearest| nearest.2.as_deref());
    let (local_time, utc_time, timestamp_source) = capture_times(&exif, longitude, timezone);
    let mut tags = ImageTags {
        version: TAG_VERSION,
        captured_at: local_time.map(|timestamp| timestamp.format("%Y:%m:%d %H:%M:%S").to_string()),
        timestamp_source,
        latitude,
        longitude,
        altitude_m: gps_altitude(&exif),
        camera_make: ascii_field(&exif, Tag::Make),
        camera_model: ascii_field(&exif, Tag::Model),
        lens_model: ascii_field(&exif, Tag::LensModel),
        ..ImageTags::default()
    };
    if let Some((town, country_code, _timezone, country, region, distance, nearby_towns)) = nearest {
        tags.country_code = Some(country_code);
        tags.country = country;
        tags.region = region;
        tags.nearest_town = Some(town);
        tags.nearby_towns = nearby_towns;
        tags.nearest_town_distance_km = Some(distance);
    }
    if let Some(local_time) = local_time {
        tags.season = Some(season(local_time.month(), latitude).to_string());
    }
    if let (Some(timestamp), Some(latitude), Some(longitude)) = (utc_time, latitude, longitude) {
        let (elevation, solar_hour) = solar_position(timestamp, latitude, longitude);
        let (time_of_day, phase) = solar_tags(elevation, solar_hour);
        tags.solar_elevation_deg = Some(elevation);
        tags.time_of_day = Some(time_of_day.to_string());
        tags.solar_phase = Some(phase.to_string());
    } else if let Some(local_time) = local_time {
        tags.time_of_day = Some(
            match local_time.hour() {
                5..=8 => "dawn",
                9..=16 => "daytime",
                17..=21 => "evening",
                _ => "nighttime",
            }
            .to_string(),
        );
    }
    tags.search_text = build_search_text(&tags);
    Ok(tags)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seasons_follow_hemisphere() {
        assert_eq!(season(1, Some(55.0)), "winter");
        assert_eq!(season(1, Some(-35.0)), "summer");
        assert_eq!(season(7, Some(55.0)), "summer");
    }

    #[test]
    fn solar_noon_and_midnight_are_distinct() {
        let noon = Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap();
        let midnight = Utc.with_ymd_and_hms(2026, 6, 21, 0, 0, 0).unwrap();
        let (noon_elevation, _) = solar_position(noon, 0.0, 0.0);
        let (midnight_elevation, _) = solar_position(midnight, 0.0, 0.0);
        assert!(noon_elevation > 60.0);
        assert!(midnight_elevation < -60.0);
    }

    #[test]
    fn low_winter_sun_after_solar_noon_is_evening() {
        let timestamp = Utc.with_ymd_and_hms(2015, 12, 22, 13, 25, 29).unwrap();
        let (elevation, solar_hour) = solar_position(timestamp, 56.381, 10.511);
        let (time_of_day, phase) = solar_tags(elevation, solar_hour);
        assert_eq!(time_of_day, "evening");
        assert_eq!(phase, "low sun");
    }

    #[test]
    fn solar_time_uses_five_description_labels() {
        assert_eq!(solar_tags(-7.0, 2.0).0, "nighttime");
        assert_eq!(solar_tags(-1.0, 6.0).0, "dawn");
        assert_eq!(solar_tags(5.0, 8.0).0, "dawn");
        assert_eq!(solar_tags(20.0, 12.0).0, "daytime");
        assert_eq!(solar_tags(5.0, 18.0).0, "evening");
        assert_eq!(solar_tags(-1.0, 18.0).0, "dusk");
    }

    #[test]
    fn search_text_contains_semantic_time_tags() {
        let tags = ImageTags {
            version: TAG_VERSION,
            country: Some("Denmark".to_string()),
            region: Some("Region Zealand".to_string()),
            nearest_town: Some("Roskilde".to_string()),
            nearby_towns: vec!["Roskilde".to_string(), "Køge".to_string(), "Holbæk".to_string()],
            nearest_town_distance_km: Some(10.1),
            season: Some("winter".to_string()),
            time_of_day: Some("evening".to_string()),
            solar_phase: Some("civil twilight".to_string()),
            camera_make: Some("Microsoft".to_string()),
            camera_model: Some("Lumia 930".to_string()),
            altitude_m: Some(42.0),
            ..ImageTags::default()
        };
        let text = build_search_text(&tags).unwrap();
        assert!(text.starts_with("Photographed near Roskilde, Køge, and Holbæk in Region Zealand, Denmark"));
        assert!(text.contains("winter"));
        assert!(text.contains("evening"));
        assert!(text.contains("42 metres"));
        assert!(!text.contains("civil twilight"));
        assert!(!text.contains("Microsoft"));
        assert!(!text.contains("Lumia"));
    }

    #[test]
    fn search_text_uses_only_city_when_within_ten_kilometres() {
        let tags = ImageTags {
            nearest_town: Some("Copenhagen".to_string()),
            nearby_towns: vec![
                "Copenhagen".to_string(),
                "Roskilde".to_string(),
                "Helsingør".to_string(),
            ],
            nearest_town_distance_km: Some(10.0),
            region: Some("Capital Region".to_string()),
            country: Some("Denmark".to_string()),
            ..ImageTags::default()
        };

        assert_eq!(
            build_search_text(&tags).unwrap(),
            "Photographed in Copenhagen, Capital Region, Denmark."
        );
    }

    #[test]
    fn search_text_does_not_repeat_city_as_region() {
        let tags = ImageTags {
            nearest_town: Some("Stockholm".to_string()),
            nearest_town_distance_km: Some(2.0),
            region: Some("Stockholm".to_string()),
            country: Some("Sweden".to_string()),
            ..ImageTags::default()
        };

        assert_eq!(build_search_text(&tags).unwrap(), "Photographed in Stockholm, Sweden.");
    }

    #[test]
    fn nearby_town_lists_use_natural_conjunctions() {
        assert_eq!(human_list(&["Randers".into(), "Aarhus".into()]), "Randers and Aarhus");
        assert_eq!(
            human_list(&["Randers".into(), "Aarhus".into(), "Grenaa".into()]),
            "Randers, Aarhus, and Grenaa"
        );
    }
}
