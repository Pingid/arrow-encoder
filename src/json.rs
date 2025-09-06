use arrow::{
    array::{ArrayBuilder, ArrayRef},
    datatypes::{DataType as DT, Field, IntervalUnit, TimeUnit},
    error::ArrowError,
};
use base64::Engine;
use chrono::{FixedOffset, NaiveDateTime, TimeZone, Timelike};
use half::f16;
use serde_json::Value as Json;
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use super::{BuilderFromField, EncodeBuilder as EB, EncodeBuilderError, RowBuilder};

#[derive(Debug, thiserror::Error)]
pub enum JsonEncodeError {
    #[error("Field is not nullable but value is null")]
    UnexpectedNullValue,
    #[error("Unexpected type for field {0}: expected {2}, found {1}")]
    UnexpectedType(String, DT, serde_json::Value),
    #[error("Unsupported data type: {0}")]
    UnsupportedType(DT),
    #[error("Failed to encode RecordBatch: {0}")]
    ArrowError(#[from] ArrowError),
    #[error("Failed to serialize value: {0}")]
    SerdeError(#[from] serde_json::Error),
    #[error("Failed to parse timestamp: {0}")]
    TimestampParseError(#[from] chrono::ParseError),
    #[error("Invalid date value: {0}")]
    InvalidDateValue(String),
    #[error("Invalid time value: {0}")]
    InvalidTimeValue(String),
    #[error("Invalid duration value: {0}")]
    InvalidDurationValue(String),
    #[error("Invalid interval value: {0}")]
    InvalidIntervalValue(String),
    #[error("Failed to create EncodeBuilder: {0}")]
    EncodeBuilderError(#[from] EncodeBuilderError),
}

pub struct JsonEncodeBuilder(EB);

impl Deref for JsonEncodeBuilder {
    type Target = EB;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for JsonEncodeBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl JsonEncodeBuilder {
    pub fn try_new(data_type: &DT) -> Result<Self, JsonEncodeError> {
        Ok(Self(EB::try_new(data_type)?))
    }
    pub fn try_with_capacity(data_type: &DT, capacity: usize) -> Result<Self, JsonEncodeError> {
        Ok(Self(EB::try_with_capacity(data_type, capacity)?))
    }
}

impl BuilderFromField for JsonEncodeBuilder {
    type Error = JsonEncodeError;
    fn try_from_field(field: &Arc<Field>) -> Result<Self, Self::Error> {
        JsonEncodeBuilder::try_new(field.data_type())
    }
}

impl RowBuilder for JsonEncodeBuilder {
    type Error = JsonEncodeError;
    type Row = serde_json::Value;

    fn append_row(&mut self, field: &Arc<Field>, value: &Self::Row) -> Result<(), Self::Error> {
        let value = value.get(field.name()).unwrap_or(&Json::Null);
        append_json(&mut self.0, field, value)
    }

    fn finish(&mut self) -> ArrayRef {
        ArrayBuilder::finish(&mut self.0)
    }
}

/// Appends a JSON value to the appropriate Arrow builder based on the field's data type.
pub fn append_json(
    encoder: &mut EB,
    field: &Arc<Field>,
    value: &Json,
) -> Result<(), JsonEncodeError> {
    if value.is_null() {
        if !field.is_nullable() {
            return Err(JsonEncodeError::UnexpectedNullValue);
        }
        return Ok(encoder.append_null());
    }

    let unexpected = || {
        JsonEncodeError::UnexpectedType(
            field.name().to_string(),
            field.data_type().clone(),
            value.clone(),
        )
    };

    match (encoder, field.data_type(), value) {
        (EB::Null(b), DT::Null, Json::Null) => {
            b.append_null();
            Ok(())
        }
        (EB::Bool(b), DT::Boolean, Json::Bool(v)) => {
            b.append_value(*v);
            Ok(())
        }
        // Signed integers
        (EB::I8(b), DT::Int8, Json::Number(n)) => {
            let val = n.as_i64().ok_or_else(unexpected)?;
            if val < i8::MIN as i64 || val > i8::MAX as i64 {
                return Err(unexpected());
            }
            b.append_value(val as i8);
            Ok(())
        }
        (EB::I16(b), DT::Int16, Json::Number(n)) => {
            let val = n.as_i64().ok_or_else(unexpected)?;
            if val < i16::MIN as i64 || val > i16::MAX as i64 {
                return Err(unexpected());
            }
            b.append_value(val as i16);
            Ok(())
        }
        (EB::I32(b), DT::Int32, Json::Number(n)) => {
            let val = n.as_i64().ok_or_else(unexpected)?;
            if val < i32::MIN as i64 || val > i32::MAX as i64 {
                return Err(unexpected());
            }
            b.append_value(val as i32);
            Ok(())
        }
        (EB::I64(b), DT::Int64, Json::Number(n)) => {
            b.append_value(n.as_i64().ok_or_else(unexpected)?);
            Ok(())
        }
        // Unsigned integers
        (EB::UI8(b), DT::UInt8, Json::Number(n)) => {
            let val = n.as_u64().ok_or_else(unexpected)?;
            if val > u8::MAX as u64 {
                return Err(unexpected());
            }
            b.append_value(val as u8);
            Ok(())
        }
        (EB::UI16(b), DT::UInt16, Json::Number(n)) => {
            let val = n.as_u64().ok_or_else(unexpected)?;
            if val > u16::MAX as u64 {
                return Err(unexpected());
            }
            b.append_value(val as u16);
            Ok(())
        }
        (EB::UI32(b), DT::UInt32, Json::Number(n)) => {
            let val = n.as_u64().ok_or_else(unexpected)?;
            if val > u32::MAX as u64 {
                return Err(unexpected());
            }
            b.append_value(val as u32);
            Ok(())
        }
        (EB::UI64(b), DT::UInt64, Json::Number(n)) => {
            b.append_value(n.as_u64().ok_or_else(unexpected)?);
            Ok(())
        }
        // Floats
        (EB::F16(b), DT::Float16, Json::Number(n)) => {
            b.append_value(f16::from_f64(n.as_f64().ok_or_else(unexpected)?));
            Ok(())
        }
        (EB::F32(b), DT::Float32, Json::Number(n)) => {
            b.append_value(n.as_f64().ok_or_else(unexpected)? as f32);
            Ok(())
        }
        (EB::F64(b), DT::Float64, Json::Number(n)) => {
            b.append_value(n.as_f64().ok_or_else(unexpected)?);
            Ok(())
        }

        // Timestamp types
        (EB::TsSecond(b), DT::Timestamp(TimeUnit::Second, _tz), v) => {
            let dt = parse_timestamp_from_json(v)?;
            b.append_value(dt.timestamp());
            Ok(())
        }
        (EB::TsMs(b), DT::Timestamp(TimeUnit::Millisecond, _tz), v) => {
            let dt = parse_timestamp_from_json(v)?;
            b.append_value(dt.timestamp_millis());
            Ok(())
        }
        (EB::TsMicro(b), DT::Timestamp(TimeUnit::Microsecond, _tz), v) => {
            let dt = parse_timestamp_from_json(v)?;
            b.append_value(dt.timestamp_micros());
            Ok(())
        }
        (EB::TsNano(b), DT::Timestamp(TimeUnit::Nanosecond, _tz), v) => {
            let dt = parse_timestamp_from_json(v)?;
            b.append_value(dt.timestamp_nanos_opt().unwrap_or(0));
            Ok(())
        }

        // Date types
        (EB::Date32(b), DT::Date32, v) => {
            let date = parse_date_from_json(v)?;
            let days_since_epoch =
                (date - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap()).num_days();
            b.append_value(days_since_epoch as i32);
            Ok(())
        }
        (EB::Date64(b), DT::Date64, v) => {
            let date = parse_date_from_json(v)?;
            let millis_since_epoch = (date - chrono::NaiveDate::from_ymd_opt(1970, 1, 1).unwrap())
                .num_days()
                * 86_400_000;
            b.append_value(millis_since_epoch);
            Ok(())
        }

        // Time types
        (EB::Time32Second(b), DT::Time32(TimeUnit::Second), v) => {
            let time = parse_time_from_json(v)?;
            let seconds_since_midnight = time.num_seconds_from_midnight();
            b.append_value(seconds_since_midnight as i32);
            Ok(())
        }
        (EB::Time32Ms(b), DT::Time32(TimeUnit::Millisecond), v) => {
            let time = parse_time_from_json(v)?;
            let millis_since_midnight =
                time.num_seconds_from_midnight() * 1000 + (time.nanosecond() / 1_000_000);
            b.append_value(millis_since_midnight as i32);
            Ok(())
        }
        (EB::Time64Micro(b), DT::Time64(TimeUnit::Microsecond), v) => {
            let time = parse_time_from_json(v)?;
            let micros_since_midnight = time.num_seconds_from_midnight() as i64 * 1_000_000
                + (time.nanosecond() / 1000) as i64;
            b.append_value(micros_since_midnight);
            Ok(())
        }
        (EB::Time64Nano(b), DT::Time64(TimeUnit::Nanosecond), v) => {
            let time = parse_time_from_json(v)?;
            let nanos_since_midnight =
                time.num_seconds_from_midnight() as i64 * 1_000_000_000 + time.nanosecond() as i64;
            b.append_value(nanos_since_midnight);
            Ok(())
        }

        // Duration types
        (EB::DurationSecond(b), DT::Duration(TimeUnit::Second), v) => {
            let duration_secs = parse_duration_from_json(v)?;
            b.append_value(duration_secs);
            Ok(())
        }
        (EB::DurationMs(b), DT::Duration(TimeUnit::Millisecond), v) => {
            let duration_secs = parse_duration_from_json(v)?;
            b.append_value(duration_secs * 1000);
            Ok(())
        }
        (EB::DurationMicro(b), DT::Duration(TimeUnit::Microsecond), v) => {
            let duration_secs = parse_duration_from_json(v)?;
            b.append_value(duration_secs * 1_000_000);
            Ok(())
        }
        (EB::DurationNano(b), DT::Duration(TimeUnit::Nanosecond), v) => {
            let duration_secs = parse_duration_from_json(v)?;
            b.append_value(duration_secs * 1_000_000_000);
            Ok(())
        }

        // Interval types
        (EB::IntervalYearMonth(b), DT::Interval(IntervalUnit::YearMonth), Json::Object(obj)) => {
            let years = obj.get("years").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let months = obj.get("months").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            b.append_value(years * 12 + months);
            Ok(())
        }
        (EB::IntervalYearMonth(b), DT::Interval(IntervalUnit::YearMonth), Json::Number(n)) => {
            // Assume months
            b.append_value(n.as_i64().ok_or_else(unexpected)? as i32);
            Ok(())
        }
        (EB::IntervalDayTime(b), DT::Interval(IntervalUnit::DayTime), Json::Object(obj)) => {
            let days = obj.get("days").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let millis = obj
                .get("milliseconds")
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as i32;
            let interval = arrow::datatypes::IntervalDayTimeType::make_value(days, millis);
            b.append_value(interval);
            Ok(())
        }
        (
            EB::IntervalMonthDayNano(b),
            DT::Interval(IntervalUnit::MonthDayNano),
            Json::Object(obj),
        ) => {
            let months = obj.get("months").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let days = obj.get("days").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
            let nanos = obj.get("nanoseconds").and_then(|v| v.as_i64()).unwrap_or(0);
            let interval =
                arrow::datatypes::IntervalMonthDayNanoType::make_value(months, days, nanos);
            b.append_value(interval);
            Ok(())
        }
        // Binary types
        (EB::Binary(b), DT::Binary, Json::String(s)) => {
            // Base64 decode or raw bytes from string
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(s) {
                b.append_value(&bytes);
            } else {
                // Fall back to raw bytes
                b.append_value(s.as_bytes());
            }
            Ok(())
        }
        (EB::Binary(b), DT::Binary, Json::Array(arr)) => {
            // Array of byte values
            let bytes: Result<Vec<u8>, _> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| if n <= 255 { Some(n as u8) } else { None })
                        .ok_or_else(unexpected)
                })
                .collect();
            b.append_value(&bytes?);
            Ok(())
        }
        (EB::FixedSizeBinary(b), DT::FixedSizeBinary(size), Json::String(s)) => {
            // Base64 decode or raw bytes from string
            let bytes = if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(s) {
                decoded
            } else {
                s.as_bytes().to_vec()
            };
            if bytes.len() != *size as usize {
                return Err(JsonEncodeError::UnexpectedType(
                    field.name().to_string(),
                    field.data_type().clone(),
                    value.clone(),
                ));
            }
            b.append_value(&bytes)?;
            Ok(())
        }
        (EB::FixedSizeBinary(b), DT::FixedSizeBinary(size), Json::Array(arr)) => {
            // Array of byte values
            if arr.len() != *size as usize {
                return Err(unexpected());
            }
            let bytes: Result<Vec<u8>, _> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| if n <= 255 { Some(n as u8) } else { None })
                        .ok_or_else(unexpected)
                })
                .collect();
            b.append_value(&bytes?)?;
            Ok(())
        }
        (EB::LargeBinary(b), DT::LargeBinary, Json::String(s)) => {
            // Base64 decode or raw bytes from string
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(s) {
                b.append_value(&bytes);
            } else {
                // Fall back to raw bytes
                b.append_value(s.as_bytes());
            }
            Ok(())
        }
        (EB::LargeBinary(b), DT::LargeBinary, Json::Array(arr)) => {
            // Array of byte values
            let bytes: Result<Vec<u8>, _> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| if n <= 255 { Some(n as u8) } else { None })
                        .ok_or_else(unexpected)
                })
                .collect();
            b.append_value(&bytes?);
            Ok(())
        }
        (EB::BinaryView(b), DT::BinaryView, Json::String(s)) => {
            // Base64 decode or raw bytes from string
            if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(s) {
                b.append_value(&bytes);
            } else {
                // Fall back to raw bytes
                b.append_value(s.as_bytes());
            }
            Ok(())
        }
        (EB::BinaryView(b), DT::BinaryView, Json::Array(arr)) => {
            // Array of byte values
            let bytes: Result<Vec<u8>, _> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| if n <= 255 { Some(n as u8) } else { None })
                        .ok_or_else(unexpected)
                })
                .collect();
            b.append_value(&bytes?);
            Ok(())
        }

        // String types
        (EB::Str(b), DT::Utf8, Json::String(s)) => {
            b.append_value(s);
            Ok(())
        }
        (EB::Str(b), DT::Utf8, s) => {
            b.append_value(serde_json::to_string(s)?);
            Ok(())
        }
        (EB::LargeStr(b), DT::LargeUtf8, Json::String(s)) => {
            b.append_value(s);
            Ok(())
        }
        (EB::LargeStr(b), DT::LargeUtf8, s) => {
            b.append_value(serde_json::to_string(s)?);
            Ok(())
        }
        (EB::StrView(b), DT::Utf8View, Json::String(s)) => {
            b.append_value(s);
            Ok(())
        }
        (EB::StrView(b), DT::Utf8View, s) => {
            b.append_value(serde_json::to_string(s)?);
            Ok(())
        }

        // List
        (EB::List(b), DT::List(f), Json::Array(arr)) => {
            let values_b: &mut Box<EB> = b.values();
            for elem in arr {
                if elem.is_null() {
                    if !f.is_nullable() {
                        return Err(JsonEncodeError::UnexpectedNullValue);
                    }
                    values_b.append_null();
                } else {
                    append_json(values_b, f, elem)?;
                }
            }
            b.append(true)?;
            Ok(())
        }
        (EB::FixedSizeList(b), DT::FixedSizeList(f, size), Json::Array(arr)) => {
            if arr.len() != *size as usize {
                return Err(unexpected());
            }
            let values_b: &mut Box<EB> = b.values();
            for elem in arr {
                if elem.is_null() {
                    if !f.is_nullable() {
                        return Err(JsonEncodeError::UnexpectedNullValue);
                    }
                    values_b.append_null();
                } else {
                    append_json(values_b, f, elem)?;
                }
            }
            b.append(true)?;
            Ok(())
        }

        // LargeList - same as List but with different builder
        (EB::LargeList(b), DT::LargeList(f), Json::Array(arr)) => {
            let values_b: &mut Box<EB> = b.values();
            for elem in arr {
                if elem.is_null() {
                    if !f.is_nullable() {
                        return Err(JsonEncodeError::UnexpectedNullValue);
                    }
                    values_b.append_null();
                } else {
                    append_json(values_b, f, elem)?;
                }
            }
            b.append(true)?;
            Ok(())
        }

        // Decimal types - parse from string or number
        // Decimal32 supported only with Arrow 56+
        #[cfg(feature = "arrow-56")]
        (EB::Decimal32(b), DT::Decimal32(_precision, scale), Json::Number(n)) => {
            let scaled_val =
                (n.as_f64().ok_or_else(unexpected)? * 10_f64.powi(*scale as i32)) as i32;
            b.append_value(scaled_val);
            Ok(())
        }
        #[cfg(feature = "arrow-56")]
        (EB::Decimal32(b), DT::Decimal32(_precision, scale), Json::String(s)) => {
            let num: f64 = s.parse().map_err(|_| unexpected())?;
            let scaled_val = (num * 10_f64.powi(*scale as i32)) as i32;
            b.append_value(scaled_val);
            Ok(())
        }

        (EB::Decimal128(b), DT::Decimal128(_precision, scale), Json::Number(n)) => {
            let scaled_val =
                (n.as_f64().ok_or_else(unexpected)? * 10_f64.powi(*scale as i32)) as i128;
            b.append_value(scaled_val);
            Ok(())
        }
        (EB::Decimal128(b), DT::Decimal128(_precision, scale), Json::String(s)) => {
            let num: f64 = s.parse().map_err(|_| unexpected())?;
            let scaled_val = (num * 10_f64.powi(*scale as i32)) as i128;
            b.append_value(scaled_val);
            Ok(())
        }

        (EB::Decimal256(b), DT::Decimal256(_precision, scale), Json::Number(n)) => {
            use arrow::datatypes::i256;
            let scaled_val =
                (n.as_f64().ok_or_else(unexpected)? * 10_f64.powi(*scale as i32)) as i128;
            b.append_value(i256::from_i128(scaled_val));
            Ok(())
        }
        (EB::Decimal256(b), DT::Decimal256(_precision, scale), Json::String(s)) => {
            use arrow::datatypes::i256;
            let num: f64 = s.parse().map_err(|_| unexpected())?;
            let scaled_val = (num * 10_f64.powi(*scale as i32)) as i128;
            b.append_value(i256::from_i128(scaled_val));
            Ok(())
        }

        // Dictionary - encode string values
        (EB::Dictionary(b), DT::Dictionary(_key_type, _value_type), Json::String(s)) => {
            b.append_value(s)?;
            Ok(())
        }

        // Map - encode as JSON object with key-value pairs
        (EB::Map(b), DT::Map(_field, _sorted), Json::Object(obj)) => {
            for (key, val) in obj {
                b.keys().append_value(key);
                if val.is_null() {
                    b.values().append_null();
                } else {
                    // Extract value field from map field structure
                    if let DT::Map(map_field, _) = field.data_type() {
                        if let DT::Struct(struct_fields) = map_field.data_type() {
                            if struct_fields.len() >= 2 {
                                let value_field = &struct_fields[1];
                                append_json(b.values(), value_field, val)?;
                            }
                        }
                    }
                }
            }
            b.append(true)?;
            Ok(())
        }

        // Struct
        (EB::Struct(b), DT::Struct(fields), Json::Object(map)) => {
            for (i, builder) in b.builders.iter_mut().enumerate() {
                let field = fields.get(i).unwrap();
                let value = map.get(field.name()).unwrap_or(&Json::Null);

                // Check for null and append
                if value.is_null() && !field.is_nullable() {
                    return Err(JsonEncodeError::UnexpectedNullValue);
                }

                append_json(builder, field, value)?;
            }
            b.append(true);
            Ok(())
        }
        _ => Err(unexpected()),
    }
}

// Helper functions for parsing date/time values
fn parse_timestamp_from_json(
    value: &Json,
) -> Result<chrono::DateTime<chrono::FixedOffset>, JsonEncodeError> {
    match value {
        Json::String(s) => {
            // Try parsing ISO 8601 first, then RFC 2822
            chrono::DateTime::parse_from_rfc3339(s)
                .or_else(|_| chrono::DateTime::parse_from_rfc2822(s))
                .or_else(|_| {
                    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                        .map(|naive| FixedOffset::east_opt(0).unwrap().from_utc_datetime(&naive))
                })
                .map_err(JsonEncodeError::TimestampParseError)
        }
        // 2025-09-02T21:15:22.276
        Json::Number(n) => {
            let timestamp = n.as_i64().ok_or_else(|| {
                JsonEncodeError::InvalidDateValue("Invalid timestamp number".to_string())
            })?;

            // Detect if timestamp is in seconds or milliseconds
            // Timestamps after year 2001 (978307200) in seconds would be > 978307200
            // Timestamps in milliseconds would be much larger (> 978307200000)
            let (secs, nanos) = if timestamp > 1_000_000_000_000 {
                // Likely milliseconds - convert to seconds and nanoseconds
                let millis = timestamp % 1000;
                (timestamp / 1000, (millis * 1_000_000) as u32)
            } else {
                // Likely seconds
                (timestamp, 0)
            };

            chrono::DateTime::from_timestamp(secs, nanos)
                .map(|dt| dt.fixed_offset())
                .ok_or_else(|| JsonEncodeError::InvalidDateValue("Invalid timestamp".to_string()))
        }
        _ => Err(JsonEncodeError::InvalidDateValue(
            "Invalid timestamp format".to_string(),
        )),
    }
}

fn parse_date_from_json(value: &Json) -> Result<chrono::NaiveDate, JsonEncodeError> {
    match value {
        Json::String(s) => {
            // Try parsing date string
            chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
                .or_else(|_| chrono::NaiveDate::parse_from_str(s, "%m/%d/%Y"))
                .or_else(|_| chrono::NaiveDate::parse_from_str(s, "%d/%m/%Y"))
                .map_err(|e| {
                    JsonEncodeError::InvalidDateValue(format!("Failed to parse date: {}", e))
                })
        }
        Json::Number(n) => {
            // Assume days since Unix epoch
            let days = n.as_i64().ok_or_else(|| {
                JsonEncodeError::InvalidDateValue("Invalid date number".to_string())
            })?;
            chrono::NaiveDate::from_num_days_from_ce_opt(days as i32 + 719163) // Unix epoch is 719163 days from CE
                .ok_or_else(|| JsonEncodeError::InvalidDateValue("Invalid date value".to_string()))
        }
        _ => Err(JsonEncodeError::InvalidDateValue(
            "Invalid date format".to_string(),
        )),
    }
}

fn parse_time_from_json(value: &Json) -> Result<chrono::NaiveTime, JsonEncodeError> {
    match value {
        Json::String(s) => {
            // Try parsing time string
            chrono::NaiveTime::parse_from_str(s, "%H:%M:%S")
                .or_else(|_| chrono::NaiveTime::parse_from_str(s, "%H:%M:%S%.f"))
                .or_else(|_| chrono::NaiveTime::parse_from_str(s, "%H:%M"))
                .map_err(|e| {
                    JsonEncodeError::InvalidTimeValue(format!("Failed to parse time: {}", e))
                })
        }
        Json::Number(n) => {
            // Assume seconds since midnight
            let seconds = n.as_i64().ok_or_else(|| {
                JsonEncodeError::InvalidTimeValue("Invalid time number".to_string())
            })?;
            chrono::NaiveTime::from_num_seconds_from_midnight_opt(seconds as u32, 0)
                .ok_or_else(|| JsonEncodeError::InvalidTimeValue("Invalid time value".to_string()))
        }
        _ => Err(JsonEncodeError::InvalidTimeValue(
            "Invalid time format".to_string(),
        )),
    }
}

fn parse_duration_from_json(value: &Json) -> Result<i64, JsonEncodeError> {
    match value {
        Json::String(s) => {
            // Try parsing ISO 8601 duration format (P[n]Y[n]M[n]DT[n]H[n]M[n]S)
            // For simplicity, we'll support basic formats like "1h30m", "90s", etc.
            if s.ends_with('s') {
                s[..s.len() - 1].parse::<i64>().map_err(|e| {
                    JsonEncodeError::InvalidDurationValue(format!("Failed to parse seconds: {}", e))
                })
            } else if s.ends_with('m') {
                s[..s.len() - 1]
                    .parse::<i64>()
                    .map(|m| m * 60)
                    .map_err(|e| {
                        JsonEncodeError::InvalidDurationValue(format!(
                            "Failed to parse minutes: {}",
                            e
                        ))
                    })
            } else if s.ends_with('h') {
                s[..s.len() - 1]
                    .parse::<i64>()
                    .map(|h| h * 3600)
                    .map_err(|e| {
                        JsonEncodeError::InvalidDurationValue(format!(
                            "Failed to parse hours: {}",
                            e
                        ))
                    })
            } else {
                Err(JsonEncodeError::InvalidDurationValue(
                    "Unsupported duration format".to_string(),
                ))
            }
        }
        Json::Number(n) => n.as_i64().ok_or_else(|| {
            JsonEncodeError::InvalidDurationValue("Invalid duration number".to_string())
        }),
        _ => Err(JsonEncodeError::InvalidDurationValue(
            "Invalid duration format".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use arrow::{array::RecordBatch, datatypes::Schema};
    use serde_json::json;

    // Test utilities
    fn schema_with(field: Field) -> Arc<Schema> {
        Arc::new(Schema::new(vec![field]))
    }

    fn encode_single(
        schema: Arc<Schema>,
        value: Json,
    ) -> Result<Vec<RecordBatch>, JsonEncodeError> {
        encode_values(schema, &[value])
    }

    fn assert_encode_error<F>(schema: Arc<Schema>, value: Json, error_check: F)
    where
        F: Fn(&JsonEncodeError) -> bool,
    {
        let result = encode_single(schema, value);
        assert!(result.is_err());
        assert!(error_check(&result.unwrap_err()));
    }

    // Integer boundary tests
    #[test]
    fn test_i8_boundaries() {
        let schema = schema_with(Field::new("i8", DT::Int8, false));
        encode_single(schema.clone(), json!({"i8": 127})).unwrap();
        encode_single(schema.clone(), json!({"i8": -128})).unwrap();
        assert_encode_error(schema.clone(), json!({"i8": 128}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
        assert_encode_error(schema, json!({"i8": -129}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_i16_boundaries() {
        let schema = schema_with(Field::new("i16", DT::Int16, false));
        encode_single(schema.clone(), json!({"i16": 32767})).unwrap();
        encode_single(schema.clone(), json!({"i16": -32768})).unwrap();
        assert_encode_error(schema.clone(), json!({"i16": 32768}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
        assert_encode_error(schema, json!({"i16": -32769}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_i32_boundaries() {
        let schema = schema_with(Field::new("i32", DT::Int32, false));
        encode_single(schema.clone(), json!({"i32": 2147483647})).unwrap();
        encode_single(schema.clone(), json!({"i32": -2147483648i64})).unwrap();
        assert_encode_error(schema.clone(), json!({"i32": 2147483648i64}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
        assert_encode_error(schema, json!({"i32": -2147483649i64}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_u8_boundaries() {
        let schema = schema_with(Field::new("u8", DT::UInt8, false));
        encode_single(schema.clone(), json!({"u8": 255})).unwrap();
        encode_single(schema.clone(), json!({"u8": 0})).unwrap();
        assert_encode_error(schema, json!({"u8": 256}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_u16_boundaries() {
        let schema = schema_with(Field::new("u16", DT::UInt16, false));
        encode_single(schema.clone(), json!({"u16": 65535})).unwrap();
        encode_single(schema.clone(), json!({"u16": 0})).unwrap();
        assert_encode_error(schema, json!({"u16": 65536}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_u32_boundaries() {
        let schema = schema_with(Field::new("u32", DT::UInt32, false));
        encode_single(schema.clone(), json!({"u32": 4294967295u64})).unwrap();
        encode_single(schema.clone(), json!({"u32": 0})).unwrap();
        assert_encode_error(schema, json!({"u32": 4294967296u64}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    // Float precision tests
    #[test]
    fn test_float16_precision() {
        let schema = schema_with(Field::new("f16", DT::Float16, false));
        encode_single(schema.clone(), json!({"f16": 65504.0})).unwrap(); // max f16
        encode_single(schema.clone(), json!({"f16": -65504.0})).unwrap();
        encode_single(schema, json!({"f16": 0.00006103515625})).unwrap(); // min positive normal
    }

    #[test]
    fn test_float32_special_values() {
        let schema = schema_with(Field::new("f32", DT::Float32, false));
        encode_single(schema.clone(), json!({"f32": 3.4028235e38})).unwrap();
        encode_single(schema.clone(), json!({"f32": -3.4028235e38})).unwrap();
        encode_single(schema, json!({"f32": 1.175494e-38})).unwrap();
    }

    // Timestamp tests
    #[test]
    fn test_timestamp_rfc3339() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Second, None),
            false,
        ));
        encode_single(schema, json!({"ts": "2025-09-02T21:15:22Z"})).unwrap();
    }

    #[test]
    fn test_timestamp_rfc2822() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Second, None),
            false,
        ));
        encode_single(schema, json!({"ts": "Tue, 02 Sep 2025 21:15:22 +0000"})).unwrap();
    }

    #[test]
    fn test_timestamp_naive() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Second, None),
            false,
        ));
        encode_single(schema, json!({"ts": "2025-09-02T21:15:22.276"})).unwrap();
    }

    #[test]
    fn test_timestamp_unix() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Second, None),
            false,
        ));
        encode_single(schema, json!({"ts": 1756847722})).unwrap();
    }

    #[test]
    fn test_timestamp_milliseconds() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Millisecond, None),
            false,
        ));
        encode_single(schema, json!({"ts": "2025-09-02T21:15:22.276Z"})).unwrap();
    }

    #[test]
    fn test_timestamp_microseconds() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Microsecond, None),
            false,
        ));
        encode_single(schema, json!({"ts": "2025-09-02T21:15:22.276123Z"})).unwrap();
    }

    #[test]
    fn test_timestamp_nanoseconds() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ));
        encode_single(schema, json!({"ts": "2025-09-02T21:15:22.276123456Z"})).unwrap();
    }

    // Date tests
    #[test]
    fn test_date32_string_formats() {
        let schema = schema_with(Field::new("date", DT::Date32, false));
        encode_single(schema.clone(), json!({"date": "2025-09-02"})).unwrap();
        encode_single(schema.clone(), json!({"date": "09/02/2025"})).unwrap();
        encode_single(schema, json!({"date": "02/09/2025"})).unwrap();
    }

    #[test]
    fn test_date32_numeric() {
        let schema = schema_with(Field::new("date", DT::Date32, false));
        encode_single(schema, json!({"date": 20335})).unwrap(); // days since epoch
    }

    #[test]
    fn test_date64_string() {
        let schema = schema_with(Field::new("date", DT::Date64, false));
        encode_single(schema, json!({"date": "2025-09-02"})).unwrap();
    }

    // Time tests
    #[test]
    fn test_time32_second() {
        let schema = schema_with(Field::new("time", DT::Time32(TimeUnit::Second), false));
        encode_single(schema.clone(), json!({"time": "21:15:22"})).unwrap();
        encode_single(schema, json!({"time": 76522})).unwrap(); // seconds since midnight
    }

    #[test]
    fn test_time32_millisecond() {
        let schema = schema_with(Field::new("time", DT::Time32(TimeUnit::Millisecond), false));
        encode_single(schema.clone(), json!({"time": "21:15:22.276"})).unwrap();
        encode_single(schema, json!({"time": "21:15:22"})).unwrap();
    }

    #[test]
    fn test_time64_microsecond() {
        let schema = schema_with(Field::new("time", DT::Time64(TimeUnit::Microsecond), false));
        encode_single(schema.clone(), json!({"time": "21:15:22.276123"})).unwrap();
        encode_single(schema, json!({"time": "21:15"})).unwrap();
    }

    #[test]
    fn test_time64_nanosecond() {
        let schema = schema_with(Field::new("time", DT::Time64(TimeUnit::Nanosecond), false));
        encode_single(schema, json!({"time": "21:15:22.276123456"})).unwrap();
    }

    // Duration tests
    #[test]
    fn test_duration_seconds() {
        let schema = schema_with(Field::new("dur", DT::Duration(TimeUnit::Second), false));
        encode_single(schema.clone(), json!({"dur": "90s"})).unwrap();
        encode_single(schema, json!({"dur": 90})).unwrap();
    }

    #[test]
    fn test_duration_minutes() {
        let schema = schema_with(Field::new("dur", DT::Duration(TimeUnit::Second), false));
        encode_single(schema, json!({"dur": "2m"})).unwrap();
    }

    #[test]
    fn test_duration_hours() {
        let schema = schema_with(Field::new("dur", DT::Duration(TimeUnit::Second), false));
        encode_single(schema, json!({"dur": "1h"})).unwrap();
    }

    #[test]
    fn test_duration_milliseconds() {
        let schema = schema_with(Field::new(
            "dur",
            DT::Duration(TimeUnit::Millisecond),
            false,
        ));
        encode_single(schema.clone(), json!({"dur": "90s"})).unwrap();
        encode_single(schema, json!({"dur": 90000})).unwrap();
    }

    // Interval tests
    #[test]
    fn test_interval_year_month_object() {
        let schema = schema_with(Field::new(
            "interval",
            DT::Interval(IntervalUnit::YearMonth),
            false,
        ));
        encode_single(schema, json!({"interval": {"years": 2, "months": 3}})).unwrap();
    }

    #[test]
    fn test_interval_year_month_number() {
        let schema = schema_with(Field::new(
            "interval",
            DT::Interval(IntervalUnit::YearMonth),
            false,
        ));
        encode_single(schema, json!({"interval": 27})).unwrap(); // 27 months
    }

    #[test]
    fn test_interval_day_time() {
        let schema = schema_with(Field::new(
            "interval",
            DT::Interval(IntervalUnit::DayTime),
            false,
        ));
        encode_single(
            schema,
            json!({"interval": {"days": 5, "milliseconds": 3600000}}),
        )
        .unwrap();
    }

    #[test]
    fn test_interval_month_day_nano() {
        let schema = schema_with(Field::new(
            "interval",
            DT::Interval(IntervalUnit::MonthDayNano),
            false,
        ));
        encode_single(
            schema,
            json!({"interval": {"months": 1, "days": 15, "nanoseconds": 1000000}}),
        )
        .unwrap();
    }

    // Binary tests
    #[test]
    fn test_binary_base64() {
        let schema = schema_with(Field::new("bin", DT::Binary, false));
        encode_single(schema, json!({"bin": "SGVsbG8gV29ybGQ="})).unwrap(); // "Hello World"
    }

    #[test]
    fn test_binary_raw_string() {
        let schema = schema_with(Field::new("bin", DT::Binary, false));
        encode_single(schema, json!({"bin": "raw bytes"})).unwrap();
    }

    #[test]
    fn test_binary_array() {
        let schema = schema_with(Field::new("bin", DT::Binary, false));
        encode_single(schema, json!({"bin": [72, 101, 108, 108, 111]})).unwrap(); // "Hello"
    }

    #[test]
    fn test_fixed_size_binary_base64() {
        let schema = schema_with(Field::new("bin", DT::FixedSizeBinary(5), false));
        encode_single(schema, json!({"bin": "SGVsbG8="})).unwrap(); // "Hello" (5 bytes)
    }

    #[test]
    fn test_fixed_size_binary_array() {
        let schema = schema_with(Field::new("bin", DT::FixedSizeBinary(3), false));
        encode_single(schema.clone(), json!({"bin": [65, 66, 67]})).unwrap();
        assert_encode_error(schema, json!({"bin": [65, 66]}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_large_binary() {
        let schema = schema_with(Field::new("bin", DT::LargeBinary, false));
        encode_single(schema.clone(), json!({"bin": "SGVsbG8gV29ybGQ="})).unwrap();
        encode_single(schema, json!({"bin": [72, 101, 108, 108, 111]})).unwrap();
    }

    #[test]
    fn test_binary_view() {
        let schema = schema_with(Field::new("bin", DT::BinaryView, false));
        encode_single(schema.clone(), json!({"bin": "SGVsbG8gV29ybGQ="})).unwrap();
        encode_single(schema, json!({"bin": [72, 101, 108, 108, 111]})).unwrap();
    }

    // String tests
    #[test]
    fn test_utf8_string() {
        let schema = schema_with(Field::new("str", DT::Utf8, false));
        encode_single(schema.clone(), json!({"str": "Hello"})).unwrap();
        encode_single(schema, json!({"str": ""})).unwrap();
    }

    #[test]
    fn test_utf8_json_object() {
        let schema = schema_with(Field::new("str", DT::Utf8, false));
        encode_single(schema, json!({"str": {"nested": "value"}})).unwrap();
    }

    #[test]
    fn test_utf8_json_array() {
        let schema = schema_with(Field::new("str", DT::Utf8, false));
        encode_single(schema, json!({"str": [1, 2, 3]})).unwrap();
    }

    #[test]
    fn test_large_utf8() {
        let schema = schema_with(Field::new("str", DT::LargeUtf8, false));
        encode_single(schema, json!({"str": "Large string"})).unwrap();
    }

    #[test]
    fn test_utf8_view() {
        let schema = schema_with(Field::new("str", DT::Utf8View, false));
        encode_single(schema, json!({"str": "View string"})).unwrap();
    }

    // List tests
    #[test]
    fn test_list_i32() {
        let schema = schema_with(Field::new(
            "list",
            DT::List(Arc::new(Field::new("item", DT::Int32, false))),
            false,
        ));
        encode_single(schema, json!({"list": [1, 2, 3, 4, 5]})).unwrap();
    }

    #[test]
    fn test_list_nullable_elements() {
        let schema = schema_with(Field::new(
            "list",
            DT::List(Arc::new(Field::new("item", DT::Int32, true))),
            false,
        ));
        encode_single(schema, json!({"list": [1, null, 3, null, 5]})).unwrap();
    }

    #[test]
    fn test_list_non_nullable_with_null() {
        let schema = schema_with(Field::new(
            "list",
            DT::List(Arc::new(Field::new("item", DT::Int32, false))),
            false,
        ));
        assert_encode_error(schema, json!({"list": [1, null, 3]}), |e| {
            matches!(e, JsonEncodeError::UnexpectedNullValue)
        });
    }

    #[test]
    fn test_list_empty() {
        let schema = schema_with(Field::new(
            "list",
            DT::List(Arc::new(Field::new("item", DT::Int32, false))),
            false,
        ));
        encode_single(schema, json!({"list": []})).unwrap();
    }

    // Null tests
    #[test]
    fn test_null_type() {
        let schema = schema_with(Field::new("null", DT::Null, true));
        encode_single(schema, json!({"null": null})).unwrap();
    }

    #[test]
    fn test_nullable_field_with_value() {
        let schema = schema_with(Field::new("val", DT::Int32, true));
        encode_single(schema.clone(), json!({"val": 42})).unwrap();
        encode_single(schema, json!({"val": null})).unwrap();
    }

    #[test]
    fn test_non_nullable_missing_field() {
        let schema = schema_with(Field::new("val", DT::Int32, false));
        assert_encode_error(schema, json!({}), |e| {
            matches!(e, JsonEncodeError::UnexpectedNullValue)
        });
    }

    // Struct tests
    #[test]
    fn test_struct_simple() {
        let struct_type = DT::Struct(
            vec![
                Field::new("name", DT::Utf8, false),
                Field::new("age", DT::Int32, false),
                Field::new("active", DT::Boolean, false),
            ]
            .into(),
        );
        let schema = schema_with(Field::new("person", struct_type, false));
        encode_single(
            schema,
            json!({"person": {"name": "Alice", "age": 30, "active": true}}),
        )
        .unwrap();
    }

    #[test]
    fn test_struct_nullable_fields() {
        let struct_type = DT::Struct(
            vec![
                Field::new("name", DT::Utf8, false),
                Field::new("age", DT::Int32, true),
                Field::new("email", DT::Utf8, true),
            ]
            .into(),
        );
        let schema = schema_with(Field::new("person", struct_type, false));
        encode_single(
            schema.clone(),
            json!({"person": {"name": "Bob", "age": null, "email": "bob@example.com"}}),
        )
        .unwrap();
        encode_single(
            schema,
            json!({"person": {"name": "Charlie", "age": 25, "email": null}}),
        )
        .unwrap();
    }

    #[test]
    fn test_struct_nested() {
        let address_type = DT::Struct(
            vec![
                Field::new("street", DT::Utf8, false),
                Field::new("city", DT::Utf8, false),
                Field::new("zip", DT::Int32, true),
            ]
            .into(),
        );
        let person_type = DT::Struct(
            vec![
                Field::new("name", DT::Utf8, false),
                Field::new("address", address_type, false),
            ]
            .into(),
        );
        let schema = schema_with(Field::new("person", person_type, false));
        encode_single(
            schema,
            json!({
                "person": {
                    "name": "Alice",
                    "address": {
                        "street": "123 Main St",
                        "city": "Springfield",
                        "zip": 12345
                    }
                }
            }),
        )
        .unwrap();
    }

    #[test]
    fn test_struct_missing_required_field() {
        let struct_type = DT::Struct(
            vec![
                Field::new("name", DT::Utf8, false),
                Field::new("age", DT::Int32, false),
            ]
            .into(),
        );
        let schema = schema_with(Field::new("person", struct_type, false));
        assert_encode_error(schema, json!({"person": {"name": "Alice"}}), |e| {
            matches!(e, JsonEncodeError::UnexpectedNullValue)
        });
    }

    #[test]
    fn test_struct_with_list() {
        let struct_type = DT::Struct(
            vec![
                Field::new("name", DT::Utf8, false),
                Field::new(
                    "scores",
                    DT::List(Arc::new(Field::new("item", DT::Int32, false))),
                    false,
                ),
            ]
            .into(),
        );
        let schema = schema_with(Field::new("student", struct_type, false));
        encode_single(
            schema,
            json!({"student": {"name": "Alice", "scores": [90, 85, 88, 92]}}),
        )
        .unwrap();
    }

    #[test]
    fn test_struct_empty_object() {
        let struct_type = DT::Struct(
            vec![
                Field::new("opt1", DT::Utf8, true),
                Field::new("opt2", DT::Int32, true),
            ]
            .into(),
        );
        let schema = schema_with(Field::new("config", struct_type, false));
        encode_single(schema, json!({"config": {}})).unwrap();
    }

    // Error tests
    #[test]
    fn test_type_mismatch_bool() {
        let schema = schema_with(Field::new("bool", DT::Boolean, false));
        assert_encode_error(schema, json!({"bool": "true"}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_type_mismatch_number() {
        let schema = schema_with(Field::new("i32", DT::Int32, false));
        assert_encode_error(schema, json!({"i32": "123"}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_invalid_binary_byte_value() {
        let schema = schema_with(Field::new("bin", DT::Binary, false));
        assert_encode_error(schema, json!({"bin": [256]}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    // Additional edge case tests
    #[test]
    fn test_list_deeply_nested() {
        let inner_list = DT::List(Arc::new(Field::new("item", DT::Int32, false)));
        let outer_list = DT::List(Arc::new(Field::new("inner", inner_list, false)));
        let schema = schema_with(Field::new("nested_list", outer_list, false));
        encode_single(schema, json!({"nested_list": [[1, 2], [3, 4, 5], []]})).unwrap();
    }

    #[test]
    fn test_empty_string_variants() {
        let utf8_schema = schema_with(Field::new("str", DT::Utf8, false));
        encode_single(utf8_schema, json!({"str": ""})).unwrap();

        let large_utf8_schema = schema_with(Field::new("str", DT::LargeUtf8, false));
        encode_single(large_utf8_schema, json!({"str": ""})).unwrap();

        let utf8_view_schema = schema_with(Field::new("str", DT::Utf8View, false));
        encode_single(utf8_view_schema, json!({"str": ""})).unwrap();
    }

    #[test]
    fn test_binary_empty() {
        let schema = schema_with(Field::new("bin", DT::Binary, false));
        encode_single(schema.clone(), json!({"bin": ""})).unwrap(); // empty string
        encode_single(schema, json!({"bin": []})).unwrap(); // empty array
    }

    #[test]
    fn test_timestamp_edge_cases() {
        let schema = schema_with(Field::new(
            "ts",
            DT::Timestamp(TimeUnit::Nanosecond, None),
            false,
        ));
        encode_single(schema.clone(), json!({"ts": "1970-01-01T00:00:00Z"})).unwrap(); // epoch
        encode_single(schema, json!({"ts": 0})).unwrap(); // unix timestamp 0
    }

    #[test]
    fn test_float_special_values() {
        // Test with JSON numbers that might cause precision issues
        let f32_schema = schema_with(Field::new("f32", DT::Float32, false));
        encode_single(f32_schema, json!({"f32": 0.1})).unwrap(); // Known precision issue in f32

        let f64_schema = schema_with(Field::new("f64", DT::Float64, false));
        encode_single(f64_schema, json!({"f64": 1.7976931348623157e308})).unwrap(); // near f64::MAX
    }

    #[test]
    fn test_interval_edge_cases() {
        // Test IntervalYearMonth with zero values
        let schema = schema_with(Field::new(
            "interval",
            DT::Interval(IntervalUnit::YearMonth),
            false,
        ));
        encode_single(
            schema.clone(),
            json!({"interval": {"years": 0, "months": 0}}),
        )
        .unwrap();
        encode_single(schema, json!({"interval": 0})).unwrap();

        // Test IntervalDayTime with zero values
        let schema = schema_with(Field::new(
            "interval",
            DT::Interval(IntervalUnit::DayTime),
            false,
        ));
        encode_single(schema, json!({"interval": {"days": 0, "milliseconds": 0}})).unwrap();
    }

    #[test]
    fn test_struct_with_all_nullable() {
        let struct_type = DT::Struct(
            vec![
                Field::new("opt1", DT::Utf8, true),
                Field::new("opt2", DT::Int32, true),
                Field::new("opt3", DT::Boolean, true),
            ]
            .into(),
        );
        let schema = schema_with(Field::new("optional_struct", struct_type, false));
        encode_single(
            schema,
            json!({"optional_struct": {"opt1": null, "opt2": null, "opt3": null}}),
        )
        .unwrap();
    }

    #[test]
    fn test_list_of_strings() {
        let schema = schema_with(Field::new(
            "string_list",
            DT::List(Arc::new(Field::new("item", DT::Utf8, false))),
            false,
        ));
        encode_single(schema, json!({"string_list": ["hello", "world", "!"]})).unwrap();
    }

    #[test]
    fn test_large_types_consistency() {
        // Test that LargeBinary and Binary handle the same data consistently
        let binary_schema = schema_with(Field::new("bin", DT::Binary, false));
        let large_binary_schema = schema_with(Field::new("bin", DT::LargeBinary, false));
        let binary_view_schema = schema_with(Field::new("bin", DT::BinaryView, false));

        let test_data = json!({"bin": "SGVsbG8gV29ybGQ="});
        encode_single(binary_schema, test_data.clone()).unwrap();
        encode_single(large_binary_schema, test_data.clone()).unwrap();
        encode_single(binary_view_schema, test_data).unwrap();
    }

    // Tests for new DataType variants

    #[test]
    fn test_fixed_size_list() {
        let schema = schema_with(Field::new(
            "fixed_list",
            DT::FixedSizeList(Arc::new(Field::new("item", DT::Int32, false)), 3),
            false,
        ));
        encode_single(schema.clone(), json!({"fixed_list": [1, 2, 3]})).unwrap();

        // Test wrong size - should fail
        assert_encode_error(schema, json!({"fixed_list": [1, 2]}), |e| {
            matches!(e, JsonEncodeError::UnexpectedType(_, _, _))
        });
    }

    #[test]
    fn test_large_list() {
        let schema = schema_with(Field::new(
            "large_list",
            DT::LargeList(Arc::new(Field::new("item", DT::Utf8, false))),
            false,
        ));
        encode_single(
            schema,
            json!({"large_list": ["hello", "world", "large", "list"]}),
        )
        .unwrap();
    }

    #[cfg(feature = "arrow-56")]
    #[test]
    fn test_decimal32() {
        let schema = schema_with(Field::new("dec32", DT::Decimal32(6, 2), false));
        encode_single(schema.clone(), json!({"dec32": 123.45})).unwrap();
        encode_single(schema, json!({"dec32": "456.78"})).unwrap();
    }

    #[test]
    fn test_decimal128() {
        let schema = schema_with(Field::new("dec128", DT::Decimal128(28, 6), false));
        encode_single(schema.clone(), json!({"dec128": 123456789.123456})).unwrap();
        encode_single(schema, json!({"dec128": "987654321.654321"})).unwrap();
    }

    #[test]
    fn test_decimal256() {
        let schema = schema_with(Field::new("dec256", DT::Decimal256(38, 8), false));
        encode_single(schema.clone(), json!({"dec256": 12345678901234.12345678})).unwrap();
        encode_single(schema, json!({"dec256": "98765432109876.87654321"})).unwrap();
    }

    #[test]
    fn test_dictionary_string() {
        let schema = schema_with(Field::new(
            "dict",
            DT::Dictionary(Box::new(DT::Int32), Box::new(DT::Utf8)),
            false,
        ));
        encode_single(schema.clone(), json!({"dict": "apple"})).unwrap();
        encode_single(schema.clone(), json!({"dict": "banana"})).unwrap();
        encode_single(schema, json!({"dict": "apple"})).unwrap(); // repeat value
    }

    #[test]
    fn test_map_simple() {
        let entries_field = Field::new(
            "entries",
            DT::Struct(
                vec![
                    Field::new("key", DT::Utf8, false),
                    Field::new("value", DT::Int32, true),
                ]
                .into(),
            ),
            false,
        );

        let schema = schema_with(Field::new(
            "map",
            DT::Map(Arc::new(entries_field), false),
            false,
        ));
        encode_single(
            schema,
            json!({"map": {"name": 100, "age": 25, "score": null}}),
        )
        .unwrap();
    }

    #[test]
    fn test_nested_structures() {
        // Test FixedSizeList of Structs
        let struct_type = DT::Struct(
            vec![
                Field::new("x", DT::Int32, false),
                Field::new("y", DT::Int32, false),
            ]
            .into(),
        );

        let schema = schema_with(Field::new(
            "points",
            DT::FixedSizeList(Arc::new(Field::new("point", struct_type, false)), 2),
            false,
        ));
        encode_single(
            schema,
            json!({"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}),
        )
        .unwrap();
    }

    #[test]
    fn test_decimal_precision_edge_cases() {
        // Test zero scale
        let schema = schema_with(Field::new("dec", DT::Decimal128(10, 0), false));
        encode_single(schema.clone(), json!({"dec": 12345})).unwrap();
        encode_single(schema, json!({"dec": "67890"})).unwrap();
    }

    #[test]
    fn test_large_list_empty() {
        let schema = schema_with(Field::new(
            "large_list",
            DT::LargeList(Arc::new(Field::new("item", DT::Boolean, false))),
            false,
        ));
        encode_single(schema, json!({"large_list": []})).unwrap();
    }

    #[test]
    fn test_large_list_with_nullable_elements() {
        let schema = schema_with(Field::new(
            "nullable_large_list",
            DT::LargeList(Arc::new(Field::new("item", DT::Float64, true))),
            false,
        ));
        encode_single(
            schema,
            json!({"nullable_large_list": [1.1, null, 3.3, null, 5.5]}),
        )
        .unwrap();
    }

    #[test]
    fn test_dictionary_with_nulls() {
        let schema = schema_with(Field::new(
            "nullable_dict",
            DT::Dictionary(Box::new(DT::Int32), Box::new(DT::Utf8)),
            true,
        ));
        // This will test null handling for dictionary fields
        encode_single(schema, json!({"nullable_dict": null})).unwrap();
    }

    fn encode_values(
        schema: Arc<Schema>,
        values: &[serde_json::Value],
    ) -> Result<Vec<RecordBatch>, JsonEncodeError> {
        let batches = FieldBatchEncoder::<JsonEncodeBuilder, _>::from_schema(
            schema,
            values.iter().cloned(),
            1024,
        )?
        .collect::<Result<Vec<_>, JsonEncodeError>>()?;
        let rows = batches.iter().map(|b| b.num_rows()).sum::<usize>();
        assert_eq!(rows, values.len(), "Batch count mismatch");
        Ok(batches)
    }
}
