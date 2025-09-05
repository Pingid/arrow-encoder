use crate::EncodeBuilder as EB;
use arrow::datatypes::{DataType, Field, IntervalUnit, TimeUnit};
use byteorder::{BigEndian, NetworkEndian, ReadBytesExt};
use chrono::{Duration, NaiveDate};
use fallible_iterator::FallibleIterator;
use postgres_protocol::types::{
    Range, RangeBound, array_from_sql, inet_from_sql, point_from_sql, range_from_sql,
};
use postgres_types::{FromSql, Kind, Type};
use std::io::Cursor;

#[derive(Debug, thiserror::Error)]
pub enum PgErr {
    #[error("Failed to encode column: {0}")]
    Encode(String),
    #[error("Failed to decode Postgres value: {0}")]
    Decode(Type, String),
    #[error("Unsupported Postgres type for encoding: {0}")]
    Unsupported(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

// Helper functions for common error patterns
#[inline]
fn err_unsupported(tp: &Type, encoder: &str) -> PgErr {
    PgErr::Encode(format!(
        "Unsupported type {} for {} encoder",
        tp.name(),
        encoder
    ))
}

#[inline]
fn err_invalid_len(expected: &str, got: usize) -> PgErr {
    PgErr::Encode(format!(
        "Invalid length: expected {}, got {}",
        expected, got
    ))
}

#[inline]
fn placeholder(prefix: &str, tp: &Type) -> String {
    format!("<{}:{}>", prefix, tp.name())
}

pub fn encode_pg_column(encoder: &mut EB, tp: &Type, value: Option<&[u8]>) -> Result<(), PgErr> {
    if value.is_none() {
        return Ok(encoder.append_null());
    }

    let bytes = value.unwrap();
    let name = tp.name();

    match (encoder, name) {
        // Basic types
        (EB::Bool(b), _) => b.append_value(bytes[0] != 0),
        (EB::Binary(b), _) => b.append_value(bytes),

        // Numbers
        (EB::I8(_), _) => return Err(err_unsupported(tp, "I8")),
        (EB::I16(b), _) => b.append_value(from_sql::<i16>(tp, bytes)?),
        (EB::I32(b), _) => b.append_value(from_sql::<i32>(tp, bytes)?),
        (EB::I64(b), _) => b.append_value(from_sql::<i64>(tp, bytes)?),
        (EB::F32(b), _) => b.append_value(from_sql::<f32>(tp, bytes)?),
        (EB::F64(b), _) => b.append_value(from_sql::<f64>(tp, bytes)?),
        (EB::UI32(b), _) => b.append_value(from_sql::<u32>(tp, bytes)?),

        // Strings
        (EB::Str(b), "name" | "text" | "varchar" | "bpchar" | "xml" | "jsonpath") => {
            b.append_value(from_sql::<String>(tp, bytes)?)
        }
        (EB::Str(b), "bit" | "varbit") => b.append_value(bit_string_from_sql(bytes)?),
        (EB::Str(b), "json" | "jsonb") => b.append_value(from_sql::<String>(tp, bytes)?),
        (EB::Str(b), "uuid") => b.append_value(fmt_uuid(bytes)?),
        (EB::Str(b), "numeric") => {
            let (value, _scale) = numeric_from_sql(bytes)?;
            b.append_value(value)
        }
        (EB::Str(b), "money") => {
            b.append_value(format!("{:.2}", from_sql::<i64>(tp, bytes)? as f64 / 100.0))
        }
        // Geometry types (as strings)
        (EB::Str(b), "point") => {
            let point_val = from_decoder(point_from_sql, bytes)?;
            b.append_value(format!("({},{})", point_val.x(), point_val.y()))
        }
        (EB::Str(b), "lseg") => {
            let value = from_decoder(lseg_from_sql, bytes)?;
            b.append_value(format!("[{:?}, {:?}]", value.0, value.1))
        }
        (EB::Str(b), "path") => {
            // PostgreSQL path format: ((x1,y1),(x2,y2),...) or [(x1,y1),(x2,y2),...]
            let mut rdr = Cursor::new(bytes);
            let closed = rdr.read_u8()? != 0; // 1 for closed path, 0 for open
            let num_points = rdr.read_i32::<BigEndian>()?;
            let mut points = Vec::with_capacity(num_points as usize);

            for _ in 0..num_points {
                let x = rdr.read_f64::<BigEndian>()?;
                let y = rdr.read_f64::<BigEndian>()?;
                points.push(format!("({},{})", x, y));
            }

            let joined = points.join(",");
            if closed {
                b.append_value(format!("({})", joined)) // Closed path uses parentheses
            } else {
                b.append_value(format!("[{}]", joined)) // Open path uses brackets
            }
        }
        (EB::Str(b), "box") => {
            // PostgreSQL box format: ((x1,y1),(x2,y2))
            let mut rdr = Cursor::new(bytes);
            let x1 = rdr.read_f64::<BigEndian>()?;
            let y1 = rdr.read_f64::<BigEndian>()?;
            let x2 = rdr.read_f64::<BigEndian>()?;
            let y2 = rdr.read_f64::<BigEndian>()?;
            b.append_value(format!("(({},{}),({},{}))", x2, y2, x1, y1))
        }
        (EB::Str(b), "polygon") => {
            let value = from_decoder(polygon_from_sql, bytes)?;
            let formatted = value
                .into_iter()
                .map(|x| format!("({}, {})", x[0], x[1]))
                .collect::<Vec<_>>();
            b.append_value(format!("({})", formatted.join(", ")))
        }
        (EB::Str(b), "line") => {
            let value = from_decoder(line_from_sql, bytes)?;
            b.append_value(format!("{{{}, {}, {}}}", value.0, value.1, value.2))
        }
        (EB::Str(b), "circle") => {
            let value = from_decoder(circle_from_sql, bytes)?;
            b.append_value(format!("<({}, {}), {}>", value.0.0, value.0.1, value.1))
        }
        // Network types
        (EB::Str(b), "cidr" | "inet") => {
            let inet_val = from_decoder(inet_from_sql, bytes)?;
            b.append_value(format!("{}/{}", inet_val.addr(), inet_val.netmask()))
        }
        (EB::Str(b), "macaddr") => {
            // Use fmt_mac for consistency
            b.append_value(fmt_mac(bytes)?)
        }
        (EB::Str(b), "macaddr8") => b.append_value(fmt_mac(bytes)?),
        // Range types (as strings)
        (EB::Str(b), "int4range") => {
            let range = display_range_from_sql(tp, bytes, |byts| i32_from_slice(byts))?;
            b.append_value(range)
        }
        (EB::Str(b), "numrange") => {
            let range =
                display_range_from_sql(tp, bytes, |byts| numeric_from_sql(byts).map(|(n, _)| n))?;
            b.append_value(range)
        }
        (EB::Str(b), "tsrange") => {
            let range = display_range_from_sql(tp, bytes, |byts| {
                // Read timestamp as microseconds since 2000-01-01
                let mut rdr = Cursor::new(byts);
                let micros = rdr.read_i64::<NetworkEndian>()?;
                // Convert to seconds and format
                let pg_epoch = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap();
                let duration = Duration::microseconds(micros);
                let timestamp = pg_epoch + duration;
                Ok(timestamp.format("%Y-%m-%d %H:%M:%S%.f").to_string())
            })?;
            b.append_value(range)
        }
        (EB::Str(b), "tstzrange") => {
            let range = display_range_from_sql(tp, bytes, |byts| {
                // Read timestamp as microseconds since 2000-01-01 UTC
                let mut rdr = Cursor::new(byts);
                let micros = rdr.read_i64::<NetworkEndian>()?;
                // Convert to seconds and format
                let pg_epoch = chrono::NaiveDate::from_ymd_opt(2000, 1, 1)
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap();
                let duration = Duration::microseconds(micros);
                let timestamp = pg_epoch + duration;
                Ok(timestamp.format("%Y-%m-%d %H:%M:%S%.f+00").to_string())
            })?;
            b.append_value(range)
        }
        (EB::Str(b), "daterange") => {
            let range = display_range_from_sql(tp, bytes, |byts| {
                // Read date as i32 (days since 2000-01-01)
                i32_from_slice(byts).map(|days| {
                    let base = NaiveDate::from_ymd_opt(2000, 1, 1).unwrap();
                    base.checked_add_signed(Duration::days(days as i64))
                        .unwrap()
                        .to_string()
                })
            })?;
            b.append_value(range)
        }
        (EB::Str(b), "int8range") => {
            let range = display_range_from_sql(tp, bytes, |byts| i64_from_slice(byts))?;
            b.append_value(range)
        }
        // Time with timezone (as string since it includes zone info)
        (EB::Str(b), "timetz") => {
            let mut rdr = Cursor::new(bytes);
            let microseconds = rdr.read_i64::<NetworkEndian>()?;
            let _tz_offset_secs = rdr.read_i32::<NetworkEndian>()?;

            // Convert microseconds to time components
            let total_secs = microseconds / 1_000_000;
            let hours = total_secs / 3600;
            let minutes = (total_secs % 3600) / 60;
            let seconds = total_secs % 60;
            let micros = microseconds % 1_000_000;

            // Format as HH:MM:SS.ffffff
            if micros > 0 {
                b.append_value(format!(
                    "{:02}:{:02}:{:02}.{:06}",
                    hours, minutes, seconds, micros
                ))
            } else {
                b.append_value(format!("{:02}:{:02}:{:02}", hours, minutes, seconds))
            }
        }
        // Default string handler for unknown string types
        (EB::Str(b), _) => b.append_value(placeholder("unsupported", tp)),

        // Date and time
        (EB::Date32(b), _) => match tp.name() {
            "date" => {
                // PG date is days since 2000-01-01, Arrow Date32 is days since 1970-01-01
                let pg_days = from_sql::<i32>(tp, bytes)?;
                const PG_EPOCH_OFFSET: i32 = 10957; // days between 1970-01-01 and 2000-01-01
                b.append_value(pg_days + PG_EPOCH_OFFSET)
            }
            _ => return Err(err_unsupported(tp, "Date32")),
        },
        (EB::Time64Micro(b), _) => match tp.name() {
            "time" => b.append_value(from_sql::<i64>(tp, bytes)?),
            "timetz" => {
                let mut rdr = Cursor::new(bytes);
                let microseconds = rdr.read_i64::<NetworkEndian>()?;
                // For Time64Micro, we just store the microseconds part (ignoring timezone)
                b.append_value(microseconds)
            }
            _ => return Err(err_unsupported(tp, "Time64Micro")),
        },
        (EB::TsMs(b), _) => match tp.name() {
            "timestamp" => {
                // Decode timestamp as microseconds since 2000-01-01
                let mut rdr = Cursor::new(bytes);
                let micros = rdr.read_i64::<NetworkEndian>()?;
                // Convert from postgres epoch (2000-01-01) to unix epoch (1970-01-01)
                let pg_epoch_offset = 946684800000i64; // milliseconds between epochs
                let millis = micros / 1000 + pg_epoch_offset;
                b.append_value(millis)
            }
            "timestamptz" => {
                // Same as timestamp for storage (postgres stores as UTC)
                let mut rdr = Cursor::new(bytes);
                let micros = rdr.read_i64::<NetworkEndian>()?;
                let pg_epoch_offset = 946684800000i64;
                let millis = micros / 1000 + pg_epoch_offset;
                b.append_value(millis)
            }
            _ => return Err(err_unsupported(tp, "TsMs")),
        },
        (EB::IntervalMonthDayNano(b), _) => match tp.name() {
            "interval" => {
                let (months, days, nanos) = interval_from_sql(bytes)?;
                b.append_value(arrow::datatypes::IntervalMonthDayNano::new(
                    months, days, nanos,
                ))
            }
            _ => return Err(err_unsupported(tp, "IntervalMonthDayNano")),
        },

        // Decimal types - handle numeric
        #[cfg(feature = "arrow-56")]
        (EB::Decimal32(_b), _) => match tp.name() {
            "numeric" => {
                // Note: We need the target scale from the Arrow field, not the PG numeric scale
                // For now, we'll just encode as string in the Str builder above
                return Err(err_unsupported(tp, "Decimal32"));
            }
            _ => return Err(err_unsupported(tp, "Decimal32")),
        },
        (EB::Decimal128(_b), _) => match tp.name() {
            "numeric" => {
                // Note: We need the target scale from the Arrow field, not the PG numeric scale
                // For now, we'll just encode as string in the Str builder above
                return Err(err_unsupported(tp, "Decimal128"));
            }
            _ => return Err(err_unsupported(tp, "Decimal128")),
        },
        (EB::Decimal256(_b), _) => match tp.name() {
            "numeric" => {
                // Note: We need the target scale from the Arrow field, not the PG numeric scale
                // For now, we'll just encode as string in the Str builder above
                return Err(err_unsupported(tp, "Decimal256"));
            }
            _ => return Err(err_unsupported(tp, "Decimal256")),
        },

        // Fixed size binary types
        (EB::FixedSizeBinary(_b), _) => return Err(err_unsupported(tp, "FixedSizeBinary")),

        // List types - handle PostgreSQL arrays properly
        (EB::List(b), _) => match tp.kind() {
            &postgres_types::Kind::Array(ref inner_tp) => decode_pg_array(b, inner_tp, bytes)?,
            _ => return Err(err_unsupported(tp, "List")),
        },

        // Fallback for any other encoder type
        _ => {
            return Err(PgErr::Encode(format!(
                "Unsupported encoder type for PostgreSQL type {}",
                tp.name()
            )));
        }
    }
    Ok(())
}

#[inline]
fn from_sql<'a, T: FromSql<'a>>(tp: &Type, bytes: &'a [u8]) -> Result<T, PgErr> {
    T::from_sql(tp, bytes).map_err(|e| {
        decode_error(
            tp,
            format!("Error decoding {:?}: {}", std::any::type_name::<T>(), e),
        )
    })
}

#[inline]
fn from_decoder<F, T, E>(decoder: F, bytes: &[u8]) -> Result<T, PgErr>
where
    F: FnOnce(&[u8]) -> Result<T, E>,
    E: std::fmt::Display,
{
    decoder(bytes).map_err(|e| PgErr::Encode(format!("Error decoding with custom decoder: {}", e)))
}

#[inline]
fn decode_error(tp: &Type, message: impl Into<String>) -> PgErr {
    PgErr::Decode(tp.clone(), message.into())
}

pub fn get_arrow_type_from_pg_oid(oid: u32) -> Result<DataType, PgErr> {
    let tp = Type::from_oid(oid).ok_or(PgErr::Unsupported(oid.to_string()))?;
    get_arrow_type_from_pg_type(&tp)
}

// Helper to get the element type from an array type
fn get_array_element_type(tp: &Type) -> Option<&Type> {
    match tp.kind() {
        &Kind::Array(ref inner) => Some(inner),
        _ => None,
    }
}

// Type mapping helper
fn is_text_type(tp: &Type) -> bool {
    matches!(
        tp,
        &Type::NAME
            | &Type::CHAR
            | &Type::BPCHAR
            | &Type::VARCHAR
            | &Type::TEXT
            | &Type::CSTRING
            | &Type::UNKNOWN
            | &Type::JSON
            | &Type::JSONB
            | &Type::JSONPATH
            | &Type::XML
            | &Type::INET
            | &Type::CIDR
            | &Type::MACADDR
            | &Type::MACADDR8
            | &Type::TS_VECTOR
            | &Type::TSQUERY
            | &Type::GTS_VECTOR
            | &Type::REFCURSOR
            | &Type::REGPROC
            | &Type::REGPROCEDURE
            | &Type::REGOPER
            | &Type::REGOPERATOR
            | &Type::REGCLASS
            | &Type::REGTYPE
            | &Type::REGCONFIG
            | &Type::REGDICTIONARY
            | &Type::REGNAMESPACE
            | &Type::REGROLE
            | &Type::REGCOLLATION
            | &Type::POINT
            | &Type::LSEG
            | &Type::PATH
            | &Type::BOX
            | &Type::POLYGON
            | &Type::LINE
            | &Type::CIRCLE
            | &Type::INT4_RANGE
            | &Type::NUM_RANGE
            | &Type::TS_RANGE
            | &Type::TSTZ_RANGE
            | &Type::DATE_RANGE
            | &Type::INT8_RANGE
            | &Type::INT4MULTI_RANGE
            | &Type::NUMMULTI_RANGE
            | &Type::TSMULTI_RANGE
            | &Type::TSTZMULTI_RANGE
            | &Type::DATEMULTI_RANGE
            | &Type::INT8MULTI_RANGE
            | &Type::MONEY
            | &Type::NUMERIC
            | &Type::BIT
            | &Type::VARBIT
            | &Type::TIMETZ
            | &Type::UUID
    )
}

#[allow(dead_code)]
fn is_array_text_type(tp: &Type) -> bool {
    matches!(
        tp,
        &Type::BOOL_ARRAY
            | &Type::BYTEA_ARRAY
            | &Type::CHAR_ARRAY
            | &Type::NAME_ARRAY
            | &Type::BPCHAR_ARRAY
            | &Type::VARCHAR_ARRAY
            | &Type::TEXT_ARRAY
            | &Type::CSTRING_ARRAY
            | &Type::JSON_ARRAY
            | &Type::JSONB_ARRAY
            | &Type::JSONPATH_ARRAY
            | &Type::XML_ARRAY
            | &Type::INET_ARRAY
            | &Type::CIDR_ARRAY
            | &Type::MACADDR_ARRAY
            | &Type::MACADDR8_ARRAY
            | &Type::TS_VECTOR_ARRAY
            | &Type::TSQUERY_ARRAY
            | &Type::GTS_VECTOR_ARRAY
            | &Type::REGPROC_ARRAY
            | &Type::REGPROCEDURE_ARRAY
            | &Type::REGOPER_ARRAY
            | &Type::REGOPERATOR_ARRAY
            | &Type::REGCLASS_ARRAY
            | &Type::REGTYPE_ARRAY
            | &Type::REGCONFIG_ARRAY
            | &Type::REGDICTIONARY_ARRAY
            | &Type::REGNAMESPACE_ARRAY
            | &Type::REGROLE_ARRAY
            | &Type::REGCOLLATION_ARRAY
            | &Type::MONEY_ARRAY
            | &Type::INT2_ARRAY
            | &Type::INT4_ARRAY
            | &Type::INT8_ARRAY
            | &Type::OID_ARRAY
            | &Type::XID_ARRAY
            | &Type::CID_ARRAY
            | &Type::FLOAT4_ARRAY
            | &Type::FLOAT8_ARRAY
            | &Type::NUMERIC_ARRAY
            | &Type::BIT_ARRAY
            | &Type::VARBIT_ARRAY
            | &Type::DATE_ARRAY
            | &Type::TIME_ARRAY
            | &Type::TIMESTAMP_ARRAY
            | &Type::TIMESTAMPTZ_ARRAY
            | &Type::TIMETZ_ARRAY
            | &Type::INTERVAL_ARRAY
            | &Type::UUID_ARRAY
            | &Type::POINT_ARRAY
            | &Type::LSEG_ARRAY
            | &Type::PATH_ARRAY
            | &Type::BOX_ARRAY
            | &Type::POLYGON_ARRAY
            | &Type::LINE_ARRAY
            | &Type::CIRCLE_ARRAY
            | &Type::INT4_RANGE_ARRAY
            | &Type::NUM_RANGE_ARRAY
            | &Type::TS_RANGE_ARRAY
            | &Type::TSTZ_RANGE_ARRAY
            | &Type::DATE_RANGE_ARRAY
            | &Type::INT8_RANGE_ARRAY
            | &Type::INT4MULTI_RANGE_ARRAY
            | &Type::NUMMULTI_RANGE_ARRAY
            | &Type::TSMULTI_RANGE_ARRAY
            | &Type::TSTZMULTI_RANGE_ARRAY
            | &Type::DATEMULTI_RANGE_ARRAY
            | &Type::INT8MULTI_RANGE_ARRAY
    )
}

pub fn get_arrow_type_from_pg_type(tp: &Type) -> Result<DataType, PgErr> {
    use std::sync::Arc;
    let value = match tp {
        // booleans / binary
        &Type::BOOL => DataType::Boolean,
        &Type::BYTEA => DataType::Binary,

        tp if is_text_type(tp) => DataType::Utf8,

        // integers
        &Type::INT2 => DataType::Int16,
        &Type::INT4 => DataType::Int32,
        &Type::INT8 => DataType::Int64,

        // system ids (unsigned 32)
        &Type::OID | &Type::XID | &Type::CID => DataType::UInt32,

        // WAL LSN is 64-bit
        &Type::PG_LSN => DataType::UInt64,

        // floats
        &Type::FLOAT4 => DataType::Float32,
        &Type::FLOAT8 => DataType::Float64,

        // Note: NUMERIC, BIT, VARBIT are handled in is_text_type

        // dates/times (use milliseconds to match old implementation)
        &Type::DATE => DataType::Date32,
        &Type::TIME => DataType::Time64(TimeUnit::Microsecond),
        &Type::TIMESTAMP => DataType::Timestamp(TimeUnit::Millisecond, None),
        &Type::TIMESTAMPTZ => DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
        // Note: TIMETZ is handled in is_text_type
        // INTERVAL -> Arrow month-day-nano interval
        &Type::INTERVAL => DataType::Interval(IntervalUnit::MonthDayNano),

        // Arrays - map to List types
        tp if get_array_element_type(tp).is_some() => {
            if let Some(element_tp) = get_array_element_type(tp) {
                match get_arrow_type_from_pg_type(element_tp) {
                    Ok(element_dt) => {
                        DataType::List(Arc::new(Field::new("item", element_dt, true)))
                    }
                    Err(_) => DataType::Utf8, // Fallback to string representation
                }
            } else {
                DataType::Utf8
            }
        }

        t => Err(PgErr::Unsupported(t.to_string()))?,
    };
    Ok(value)
}

// Helper functions for complex type decoding
#[inline]
fn numeric_from_sql(bytes: &[u8]) -> Result<(String, i16), PgErr> {
    let mut rdr = Cursor::new(bytes);
    let num_digits = rdr.read_i16::<BigEndian>()?;
    let weight = rdr.read_i16::<BigEndian>()?;
    let sign = rdr.read_i16::<BigEndian>()?;
    let dscale = rdr.read_i16::<BigEndian>()?;

    if num_digits == 0 {
        return Ok(("0".to_string(), dscale));
    }

    // Read all digits
    let mut digits = Vec::with_capacity(num_digits as usize);
    for _ in 0..num_digits {
        digits.push(rdr.read_i16::<BigEndian>()?);
    }

    // Convert to string representation
    let mut result = String::new();

    // Add sign
    if sign == 0x4000 {
        result.push('-');
    }

    // Calculate the position of decimal point
    let decimal_pos = (weight + 1) * 4;

    // Convert digits to string
    let mut all_digits = String::new();
    for (i, &digit) in digits.iter().enumerate() {
        if i == 0 {
            // First digit, no leading zeros
            all_digits.push_str(&digit.to_string());
        } else {
            // Subsequent digits need leading zeros
            all_digits.push_str(&format!("{:04}", digit));
        }
    }

    // Trim trailing zeros
    let all_digits = all_digits.trim_end_matches('0');
    if all_digits.is_empty() {
        return Ok(("0".to_string(), dscale));
    }

    // Insert decimal point if needed
    if decimal_pos <= 0 {
        // Number is < 1
        result.push_str("0.");
        for _ in 0..(-decimal_pos) {
            result.push('0');
        }
        result.push_str(all_digits);
    } else if decimal_pos as usize >= all_digits.len() {
        // No fractional part
        result.push_str(all_digits);
        for _ in 0..(decimal_pos as usize - all_digits.len()) {
            result.push('0');
        }
    } else {
        // Split at decimal point
        let (int_part, frac_part) = all_digits.split_at(decimal_pos as usize);
        result.push_str(int_part);
        if !frac_part.is_empty() {
            result.push('.');
            result.push_str(frac_part);
        }
    }

    Ok((result, dscale))
}

#[inline]
fn interval_from_sql(bytes: &[u8]) -> Result<(i32, i32, i64), PgErr> {
    let mut rdr = Cursor::new(bytes);
    let microseconds = rdr.read_i64::<NetworkEndian>()?;
    let days = rdr.read_i32::<NetworkEndian>()?;
    let months = rdr.read_i32::<NetworkEndian>()?;
    let nanoseconds = microseconds * 1_000;
    Ok((months, days, nanoseconds))
}

#[inline]
fn bit_string_from_sql(mut buf: &[u8]) -> Result<String, PgErr> {
    let bit_length = buf.read_i32::<BigEndian>()? as usize;
    let mut bit_string = String::with_capacity(bit_length);

    for &byte in buf {
        for i in (0..8).rev() {
            if bit_string.len() < bit_length {
                bit_string.push(if (byte & (1 << i)) != 0 { '1' } else { '0' });
            }
        }
    }

    Ok(bit_string)
}

// Helper functions for formatting
#[inline]
fn fmt_uuid(bytes: &[u8]) -> Result<String, PgErr> {
    if bytes.len() != 16 {
        return Err(err_invalid_len("16 bytes", bytes.len()));
    }
    Ok(format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    ))
}

#[inline]
fn fmt_mac(bytes: &[u8]) -> Result<String, PgErr> {
    match bytes.len() {
        6 => Ok(bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(":")),
        8 => Ok(bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<Vec<_>>()
            .join(":")),
        _ => Err(err_invalid_len("6 or 8 bytes", bytes.len())),
    }
}

// Geometry helper functions
#[inline]
fn line_from_sql(bytes: &[u8]) -> Result<(f64, f64, f64), PgErr> {
    let mut rdr = Cursor::new(bytes);
    let a = rdr.read_f64::<BigEndian>()?;
    let b = rdr.read_f64::<BigEndian>()?;
    let c = rdr.read_f64::<BigEndian>()?;
    Ok((a, b, c))
}

#[inline]
fn lseg_from_sql(bytes: &[u8]) -> Result<((f64, f64), (f64, f64)), PgErr> {
    let mut rdr = Cursor::new(bytes);
    let a = rdr.read_f64::<BigEndian>()?;
    let b = rdr.read_f64::<BigEndian>()?;
    let c = rdr.read_f64::<BigEndian>()?;
    let d = rdr.read_f64::<BigEndian>()?;
    Ok(((a, b), (c, d)))
}

#[inline]
fn polygon_from_sql(bytes: &[u8]) -> Result<Vec<[f64; 2]>, PgErr> {
    let mut rdr = Cursor::new(bytes);
    let num_points = rdr.read_i32::<BigEndian>()?;
    let mut points: Vec<[f64; 2]> = Vec::with_capacity(num_points as usize);
    for _ in 0..num_points {
        let x = rdr.read_f64::<BigEndian>()?;
        let y = rdr.read_f64::<BigEndian>()?;
        points.push([x, y]);
    }
    Ok(points)
}

#[inline]
fn circle_from_sql(bytes: &[u8]) -> Result<((f64, f64), f64), PgErr> {
    let mut rdr = Cursor::new(bytes);
    let x = rdr.read_f64::<BigEndian>()?;
    let y = rdr.read_f64::<BigEndian>()?;
    let radius = rdr.read_f64::<BigEndian>()?;
    Ok(((x, y), radius))
}

// Range helper functions
#[inline]
fn i32_from_slice(bytes: &[u8]) -> Result<i32, PgErr> {
    if bytes.len() < 4 {
        return Err(PgErr::Encode("Insufficient bytes to convert to i32".into()));
    }
    let array: [u8; 4] = bytes[0..4]
        .try_into()
        .map_err(|_| PgErr::Encode("Failed to convert bytes to i32".into()))?;
    Ok(i32::from_be_bytes(array))
}

#[inline]
fn i64_from_slice(bytes: &[u8]) -> Result<i64, PgErr> {
    if bytes.len() < 8 {
        return Err(PgErr::Encode("Insufficient bytes to convert to i64".into()));
    }
    let array: [u8; 8] = bytes[0..8]
        .try_into()
        .map_err(|_| PgErr::Encode("Failed to convert bytes to i64".into()))?;
    Ok(i64::from_be_bytes(array))
}

#[inline]
fn display_range_from_sql<R: std::fmt::Display>(
    tp: &Type,
    buf: &[u8],
    f: impl Fn(&[u8]) -> Result<R, PgErr>,
) -> Result<String, PgErr> {
    let range = range_from_sql(buf)
        .map_err(|e| decode_error(tp, format!("Error decoding range: {}", e)))?;
    display_range(range, f)
}

#[inline]
fn display_range<R: std::fmt::Display>(
    range: Range<'_>,
    f: impl Fn(&[u8]) -> Result<R, PgErr>,
) -> Result<String, PgErr> {
    match range {
        Range::Empty => Ok("empty".to_string()),
        Range::Nonempty(lower, upper) => {
            let lower_str = match lower {
                RangeBound::Inclusive(v) => match v {
                    Some(x) => format!("[{}", f(x)?),
                    None => "[-inf".to_string(),
                },
                RangeBound::Exclusive(v) => match v {
                    Some(x) => format!("({}", f(x)?),
                    None => "(-inf".to_string(),
                },
                RangeBound::Unbounded => "(-inf".to_string(),
            };

            let upper_str = match upper {
                RangeBound::Inclusive(v) => match v {
                    Some(x) => format!(", {}]", f(x)?),
                    None => ", inf]".to_string(),
                },
                RangeBound::Exclusive(v) => match v {
                    Some(x) => format!(", {})", f(x)?),
                    None => ", inf)".to_string(),
                },
                RangeBound::Unbounded => ", inf)".to_string(),
            };

            Ok(format!("{}{}", lower_str, upper_str))
        }
    }
}

// Helper function to decode PostgreSQL arrays
fn decode_pg_array(
    builder: &mut crate::ListBuilderWrapper,
    inner_tp: &Type,
    bytes: &[u8],
) -> Result<(), PgErr> {
    let arr = array_from_sql(bytes)
        .map_err(|e| PgErr::Decode(inner_tp.clone(), format!("Error decoding array: {}", e)))?;

    let values_builder = builder.values();
    let mut array_values = arr.values();

    // Iterate over array values
    while let Some(item) = array_values
        .next()
        .map_err(|e| PgErr::Decode(inner_tp.clone(), format!("Error iterating array: {:?}", e)))?
    {
        encode_pg_column(values_builder, inner_tp, item)?;
    }

    builder
        .append(true)
        .map_err(|e| PgErr::Encode(format!("Failed to append array: {}", e)))?;
    Ok(())
}
