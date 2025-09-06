use crate::ConnectArrowError;
use arrow::{
    array::{ArrayBuilder, BooleanBuilder, GenericListBuilder},
    datatypes::{DataType, Field, IntervalMonthDayNano, IntervalUnit, TimeUnit},
};
use byteorder::{BigEndian, NetworkEndian, ReadBytesExt};
use chrono::{DateTime, NaiveDateTime, Utc};
use fallible_iterator::{FallibleIterator, IntoFallibleIterator};
use postgres_protocol::types::{
    Range, RangeBound, array_from_sql, float4_from_sql, inet_from_sql, point_from_sql,
    range_from_sql,
};
use postgres_types::{FromSql, Kind, Type};
use std::{any::type_name, io::Cursor, sync::Arc};

pub fn get_pg_arrow_field_from_typname(tp_name: &str) -> DataType {
    match tp_name {
        "bool" => DataType::Boolean,
        "bytea" => DataType::Binary,

        // Numbers
        "int2" => DataType::Int16,
        "int4" => DataType::Int32,
        "int8" => DataType::Int64,
        "float4" => DataType::Float32,
        "float8" => DataType::Float64,
        "numeric" => DataType::Decimal128(38, 6),
        "oid" => DataType::UInt32,
        "money" => DataType::Float64,

        // Strings
        "name" | "text" | "json" | "varchar" | "bpchar" | "bit" | "varbit" | "uuid" | "jsonb"
        | "xml" | "jsonpath" => DataType::Utf8,

        // Geometry
        "point" | "lseg" | "path" | "box" | "polygon" | "line" | "circle" => DataType::Utf8,

        // Network
        "cidr" | "macaddr8" | "macaddr" | "inet" => DataType::Utf8,

        // Date and time
        "date" => DataType::Date32,
        "time" => DataType::Time64(TimeUnit::Microsecond),
        // Timestamps require Unix epoch base
        "timestamp" => DataType::Timestamp(TimeUnit::Millisecond, None),
        "timestamptz" => DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())), // Assume UTC storage
        "interval" => DataType::Interval(IntervalUnit::MonthDayNano),
        "timetz" => DataType::Time64(TimeUnit::Microsecond),

        // Range types
        "int4range" | "numrange" | "tsrange" | "tstzrange" | "daterange" | "int8range" => {
            DataType::Utf8
        }

        // Arrays
        tp if tp.starts_with("_") => DataType::List(Arc::new(Field::new(
            "item",
            get_pg_arrow_field_from_typname(&tp[1..]),
            true,
        ))),
        _ => DataType::Binary,
    }
}

pub fn write_pg_column_to_arrow_array(
    writer: &mut Box<dyn ArrayBuilder>,
    tp: &Type,
    bytes: Option<&[u8]>,
) -> Result<(), ConnectArrowError> {
    match tp.name() {
        // Numbers
        "bool" => {
            let b = cast::<BooleanBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<bool>(tp, bytes)?)),
            }
        }
        "bytea" => {
            let b = cast::<arrow::array::BinaryBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(bytes)),
            }
        }
        "int2" => {
            let b = cast::<arrow::array::Int16Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<i16>(tp, bytes)?)),
            }
        }
        "int4" => {
            let b = cast::<arrow::array::Int32Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<i32>(tp, bytes)?)),
            }
        }
        "int8" => {
            let b = cast::<arrow::array::Int64Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<i64>(tp, bytes)?)),
            }
        }

        "float4" => {
            let b = cast::<arrow::array::Float32Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_decoder(float4_from_sql, tp, bytes)?)),
            }
        }
        "float8" => {
            let b = cast::<arrow::array::Float64Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<f64>(tp, bytes)?)),
            }
        }
        "numeric" => {
            let b = cast::<arrow::array::Decimal128Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_decoder(numeric_from_sql, tp, bytes)?)),
            }
        }
        "oid" => {
            let b = cast::<arrow::array::UInt32Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<u32>(tp, bytes)?)),
            }
        }
        "money" => {
            let b = cast::<arrow::array::Float64Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value((from_sql::<i64>(tp, bytes)?) as f64 / 100.0)),
            }
        }

        // Strings
        "name" | "text" | "varchar" | "bpchar" | "xml" | "jsonpath" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<String>(tp, bytes)?)),
            }
        }
        "json" | "jsonb" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    Ok(b.append_value(from_sql::<serde_json::Value>(tp, bytes)?.to_string()))
                }
            }
        }
        "uuid" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<uuid::Uuid>(tp, bytes)?.to_string())),
            }
        }

        "bit" | "varbit" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(bit_string_from_sql(bytes)?)),
            }
        }

        // Geometry
        "point" => {
            let bldr = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(bldr.append_null()),
                Some(bytes) => {
                    let value = from_decoder(point_from_sql, tp, bytes)?;
                    Ok(bldr.append_value(&format!("({}, {})", value.x(), value.y())))
                }
            }
        }
        "lseg" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let value = from_decoder(lseg_from_sql, tp, bytes)?;
                    Ok(b.append_value(&format!("[{:?}, {:?}]", value.0, value.1)))
                }
            }
        }
        "path" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let value = from_sql::<geo_types::LineString<f64>>(tp, bytes)?;
                    let value = value
                        .into_points()
                        .into_iter()
                        .map(|x| format!("({}, {})", x.x(), x.y()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    Ok(b.append_value(&format!("({})", value)))
                }
            }
        }
        "box" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let bx = from_sql::<geo_types::Rect<f64>>(tp, bytes)?;
                    Ok(b.append_value(format!(
                        "({}, {}), ({}, {})",
                        bx.min().x,
                        bx.min().y,
                        bx.max().x,
                        bx.max().y
                    )))
                }
            }
        }
        "polygon" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let value = from_decoder(polygon_from_sql, tp, bytes)?;
                    let formatted = value
                        .into_iter()
                        .map(|x| format!("({}, {})", x[0], x[1]))
                        .collect::<Vec<_>>();
                    Ok(b.append_value(&format!("({})", formatted.join(", "))))
                }
            }
        }
        "line" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let value = from_decoder(line_from_sql, tp, bytes)?;
                    Ok(b.append_value(&format!("{{{}, {}, {}}}", value.0, value.1, value.2)))
                }
            }
        }
        "circle" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let value = from_decoder(line_from_sql, tp, bytes)?;
                    Ok(b.append_value(&format!("<({}, {}), {}>", value.0, value.1, value.2)))
                }
            }
        }

        // Network
        "cidr" | "inet" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let value = from_decoder(inet_from_sql, tp, bytes)?;
                    Ok(b.append_value(&format!("{}/{}", value.addr(), value.netmask())))
                }
            }
        }
        "macaddr" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let value = from_decoder(eui48::MacAddress::from_bytes, tp, bytes)?;
                    b.append_value(&format!("{}", value.to_hex_string()));
                }
            }
            Ok(())
        }
        "macaddr8" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let value = from_decoder(slice_from_sql::<8>, tp, bytes)?;
                    let formatted: Vec<String> = value
                        .into_iter()
                        .map(|x| format!("{:02x}", x))
                        .collect::<Vec<_>>();
                    b.append_value(&format!("{}", formatted.join(":")));
                }
            }
            Ok(())
        }

        // Date and time
        "date" => {
            let b = cast::<arrow::array::Date32Builder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<i32>(tp, bytes)?)),
            }
        }
        "time" => {
            let b = cast::<arrow::array::Time64MicrosecondBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => Ok(b.append_value(from_sql::<i64>(tp, bytes)?)),
            }
        }
        "timestamp" => {
            let b = cast::<arrow::array::TimestampMillisecondBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let naive_dt = from_sql::<NaiveDateTime>(tp, bytes)?;
                    Ok(b.append_value(naive_dt.and_utc().timestamp_millis())) // Convert assuming UTC for calculation, then get Unix micros
                }
            }
        }
        "timestamptz" => {
            let b = cast::<arrow::array::TimestampMillisecondBuilder>(writer);
            match bytes {
                None => Ok(b.append_null()),
                Some(bytes) => {
                    let dt_utc = from_sql::<DateTime<Utc>>(tp, bytes)?;
                    Ok(b.append_value(dt_utc.timestamp_millis()))
                }
            }
        }
        "interval" => {
            let b = cast::<arrow::array::IntervalMonthDayNanoBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let value = from_decoder(interval_from_sql, tp, bytes)?;
                    b.append_value(IntervalMonthDayNano::new(value.0, value.1, value.2));
                }
            }
            Ok(())
        }
        "timetz" => {
            let b = cast::<arrow::array::Time64MicrosecondBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let value = from_decoder(timetz_from_sql, tp, bytes)?;
                    b.append_value(value);
                }
            }
            Ok(())
        }

        // Range types
        "int4range" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let range = display_range_from_sql(tp, bytes, |byts| i32_from_slice(byts))?;
                    b.append_value(&range);
                }
            }
            Ok(())
        }
        "numrange" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let range = display_range_from_sql(tp, bytes, |byts| {
                        numeric_from_sql(byts).map(|n| n.to_string())
                    })?;
                    b.append_value(&range);
                }
            }
            Ok(())
        }
        "tsrange" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let range = display_range_from_sql(tp, bytes, |byts| {
                        from_sql::<NaiveDateTime>(tp, byts).map(|dt| dt.to_string())
                    })?;
                    b.append_value(&range);
                }
            }
            Ok(())
        }
        "tstzrange" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let range = display_range_from_sql(tp, bytes, |byts| {
                        from_sql::<DateTime<Utc>>(tp, byts).map(|dt| dt.to_string())
                    })?;
                    b.append_value(&range);
                }
            }
            Ok(())
        }
        "daterange" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let range = display_range_from_sql(tp, bytes, |byts| {
                        from_sql::<chrono::NaiveDate>(tp, byts).map(|d| d.to_string())
                    })?;
                    b.append_value(&range);
                }
            }
            Ok(())
        }
        "int8range" => {
            let b = cast::<arrow::array::StringBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => {
                    let range = display_range_from_sql(tp, bytes, |byts| i64_from_slice(byts))?;
                    b.append_value(&range);
                }
            }
            Ok(())
        }

        typname if typname.starts_with("_") => {
            let b = cast::<GenericListBuilder<i32, Box<dyn ArrayBuilder>>>(writer);
            match bytes {
                None => b.append(false),
                Some(bytes) => {
                    let arr = array_from_sql(bytes)
                        .map_err(|e| decode_error(tp, format!("Error decoding array: {}", e)))?;
                    let mut iter = arr.values().into_fallible_iter();
                    let inner_tp = match tp.kind() {
                        Kind::Array(inner) => inner,
                        _ => {
                            return Err(decode_error(tp, "Expected array type but found non-array"));
                        }
                    };
                    while let Some(item) = iter
                        .next()
                        .map_err(|e| decode_error(tp, format!("Error iterating array: {:?}", e)))?
                    {
                        write_pg_column_to_arrow_array(b.values(), inner_tp, item)?;
                    }
                    b.append(true);
                }
            }
            Ok(())
        }
        // Unknown
        _ => {
            let b = cast::<arrow::array::BinaryBuilder>(writer);
            match bytes {
                None => b.append_null(),
                Some(bytes) => b.append_value(bytes),
            }
            Ok(())
        }
    }
}

#[inline]
fn from_decoder<T, E: std::fmt::Display>(
    f: impl Fn(&[u8]) -> Result<T, E>,
    tp: &Type,
    bytes: &[u8],
) -> Result<T, ConnectArrowError> {
    f(bytes).map_err(|e| decode_error(tp, format!("Error decoding {}: {}", tp.name(), e)))
}

#[inline]
fn from_sql<'a, T: FromSql<'a>>(tp: &Type, bytes: &'a [u8]) -> Result<T, ConnectArrowError> {
    T::from_sql(tp, bytes).map_err(|e| {
        decode_error(
            tp,
            format!("Error decoding {:?}: {}", std::any::type_name::<T>(), e),
        )
    })
}

#[inline]
fn numeric_from_sql(bytes: &[u8]) -> Result<i128, ConnectArrowError> {
    let mut rdr = Cursor::new(bytes);
    let num_digits = rdr.read_i16::<BigEndian>()? as usize;
    let _weight = rdr.read_i16::<BigEndian>()?;
    let sign = rdr.read_i16::<BigEndian>()?;
    let _scale = rdr.read_i16::<BigEndian>()?;

    let mut value: i128 = 0;
    for _ in 0..num_digits {
        let digit = rdr.read_i16::<BigEndian>()? as i128;
        value = value * 10_000 + digit; // Each digit is base-10000
    }
    if sign == 0x4000 {
        value = -value;
    }
    Ok(value)
}

#[inline]
fn line_from_sql(bytes: &[u8]) -> Result<(f64, f64, f64), ConnectArrowError> {
    let mut rdr = Cursor::new(bytes);
    let a = rdr.read_f64::<BigEndian>()?;
    let b = rdr.read_f64::<BigEndian>()?;
    let c = rdr.read_f64::<BigEndian>()?;
    Ok((a, b, c))
}

#[inline]
fn lseg_from_sql(bytes: &[u8]) -> Result<((f64, f64), (f64, f64)), ConnectArrowError> {
    let mut rdr = Cursor::new(bytes);
    let a = rdr.read_f64::<BigEndian>()?;
    let b = rdr.read_f64::<BigEndian>()?;
    let c = rdr.read_f64::<BigEndian>()?;
    let d = rdr.read_f64::<BigEndian>()?;
    Ok(((a, b), (c, d)))
}

#[inline]
fn polygon_from_sql(bytes: &[u8]) -> Result<Vec<[f64; 2]>, ConnectArrowError> {
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
fn slice_from_sql<const N: usize>(bytes: &[u8]) -> Result<[u8; N], ConnectArrowError> {
    let mut value = [0; N];
    value.copy_from_slice(bytes);
    Ok(value)
}

#[inline]
fn interval_from_sql(bytes: &[u8]) -> Result<(i32, i32, i64), ConnectArrowError> {
    let mut rdr = Cursor::new(bytes);
    let microseconds = rdr.read_i64::<NetworkEndian>()?;
    let days = rdr.read_i32::<NetworkEndian>()?;
    let months = rdr.read_i32::<NetworkEndian>()?;
    let nanoseconds = microseconds * 1_000;
    Ok((months, days, nanoseconds))
}

#[inline]
fn timetz_from_sql(bytes: &[u8]) -> Result<i64, ConnectArrowError> {
    let mut rdr = Cursor::new(bytes);
    let microseconds = rdr.read_i64::<NetworkEndian>()?;
    Ok(microseconds)
}

#[inline]
fn i32_from_slice(bytes: &[u8]) -> Result<i32, ConnectArrowError> {
    if bytes.len() < 4 {
        return Err(ConnectArrowError::DecodeError {
            col_type: "range".to_string(),
            message: "Insufficient bytes to convert to i32".to_string(),
        });
    }
    let array: [u8; 4] = bytes[0..4]
        .try_into()
        .map_err(|_| ConnectArrowError::DecodeError {
            col_type: "range".to_string(),
            message: "Failed to convert bytes to i32".to_string(),
        })?;
    Ok(i32::from_be_bytes(array))
}

#[inline]
fn i64_from_slice(bytes: &[u8]) -> Result<i64, ConnectArrowError> {
    if bytes.len() < 8 {
        return Err(ConnectArrowError::DecodeError {
            col_type: "range".to_string(),
            message: "Insufficient bytes to convert to i64".to_string(),
        });
    }
    let array: [u8; 8] = bytes[0..8]
        .try_into()
        .map_err(|_| ConnectArrowError::DecodeError {
            col_type: "range".to_string(),
            message: "Failed to convert bytes to i64".to_string(),
        })?;
    Ok(i64::from_be_bytes(array))
}

#[inline]
pub fn bit_string_from_sql(mut buf: &[u8]) -> Result<String, ConnectArrowError> {
    let bit_length = buf.read_i32::<BigEndian>()? as usize; // Read the total number of bits
    let mut bit_string = String::with_capacity(bit_length);

    for &byte in buf {
        for i in (0..8).rev() {
            // Process each bit from MSB to LSB
            if bit_string.len() < bit_length {
                bit_string.push(if (byte & (1 << i)) != 0 { '1' } else { '0' });
            }
        }
    }

    Ok(bit_string)
}

#[inline]
fn display_range_from_sql<R: std::fmt::Display>(
    tp: &Type,
    buf: &[u8],
    f: impl Fn(&[u8]) -> Result<R, ConnectArrowError>,
) -> Result<String, ConnectArrowError> {
    let range = range_from_sql(buf)
        .map_err(|e| decode_error(tp, format!("Error decoding range: {}", e)))?;
    display_range(range, f)
}

#[inline]
fn display_range<R: std::fmt::Display>(
    range: Range<'_>,
    f: impl Fn(&[u8]) -> Result<R, ConnectArrowError>,
) -> Result<String, ConnectArrowError> {
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

/// Helper to create a standardized decode error.
#[inline]
fn decode_error(tp: &Type, message: impl Into<String>) -> ConnectArrowError {
    ConnectArrowError::DecodeError {
        col_type: tp.name().to_string(),
        message: message.into(),
    }
}

#[inline]
pub fn cast<'a, T: ArrayBuilder>(builder: &'a mut Box<dyn ArrayBuilder>) -> &'a mut T {
    builder
        .as_any_mut()
        .downcast_mut::<T>()
        .expect(&format!("bad cast to {}", type_name::<T>()))
}

// pub fn write_struct_null<A: ArrayBuilder>(
//     builder: &mut StructBuilder,
//     f: impl Fn(&mut A),
// ) -> Result<(), ConnectArrowError> {
//     for i in 0..builder.len() {
//         f(builder.field_builder::<A>(i).unwrap());
//     }
//     builder.append(false);
//     Ok(())
// }

// pub fn write_struct_value<A: ArrayBuilder, T, R: IntoIterator<Item = T>>(
//     builder: &mut StructBuilder,
//     value: R,
//     f: impl Fn(&mut A, T) -> Result<(), ConnectArrowError>,
// ) -> Result<(), ConnectArrowError> {
//     for (i, value) in value.into_iter().enumerate() {
//         f(builder.field_builder::<A>(i).unwrap(), value)?;
//     }
//     builder.append(true);
//     Ok(())
// }
