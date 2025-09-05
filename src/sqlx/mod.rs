use arrow::array::{ArrayBuilder, ArrayRef};
use arrow::datatypes::{Field, Schema};
use arrow::error::ArrowError;
use postgres_types::Type;
use sqlx::postgres::{PgRow, PgStatement};
use sqlx::{Column, Row, Statement};
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    ColumnEncoder, ColumnFactory, EncodeBuilder as EB, EncodeBuilderError, PgErr, RowBatcher,
    encode_pg_column, get_arrow_type_from_pg_oid,
};

#[derive(Debug, thiserror::Error)]
pub enum PostgresEncodeError {
    #[error("Field is not nullable but value is null")]
    UnexpectedNullValue,
    #[error("Failed to get builder: {0}")]
    EncodeBuilderError(#[from] EncodeBuilderError),
    #[error("Unsupported Postgres type for encoding: {0}")]
    UnsupportedType(String),
    #[error("PgErr: {0}")]
    PgErr(#[from] PgErr),
    #[error("Arrow error: {0}")]
    ArrowError(#[from] ArrowError),
    #[error("SQLx error: {0}")]
    SqlxError(#[from] sqlx::Error),
    #[error("Missing column in PgCtx for field: {0}")]
    MissingColumn(String),
}

pub struct PgColumn {
    builder: EB,
    pg_type: Type,
    col_name: String, // lookup in PgRow
}

impl ColumnEncoder<PgRow> for PgColumn {
    type Error = PostgresEncodeError;

    fn append(&mut self, row: &PgRow) -> Result<(), Self::Error> {
        let raw = row
            .try_get_raw(self.col_name.as_str())
            .map_err(|_| PostgresEncodeError::MissingColumn(self.col_name.clone()))?;
        let bytes = raw.as_bytes().ok();
        encode_pg_column(&mut self.builder, &self.pg_type, bytes).map_err(From::from)
    }

    fn finish(&mut self) -> ArrayRef {
        ArrayBuilder::finish(&mut self.builder)
    }
}

/// Context built from a prepared statement: maps column name -> PG type
pub struct PgCtx {
    by_name: HashMap<String, Type>,
}

pub struct PgFactory;

impl ColumnFactory<PgRow, PgCtx> for PgFactory {
    type Error = PostgresEncodeError;
    type Col = PgColumn;

    fn make(field: &Field, capacity: usize, ctx: &PgCtx) -> Result<Self::Col, Self::Error> {
        let pg_type = ctx
            .by_name
            .get(field.name())
            .ok_or_else(|| PostgresEncodeError::MissingColumn(field.name().to_string()))?
            .clone();

        let builder = EB::try_with_capacity(field.data_type(), capacity)?;
        Ok(PgColumn {
            builder,
            pg_type,
            col_name: field.name().to_string(),
        })
    }
}

/* ------------ helpers to build schema + ctx from PgStatement ------------ */

fn schema_and_ctx_from_stmt(
    stmt: &PgStatement<'_>,
) -> Result<(Arc<Schema>, PgCtx), PostgresEncodeError> {
    let mut fields = Vec::with_capacity(stmt.columns().len());
    let mut by_name = HashMap::with_capacity(stmt.columns().len());

    for col in stmt.columns() {
        let oid = col
            .type_info()
            .oid()
            .ok_or_else(|| {
                PostgresEncodeError::UnsupportedType(format!(
                    "Unknown OID for column {}",
                    col.name()
                ))
            })?
            .0;

        let pg_type = Type::from_oid(oid)
            .ok_or_else(|| PostgresEncodeError::UnsupportedType(format!("Unknown OID: {}", oid)))?;
        let arrow_dt = get_arrow_type_from_pg_oid(oid)?;

        fields.push(Field::new(col.name(), arrow_dt, true));
        by_name.insert(col.name().to_string(), pg_type);
    }

    Ok((Arc::new(Schema::new(fields)), PgCtx { by_name }))
}

pub fn create_batcher(
    stmt: &PgStatement<'_>,
    batch_size: usize,
) -> Result<RowBatcher<PgRow, PgColumn>, PostgresEncodeError> {
    let (schema, ctx) = schema_and_ctx_from_stmt(stmt)?;
    RowBatcher::<PgRow, PgColumn>::from_schema_with::<PgFactory, _>(schema, batch_size, &ctx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode_stream;
    use arrow::array::*;
    use futures::{StreamExt, TryStreamExt};
    use sqlx::{Executor, PgPool};

    const DB_URL: &str = "postgres://test:test@localhost/test";

    async fn setup_db() -> PgPool {
        PgPool::connect(DB_URL).await.unwrap()
    }

    #[tokio::test]
    async fn test_basic_bool_encoding() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT col_bool FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 1);

        let bool_array = batch.column(0).as_boolean();
        assert_eq!(bool_array.value(0), true);
    }

    #[tokio::test]
    async fn test_numeric_types() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare(
                "SELECT col_int2, col_int4, col_int8, col_float4, col_float8, col_oid FROM example",
            )
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 6);

        let int16_array = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Int16Type>();
        assert_eq!(int16_array.value(0), 123);

        let int32_array = batch
            .column(1)
            .as_primitive::<arrow::datatypes::Int32Type>();
        assert_eq!(int32_array.value(0), 123456);

        let int64_array = batch
            .column(2)
            .as_primitive::<arrow::datatypes::Int64Type>();
        assert_eq!(int64_array.value(0), 1234567890123);

        let float32_array = batch
            .column(3)
            .as_primitive::<arrow::datatypes::Float32Type>();
        assert!((float32_array.value(0) - 1.23).abs() < 0.01);

        let float64_array = batch
            .column(4)
            .as_primitive::<arrow::datatypes::Float64Type>();
        assert!((float64_array.value(0) - 3.14159).abs() < 0.00001);

        let oid_array = batch
            .column(5)
            .as_primitive::<arrow::datatypes::UInt32Type>();
        assert_eq!(oid_array.value(0), 12345);
    }

    #[tokio::test]
    async fn test_string_types() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_text, col_name, col_varchar, col_char_alt FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 4);

        let text_array = batch.column(0).as_string::<i32>();
        assert_eq!(text_array.value(0), "example text");

        let name_array = batch.column(1).as_string::<i32>();
        assert_eq!(name_array.value(0), "example_name");

        let varchar_array = batch.column(2).as_string::<i32>();
        assert_eq!(varchar_array.value(0), "sample varchar");

        let char_array = batch.column(3).as_string::<i32>();
        assert_eq!(char_array.value(0), "X");
    }

    #[tokio::test]
    async fn test_binary_types() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT col_bytea FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        let binary_array = batch.column(0).as_binary::<i32>();
        let expected = vec![0xDE, 0xAD, 0xBE, 0xEF];
        assert_eq!(binary_array.value(0), expected);
    }

    #[tokio::test]
    async fn test_json_types() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_json, col_jsonb FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 2);

        let json_array = batch.column(0).as_string::<i32>();
        assert!(json_array.value(0).contains("key"));

        let jsonb_array = batch.column(1).as_string::<i32>();
        assert!(jsonb_array.value(0).contains("jsonb_key"));
    }

    #[tokio::test]
    async fn test_uuid_type() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT col_uuid FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        let uuid_array = batch.column(0).as_string::<i32>();
        assert!(uuid_array.value(0).contains("a0eebc99"));
    }

    #[tokio::test]
    async fn test_numeric_type() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_numeric FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        let numeric_array = batch.column(0).as_string::<i32>();
        assert!(numeric_array.value(0).contains("9876"));
    }

    #[tokio::test]
    async fn test_money_type() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT col_money FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        let money_array = batch.column(0).as_string::<i32>();
        assert!(money_array.value(0).contains("123.45"));
    }

    #[tokio::test]
    async fn test_date_time_types() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_date, col_time, col_timestamp, col_timestamptz FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 4);

        let date_array = batch
            .column(0)
            .as_primitive::<arrow::datatypes::Date32Type>();
        assert!(date_array.value(0) > 0); // Should be a valid date

        let time_array = batch
            .column(1)
            .as_primitive::<arrow::datatypes::Time64MicrosecondType>();
        assert!(time_array.value(0) > 0); // Should be a valid time

        let timestamp_array = batch
            .column(2)
            .as_primitive::<arrow::datatypes::TimestampMillisecondType>();
        assert!(timestamp_array.value(0) > 0); // Should be a valid timestamp

        let timestamptz_array = batch
            .column(3)
            .as_primitive::<arrow::datatypes::TimestampMillisecondType>();
        assert!(timestamptz_array.value(0) > 0); // Should be a valid timestamp with timezone
    }

    #[tokio::test]
    async fn test_interval_type() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_interval FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        let interval_array = batch
            .column(0)
            .as_primitive::<arrow::datatypes::IntervalMonthDayNanoType>();
        assert!(interval_array.value(0).days > 0); // Should have some days
    }

    #[tokio::test]
    async fn test_network_types() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_cidr, col_inet, col_macaddr, col_macaddr8 FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 4);

        let cidr_array = batch.column(0).as_string::<i32>();
        assert!(cidr_array.value(0).contains("192.168.1.0"));

        let inet_array = batch.column(1).as_string::<i32>();
        assert!(inet_array.value(0).contains("192.168.100.128"));

        let macaddr_array = batch.column(2).as_string::<i32>();
        assert!(macaddr_array.value(0).contains("08:00:2b"));

        let macaddr8_array = batch.column(3).as_string::<i32>();
        assert!(macaddr8_array.value(0).contains("01:23:45"));
    }

    #[tokio::test]
    async fn test_geometry_types() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT col_point, col_lseg, col_path, col_box, col_polygon, col_line, col_circle FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 7);

        // All geometry types should be encoded as strings
        for i in 0..7 {
            let geom_array = batch.column(i).as_string::<i32>();
            assert!(!geom_array.value(0).is_empty());
        }
    }

    #[tokio::test]
    async fn test_range_types() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT col_int4range, col_numrange, col_tsrange, col_tstzrange, col_daterange, col_int8range FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 6);

        // All range types should be encoded as strings
        for i in 0..6 {
            let range_array = batch.column(i).as_string::<i32>();
            assert!(!range_array.value(0).is_empty());
        }
    }

    #[tokio::test]
    async fn test_bit_types() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_bit, col_varbit FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 2);

        let bit_array = batch.column(0).as_string::<i32>();
        assert_eq!(bit_array.value(0), "1");

        let varbit_array = batch.column(1).as_string::<i32>();
        assert_eq!(varbit_array.value(0), "10101");
    }

    #[tokio::test]
    async fn test_array_types() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_bool_array, col_int4_array, col_text_array FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        assert_eq!(batch.num_columns(), 3);

        // Check array types are properly handled
        // col_bool_array should be List<Boolean>
        assert!(matches!(
            batch.column(0).data_type(),
            arrow::datatypes::DataType::List(_)
        ));

        // col_int4_array should be List<Int32>
        assert!(matches!(
            batch.column(1).data_type(),
            arrow::datatypes::DataType::List(_)
        ));

        // col_text_array should be List<Utf8>
        assert!(matches!(
            batch.column(2).data_type(),
            arrow::datatypes::DataType::List(_)
        ));
    }

    #[tokio::test]
    async fn test_xml_type() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT col_xml FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        let xml_array = batch.column(0).as_string::<i32>();
        assert!(xml_array.value(0).contains("<note>"));
    }

    #[tokio::test]
    async fn test_jsonpath_type() {
        let pool = setup_db().await;
        let stmt = pool
            .prepare("SELECT col_jsonpath FROM example")
            .await
            .unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;

        let batch = &batches[0].as_ref().unwrap();
        let jsonpath_array = batch.column(0).as_string::<i32>();
        assert!(jsonpath_array.value(0).contains("$.\"store\""));
    }

    #[tokio::test]
    async fn test_all_types() {
        let pool = setup_db().await;
        let stmt = pool.prepare("SELECT * FROM example").await.unwrap();
        let batcher = create_batcher(&stmt, 10).unwrap();

        let batches = encode_stream(batcher, stmt.query().fetch(&pool).map_err(From::from));
        let batches = batches.collect::<Vec<_>>().await;
        for batch in batches {
            let batch = batch.unwrap();
            for i in 0..batch.num_columns() {
                let col = batch.column(i);
                println!("{:?}", col.data_type());
            }
        }
    }
}
