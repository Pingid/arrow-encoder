### "./src/lib.rs"
```rs
// Select Arrow crate version via features and expose it as `arrow`
// Only one Arrow version feature should be enabled
#[cfg(any(
    all(feature = "arrow-54", feature = "arrow-55"),
    all(feature = "arrow-54", feature = "arrow-56"),
    all(feature = "arrow-55", feature = "arrow-56"),
))]
compile_error!("Features 'arrow-54', 'arrow-55' and 'arrow-56' are mutually exclusive");

#[cfg(feature = "arrow-54")]
extern crate arrow_v54 as arrow;
#[cfg(feature = "arrow-55")]
extern crate arrow_v55 as arrow;

#[cfg(feature = "arrow-56")]
extern crate arrow_v56 as arrow;

use futures::stream;
use futures::{Stream, TryStreamExt};
use std::any::Any;
use std::marker::PhantomData;
use std::sync::Arc;

use arrow::{
    array::{ArrayRef, RecordBatch},
    datatypes::{Field, Schema},
    error::ArrowError,
};

#[cfg(feature = "json")]
mod json;
#[cfg(feature = "json")]
pub use json::*;

#[cfg(feature = "postgres")]
mod postgres;
#[cfg(feature = "postgres")]
pub use postgres::*;

#[cfg(feature = "postgres")]
mod sqlx;
#[cfg(feature = "postgres")]
pub use sqlx::*;

/// Per-column encoder that knows how to pull its value from a row `R`.
pub trait ColumnEncoder<R> {
    type Error;
    fn append(&mut self, row: &R) -> Result<(), Self::Error>;
    fn finish(&mut self) -> ArrayRef;
}

/// Factory to build a column encoder from a schema field + capacity + optional context.
pub trait ColumnFactory<R, Ctx = ()> {
    type Error;
    type Col: ColumnEncoder<R, Error = Self::Error>;
    fn make(field: &Field, capacity: usize, ctx: &Ctx) -> Result<Self::Col, Self::Error>;
}

/// Batches rows into Arrow RecordBatches using a set of per-column encoders.
pub struct RowBatcher<R, C> {
    schema: Arc<Schema>,
    cols: Vec<C>,
    batch_size: usize,
    rows: usize,
    _marker: PhantomData<R>,
}

impl<R, C: ColumnEncoder<R>> RowBatcher<R, C> {
    pub fn new(schema: Arc<Schema>, cols: Vec<C>, batch_size: usize) -> Self {
        Self {
            schema,
            cols,
            batch_size,
            rows: 0,
            _marker: PhantomData,
        }
    }

    /// Build from a schema using a factory. Function-generic → no E0207.
    pub fn from_schema_with<F, Ctx>(
        schema: Arc<Schema>,
        batch_size: usize,
        ctx: &Ctx,
    ) -> Result<Self, F::Error>
    where
        F: ColumnFactory<R, Ctx, Col = C, Error = C::Error>,
    {
        let cols = schema
            .fields()
            .iter()
            .map(|f| F::make(f, batch_size, ctx))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self::new(schema, cols, batch_size))
    }

    pub fn schema(&self) -> &Arc<Schema> {
        &self.schema
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.rows >= self.batch_size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn push_row(&mut self, row: &R) -> Result<(), C::Error> {
        for col in self.cols.iter_mut() {
            col.append(row)?;
        }
        self.rows += 1;
        Ok(())
    }

    pub fn finish_batch(&mut self) -> Result<RecordBatch, C::Error>
    where
        C::Error: From<ArrowError>,
    {
        let arrays: Vec<ArrayRef> = self.cols.iter_mut().map(|c| c.finish()).collect();
        self.rows = 0;
        RecordBatch::try_new(self.schema.clone(), arrays).map_err(From::from)
    }
}
/// Iterator adapter: turns an iterator of rows into batches.
pub struct FieldBatchEncoder<R, C: ColumnEncoder<R>, V: Iterator<Item = R>> {
    values: V,
    batcher: RowBatcher<R, C>,
}

impl<R, C, V> FieldBatchEncoder<R, C, V>
where
    C: ColumnEncoder<R>,
    V: Iterator<Item = R>,
{
    /// Build from schema + factory (function-generic).
    pub fn from_schema_with<F, Ctx>(
        schema: Arc<Schema>,
        values: V,
        batch_size: usize,
        ctx: &Ctx,
    ) -> Result<Self, F::Error>
    where
        F: ColumnFactory<R, Ctx, Col = C, Error = C::Error>,
    {
        let batcher = RowBatcher::<R, C>::from_schema_with::<F, Ctx>(schema, batch_size, ctx)?;
        Ok(Self { values, batcher })
    }

    pub fn new(schema: Arc<Schema>, cols: Vec<C>, values: V, batch_size: usize) -> Self {
        let batcher = RowBatcher::new(schema, cols, batch_size);
        Self { values, batcher }
    }

    /// Fills up to `batch_size` rows or until the input iterator is exhausted.
    pub fn write_next(&mut self) -> Result<Option<RecordBatch>, C::Error>
    where
        C::Error: From<ArrowError>,
    {
        while !self.batcher.is_full() {
            match self.values.next() {
                Some(row) => self.batcher.push_row(&row)?,
                None => break,
            }
        }

        if self.batcher.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.batcher.finish_batch()?))
        }
    }
}

impl<R, E, C, V> Iterator for FieldBatchEncoder<R, C, V>
where
    C: ColumnEncoder<R, Error = E>,
    V: Iterator<Item = R>,
    E: From<ArrowError>,
{
    type Item = Result<RecordBatch, E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.write_next().transpose()
    }
}

/// Async variant over a `Stream` of rows.
pub fn encode_stream<R, C, E, S>(
    batcher: RowBatcher<R, C>,
    values: S,
) -> impl Stream<Item = Result<RecordBatch, E>>
where
    C: ColumnEncoder<R, Error = E>,
    S: Stream<Item = Result<R, E>>,
    E: From<ArrowError>,
{
    stream::try_unfold(
        (Box::pin(values), batcher),
        |(mut values_stream, mut batcher)| async move {
            let mut count = 0;
            let batch_size = batcher.batch_size;

            while count < batch_size {
                match values_stream.try_next().await {
                    Ok(Some(row)) => {
                        batcher.push_row(&row)?;
                        count += 1;
                    }
                    Ok(None) => break,
                    Err(e) => return Err(e),
                }
            }

            if count == 0 {
                Ok(None)
            } else {
                let batch = batcher.finish_batch()?;
                Ok(Some((batch, (values_stream, batcher))))
            }
        },
    )
}

macro_rules! make_encoder {
    ($name:ident { $(($variant:ident, $builder:ty)),* $(,)? }) => {
        #[derive(Debug)]
        pub enum $name {
            $( $variant($builder) ),*
        }

        impl $name {
            pub fn append_null(&mut self) {
                match self {
                    $( $name::$variant(builder) => builder.append_null(), )*
                }
            }
        }

        impl ArrayBuilder for $name {
            fn as_any(&self) -> &dyn Any {
                match self { $( $name::$variant(b) => b.as_any(), )* }
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                match self { $( $name::$variant(b) => b.as_any_mut(), )* }
            }
            fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
                match *self { $( $name::$variant(b) => Box::new(b).into_box_any(), )* }
            }
            fn len(&self) -> usize {
                match self { $( $name::$variant(b) => b.len(), )* }
            }
            fn finish(&mut self) -> ArrayRef {
                match self { $( $name::$variant(b) => Arc::new(b.finish()), )* }
            }
            fn finish_cloned(&self) -> ArrayRef {
                match self { $( $name::$variant(b) => Arc::new(b.finish_cloned()), )* }
            }
        }

        impl ArrayBuilder for Box<$name> {
            fn as_any(&self) -> &dyn Any {
                match self.as_ref() { $( $name::$variant(b) => b.as_any(), )* }
            }
            fn as_any_mut(&mut self) -> &mut dyn Any {
                match self.as_mut() { $( $name::$variant(b) => b.as_any_mut(), )* }
            }
            fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
                unimplemented!()
            }
            fn len(&self) -> usize {
                match self.as_ref() { $( $name::$variant(b) => b.len(), )* }
            }
            fn finish(&mut self) -> ArrayRef {
                match self.as_mut() { $( $name::$variant(b) => Arc::new(b.finish()), )* }
            }
            fn finish_cloned(&self) -> ArrayRef {
                match self.as_ref() { $( $name::$variant(b) => Arc::new(b.finish_cloned()), )* }
            }
        }
    };
}

use arrow::array::MapFieldNames;
use arrow::array::StructArray;
use arrow::array::*;
use arrow::datatypes::IntervalUnit::*;
use arrow::datatypes::TimeUnit::*;
use arrow::datatypes::{
    DataType::{self, *},
    Fields, Int32Type,
};

make_encoder!(EncodeBuilder {
    (Null, NullBuilder),
    (Bool, BooleanBuilder),
    // Number variants
    (I8, Int8Builder),
    (I16, Int16Builder),
    (I32, Int32Builder),
    (I64, Int64Builder),
    (UI8, UInt8Builder),
    (UI16, UInt16Builder),
    (UI32, UInt32Builder),
    (UI64, UInt64Builder),
    (F16, Float16Builder),
    (F32, Float32Builder),
    (F64, Float64Builder),

    // String variants
    (Str, StringBuilder),
    (LargeStr, LargeStringBuilder),
    (StrView, StringViewBuilder),

    // Timestamp variants
    (TsSecond, TimestampSecondBuilder),
    (TsMs, TimestampMillisecondBuilder),
    (TsMicro, TimestampMicrosecondBuilder),
    (TsNano, TimestampNanosecondBuilder),

    // Date variants
    (Date32, Date32Builder),
    (Date64, Date64Builder),

    // Time variants
    (Time32Second, Time32SecondBuilder),
    (Time32Ms, Time32MillisecondBuilder),
    (Time64Micro, Time64MicrosecondBuilder),
    (Time64Nano, Time64NanosecondBuilder),

    // Duration variants
    (DurationSecond, DurationSecondBuilder),
    (DurationMs, DurationMillisecondBuilder),
    (DurationMicro, DurationMicrosecondBuilder),
    (DurationNano, DurationNanosecondBuilder),

    // Interval variants
    (IntervalYearMonth, IntervalYearMonthBuilder),
    (IntervalDayTime, IntervalDayTimeBuilder),
    (IntervalMonthDayNano, IntervalMonthDayNanoBuilder),

    // Binary variants
    (Binary, BinaryBuilder),
    (FixedSizeBinary, FixedSizeBinaryBuilder),
    (LargeBinary, LargeBinaryBuilder),
    (BinaryView, BinaryViewBuilder),

    // Decimal variants
    (Decimal32, Decimal32Builder),
    (Decimal128, Decimal128Builder),
    (Decimal256, Decimal256Builder),

    // List variants
    (List, ListBuilderWrapper),
    (FixedSizeList, FixedSizeListBuilderWrapper),
    (LargeList, LargeListBuilderWrapper),

    // Complex variants
    (Struct, StructBuilderWrapper),
    (Dictionary, DictionaryBuilderWrapper),
    (Map, MapBuilderWrapper),
});

// Arrow < 56 does not provide Decimal32Builder. Provide a compatibility alias
// so the enum compiles, but Decimal32 code paths are gated elsewhere.
#[cfg(not(feature = "arrow-56"))]
type Decimal32Builder = Decimal128Builder;

#[derive(Debug)]
pub struct ListBuilderWrapper {
    builder: ListBuilder<Box<EncodeBuilder>>,
    field: Arc<Field>,
}

impl ListBuilderWrapper {
    pub fn new(builder: ListBuilder<Box<EncodeBuilder>>, field: Arc<Field>) -> Self {
        Self { builder, field }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false);
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid);
        Ok(())
    }
}

impl ArrayBuilder for ListBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }

    fn len(&self) -> usize {
        self.builder.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let list_array = self.builder.finish();
        let offsets = list_array.offsets().clone();
        let values = list_array.values().clone();
        let nulls = list_array.nulls().cloned();
        Arc::new(ListArray::new(self.field.clone(), offsets, values, nulls))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let list_array = self.builder.finish_cloned();
        let offsets = list_array.offsets().clone();
        let values = list_array.values().clone();
        let nulls = list_array.nulls().cloned();
        Arc::new(ListArray::new(self.field.clone(), offsets, values, nulls))
    }
}

#[derive(Debug)]
pub struct FixedSizeListBuilderWrapper {
    builder: FixedSizeListBuilder<Box<EncodeBuilder>>,
    field: Arc<Field>,
    size: i32,
}

impl FixedSizeListBuilderWrapper {
    pub fn new(
        builder: FixedSizeListBuilder<Box<EncodeBuilder>>,
        field: Arc<Field>,
        size: i32,
    ) -> Self {
        Self {
            builder,
            field,
            size,
        }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false);
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid);
        Ok(())
    }
}

impl ArrayBuilder for FixedSizeListBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }

    fn len(&self) -> usize {
        self.builder.len()
    }

    fn finish(&mut self) -> ArrayRef {
        let array = self.builder.finish();
        let values = array.values().clone();
        let nulls = array.nulls().cloned();
        let size = self.size;
        Arc::new(FixedSizeListArray::new(
            self.field.clone(),
            size,
            values,
            nulls,
        ))
    }

    fn finish_cloned(&self) -> ArrayRef {
        let array = self.builder.finish_cloned();
        let values = array.values().clone();
        let nulls = array.nulls().cloned();
        let size = self.size;
        Arc::new(FixedSizeListArray::new(
            self.field.clone(),
            size,
            values,
            nulls,
        ))
    }
}

// Wrapper for StructBuilder to work with EncodeBuilder field builders
#[derive(Debug)]
pub struct StructBuilderWrapper {
    // builder: StructBuilder,
    fields: Fields,
    builders: Vec<Box<EncodeBuilder>>,
    null_buffer_builder: NullBufferBuilder,
}

impl ArrayBuilder for StructBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }

    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }

    fn len(&self) -> usize {
        self.null_buffer_builder.len()
    }

    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.finnish())
    }

    fn finish_cloned(&self) -> ArrayRef {
        Arc::new(self.finish_cloned())
    }
}

impl StructBuilderWrapper {
    pub fn new(fields: Fields, builders: Vec<Box<EncodeBuilder>>) -> Self {
        if fields.len() != builders.len() {
            panic!("Number of fields is not equal to the number of field_builders.");
        }
        Self {
            fields,
            builders,
            null_buffer_builder: NullBufferBuilder::new(0),
        }
    }

    pub fn append(&mut self, is_valid: bool) {
        self.null_buffer_builder.append(is_valid);
    }

    pub fn append_null(&mut self) {
        self.append(false);
    }

    pub fn finnish(&mut self) -> StructArray {
        if self.fields.is_empty() {
            return StructArray::new_empty_fields(self.len(), self.null_buffer_builder.finish());
        }

        let arrays = self.builders.iter_mut().map(|f| f.finish()).collect();
        let nulls = self.null_buffer_builder.finish();
        StructArray::new(self.fields.clone(), arrays, nulls)
    }

    pub fn finish_cloned(&self) -> StructArray {
        if self.fields.is_empty() {
            return StructArray::new_empty_fields(
                self.len(),
                self.null_buffer_builder.finish_cloned(),
            );
        }

        let arrays = self.builders.iter().map(|f| f.finish_cloned()).collect();
        let nulls = self.null_buffer_builder.finish_cloned();
        StructArray::new(self.fields.clone(), arrays, nulls)
    }
}

// Wrapper for LargeListBuilder
#[derive(Debug)]
pub struct LargeListBuilderWrapper {
    builder: LargeListBuilder<Box<EncodeBuilder>>,
    field: Arc<Field>,
}

impl LargeListBuilderWrapper {
    pub fn new(builder: LargeListBuilder<Box<EncodeBuilder>>, field: Arc<Field>) -> Self {
        Self { builder, field }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false);
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid);
        Ok(())
    }
}

impl ArrayBuilder for LargeListBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }
    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }
    fn len(&self) -> usize {
        self.builder.len()
    }
    fn finish(&mut self) -> ArrayRef {
        let list = self.builder.finish();
        let offsets = list.offsets().clone();
        let values = list.values().clone();
        let nulls = list.nulls().cloned();
        Arc::new(LargeListArray::new(
            self.field.clone(),
            offsets,
            values,
            nulls,
        ))
    }
    fn finish_cloned(&self) -> ArrayRef {
        let list = self.builder.finish_cloned();
        let offsets = list.offsets().clone();
        let values = list.values().clone();
        let nulls = list.nulls().cloned();
        Arc::new(LargeListArray::new(
            self.field.clone(),
            offsets,
            values,
            nulls,
        ))
    }
}

// Wrapper for Dictionary builders
pub struct DictionaryBuilderWrapper {
    builder: StringDictionaryBuilder<Int32Type>,
}

impl std::fmt::Debug for DictionaryBuilderWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DictionaryBuilderWrapper")
    }
}

impl DictionaryBuilderWrapper {
    pub fn new(builder: StringDictionaryBuilder<Int32Type>) -> Self {
        Self { builder }
    }

    pub fn append_null(&mut self) {
        self.builder.append_null();
    }

    pub fn append_value(&mut self, value: &str) -> Result<(), ArrowError> {
        self.builder.append_value(value);
        Ok(())
    }
}

impl ArrayBuilder for DictionaryBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }
    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }
    fn len(&self) -> usize {
        self.builder.len()
    }
    fn finish(&mut self) -> ArrayRef {
        Arc::new(self.builder.finish())
    }
    fn finish_cloned(&self) -> ArrayRef {
        Arc::new(self.builder.finish_cloned())
    }
}

// Wrapper for MapBuilder
#[derive(Debug)]
pub struct MapBuilderWrapper {
    builder: MapBuilder<StringBuilder, Box<EncodeBuilder>>,
    field: Arc<Field>,
}

impl MapBuilderWrapper {
    pub fn new(builder: MapBuilder<StringBuilder, Box<EncodeBuilder>>, field: Arc<Field>) -> Self {
        Self { builder, field }
    }

    pub fn append_null(&mut self) {
        self.builder.append(false).unwrap();
    }

    pub fn keys(&mut self) -> &mut StringBuilder {
        self.builder.keys()
    }

    pub fn values(&mut self) -> &mut Box<EncodeBuilder> {
        self.builder.values()
    }

    pub fn append(&mut self, is_valid: bool) -> Result<(), ArrowError> {
        self.builder.append(is_valid)
    }
}

impl ArrayBuilder for MapBuilderWrapper {
    fn as_any(&self) -> &dyn Any {
        &self.builder as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.builder as &mut dyn Any
    }
    fn into_box_any(self: Box<Self>) -> Box<dyn Any> {
        Box::new(self.builder)
    }
    fn len(&self) -> usize {
        self.builder.len()
    }
    fn finish(&mut self) -> ArrayRef {
        let map = self.builder.finish();
        let offsets = map.offsets().clone();
        let entries = map.entries();
        let nulls = map.nulls().cloned();
        let sorted = match self.field.data_type() {
            DataType::Map(_, s) => *s,
            _ => false,
        };
        let rebuilt_entries: StructArray = match self.field.data_type() {
            DataType::Map(entry_field, _) => match entry_field.data_type() {
                DataType::Struct(struct_fields) => {
                    let key = entries.column(0).clone();
                    let value = entries.column(1).clone();
                    StructArray::new(
                        struct_fields.clone(),
                        vec![key, value],
                        entries.nulls().cloned(),
                    )
                }
                _ => entries.clone(),
            },
            _ => entries.clone(),
        };
        Arc::new(MapArray::new(
            self.field.clone(),
            offsets,
            rebuilt_entries,
            nulls,
            sorted,
        ))
    }
    fn finish_cloned(&self) -> ArrayRef {
        let map = self.builder.finish_cloned();
        let offsets = map.offsets().clone();
        let entries = map.entries();
        let nulls = map.nulls().cloned();
        let sorted = match self.field.data_type() {
            DataType::Map(_, s) => *s,
            _ => false,
        };
        let rebuilt_entries: StructArray = match self.field.data_type() {
            DataType::Map(entry_field, _) => match entry_field.data_type() {
                DataType::Struct(struct_fields) => {
                    let key = entries.column(0).clone();
                    let value = entries.column(1).clone();
                    StructArray::new(
                        struct_fields.clone(),
                        vec![key, value],
                        entries.nulls().cloned(),
                    )
                }
                _ => entries.clone(),
            },
            _ => entries.clone(),
        };
        Arc::new(MapArray::new(
            self.field.clone(),
            offsets,
            rebuilt_entries,
            nulls,
            sorted,
        ))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EncodeBuilderError {
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(DataType),
    #[error("Arrow error: {0}")]
    ArrowError(#[from] ArrowError),
}

impl EncodeBuilder {
    pub fn try_new(data_type: &DataType) -> Result<Self, EncodeBuilderError> {
        match data_type {
            Null => Ok(EncodeBuilder::Null(NullBuilder::new())),
            Boolean => Ok(EncodeBuilder::Bool(BooleanBuilder::new())),
            Int8 => Ok(EncodeBuilder::I8(Int8Builder::new())),
            Int16 => Ok(EncodeBuilder::I16(Int16Builder::new())),
            Int32 => Ok(EncodeBuilder::I32(Int32Builder::new())),
            Int64 => Ok(EncodeBuilder::I64(Int64Builder::new())),
            UInt8 => Ok(EncodeBuilder::UI8(UInt8Builder::new())),
            UInt16 => Ok(EncodeBuilder::UI16(UInt16Builder::new())),
            UInt32 => Ok(EncodeBuilder::UI32(UInt32Builder::new())),
            UInt64 => Ok(EncodeBuilder::UI64(UInt64Builder::new())),
            Float16 => Ok(EncodeBuilder::F16(Float16Builder::new())),
            Float32 => Ok(EncodeBuilder::F32(Float32Builder::new())),
            Float64 => Ok(EncodeBuilder::F64(Float64Builder::new())),
            // Timestamp variants
            Timestamp(Second, tz) => Ok(EncodeBuilder::TsSecond(
                TimestampSecondBuilder::new().with_timezone_opt(tz.clone()),
            )),
            Timestamp(Millisecond, tz) => Ok(EncodeBuilder::TsMs(
                TimestampMillisecondBuilder::new().with_timezone_opt(tz.clone()),
            )),
            Timestamp(Microsecond, tz) => Ok(EncodeBuilder::TsMicro(
                TimestampMicrosecondBuilder::new().with_timezone_opt(tz.clone()),
            )),
            Timestamp(Nanosecond, tz) => Ok(EncodeBuilder::TsNano(
                TimestampNanosecondBuilder::new().with_timezone_opt(tz.clone()),
            )),

            // Date variants
            Date32 => Ok(EncodeBuilder::Date32(Date32Builder::new())),
            Date64 => Ok(EncodeBuilder::Date64(Date64Builder::new())),

            // Time variants
            Time32(Second) => Ok(EncodeBuilder::Time32Second(Time32SecondBuilder::new())),
            Time32(Millisecond) => Ok(EncodeBuilder::Time32Ms(Time32MillisecondBuilder::new())),
            Time64(Microsecond) => Ok(EncodeBuilder::Time64Micro(Time64MicrosecondBuilder::new())),
            Time64(Nanosecond) => Ok(EncodeBuilder::Time64Nano(Time64NanosecondBuilder::new())),

            // Duration variants
            Duration(Second) => Ok(EncodeBuilder::DurationSecond(DurationSecondBuilder::new())),
            Duration(Millisecond) => {
                Ok(EncodeBuilder::DurationMs(DurationMillisecondBuilder::new()))
            }
            Duration(Microsecond) => Ok(EncodeBuilder::DurationMicro(
                DurationMicrosecondBuilder::new(),
            )),
            Duration(Nanosecond) => {
                Ok(EncodeBuilder::DurationNano(DurationNanosecondBuilder::new()))
            }

            // Interval variants
            Interval(YearMonth) => Ok(EncodeBuilder::IntervalYearMonth(
                IntervalYearMonthBuilder::new(),
            )),
            Interval(DayTime) => Ok(EncodeBuilder::IntervalDayTime(IntervalDayTimeBuilder::new())),
            Interval(MonthDayNano) => Ok(EncodeBuilder::IntervalMonthDayNano(
                IntervalMonthDayNanoBuilder::new(),
            )),

            // Binary variants
            Binary => Ok(EncodeBuilder::Binary(BinaryBuilder::new())),
            FixedSizeBinary(size) => Ok(EncodeBuilder::FixedSizeBinary(
                FixedSizeBinaryBuilder::new(*size),
            )),
            LargeBinary => Ok(EncodeBuilder::LargeBinary(LargeBinaryBuilder::new())),
            BinaryView => Ok(EncodeBuilder::BinaryView(BinaryViewBuilder::new())),

            // String variants
            Utf8 => Ok(EncodeBuilder::Str(StringBuilder::new())),
            LargeUtf8 => Ok(EncodeBuilder::LargeStr(LargeStringBuilder::new())),
            Utf8View => Ok(EncodeBuilder::StrView(StringViewBuilder::new())),

            // List variants
            List(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = ListBuilder::new(inner_builder);
                Ok(EncodeBuilder::List(ListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Decimal variants - use Decimal128Builder for smaller decimals
            #[cfg(feature = "arrow-56")]
            Decimal32(precision, scale) => Ok(EncodeBuilder::Decimal32(
                Decimal32Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal128(precision, scale) => Ok(EncodeBuilder::Decimal128(
                Decimal128Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal256(precision, scale) => Ok(EncodeBuilder::Decimal256(
                Decimal256Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),

            // Additional List variants
            FixedSizeList(field, size) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = FixedSizeListBuilder::new(inner_builder, *size);
                Ok(EncodeBuilder::FixedSizeList(
                    FixedSizeListBuilderWrapper::new(list_builder, field.clone(), *size),
                ))
            }
            LargeList(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = LargeListBuilder::new(inner_builder);
                Ok(EncodeBuilder::LargeList(LargeListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Struct variant
            Struct(fields) => {
                let field_builders: Vec<Box<EncodeBuilder>> = fields
                    .iter()
                    .map(|f| EncodeBuilder::try_new(f.data_type()).map(Box::new))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(EncodeBuilder::Struct(StructBuilderWrapper::new(
                    fields.clone(),
                    field_builders,
                )))
            }

            // Dictionary variant
            Dictionary(_key_type, value_type) => match value_type.as_ref() {
                Utf8 => {
                    let dict_builder = StringDictionaryBuilder::<Int32Type>::new();
                    Ok(EncodeBuilder::Dictionary(DictionaryBuilderWrapper::new(
                        dict_builder,
                    )))
                }
                _ => Err(EncodeBuilderError::UnsupportedDataType(data_type.clone())),
            },

            // Map variant
            Map(field, _sorted) => {
                if let Struct(map_fields) = field.data_type() {
                    if map_fields.len() == 2 {
                        let key_field = &map_fields[0];
                        let value_field = &map_fields[1];

                        let field_names = MapFieldNames {
                            entry: field.name().to_string(),
                            key: key_field.name().to_string(),
                            value: value_field.name().to_string(),
                        };

                        let key_builder = StringBuilder::new();
                        let value_builder =
                            Box::new(EncodeBuilder::try_new(value_field.data_type())?);
                        let map_builder =
                            MapBuilder::new(Some(field_names), key_builder, value_builder);

                        Ok(EncodeBuilder::Map(MapBuilderWrapper::new(
                            map_builder,
                            field.clone(),
                        )))
                    } else {
                        Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                    }
                } else {
                    Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                }
            }

            dt => Err(EncodeBuilderError::UnsupportedDataType(dt.clone())),
        }
    }

    pub fn try_with_capacity(
        data_type: &DataType,
        capacity: usize,
    ) -> Result<Self, EncodeBuilderError> {
        match data_type {
            Null => Ok(EncodeBuilder::Null(NullBuilder::new())),
            Boolean => Ok(EncodeBuilder::Bool(BooleanBuilder::new())),
            Int8 => Ok(EncodeBuilder::I8(Int8Builder::new())),
            Int16 => Ok(EncodeBuilder::I16(Int16Builder::new())),
            Int32 => Ok(EncodeBuilder::I32(Int32Builder::new())),
            Int64 => Ok(EncodeBuilder::I64(Int64Builder::new())),
            UInt8 => Ok(EncodeBuilder::UI8(UInt8Builder::new())),
            UInt16 => Ok(EncodeBuilder::UI16(UInt16Builder::new())),
            UInt32 => Ok(EncodeBuilder::UI32(UInt32Builder::new())),
            UInt64 => Ok(EncodeBuilder::UI64(UInt64Builder::new())),
            Float16 => Ok(EncodeBuilder::F16(Float16Builder::new())),
            Float32 => Ok(EncodeBuilder::F32(Float32Builder::new())),
            Float64 => Ok(EncodeBuilder::F64(Float64Builder::new())),
            // Timestamp variants
            Timestamp(Second, tz) => Ok(EncodeBuilder::TsSecond(
                TimestampSecondBuilder::new().with_timezone_opt(tz.clone()),
            )),
            Timestamp(Millisecond, tz) => Ok(EncodeBuilder::TsMs(
                TimestampMillisecondBuilder::new().with_timezone_opt(tz.clone()),
            )),
            Timestamp(Microsecond, tz) => Ok(EncodeBuilder::TsMicro(
                TimestampMicrosecondBuilder::new().with_timezone_opt(tz.clone()),
            )),
            Timestamp(Nanosecond, tz) => Ok(EncodeBuilder::TsNano(
                TimestampNanosecondBuilder::new().with_timezone_opt(tz.clone()),
            )),

            // Date variants
            Date32 => Ok(EncodeBuilder::Date32(Date32Builder::new())),
            Date64 => Ok(EncodeBuilder::Date64(Date64Builder::with_capacity(
                capacity,
            ))),

            // Time variants
            Time32(Second) => Ok(EncodeBuilder::Time32Second(Time32SecondBuilder::new())),
            Time32(Millisecond) => Ok(EncodeBuilder::Time32Ms(Time32MillisecondBuilder::new())),
            Time64(Microsecond) => Ok(EncodeBuilder::Time64Micro(Time64MicrosecondBuilder::new())),
            Time64(Nanosecond) => Ok(EncodeBuilder::Time64Nano(Time64NanosecondBuilder::new())),

            // Duration variants
            Duration(Second) => Ok(EncodeBuilder::DurationSecond(DurationSecondBuilder::new())),
            Duration(Millisecond) => {
                Ok(EncodeBuilder::DurationMs(DurationMillisecondBuilder::new()))
            }
            Duration(Microsecond) => Ok(EncodeBuilder::DurationMicro(
                DurationMicrosecondBuilder::new(),
            )),
            Duration(Nanosecond) => {
                Ok(EncodeBuilder::DurationNano(DurationNanosecondBuilder::new()))
            }

            // Interval variants
            Interval(YearMonth) => Ok(EncodeBuilder::IntervalYearMonth(
                IntervalYearMonthBuilder::new(),
            )),
            Interval(DayTime) => Ok(EncodeBuilder::IntervalDayTime(IntervalDayTimeBuilder::new())),
            Interval(MonthDayNano) => Ok(EncodeBuilder::IntervalMonthDayNano(
                IntervalMonthDayNanoBuilder::new(),
            )),

            // Binary variants
            Binary => Ok(EncodeBuilder::Binary(BinaryBuilder::with_capacity(
                capacity, 1024,
            ))),
            FixedSizeBinary(size) => Ok(EncodeBuilder::FixedSizeBinary(
                FixedSizeBinaryBuilder::with_capacity(capacity, *size),
            )),
            LargeBinary => Ok(EncodeBuilder::LargeBinary(
                LargeBinaryBuilder::with_capacity(capacity, 1024),
            )),
            BinaryView => Ok(EncodeBuilder::BinaryView(BinaryViewBuilder::with_capacity(
                capacity,
            ))),

            // String variants
            Utf8 => Ok(EncodeBuilder::Str(StringBuilder::with_capacity(
                capacity, 1024,
            ))),
            LargeUtf8 => Ok(EncodeBuilder::LargeStr(LargeStringBuilder::with_capacity(
                capacity, 1024,
            ))),
            Utf8View => Ok(EncodeBuilder::StrView(StringViewBuilder::with_capacity(
                capacity,
            ))),

            // List variants
            List(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder = ListBuilder::with_capacity(inner_builder, capacity);
                Ok(EncodeBuilder::List(ListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Additional List variants
            FixedSizeList(field, size) => {
                let inner_builder = Box::new(EncodeBuilder::try_new(field.data_type())?);
                let list_builder =
                    FixedSizeListBuilder::with_capacity(inner_builder, *size, capacity);
                Ok(EncodeBuilder::FixedSizeList(
                    FixedSizeListBuilderWrapper::new(list_builder, field.clone(), *size),
                ))
            }
            LargeList(field) => {
                let inner_builder = Box::new(EncodeBuilder::try_with_capacity(
                    field.data_type(),
                    capacity,
                )?);
                let list_builder = LargeListBuilder::with_capacity(inner_builder, capacity);
                Ok(EncodeBuilder::LargeList(LargeListBuilderWrapper::new(
                    list_builder,
                    field.clone(),
                )))
            }

            // Decimal variants - same as try_new since they don't use capacity
            #[cfg(feature = "arrow-56")]
            Decimal32(precision, scale) => Ok(EncodeBuilder::Decimal32(
                Decimal32Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal128(precision, scale) => Ok(EncodeBuilder::Decimal128(
                Decimal128Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),
            Decimal256(precision, scale) => Ok(EncodeBuilder::Decimal256(
                Decimal256Builder::new().with_precision_and_scale(*precision, *scale)?,
            )),

            // Struct variant
            Struct(fields) => {
                let field_builders: Vec<Box<EncodeBuilder>> = fields
                    .iter()
                    .map(|f| {
                        EncodeBuilder::try_with_capacity(f.data_type(), capacity).map(Box::new)
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(EncodeBuilder::Struct(StructBuilderWrapper::new(
                    fields.clone(),
                    field_builders,
                )))
            }

            // Dictionary variant - use capacity
            Dictionary(_key_type, value_type) => match value_type.as_ref() {
                Utf8 => {
                    let dict_builder = StringDictionaryBuilder::<Int32Type>::with_capacity(
                        capacity, capacity, capacity,
                    );
                    Ok(EncodeBuilder::Dictionary(DictionaryBuilderWrapper::new(
                        dict_builder,
                    )))
                }
                _ => Err(EncodeBuilderError::UnsupportedDataType(data_type.clone())),
            },

            // Map variant - use capacity
            Map(field, _sorted) => {
                if let Struct(map_fields) = field.data_type() {
                    if map_fields.len() == 2 {
                        let key_field = &map_fields[0];
                        let value_field = &map_fields[1];

                        // Create field names that match the schema
                        let field_names = MapFieldNames {
                            entry: field.name().to_string(),
                            key: key_field.name().to_string(),
                            value: value_field.name().to_string(),
                        };

                        let key_builder = StringBuilder::with_capacity(capacity, capacity * 10);
                        let value_builder = Box::new(EncodeBuilder::try_with_capacity(
                            value_field.data_type(),
                            capacity,
                        )?);
                        let map_builder = MapBuilder::with_capacity(
                            Some(field_names),
                            key_builder,
                            value_builder,
                            capacity,
                        );

                        Ok(EncodeBuilder::Map(MapBuilderWrapper::new(
                            map_builder,
                            field.clone(),
                        )))
                    } else {
                        Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                    }
                } else {
                    Err(EncodeBuilderError::UnsupportedDataType(data_type.clone()))
                }
            }

            dt => Err(EncodeBuilderError::UnsupportedDataType(dt.clone())),
        }
    }
}

```
### "./src/json.rs"
```rs
use arrow::{
    array::{ArrayBuilder, ArrayRef},
    datatypes::{DataType as DT, Field, IntervalUnit, TimeUnit},
    error::ArrowError,
};
use base64::Engine;
use chrono::{FixedOffset, NaiveDateTime, TimeZone, Timelike};
use half::f16;
use serde_json::Value as Json;
use std::sync::Arc;

use super::{ColumnEncoder, ColumnFactory, EncodeBuilder as EB, EncodeBuilderError};

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

pub struct JsonColumn {
    builder: EB,
    field: Arc<Field>,
}

impl ColumnEncoder<Json> for JsonColumn {
    type Error = JsonEncodeError;

    fn append(&mut self, row: &Json) -> Result<(), Self::Error> {
        // pick value for this column from the row by field name
        let v = row.get(self.field.name()).unwrap_or(&Json::Null);
        append_json(&mut self.builder, &self.field, v)
    }

    fn finish(&mut self) -> ArrayRef {
        ArrayBuilder::finish(&mut self.builder)
    }
}

pub struct JsonFactory;

impl ColumnFactory<Json> for JsonFactory {
    type Error = JsonEncodeError;
    type Col = JsonColumn;

    fn make(field: &Field, capacity: usize, _ctx: &()) -> Result<Self::Col, Self::Error> {
        let builder = EB::try_with_capacity(field.data_type(), capacity)?;
        Ok(JsonColumn {
            builder,
            field: Arc::new(field.clone()),
        })
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
            // Assume Unix timestamp in seconds
            let timestamp = n.as_i64().ok_or_else(|| {
                JsonEncodeError::InvalidDateValue("Invalid timestamp number".to_string())
            })?;
            chrono::DateTime::from_timestamp(timestamp, 0)
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
        values: &[Json],
    ) -> Result<Vec<RecordBatch>, JsonEncodeError> {
        // Turn the slice into an iterator of Json rows
        let rows = values.iter().cloned();

        // Build encoder and collect all batches
        let batches =
            FieldBatchEncoder::<Json, JsonColumn, _>::from_schema_with::<JsonFactory, ()>(
                schema,
                rows,
                1024,
                &(),
            )?
            .collect::<Result<Vec<_>, JsonEncodeError>>()?;

        let row_count: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(row_count, values.len(), "Batch count mismatch");
        Ok(batches)
    }
}

```
### "./src/sqlx/mod.rs"
```rs
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
        let raw = row.try_get_raw(self.col_name.as_str()).ok();
        let bytes = raw.as_ref().and_then(|v| v.as_bytes().ok());
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

```
### "./src/postgres.rs"
```rs
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
        (EB::I8(_), _) => unimplemented!("I8"),
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
        (EB::Str(b), "numeric") => b.append_value(numeric_from_sql(bytes)?.to_string()),
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
            let value = from_decoder(line_from_sql, bytes)?;
            b.append_value(format!("<({}, {}), {}>", value.0, value.1, value.2))
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
            let range = display_range_from_sql(tp, bytes, |byts| {
                numeric_from_sql(byts).map(|n| n.to_string())
            })?;
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
            "date" => b.append_value(from_sql::<i32>(tp, bytes)?),
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
fn numeric_from_sql(bytes: &[u8]) -> Result<i128, PgErr> {
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

```
